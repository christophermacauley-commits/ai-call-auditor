import os
import time
import subprocess
import re
import sqlite3
import shutil
from faster_whisper import WhisperModel

CALLS_FOLDER = "calls"
TRANSCRIPTS_FOLDER = "transcripts"
REPORTS_FOLDER = "reports"
DB_FILE = "calls.db"

WHISPER_MODEL = "small"
OLLAMA_MODEL = "llama3.2:3b"
SCAN_INTERVAL_SECONDS = 5
OLLAMA_TIMEOUT_SECONDS = 300
AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a")

TRANSCRIPTION_START_PROGRESS = 5
TRANSCRIPTION_DONE_PROGRESS = 75
AI_START_PROGRESS = 80
AI_DONE_PROGRESS = 95

model = None


def read_text(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def clean_text(text):
    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)


def get_model():
    global model
    if model is None:
        model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8"
        )
    return model


def ensure_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        call_name TEXT,
        transcript TEXT,
        report TEXT,
        score INTEGER,
        risk TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS processing_state (
        call_name TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        status TEXT NOT NULL,
        progress INTEGER NOT NULL DEFAULT 0,
        message TEXT,
        attempts INTEGER NOT NULL DEFAULT 0,
        error TEXT,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def set_processing_state(call_name, filename, status, progress=0, message=None, error=None):
    ensure_db()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("SELECT attempts FROM processing_state WHERE call_name=?", (call_name,))
    row = c.fetchone()
    attempts = row[0] if row else 0

    if status in ("processing", "retry"):
        attempts += 1

    if row:
        c.execute("""
            UPDATE processing_state
            SET filename=?,
                status=?,
                progress=?,
                message=?,
                attempts=?,
                error=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE call_name=?
        """, (filename, status, int(progress), message, attempts, error, call_name))
    else:
        c.execute("""
            INSERT INTO processing_state
                (call_name, filename, status, progress, message, attempts, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (call_name, filename, status, int(progress), message, attempts, error))

    conn.commit()
    conn.close()


def save_to_db(call_name, transcript, report, score, risk):
    ensure_db()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("DELETE FROM calls WHERE call_name=?", (call_name,))
    c.execute("""
        INSERT INTO calls (call_name, transcript, report, score, risk)
        VALUES (?, ?, ?, ?, ?)
    """, (call_name, transcript, report, score, risk))

    conn.commit()
    conn.close()


def parse_report(report_text):
    score = None
    risk = None

    for line in report_text.splitlines():
        upper = line.upper().strip()

        if upper.startswith("SCORE:"):
            nums = re.findall(r"\d+", line)
            if nums:
                score = int(nums[0])

        if upper.startswith("RISK:"):
            if "HIGH" in upper:
                risk = "HIGH"
            elif "MEDIUM" in upper:
                risk = "MEDIUM"
            elif "LOW" in upper:
                risk = "LOW"

    return score, risk


def get_ollama_command():
    candidates = [
        "/usr/local/bin/ollama",
        "/opt/homebrew/bin/ollama",
        shutil.which("ollama"),
    ]

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    raise FileNotFoundError("Ollama executable not found.")


def run_ollama(prompt):
    ollama = get_ollama_command()

    result = subprocess.run(
        [ollama, "run", OLLAMA_MODEL],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=OLLAMA_TIMEOUT_SECONDS
    )

    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"Ollama failed: {error}")

    return clean_text(result.stdout.decode("utf-8", errors="ignore"))


def transcribe(file_path, call_name, filename):
    set_processing_state(call_name, filename, "transcribing", 5, "Transcribing audio")

    segments, info = get_model().transcribe(
        file_path,
        beam_size=1,
        vad_filter=True
    )

    duration = float(getattr(info, "duration", 0) or 0)
    transcript_parts = []
    last_progress = 5

    for segment in segments:
        transcript_parts.append(segment.text)

        if duration > 0:
            end_time = float(getattr(segment, "end", 0) or 0)
            ratio = max(0, min(1, end_time / duration))
            progress = 5 + int(ratio * 70)

            if progress >= last_progress + 3:
                set_processing_state(call_name, filename, "transcribing", progress, "Transcribing audio")
                last_progress = progress

    set_processing_state(call_name, filename, "transcribing", 75, "Transcription complete")

    return "\n".join(transcript_parts).strip()


def audit(transcript, progress_callback=None):
    if progress_callback:
        progress_callback(AI_START_PROGRESS, "Running AI audit")

    checklist = read_text("training/sales_task_checklist.txt")
    rubric = read_text("training/scoring_rubric.txt")
    output_format = read_text("training/audit_output_format.txt")

    prompt = f"""
You are a strict QA auditor for final expense sales calls.

RULES:
- Return ONE audit only.
- Start with SCORE.
- Do not add extra commentary.
- If a stage was not reached, mark items as NOT REACHED.
- Do NOT assume future steps.
- Do NOT copy the template literally.
- Replace every placeholder with an actual audit value.
- SCORE must be a real number like 40, not "0-100".
- RISK must be one value only: LOW, MEDIUM, or HIGH.
- PASS must be one value only: YES or NO.
- Every checklist item must be answered with YES, NO, UNCLEAR, or NOT REACHED.
- Only evaluate what actually happened in the transcript.

COMPANY STAGE ORDER:
1. PQ / Handoff
2. Who I Am / What I Do
3. Product intro / basic plan explanation
4. License number
5. Warm-up / Rapport
6. Health
7. Need
8. Share a story about yourself
9. Features / Benefits
10. Change Up
11. Quotes
12. Close / Client chooses option
13. Application information
14. Payment Date
15. Collect Banking / Payment Setup
16. Disclosures
17. Voice Signature
18. Filling out application
19. Peace of Mind
20. Cooldown

STAGE RULES:
- CALL STAGE REACHED must be the SINGLE furthest stage clearly reached.
- Do NOT guess or skip ahead.
- EARLY END must be YES unless Close / Client chooses option or later was reached.
- If transcript includes recording disclosure, agent says they are a state-licensed field underwriter, or explains what they do, CALL STAGE REACHED must be at least Who I Am / What I Do.
- If transcript includes explanation of final expense plans providing money for burial/cremation or basic plan purpose, CALL STAGE REACHED must be at least Product intro / basic plan explanation.
- Do NOT mark License number reached unless an actual license number is clearly spoken.
- If CALL STAGE REACHED is PQ / Handoff, Who I Am / What I Do, Product intro / basic plan explanation, License number, Warm-up / Rapport, Health, Need, Features / Benefits, Change Up, or Quotes, then EARLY END must be YES.
- NOT REACHED must list ONLY stages AFTER the stage reached.
- Do NOT include earlier stages in NOT REACHED.
- Do not count future script requirements as completed.

CHECKLIST RULES:
- Only mark YES if that stage was clearly completed.
- If a stage was not reached -> mark NOT REACHED.
- Do NOT mark YES for steps that occur later in the script.
- Do NOT infer or assume actions.

CHECKLIST MAPPING:
- Recording disclosure = Who I Am / What I Do
- Agent introduction = Who I Am / What I Do
- License number = License number
- Warm up / rapport = Warm-up / Rapport
- Fact finding = Warm-up / Rapport
- Existing coverage asked = Need
- Beneficiary identified = Application information
- Need amount discussed = Need
- Health questions completed = Health
- Product benefits explained = Features / Benefits
- Three options presented = Quotes
- Client chose an option = Close / Client chooses option
- Application info collected = Application information
- Payment date explained = Payment Date
- Banking/payment setup explained = Collect Banking / Payment Setup
- Peace of mind completed = Peace of Mind
- Cool down completed = Cooldown

IMPORTANT LOGIC:
- Do NOT count basic product explanation as full Features / Benefits.
- Do NOT count early script sections as Need.
- Do NOT count health questions unless Health stage was reached.
- Do NOT count existing coverage unless Need stage was reached.
- Do NOT count application steps unless Application stage was reached.
- Do NOT count payment/banking unless those stages were reached.

EARLY CALL RULE:
If the call includes:
- PQ handoff
- Agent introduction (Who I Am / What I Do)
- Product intro / explanation

BUT does NOT reach Warm-up / Rapport:

THEN:
- CALL STAGE REACHED = Product intro / basic plan explanation OR License number (if license clearly given)
- EARLY END = YES
- PASS = NO
- SCORE must NOT exceed 40
- Everything from Warm-up / Rapport forward must be NOT REACHED

SALES TASK CHECKLIST:
{checklist}

SCORING RUBRIC:
{rubric}

REQUIRED OUTPUT FORMAT:
{output_format}

TRANSCRIPT:
{transcript}

Start EXACTLY like:
SCORE:
RISK:
PASS:
"""

    report = run_ollama(prompt).strip()

    score_index = report.upper().find("SCORE:")
    if score_index > 0:
        report = report[score_index:].strip()

    unwanted_markers = [
        "**CONVERSATION QUALITY",
        "CONVERSATION QUALITY:",
        "**RELEVANCE",
        "RELEVANCE:",
        "**NOTABLE POINTS",
        "NOTABLE POINTS:",
        "**IMPROVEMENT AREAS",
        "IMPROVEMENT AREAS:",
        "OVERALL,",
    ]

    upper_report = report.upper()
    cut_positions = []

    for marker in unwanted_markers:
        pos = upper_report.find(marker)
        if pos != -1:
            cut_positions.append(pos)

    if cut_positions:
        report = report[:min(cut_positions)].strip()

    stage_order = [
        "PQ / Handoff",
        "Who I Am / What I Do",
        "Product intro / basic plan explanation",
        "License number",
        "Warm-up / Rapport",
        "Health",
        "Need",
        "Share a story about yourself",
        "Features / Benefits",
        "Change Up",
        "Quotes",
        "Close / Client chooses option",
        "Application information",
        "Payment Date",
        "Collect Banking / Payment Setup",
        "Disclosures",
        "Voice Signature",
        "Filling out application",
        "Peace of Mind",
        "Cooldown",
    ]
    stage_index = {name.upper(): i for i, name in enumerate(stage_order)}

    score_match = re.search(r"(?im)^SCORE:\s*(\d+)\b", report)
    stage_match = re.search(r"(?im)^CALL STAGE REACHED:\s*(.+)$", report)
    score_value = int(score_match.group(1)) if score_match else None

    stage_value_raw = stage_match.group(1).strip() if stage_match else ""
    stage_value_upper = stage_value_raw.upper()
    current_stage_idx = stage_index.get(stage_value_upper, 0)

    transcript_upper = transcript.upper()

    if any(token in transcript_upper for token in (
        "STATE-LICENSED FIELD UNDERWRITER",
        "STATE LICENSED FIELD UNDERWRITER",
        "FINAL EXPENSE DEPARTMENT",
        "WHAT THAT MEANS",
    )):
        current_stage_idx = max(current_stage_idx, stage_index["WHO I AM / WHAT I DO"])

    if any(token in transcript_upper for token in (
        "FINAL EXPENSE PLANS",
        "BURIAL OR CREMATION",
        "PROVIDE MONEY TO YOUR FAMILY",
        "DESIGNED TO PROVIDE MONEY",
        "PERMANENT AND WILL NEVER CHANGE",
    )):
        current_stage_idx = max(current_stage_idx, stage_index["PRODUCT INTRO / BASIC PLAN EXPLANATION"])

    has_license_phrase = "LICENSE NUMBER IS" in transcript_upper
    has_license_pattern = re.search(
        r"\bLICENSE(?:\s+NUMBER)?\s*(?:IS|#|NO\.?|NUMBER:)?\s*[A-Z]?\d{4,}\b",
        transcript_upper
    ) is not None
    if has_license_phrase or has_license_pattern:
        current_stage_idx = max(current_stage_idx, stage_index["LICENSE NUMBER"])

    forced_stage = stage_order[current_stage_idx]
    if stage_match:
        report = re.sub(r"(?im)^CALL STAGE REACHED:\s*.+$", f"CALL STAGE REACHED: {forced_stage}", report, count=1)
    else:
        report = re.sub(r"(?im)^(PASS:\s*.+)$", rf"\1\nCALL STAGE REACHED: {forced_stage}", report, count=1)

    not_reached_block = "NOT REACHED:\n" + "\n".join(f"- {stage}" for stage in stage_order[current_stage_idx + 1:])
    report = re.sub(
        r"(?ims)^NOT REACHED:\s*(?:\n-.*?)*(?=\n[A-Z][A-Z /]+:|\Z)",
        not_reached_block,
        report,
        count=1
    )

    close_index = stage_index["CLOSE / CLIENT CHOOSES OPTION"]
    close_or_application_reached = current_stage_idx >= close_index
    early_end_value = "NO" if close_or_application_reached else "YES"
    if re.search(r"(?im)^EARLY END:\s*(YES|NO)\s*$", report):
        report = re.sub(r"(?im)^EARLY END:\s*(YES|NO)\s*$", f"EARLY END: {early_end_value}", report, count=1)
    else:
        report = re.sub(
            r"(?im)^(CALL STAGE REACHED:\s*.+)$",
            rf"\1\nEARLY END: {early_end_value}",
            report,
            count=1
        )

    if not close_or_application_reached and score_value is not None and score_value > 40:
        report = re.sub(r"(?im)^SCORE:\s*\d+\b", "SCORE: 40", report, count=1)
        score_value = 40

    if (score_value is not None and score_value < 70) or not close_or_application_reached:
        report = re.sub(r"(?im)^PASS:\s*YES\s*$", "PASS: NO", report)

    if progress_callback:
        progress_callback(AI_DONE_PROGRESS, "Saving audit report")

    return report


def wait_until_file_ready(file_path, checks=3, delay=1):
    last_size = -1

    for _ in range(checks):
        if not os.path.exists(file_path):
            return False

        size = os.path.getsize(file_path)

        if size == last_size and size > 0:
            return True

        last_size = size
        time.sleep(delay)

    return os.path.exists(file_path) and os.path.getsize(file_path) > 0


def process_file(file_path):
    filename = os.path.basename(file_path)

    if not filename.lower().endswith(AUDIO_EXTENSIONS):
        return

    call_name = os.path.splitext(filename)[0]

    transcript_path = os.path.join(TRANSCRIPTS_FOLDER, f"{call_name}.txt")
    report_path = os.path.join(REPORTS_FOLDER, f"{call_name}_report.txt")

    if os.path.exists(report_path):
        set_processing_state(call_name, filename, "complete", 100, "Complete")
        return

    if not wait_until_file_ready(file_path):
        return

    try:
        print(f"Processing: {call_name}", flush=True)

        set_processing_state(call_name, filename, "processing", 1, "Starting processing")

        if os.path.exists(transcript_path):
            transcript = read_text(transcript_path)
            print(f"Resuming from transcript: {call_name}", flush=True)
        else:
            transcript = transcribe(file_path, call_name, filename)
            write_text(transcript_path, transcript)

        set_processing_state(call_name, filename, "analyzing", 80, "Running AI audit")

        report = audit(
            transcript,
            lambda progress, message: set_processing_state(
                call_name,
                filename,
                "analyzing",
                progress,
                message
            )
        )

        write_text(report_path, report)

        score, risk = parse_report(report)

        save_to_db(call_name, transcript, report, score, risk)

        set_processing_state(call_name, filename, "complete", 100, "Complete")

        print(f"Done: {call_name} | Score: {score} | Risk: {risk}", flush=True)

    except Exception as e:
        error = str(e)
        print(f"ERROR processing {call_name}: {error}", flush=True)

        transcript = read_text(transcript_path)

        failure_report = f"""SCORE: 0
RISK: HIGH
PASS: NO

CALL STAGE REACHED: Processing failed
EARLY END: YES
NOT REACHED:
- Unable to evaluate

COMPLIANCE FAILURES:
- Processing failed

SCRIPT / FLOW MISSES:
- Unable to complete audit

TASK CHECKLIST:
- Unable to evaluate

COACHING:
- Retry this audit or delete and re-upload the call.

SUMMARY:
The system could not complete the audit because: {error}

TRANSCRIPT:
{transcript}
"""

        write_text(report_path, failure_report)
        save_to_db(call_name, transcript, failure_report, 0, "HIGH")
        set_processing_state(call_name, filename, "failed", 100, "Processing failed", error)


def recover_interrupted_work():
    ensure_db()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
        UPDATE processing_state
        SET status='retry',
            progress=0,
            message='Recovered after restart',
            error='Recovered after watcher restart',
            updated_at=CURRENT_TIMESTAMP
        WHERE status IN ('processing', 'transcribing', 'analyzing')
    """)

    conn.commit()
    conn.close()


def start():
    os.makedirs(CALLS_FOLDER, exist_ok=True)
    os.makedirs(TRANSCRIPTS_FOLDER, exist_ok=True)
    os.makedirs(REPORTS_FOLDER, exist_ok=True)

    ensure_db()
    recover_interrupted_work()

    print("QA SYSTEM RUNNING...", flush=True)
    print(f"Scanning: {CALLS_FOLDER}", flush=True)
    print(f"Every {SCAN_INTERVAL_SECONDS} seconds", flush=True)
    print("Press CTRL+C to stop.", flush=True)

    while True:
        try:
            files = sorted(os.listdir(CALLS_FOLDER))

            for filename in files:
                if not filename.lower().endswith(AUDIO_EXTENSIONS):
                    continue

                file_path = os.path.join(CALLS_FOLDER, filename)

                if os.path.isfile(file_path):
                    process_file(file_path)

            time.sleep(SCAN_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("Watcher stopped.", flush=True)
            break

        except Exception as e:
            print(f"WATCHER ERROR: {e}", flush=True)
            time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    start()
