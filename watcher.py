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
- Do not summarize before SCORE.
- Do not add Conversation Quality, Relevance, Notable Points, or extra sections.
- If something is not clearly stated in the transcript, mark it NO or UNCLEAR.
- Never assume bank verification, insurance verification, application info, payment setup, or close unless clearly stated.
- Do not require exact script wording.
- Grade based on intent, compliance, process, and Straight Line sales execution.

CALL STAGE:
First identify the furthest stage reached:
PQ / Handoff, Opening, Who I Am / What I Do, Warm-up / Rapport, Health, Need, Features / Benefits, Quotes, Close, Application, Payment, Banking, Peace of Mind, Cool Down.

If Close or Application was NOT reached:
- PASS must be NO
- SCORE must NOT exceed 55

If Health was NOT reached:
- SCORE must NOT exceed 45

If the call only reached Opening or Warm-up:
- SCORE must NOT exceed 40

If the call ended early:
- EARLY END must be YES.
- Do not give credit for stages not reached.
- Do not heavily penalize stages not reached.
- A call that does not reach Close or Application should normally be PASS: NO.

CHECKLIST:
{checklist}

RUBRIC:
{rubric}

REQUIRED FORMAT:
{output_format}

TRANSCRIPT:
{transcript}

Return exactly one audit. Start exactly like this:

SCORE: <number>
RISK: <LOW/MEDIUM/HIGH>
PASS: <YES/NO>
"""

    report = run_ollama(prompt).strip()

    # Remove anything before SCORE if the model adds intro text
    score_index = report.upper().find("SCORE:")
    if score_index > 0:
        report = report[score_index:].strip()

    # Remove common unwanted extra sections if the model appends them
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

    if progress_callback:
        progress_callback(AI_DONE_PROGRESS, "Saving audit report")

    return report

    report = run_ollama(prompt)

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
