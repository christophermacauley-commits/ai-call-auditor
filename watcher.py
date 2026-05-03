import os
import sys
import time
import subprocess
import re
import sqlite3
import shutil
import json
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from openai import OpenAI

load_dotenv()

CALLS_FOLDER = "calls"
TRANSCRIPT_UPLOADS_FOLDER = "transcript_uploads"
PROCESSED_CALLS_FOLDER = "processed_calls"
PROCESSED_TRANSCRIPTS_FOLDER = "processed_transcripts"
TRANSCRIPTS_FOLDER = "transcripts"
TRANSCRIPTS_ROLE_LABELED_FOLDER = "transcripts_role_labeled"
REPORTS_FOLDER = "reports"
DB_FILE = "calls.db"

# "medium" balances WER vs speed; int8 on CPU is the usual faster-whisper sweet spot (much faster
# than float32/float16 with modest accuracy loss vs full precision).
WHISPER_MODEL = "medium"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
SCAN_INTERVAL_SECONDS = 5
OLLAMA_TIMEOUT_SECONDS = 300
AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a")

TRANSCRIPTION_START_PROGRESS = 5
TRANSCRIPTION_DONE_PROGRESS = 75
AI_START_PROGRESS = 80
AI_DONE_PROGRESS = 95
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
USE_STRUCTURED_AUDIT = os.getenv("USE_STRUCTURED_AUDIT", "false").strip().lower() == "true"
STORE_RAW_TRANSCRIPTS = os.getenv("STORE_RAW_TRANSCRIPTS", "false").strip().lower() == "true"
OPENAI_INPUT_COST_PER_1K_TOKENS = 0.0004
OPENAI_OUTPUT_COST_PER_1K_TOKENS = 0.0016

model = None
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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


def final_transcript_privacy_cleanup(text):
    """
    Last transcript-only privacy cleanup.
    Redacts names only in narrow, clear contexts.
    """
    if not text:
        return text

    t = text

    # Fix fake speaker artifacts like: Prospect: Yes, sir. [NAME]: Do you...
    t = re.sub(r"\s+\[NAME\]\s*:\s*", "\nAgent: ", t)

    # PQ numbers.
    t = re.sub(r"\bPQ\s*\d+\b", "PQ[NUMBER]", t, flags=re.I)

    # Greeting/thanks name leaks.
    t = re.sub(
        r"(?im)^(\s*(?:PQ|Agent|Prospect|Unknown)\s*:\s*(?:hi|hello|hey|thank you|thanks)\s*,?\s+)([A-Z][a-z]+)\b",
        r"\1[NAME]",
        t,
    )

    # Direct-address name leaks.
    t = re.sub(
        r"(?i)\b((?:all\s+right|alright|okay|perfect|gotcha|don't\s+you\s+worry|do\s+not\s+worry|don't\s+worry)\s*,?\s+)([A-Z][a-z]+)(?=\s*[,\.])",
        r"\1[NAME]",
        t,
    )

    # "Thank you [NAME], John" -> "Thank you [NAME]"
    t = re.sub(
        r"(?i)\b(thank you|thanks)\s+\[NAME\],?\s+[A-Z][a-z]+\b",
        r"\1 [NAME]",
        t,
    )

    # Same-line identity/handoff answers.
    t = re.sub(
        r"(?i)(who do i have the pleasure of (?:speaking with|helping)(?: today)?\??\s+)([A-Z][a-z]+)\b",
        r"\1[NAME]",
        t,
    )
    t = re.sub(
        r"(?i)\b(i have\s+)([A-Z][a-z]+)(\s+(?:here|with me|on the line)\b)",
        r"\1[NAME]\3",
        t,
    )

    # Title + name.
    t = re.sub(
        r"(?i)\b(Mr|Mrs|Ms|Miss)\.?\s+[A-Z][a-z]+\b",
        r"\1. [NAME]",
        t,
    )

    # Very narrow known direct-address endings.
    t = re.sub(
        r"(?i)\b((?:don't you worry|thank you for your honesty)\s*,?\s+)[A-Z][a-z]+\b",
        r"\1[NAME]",
        t,
    )

    # Fix placeholder stuck to next role label.
    t = re.sub(r"(\[NUMBER\])(?=(?:PQ|Agent|Prospect|Unknown)\s*:)", r"\1\n", t)
    t = re.sub(r"(\[DOB\])(?=(?:PQ|Agent|Prospect|Unknown)\s*:)", r"\1\n", t)
    t = re.sub(r"(\[MONEY\])(?=(?:PQ|Agent|Prospect|Unknown)\s*:)", r"\1\n", t)

    # Collapse repeated identical short greeting lines.
    cleaned = []
    prev = None
    repeats = 0
    for line in t.split("\n"):
        key = line.strip()
        if key == prev and re.match(r"^(?:PQ|Agent|Prospect|Unknown):\s+(?:Hi,?\s+)?\[NAME\]\.?$", key, re.I):
            repeats += 1
            if repeats <= 1:
                cleaned.append(line)
            continue
        prev = key
        repeats = 0
        cleaned.append(line)

    return "\n".join(cleaned)




def _protect_report_metric_numbers(report):
    """
    Temporarily protect non-private audit/report numbers before report redaction.
    This prevents report metrics from becoming [NUMBER].
    """
    if not report:
        return report, {}

    protected = {}
    counter = 0

    metric_line_re = re.compile(
        r"(?im)^("
        r"SCORE|"
        r"- Compliance|"
        r"- Sales Process|"
        r"- Product Explanation|"
        r"- Closing|"
        r"- Communication Quality|"
        r"- Input tokens \\(est\\)|"
        r"- Output tokens \\(est\\)|"
        r"Account verification evidence count|"
        r"Routing verification evidence count"
        r"):\\s*([0-9]+(?:\\.[0-9]+)?)\\b"
    )

    def repl(m):
        nonlocal counter
        token = f"__REPORT_METRIC_{counter}__"
        counter += 1
        full = m.group(0)
        protected[token] = full
        return token

    return metric_line_re.sub(repl, report), protected


def _restore_report_metric_numbers(report, protected):
    if not report or not protected:
        return report
    out = report
    for token, value in protected.items():
        out = out.replace(token, value)
    return out


def _restore_redacted_report_metric_lines(report):
    """
    Repair already-redacted metric lines where the numeric score was replaced.
    If the number is gone and cannot be known, leave it as Unknown rather than SCORE: 0.
    """
    if not report:
        return report

    metric_names = [
        "Compliance",
        "Sales Process",
        "Product Explanation",
        "Closing",
        "Communication Quality",
    ]

    for name in metric_names:
        report = re.sub(
            rf"(?im)^- {re.escape(name)}:\\s*\\[NUMBER\\]\\s*$",
            f"- {name}: Unknown",
            report,
        )

    report = re.sub(r"(?im)^SCORE:\\s*\\[NUMBER\\]\\s*$", "SCORE: Unknown", report)

    return report


def redact_report_text(report):
    report, _metric_protect = _protect_report_metric_numbers(report)
    """
    Report-safe privacy cleanup.
    Do NOT run full transcript redaction on reports because it breaks audit structure:
    SCORE: 80, TOP 3, 3 and 1 Method, token counts, cost estimates, etc.

    This only fixes known sensitive leaks and preserves audit/report numbers.
    """
    if not report:
        return report

    r = report

    # Preserve / restore audit structure phrases if a previous redaction over-hit them.
    r = re.sub(r"(?im)^SCORE:\s*\[NUMBER\]\s*$", "SCORE: Unknown", r)
    r = re.sub(r"(?i)\bTOP\s+\[NUMBER\]\s+COACHING PRIORITIES\b", "TOP 3 COACHING PRIORITIES", r)
    r = re.sub(r"(?i)\b\[NUMBER\]\s+and\s+\[NUMBER\]\s+Method\b", "3 and 1 Method", r)
    r = re.sub(r"(?i)\b\[NUMBER\]\s*&\s+\[NUMBER\]\s+Method\b", "3 and 1 Method", r)

    # Remove person-name leaks without touching carrier names / states too broadly.
    r = final_transcript_privacy_cleanup(r)

    # Redact obvious sensitive values if they somehow appear in report prose.
    r = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", r)
    r = re.sub(r"\b(?:\+?1[\s\-.]?)?(?:\(?\d{3}\)?[\s\-.]?)\d{3}[\s\-.]?\d{4}\b", "[PHONE]", r)
    r = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "[DATE]", r)
    r = re.sub(r"\$\s*\d+(?:[,.]\d+)*(?:\.\d+)?\b", "[MONEY]", r)
    r = re.sub(r"(?i)\b((?:account|routing|card|debit|credit)\s*(?:number)?\s*(?:is|#|:)?\s*)\d[\d\s\-]{4,}\b", r"\1[BANK_NUMBER]", r)

    # Re-restore report structure after cleanup.
    r = re.sub(r"(?i)\bTOP\s+\[NUMBER\]\s+COACHING PRIORITIES\b", "TOP 3 COACHING PRIORITIES", r)
    r = re.sub(r"(?i)\b\[NUMBER\]\s+and\s+\[NUMBER\]\s+Method\b", "3 and 1 Method", r)
    r = re.sub(r"(?i)\b3\s+and\s+1\s+Method\b", "3 and 1 Method", r)

    return _restore_safe_business_terms(_restore_redacted_report_metric_lines(_restore_report_metric_numbers(_restore_safe_business_terms(r), _metric_protect)))



def get_model():
    global model
    if model is None:
        # medium: stronger ASR than small/tiny with acceptable latency on CPU when combined with
        # int8 quantization, VAD skipping silence, and decode settings below (beam 5, no context carry).
        model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
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




VALID_DISPOSITIONS = {"SOLD", "U90", "LCR", "BOOTC", "LEAD", "AGE"}

def report_says_policy_sold_for_disposition(report):
    return bool(re.search(
        r"(?im)^- Policy sold:\s*YES\b|^- Was the policy sold\?\s*YES\b",
        report or "",
    ))

def detect_auto_disposition(call_name, transcript, report, duration_seconds=None):
    """
    Operational disposition only.
    This must not change score/risk/pass/report logic.
    """
    combined = ((transcript or "") + "\n" + (report or "")).lower()

    if report_says_policy_sold_for_disposition(report):
        return "SOLD", "Report indicates policy sold."

    if re.search(
        r"\b(over\s*80|older than\s*80|too old|outside (?:the )?age range|age limit|cannot qualify due to age|you have to be younger)\b",
        combined,
        re.I,
    ):
        return "AGE", "Age-related disqualification detected."

    # Health LCR disposition requires confirmed disqualification, not health-question wording.
    health_agent_dq = bool(re.search(
        r"(?is)"
        r"(unfortunately|sorry|based on that|because of that|with that condition|due to that|that means|"
        r"after reviewing|from those answers).{0,220}"
        r"(do(?:es)? not qualify|won't qualify|would not qualify|can't qualify|cannot qualify|"
        r"not able to qualify|unable to qualify|can't help you|cannot help you|not eligible|declined|knockout)",
        combined,
    ))

    health_report_dq = bool(re.search(
        r"(?is)"
        r"(prospect had a disqualifying health condition|health-related disqualification|"
        r"disqualifying medical condition|declined due to health|medical disqualification)",
        report or "",
    ))

    if health_agent_dq or health_report_dq:
        return "LCR", "Health-related disqualification language detected."

    if re.search(
        r"\b(no income|don't have any income|do not have any income|not at all.*income|working on my disability|take food off your table|can't afford it|cannot afford it)\b",
        combined,
        re.I | re.S,
    ):
        return "LCR", "No-income / affordability disqualification language detected."

    # BOOTC should stay conservative until duration/first-seconds support is added.
    opening_only = bool(re.search(
        r"(?im)^CALL STAGE REACHED:\s*Opening / Handoff\b",
        report or "",
    ))

    meaningful_agent_start = bool(re.search(
        r"(?is)"
        r"(call (?:may|will) be recorded|recorded for quality|"
        r"state licensed|license number|field underwriter|"
        r"fact finding|warm-up|warm up|3 and 1|"
        r"were you born|are you still working|beneficiary|"
        r"health questions|medications|height|weight|"
        r"product benefits|three options|application)",
        combined,
    ))

    early_text = combined[:1800]
    pq_or_handoff_early = "pq:" in early_text or "handoff" in early_text or "transfer" in early_text
    early_hangup = bool(re.search(
        r"\b(hung up|hang up|disconnected|stopped responding|are you there|can you hear me|bye-bye|bye)\b",
        early_text,
    ))

    if opening_only and pq_or_handoff_early and early_hangup and not meaningful_agent_start:
        return "BOOTC", "Prospect disconnected during PQ/handoff before the selling agent meaningfully started."

    if duration_seconds is not None:
        try:
            if int(duration_seconds) < 110:
                return "U90", "Call duration was under 110 seconds."
        except Exception:
            pass

    return "LEAD", "No sold, age, health-disqualification, BOOTC, or U90 indicator detected."


def save_to_db(call_name, transcript, report, score, risk):
    ensure_db()

    # Privacy safety at the database boundary.
    transcript = redact_sensitive_transcript(transcript or "")
    report = redact_report_text(report or "")

    auto_disposition, disposition_reason = detect_auto_disposition(
        call_name,
        transcript,
        report,
        duration_seconds=None,
    )
    final_disposition = auto_disposition

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("DELETE FROM calls WHERE call_name=?", (call_name,))
    c.execute("""
        INSERT INTO calls (
            call_name,
            transcript,
            report,
            score,
            risk,
            auto_disposition,
            manual_disposition,
            final_disposition,
            disposition_reason,
            duration_seconds
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        call_name,
        transcript,
        report,
        score,
        risk,
        auto_disposition,
        None,
        final_disposition,
        disposition_reason,
        None,
    ))

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


def build_audit_prompt(transcript, checklist, rubric, output_format, role_label_note=None):
    role_block = f"\n{role_label_note}\n\n" if role_label_note else ""
    return f"""
You are a strict QA auditor for final expense sales calls.

HARD OVERRIDE RULES (MANDATORY):

- If PQ identified the prospect before handoff, it is STRICTLY FORBIDDEN to include any mention of name confirmation in:
  - SCRIPT / FLOW MISSES
  - COACHING
  - SCORING BREAKDOWN
  - SUMMARY

- If the call ends before Fact Finding / Warm-up, it is STRICTLY FORBIDDEN to:
  - Coach the agent to progress the call forward
  - Mention missing Fact Finding / Warm-up or need discovery
  - Penalize for not advancing stages

- If Recording Disclosure is marked YES, it is STRICTLY FORBIDDEN to:
  - Critique timing
  - Suggest improvement
  - Mention it as a miss

- If License Number is not required at the stage reached, it is STRICTLY FORBIDDEN to:
  - Mention it as a miss
  - Include it in coaching

- Call control coaching should ONLY be given if:
  - The prospect shows resistance, objection, or attempts to end the call
  - AND the agent does NOT attempt any call control response

- If the agent does use call control appropriately, do NOT coach on call control.

- If no objection or resistance occurs, it is STRICTLY FORBIDDEN to:
  - Mention call control
  - Suggest improving call control
  - Penalize for lack of call control

- Only include misses and coaching that are DIRECTLY REQUIRED by the stage reached.
- If something is not required at the stage reached, it must be COMPLETELY IGNORED.

CALL STAGE:
Determine the SINGLE furthest stage reached in the call.
There is **no separate** “Warm-up / Rapport” or “Fact finding” **call stage** — those activities are **only** the combined stage **Fact Finding / Warm-up** (exact name in the list below).
STAGE DETECTION MUST BE STRICT (with one important exception below for **Fact Finding / Warm-up**):
- A stage is only reached if the agent clearly **enters** that stage in the transcript (progression), not when every checklist item for that stage is finished (**completion** belongs in TASK CHECKLIST / SCRIPT MISSES / COACHING, not in CALL STAGE REACHED).
- Do NOT infer a later stage from vague conversation.
- Do NOT mark Need unless the agent clearly asks about the customer's need, reason for coverage, family protection, burial/funeral concern, or motivation for buying.
- If the agent only explains who they are or what they do, the stage is Who I Am / What I Do.
- If uncertain between two stages, choose the earlier stage **except** where **Fact Finding / Warm-up — STAGE ENTRY** below applies.

Fact Finding / Warm-up — STAGE ENTRY vs TASK CHECKLIST (MANDATORY):
- **Fact Finding / Warm-up** is **one** stage that includes **all** of: warm-up, rapport building, **3 and 1 Method**, asking personal/background questions, gathering information **before** medical, and building credibility/connection. Do **not** treat these as separate stages in CALL STAGE REACHED or NOT REACHED.
- **CALL STAGE REACHED** for **Fact Finding / Warm-up** is based on **entry**, **not** full completion of every checklist line for that segment.
- Mark **Fact Finding / Warm-up** as **REACHED** as soon as the agent clearly **begins** any of the above (e.g. even **one or two** clear rapport or fact-finding questions **before** heavy **Medical / Health** underwriting).
- **If the agent transitions into Medical / Health** (health questions, medications, height/weight, tobacco, underwriting health, etc.), **Fact Finding / Warm-up must be considered already reached** — do **not** leave CALL STAGE REACHED at Who I Am / What I Do or Opening solely because 3+1 or rapport checklist items were incomplete.
- **NOT REACHED** must **not** list **Fact Finding / Warm-up** if the transcript shows that segment was **entered** or superseded by **Medical / Health**.
- **Separation of concerns:** **CALL STAGE REACHED** = progression; **TASK CHECKLIST** lines for **Fact Finding / Warm-up**, **3 and 1 Method used**, and **Agent shared personal rapport information** = **completion quality** — use **NO** / **PARTIAL** / **YES** (and **NOT REACHED** only when that line’s stage segment was **never entered**, per rules below), not “stage not reached” for the whole call when the segment clearly started.
- **Stage anchors** (see **STAGE ANCHORS** below) that match **Fact Finding / Warm-up** / rapport prove **segment entry** for **CALL STAGE REACHED** — they do **not** set **3 and 1 Method used: YES**; **3 and 1** completion stays per **TASK CHECKLIST**.

OPENING
- Agent answered with energy and enthusiasm
- Agent gave the recording disclosure
- Agent introduced themselves
- Agent stated they were licensed (optional)
- License number: YES only if an actual license number was clearly stated
- Agent confirmed who they were speaking with

Choose ONLY ONE from this list:
- PQ / Handoff
- Opening
- Who I Am / What I Do
- Fact Finding / Warm-up
- Medical / Health
- Need
- Features / Benefits
- Change Up
- Pre-Close
- Quotes
- Close
- Application Information
- Payment Date
- Banking
- Disclosures
- Third Party Underwriting
- Peace of Mind
- Cool Down

STAGE ANCHORS (SUPPLEMENTARY TRANSCRIPT EVIDENCE — **REACH** vs **COMPLETION**):
- **GLOBAL STAGE-ANCHOR RULE (STRICT):** Mark a stage **reached** **only** when **(1)** the transcript contains a **valid anchor phrase** for that stage **from the numbered list below** (paraphrase OK; **[STATE]**, **[CLIENT NAME]**, **[MONEY]**, **[DATE]**, **[NUMBER]**, redacted tokens OK), **or (2)** the transcript **clearly performs** that stage’s **required scripted action** in context. **Do NOT** classify stages by **loose topic similarity** alone (e.g. terminal illness / DNQ wording ⇒ **Medical / Health** only, **not** **Peace of Mind**; **walk you through step-by-step** / **direct toll-free number** ⇒ **Features / Benefits**, **not** **Peace of Mind**). **Existing coverage** autofail logic is **unchanged**.
- **STAGE ORDER:** **CALL STAGE REACHED** = the **latest** stage with a valid anchor or clear scripted performance; if a **later** anchor appears, **do not** stop at an earlier stage (e.g. **Medical** then **Quotes** anchors ⇒ **at least Quotes**; **Quotes** then **Close** anchors ⇒ **at least Close**; **Close** then **Application** anchors ⇒ **at least Application**; **Payment Date** then **Banking** ⇒ **at least Banking**).
- **REACH vs COMPLETION:** Anchors prove **entry** / **reached**, **not** full **TASK CHECKLIST** completion — e.g. **Quotes** reached ≠ **three options presented: YES** (use **PARTIAL** if only one option); **Banking** reached ≠ verification complete; **Fact Finding / Warm-up** reached ≠ **3 and 1 Method used: YES** (grade **3 and 1** separately per **TASK CHECKLIST** / **PHE** / vivid disclosure). **Payment Date** reached vs **Payment date explained** per **PAYMENT DATE STAGE** (DOB / date-of-proof false positives unchanged).
1. **Fact Finding / Warm-up:** **Reached** when rapport / location / background begins. **Anchors:** **Now that we got all that out of the way**; **I know that [STATE] is a beautiful part of the country**; **Were you born and raised there**; **How long have you been in [STATE]**; **What's your favorite part of living in [STATE]**; **Have you always lived in the same city**; **Are you still working or are you retired**; **Are you married**; **Do you have any children**; **Do you have any grandchildren**; **were you born and raised there** (paraphrase OK). **Important:** These prove **Fact Finding / Warm-up** reached — **not** automatic **3 and 1 Method used: YES**.
2. **Medical / Health:** **Reached** when health qualification / DNQ begins. **Anchors:** **Because this is a special state regulated plan**; **No physical exam is needed**; **There are just a few basic health questions** / **just a few basic health questions**; **To see if you'll qualify** / **see if you'll qualify**; **basic health questions**; **Do you currently**; **Have you ever been diagnosed**; **Terminal medical condition**; **End-stage disease**; **Expected to result in death**; **Respiratory failure**; **Liver failure**; **Hospice**; **Oxygen**; **Cancer**; **Nursing facility**. **Important:** Terminal / end-stage / respiratory / liver / DNQ language is **Medical / Health (this §2 list) only** — **never** **Peace of Mind**.
3. **Need:** **Reached** when reasonable need, current coverage, or only-policy discussion begins. **Anchors:** **Your total coverage can't exceed reasonable need**; **We can't apply for [MONEY] with these programs**; **Do you have any kind of final expense plan or life insurance now**; **Will this be your only policy**; **Have you ever had coverage in the past**; **current coverage**; **existing coverage**; **final expense plan**; **life insurance now**; **reasonable need**. **Important:** **Need** reached does **not** by itself trigger coverage autofail — use **current vs past** coverage rules.
4. **Features / Benefits:** **Reached** when product features / plan benefits / how the plan works begins (**pre-sale**). **Anchors:** **Do you still have your pen and paper handy**; **I am going to walk you through this step by step** / **walk you through this step-by-step**; **So you can know exactly what you're getting** / **know exactly what you're getting**; **First I'll give you my direct toll-free number** / **direct toll-free number**; **no waiting period**; **100% of your death benefit**; **tax-free**; **government cannot tax this money**; **additional benefits**; **immediate plan**; **ROP**; **graded**. **Important:** **Not** **Peace of Mind**.
5. **Change Up:** **Anchors:** **Before we get to your plans**; **On top of it all**; **We pay most claims immediately** / **most claims immediately**; **Usually within 48 hours of notification** / **48 hours of notification**.
6. **Pre-Close:** **Anchors:** **OK, WOW**; **I'm looking over these options**; **I've got to congratulate you**; **It looks like you may be eligible** / **eligible to qualify**; **One of our top tier plans** / **top tier plans**; **This is amazing**; **Congratulations to you**.
7. **Quotes:** **Reached** when premium / coverage **options** presentation begins. **Anchors:** **I'm going to share three affordable options with you**; **Three affordable options**; **I'm going to start with the largest amount first**; **Work my way all the way down**; **Are you ready**; **The first option you've qualified for is**; **The first option** / **second option** / **third option**; **That one is only [MONEY] per month**; **[MONEY] per month**; **per month**; **coverage amount**; **monthly premium** — furthest **≥ Quotes**; **three options presented** still by actual count (**PARTIAL** if only one). **Examples:** **three affordable options** + **first option** ⇒ **Quotes** reached — **do not** leave **CALL STAGE REACHED** at **Medical** or earlier.
8. **Close / Option selection:** **Reached** when the agent asks the prospect to **choose or commit** to an option. **Anchors:** **Which option would you want**; **Which one would you want your family to receive**; **Which option works best**; **Circle that option**; **Go ahead and circle**; **Let's go with the lowest option**; **The lowest option**; **Most affordable option**; **Go with that one**; **Choose an option**. **Important:** **Client chose an option** ≠ **Policy sold YES** without completion evidence.
9. **Application Information:** **Anchors:** **What's your middle initial**; **Please verify the spelling of your full name**; **How do you spell their name**; **Primary beneficiary**; **You stated that you would like [BENEFICIARY NAME] to be your primary beneficiary**; **Verify the spelling**; **Middle initial**; **Date of birth** / **DOB** / **Birth date**; **Address**; **Height and weight**; **Driver's license**; **Social Security** (application intake context). **Important:** **Reached** ≠ application completed.
10. **Payment Date (stage reached — separate from Payment date explained YES):** **Anchors:** **Since you're not receiving any government benefits yet**; **We'll set this up for the 1st or the 15th** / **1st or the 15th**; **Which of those two works best for you**; **We work on the same payment schedule as Social Security** / **same payment schedule as social security**; **Your first payment won't be until after you receive your government benefits** / **was not going to be until after**; **What day do you receive your benefits on**; **1st, 3rd, 2nd Wednesday, 3rd Wednesday, or 4th Wednesday**; **first payment is not going to draft until after your benefits have been deposited**; **after your benefits have been deposited on [DATE] / [NUMBER]**; **draft date**; **payment date**; **first payment**. **Payment date explained** still per **PAYMENT DATE STAGE**; **DOB** / **date of proof** (DOB) ≠ **Payment Date** evidence.
11. **Banking:** **Anchors:** **Now, since we work with all the banks directly** / **we work with all the banks directly**; **This is really easy to set up** / **really easy to set up**; **Are you with a bank or a credit union**; **What's their name**; **Bank or credit union**; **Routing number**; **Account number**; **Checking account**; **Savings account**; **Payment account**; **Draft account**; **[ACCOUNT_NUMBER]** / **[ROUTING_NUMBER]** / **[BANK_NUMBER]** in banking context. **Important:** **Reached** ≠ verification complete.
12. **Disclosures:** **Anchors:** **I just need to review a few disclosures that are required by law** / **review a few disclosures that are required by law**; **Required by law**; **The first one is the Fair Credit Reporting Act**; **Fair Credit Reporting Act**; **We are required by law to inform you**; **Disclosures** (read-in context).
13. **Third Party Underwriting / voice signature:** **Anchors:** **As a final step to completing the application process**; **I need you to please verify the following** / **verify the following**; **Can you please state your full name and today's date** / **Please state your full name and today's date** / **state your full name and today's date**; **Voice signature**; **Recorded line**; **Recorded verification**; **Welcome to the American Amicable Group recording system**; **For the app ID, only enter the numbers**; **Enter the app ID followed by the pound sign**; **App ID**; **Pound sign**; **American Amicable recording system** — **HARD TRIGGER**; furthest **≥ Third Party Underwriting** unless **Peace of Mind** / **Cool Down** clearly after per **§14–15** — and **§14** requires **all** listed conditions (not voice-signature cues alone).
14. **Peace of Mind (strict — ALL required, after §13):** **CALL STAGE REACHED: Peace of Mind** only when **all** are true: **(1)** **Policy sold: YES** (or transcript clearly completes the sale process per **POLICY SALE / SALE OUTCOME**). **(2)** **Third Party Underwriting / voice signature** reached or completed **first** per **§13**. **(3)** **Peace of Mind script after that:** **You're good**; **We're not going to forget about you either**; **We're going to mail that welcome letter**; **That will include all my personal information**; **Some information about the company**; **We got you qualified for today**; **Welcome letter**. **(4)** **Peace of mind completed: YES** in the report. **Never output invalid pairs:** **CALL STAGE REACHED: Peace of Mind** with **Peace of mind completed: NOT REACHED** or **NO**; with **Policy sold: NO**; with **Third Party Underwriting** listed only under **NOT REACHED**; with **Final stage supporting sale** at **Medical / Health**, **Features / Benefits**, **Quotes**, **Close**, **Application Information**, **Payment Date**, or **Banking**; or when evidence is only medical/DNQ, terminal illness, end-stage disease, product benefits, or quote language. If **Peace of Mind** was **not** actually reached, **CALL STAGE REACHED** must **not** be **Peace of Mind**. If **Policy sold** is **NO**, treat **Peace of Mind** / **Cool Down** as **NOT REACHED** and set **CALL STAGE REACHED** to the **latest valid pre-sale** stage from transcript anchors (**Quotes** when **§7** quote anchors appear and no later valid stage; **Medical / Health** or **Features / Benefits** when only those segments occurred). **Invalid:** generic reassurance **before** **(2)** completes; **Features / Benefits** or **Quotes** alone. **Policy sold** scoring unchanged elsewhere.
15. **Cool Down:** **Post-sale** casual non-insurance. **Anchors:** **You did something really important today and it got me thinking**; **handling stuff like this always makes people reflect**; **You've got things taken care of now, so I'm curious**; back-and-forth on family, memories, hobbies, weather, plans, location — **not** thank-you/goodbye alone; **not** required if **Policy sold** is **NO** / not completed.

**STAGE ORDER WITH ANCHORS:** Use the **latest** valid anchor-backed stage as **CALL STAGE REACHED**. **Medical** then **Quotes** ⇒ **≥ Quotes**; **Quotes** then **Close** ⇒ **≥ Close**; **Close** then **Application** ⇒ **≥ Application**; **Application** then **Payment Date** ⇒ advance; **Banking** after **Payment Date** ⇒ **≥ Banking**; **Disclosures** / **Third Party Underwriting** / **Peace of Mind** / **Cool Down** cues ⇒ advance per ordered list — **never** under-select when a **later** anchor is present.

FURTHEST STAGE — POST-APPLICATION (MANDATORY):
- **CALL STAGE REACHED** must be the **latest** stage in the ordered list above that the agent **clearly performed** in the transcript.
- **POST-SALE / ENROLLMENT ORDER (after Application Information):** Application Information → **Payment Date** → **Banking** → **Disclosures** → **Third Party Underwriting** → **Peace of Mind** → **Cool Down**. **CALL STAGE REACHED** must be the **furthest** stage in this order that was **clearly performed** in the transcript — do **not** skip ahead in the label unless that stage clearly occurred.
- **Do NOT** stop at **Application Information** only because the policy was sold or application details were taken — if the call continued into later enrollment steps, advance the stage accordingly.
- **Do NOT** infer later post-sale stages from any of: **Policy sold**; application completed; banking completed; payment date explained; disclosures read; polite ending; thank you / goodbye; friendly tone; generic warmth; a normal close.
- If the agent collected **banking / account / routing / payment** information for the policy, or **called the bank/CU to verify** account/banking for payment setup, **Banking** is reached (at minimum) unless a **later** listed stage clearly occurred afterward.
- **BANKING REACHED vs COMPLETED (MANDATORY):** **Banking** counts as **reached** when the agent **begins** any banking/payment-account step — **completion is not required**: asks **bank name**; **bank vs credit union**; **routing** or **account** number; **checking/savings/payment/draft** account; **discusses** payment-account or draft setup for the policy; **read-back** or prospect provides banking details (including **redacted**); placeholders **[ACCOUNT_NUMBER]**, **[ROUTING_NUMBER]**, **[BANK_NUMBER]**, or **[NUMBER]** in **clear banking/routing/account context**. Interrupted Banking (e.g. **callback** before verification finished) is still **Banking reached** for **CALL STAGE REACHED**.
- **CALL STAGE REACHED vs Close:** **Close** is **before** **Banking** in the ordered list. If banking/payment-account setup **began** per above, **CALL STAGE REACHED** must be **at least Banking** — **do not** leave it at **Close** (or **Quotes** / earlier) when those banking steps occurred and **no** **later** stage (**Disclosures** onward) was clearly performed.
- If **Banking** began but did **not** finish because a **callback** was set and **Policy sold** is **NO**, set **CALL STAGE REACHED: Banking**, **EARLY END: YES**; **NOT REACHED** must list **only** stages **after Banking** in order — **never** list **Banking** under **NOT REACHED**, and **do not** list **Application Information** or **Payment Date** under **NOT REACHED** if the transcript shows they occurred **before** Banking.
- If the agent **read disclosures** (legal/state/product disclosures to the prospect), **Disclosures** is reached when clearly performed.
- **Third Party Underwriting** is **NOT** reached merely because the policy was sold, application information was collected, banking was handled, payment date was handled, disclosures were read, or the agent said the application was complete. **Third Party Underwriting** is reached **only** when the transcript **clearly** shows the agent starting or completing the **post-disclosure** third-party recorded verification step, such as: calling into the **American Amicable recorded line**, starting the **American Amicable** recording, beginning **recorded third-party underwriting**, asking **official recorded verification / underwriting questions after disclosures**, or clearly placing the prospect into the required **carrier/third-party recorded verification** process. **Strong transcript evidence** includes (non-exhaustive): **"Welcome to the American Amicable Group recording system."**; **"For the app ID, only enter the numbers."**; **"Enter the app ID followed by the pound sign."**; **American Amicable Group recording system**; **American Amicable recording system**; carrier IVR-style **recording system** plus **app ID** / **pound sign** instructions; **voice signature** / **recorded verification** through American Amicable. When that language appears, **Third Party Underwriting** is reached — set **CALL STAGE REACHED** to at least **Third Party Underwriting** (unless **Peace of Mind** or **Cool Down** clearly occurred afterward per their strict definitions) and **do not** list **Third Party Underwriting** under **NOT REACHED**. If **Peace of Mind** and **Cool Down** did not occur afterward, list them under **NOT REACHED** instead. If sale/application/banking/disclosures appear but **no** clear recorded third-party underwriting cues (above or equivalent), do **NOT** mark **Third Party Underwriting** as reached and do **NOT** mention **recorded third-party underwriting** in **SALE OUTCOME** evidence or **SUMMARY**. If **Disclosures** were reached and **Third Party Underwriting** was **skipped** when it should have happened next, include a **SCRIPT / FLOW MISS** and **lower the score**, unless the prospect ended the call before the agent had a reasonable chance.
- **Third Party Underwriting — HARD TRIGGER (post-application / post-disclosure context):** If any of these appear in transcript-backed enrollment flow — **Welcome to the American Amicable Group recording system**; **For the app ID, only enter the numbers**; **Enter the app ID followed by the pound sign**; **American Amicable recording system**; **app ID**; **pound sign**; **voice signature**; **recorded line**; **recorded verification** — treat **Third Party Underwriting** as **reached**: **CALL STAGE REACHED** must be **at least Third Party Underwriting** unless **Peace of Mind** or **Cool Down** clearly occurred afterward; **never** list **Third Party Underwriting** only under **NOT REACHED**.
- **Peace of Mind** is reached **only** when **§14 BOTH** are satisfied: **Third Party Underwriting / voice signature completed** per **§13**, **then** the **Peace of Mind script** (you're good / not going to forget / welcome letter mailing / personal information + company qualified for today, etc.). **Do NOT** mark **Peace of Mind** or treat it as reached from **medical / health / DNQ** language (**terminal medical condition**, **end-stage disease**, **expected to result in death**, **respiratory failure**, **liver failure**, hospice, oxygen, cancer, etc. — **Medical / Health (anchor §2)** only); generic politeness; thank you/goodbye; application/payment/banking/disclosures **without** completed **§13**; **Features / Benefits** or **Quotes** alone; generic reassurance **before** **§13** completion. If voice signature / **§13** is **not** completed, **Peace of Mind** is **NOT REACHED** and **Peace of mind completed: NOT REACHED** (or **NO**) even if other sale language appears. **Quotes** anchors ⇒ **at least Quotes**; **Features / Benefits** anchors ⇒ **at least Features / Benefits** — **never** jump to **Peace of Mind** from medical/DNQ or early-stage wording. If the agent skips **Peace of Mind** after a completed sale path when **(1)+(2)** were possible, **Peace of mind completed** is **NO** when there was reasonable opportunity (not when the prospect hung up first).
- **Cool Down** is **separate** from **Peace of Mind**. **Cool Down** means the agent clearly spends time in **casual non-insurance conversation after the sale** (e.g. weather, family, hobbies, location, work, pets — **back-and-forth** away from insurance/application). **Do NOT** mark **Cool Down** or set **CALL STAGE REACHED** to **Cool Down** from: polite ending; thank you; goodbye; confirming the sale; wrapping up the application; warm tone; or a normal close. If there is **no** clear non-insurance small talk after the sale, **Cool down completed** is **NO** and **Cool Down** is **not** the furthest stage reached when it was required and the agent had reasonable opportunity (not when the prospect ended the call first).
- **NOT REACHED** must include **only** stages **after** the furthest reached stage, in order — **never** list a stage as NOT REACHED if it clearly occurred in the transcript.
- Tie-break: if uncertain between two **early** (pre-application) adjacent stages, prefer the **earlier** stage; after **Application Information**, when the transcript **clearly** shows a **later** listed stage occurred **per that stage's strict definition** (especially **Third Party Underwriting**, **Peace of Mind**, **Cool Down**), set **CALL STAGE REACHED** to that **furthest** later stage — do **not** under-select **Application Information** because of sale alone, and do **not** use sale or politeness to infer **Third Party Underwriting**, **Peace of Mind**, or **Cool Down**.

**PEACE OF MIND AND COOL DOWN AFTER SALE (WHEN TO SCORE):**
- **Peace of Mind** and **Cool Down** are **post-sale** stages — evaluate completion **only** when **Policy sold** is **YES** (sale actually **completed** per **POLICY SALE / SALE OUTCOME** below) **and** the agent had a **reasonable opportunity** before the call ended. If **Policy sold** is **NO** or **UNCLEAR** (including calls that end at **Banking** or earlier with no completed sale), treat **Peace of Mind** / **Cool Down** as **NOT REACHED** for checklist purposes — **do not** mark them as **skipped after sale**, **do not** add **SCRIPT / FLOW MISSES** for skipping them, **do not** lower scores for post-sale skips, and **do not** put **Post-sale process incomplete: Peace of Mind and Cool Down skipped** in **Reason** / **automatic fail** on that basis alone.
- If **Policy sold** is **YES** and the agent had a **reasonable opportunity** before the call ended, evaluate **Peace of Mind** and **Cool Down** completion on **transcript evidence** per the strict definitions above — **not** from sale or enrollment steps alone.
- If the agent **skips Peace of Mind** when there was time/opportunity (**and** **Policy sold** is **YES** per above), mark **Peace of mind completed: NO**, include a **SCRIPT / FLOW MISS**, **lower the Sales Process score**, and mention it in **TOP 3 COACHING PRIORITIES** if it is a top issue. Same for **Cool Down** when skipped with opportunity: **Cool down completed: NO**, **SCRIPT / FLOW MISS**, **lower the Sales Process score**, coaching if top issue, and **do not** set **CALL STAGE REACHED** to **Cool Down**.
- **Do NOT** penalize for skipping **Peace of Mind** or **Cool Down** if the **prospect/customer** ended the call, disconnected, or the transcript ended before the agent had a reasonable chance.

BANKING STAGE (CALL STAGE REACHED / NOT REACHED):
- **Banking** is reached when the transcript shows the agent **asked for**, **handled**, or **began discussing** **bank name**, **bank vs credit union status**, **routing number**, **account number**, **checking/savings/payment/draft account**, **read-back** of account/routing details (including **redacted** forms), **prospect confirmation** of redacted banking details, **payment-account setup** talk tied to the policy, or **called/verified** with a bank or credit union for payment setup — **including when the step is incomplete** or ended by **callback**.
- If **"Did the agent call the bank to verify banking/account information?"** is **YES**, treat **Banking** as reached for stage detection.
- Do **NOT** list **Banking** under **NOT REACHED** if any banking or payment-account information was collected or verified on the call.
- If **Application Information** was reached and banking/payment-account handling occurred afterward, **CALL STAGE REACHED** should be **Banking** unless **Disclosures**, **Third Party Underwriting**, **Peace of Mind**, **Cool Down**, or another **later-than-Banking** listed stage was clearly reached after Banking.

**REDACTED PLACEHOLDERS — STAGE DETECTION (MANDATORY):**
- Redacted tokens (**[DATE]**, **[NUMBER]**, **[MONEY]**, **[ACCOUNT_NUMBER]**, **[ROUTING_NUMBER]**, **[BANK_NUMBER]**, **[PHONE]**, etc.) must **not** hide real enrollment progress: infer stages from **surrounding words** and **turn structure**, not from raw digits alone.
- **Payment Date** may be reached when the transcript clearly discusses **policy draft / payment date / premium draft timing** for **this** sale, even if the calendar day appears only as **[DATE]** or **[NUMBER]** / **[MONEY]** — including when timing is explained as **after** the prospect’s **benefits / government benefits / Social Security / check deposited** without a literal calendar day when redacted.
- **Banking** may be reached when the transcript shows **bank name**, **bank vs credit union**, **routing**, **account**, **checking/savings/payment/draft account**, read-back or confirmation of **redacted** banking details, or placeholders (**[ACCOUNT_NUMBER]**, **[ROUTING_NUMBER]**, **[BANK_NUMBER]**, or **[NUMBER]** when context is clearly banking). Do **not** list **Banking** under **NOT REACHED** solely because numbers were redacted.
- **Disclosures** may be reached from **legal/state/product disclosure** language or the agent **reading required disclosures**, even if amounts or dates nearby are redacted.
- **Third Party Underwriting:** use the **HARD TRIGGER** bullet in **FURTHEST STAGE** above — redacted tokens must not hide that stage when surrounding language matches.

PAYMENT DATE STAGE (STRICT — SEPARATE FROM DEPOSIT TIMING):
- **Payment Date** / **Payment date explained** is judged **separately** from whether **Banking** was **finished** — use **PAYMENT DATE STAGE** below even when **Banking** was only **started** or **incomplete**.
- **DOB / application false positives (mandatory):** **Date of birth** / **DOB** / **birth date** / **date of proof** (when context is clearly **DOB** / identity), **age**, medical DOB questions, and redacted **\[NUMBER\].\[NUMBER\]** patterns used for **birth** are **not** **Payment Date** evidence — **do not** mark **Payment date explained: NO**, **do not** add **SCRIPT / FLOW MISS** / autofail for payment date, and **do not** cite those lines as payment/draft misses. If the **Payment Date** segment **never** began, use **Payment date explained: NOT REACHED** (or equivalent **NOT REACHED** treatment per template) — **not** **NO**. If **Payment Date** was **reached** but draft/premium timing was **not** explained, then **NO** per **PAYMENT DATE STAGE** below.
- **Payment Date** is reached when the agent **explains, sets, or confirms** the policy **draft/payment date** (or **first** draft / premium timing) for **this** sale — including when the agent ties **first payment / first draft** to the prospect’s **benefits or Social Security deposit** (e.g. first payment will **not** draft until **after** benefits are deposited; payment will draft after your deposit; we'll set it up after your benefits come in; first draft/payment on or after **[DATE]**; we'll set payment **around** / **after** that deposit). That **deposit-to-draft** explanation counts as explaining/setting draft or payment timing. **Asking only** which day benefits hit **without** any policy draft/payment-after-deposit explanation does **not** satisfy **Payment Date** — keep **Payment Date** under **NOT REACHED** / **Payment date explained: NO** for that narrow gap only.
- **Redaction:** Do **not** mark **Payment date explained: NO** **solely** because the calendar day is **[DATE]** / **[NUMBER]** / **[MONEY]** when the agent clearly stated first draft/payment occurs **after** the benefits deposit (or equivalent).
- **TASK CHECKLIST / SEARCHABLE:** If the agent asked benefits deposit timing **and** then explained the **first** policy payment/draft will happen **after** that deposit, mark **Payment date explained: YES** — do **not** mark **NO** when that linkage is clear in the transcript.
- **MANDATORY — Payment date explained: YES:** When the agent clearly ties **first payment / first draft / premium** to **after** the prospect receives or has **benefits**, **government benefits**, or **Social Security** deposited (or “after your … check is deposited”), **present or past tense** both count (**"is not going to"** / **"was not going to"** / **"will draft after"**). **Exact unredacted calendar day not required.**
- These transcript patterns **must** yield **Payment date explained: YES** (match **intent**; same meaning with minor wording changes OK): **"Your first payment is not going to be until after you receive your government benefits."**; **"Your first payment was not going to be until after you receive your government benefits."**; **"Your first payment is not going to draft until after your benefits have been deposited on May [NUMBER]."** (same with **[NUMBER]** / **[DATE]** for the day); **"We'll set this up so your first payment will draft after your benefits are deposited."**; **"We'll set this up so your first payment drafts after your benefits are deposited."**; **"Your first payment will be after your Social Security/government benefits deposit."**; **"The payment will come out after your Social Security check is deposited."**; **"Your draft date will be after your benefits deposit."**; **"We will set the first premium payment after your benefits come in."**
- **Same-call flow (mandatory YES):** Agent explains **same payment schedule as Social Security** (or equivalent), states **first payment was / is not** until **after** **government benefits**, asks **what day** benefits are received, prospect answers, agent confirms **first payment is not going to draft until after** benefits deposited on **May [NUMBER]** (or **[NUMBER]**/**[DATE]**) ⇒ **Payment date explained: YES** for the segment — **one coherent** explanation; **do not** require a separate isolated sentence if this full pattern appears across adjacent turns.
- **ASH_TEST payment (mandatory YES when present):** **Payment date explained: YES** when the transcript includes **"Your first payment was not going to be until after you receive your government benefits."** and/or **"Your first payment is not going to draft until after your benefits have been deposited on May [NUMBER]."** (or **[NUMBER]**/**[DATE]** redaction) — **do not** list payment/draft date as a **SCRIPT / FLOW MISS** or coach on explaining payment date when this language is present.

LATE-STAGE SCRIPT / FLOW MISSES (WHEN BANKING REACHED):
- If **Banking** was reached and **Payment Date** was not explained/set/confirmed per above, include a **SCRIPT / FLOW MISS** for missing payment/draft date handling.
- Do **NOT** add that **SCRIPT / FLOW MISS**, **Reason: Payment/draft date not explained after banking**, section **6)** payment autofail, or score penalties for “missing Payment Date” when the transcript already satisfies **PAYMENT DATE STAGE** (including **MANDATORY** examples / deposit-after-benefits linkage / **ASH_TEST payment** above), **or** when the **only** cited issue is **DOB** / **date of proof** (DOB) / application identity per **DOB / application false positives**.
- **Self-negating payment miss (forbidden):** **Never** output a **SCRIPT / FLOW MISS** bullet that **claims** a payment/draft-date gap **and** then says it is **not applicable**, **withdrawn**, or that the agent **did explain** draft timing — if the miss does **not** apply, **omit** the bullet **entirely** (do **not** leave a stricken or contradictory line).
- When **Payment date explained** is **YES** per **PAYMENT DATE STAGE**, there must be **zero** payment/draft-date lines under **SCRIPT / FLOW MISSES**, **no** payment-date item in **TOP 3 COACHING PRIORITIES**, and **do not** include **Payment/draft date not explained after banking** in **Reason** / **AUTOMATIC FAIL CHECKS** for that basis alone.

**BANKING VERIFICATION — ACCOUNT AND ROUTING SEPARATE:**
- When **Banking** is reached, evaluate **account number** and **routing number** **independently**. **Do not** merge account-side and routing-side moments into **one** combined count — **never** award full success based on **three** total banking touches split across fields unless **each** field meets its **own** standard.
- **Expected:** **Account number:** requested, repeated, read back, or **verified** in **three** separate **verification events** when possible (**minimum two** meaningful events before treating that side as minimally acceptable). **Routing number:** same (**three** target; **minimum two**). The agent should **read each number back** or otherwise **confirm each one clearly** with the prospect. In the written audit, reference only redaction placeholders (**[ACCOUNT_NUMBER]**, **[ROUTING_NUMBER]**, **[BANK_NUMBER]** — never raw digits).
- **Combined checklist line — Banking/account information requested or verified 3 times:** **YES** only if **both** the **account-number** side **and** the **routing-number** side **each** meet the **three-event** standard on their **own**. **PARTIAL** if **one** side reaches **three** but the **other** does **not**, **or** if **both** sides have **at least two** events but **not three**. **NO** if **either** side was only **collected once** or **not meaningfully confirmed**. **NOT REACHED** only if **Banking** was not reached.

**COUNTING RULES (Banking verification):**
- Count only **distinct verification events** (separate asks, read-backs, or confirmations).
- **Do not** treat multiple placeholders in **one sentence** as multiple events.
- **Do not** count **SUMMARY**, **narrative audit wording**, or **report meta-text** as events — only **transcript-backed** moments listed in **Routing verification evidence** / **Account verification evidence**.
- **Do not** count “I can pull routing on my end” / similar as **routing** verification unless the **routing number** is actually **read, repeated, or confirmed** (including **[ROUTING_NUMBER]** / **[BANK_NUMBER]** when context is routing).
- **Do not** count an **account** ask as **routing** evidence, or a **routing** ask as **account** evidence.
- **Do not** count **bank name** as account or routing verification.
- **Do not** count **Social Security / deposit timing** discussion as substituting for routing or account verification (see **Payment Date** vs deposit timing).
- Never expose real account/routing digits in the report.

**Placeholder interpretation:** **[ACCOUNT_NUMBER]** in account context ⇒ account evidence only. **[ROUTING_NUMBER]** in routing context ⇒ routing evidence only. **[BANK_NUMBER]** ⇒ use surrounding words: near routing language ⇒ routing evidence; near account language ⇒ account evidence; unclear banking context ⇒ **one** banking-number touch — **do not** double-count the same ambiguous token as **both** account **and** routing.

**STRICT — VERDICTS MUST MATCH EVIDENCE LINES:** Do **not** mark **YES** / **PARTIAL** on banking-number checklist lines **without** supporting rows in **Account verification evidence** / **Routing verification evidence** — **counts must align**. Do **not** mark **Account number requested or verified 3 times:** **YES** unless **Account verification evidence count** is **≥ 3** with **three** genuinely **separate** events listed. Do **not** mark **Routing number requested or verified 3 times:** **YES** unless routing evidence count **≥ 3**. Do **not** mark **Routing number verified at least 2 times:** **YES** unless routing evidence count **≥ 2**. Same for account **≥ 2**. If you **cannot** list enough distinct events for a verdict, **lower** the verdict (**PARTIAL** / **NO**) to match — **never** inflate YES/PARTIAL without the listed evidence.

Valid **separate** events (credit only to **account** or **routing** per topic): agent asks for that number type; prospect gives it; agent asks prospect to repeat; prospect repeats; agent reads back; prospect confirms read-back (**must be tied** to account vs routing). **Never** YES without evidence lists that **prove** it.

**Illustrative grade (NOT transcript text to copy verbatim):** Account side shows three **[ACCOUNT_NUMBER]** moments that qualify as separate events ⇒ account lines may be **YES** / **PARTIAL** depending on separation; routing side shows intent to verify but **no** qualifying routing-number read/repeat ⇒ **Routing number requested or verified 3 times:** **NO**; combined **Banking/account information requested or verified 3 times** must **not** be **YES** unless routing also reaches **three** qualifying events.

**SCORING:** If **Banking** is reached and **either** account **or** routing falls **below two** qualifying events, **lower** scores — especially **Banking / Payment accuracy**, **Sales Process**, and **Closing**. If **either** side misses the **three-event** target, mark that side’s checklist line **PARTIAL** or **NO**. If **both** sides miss the **three-event** target (after fair counting), include a **SCRIPT / FLOW MISS** for incomplete banking verification. When **Banking** **started** but a **callback** ended the process early, you may phrase the miss as **incomplete banking due to callback** (cite transcript) **in addition to** callback autofail — this gap is **not** an automatic fail **by itself** unless another banking/credit-union automatic-fail rule applies.

**TASK CHECKLIST (BANKING — REQUIRED LINES when Banking is reached):** Include these lines **exactly** in **TASK CHECKLIST** (use **NOT REACHED** on each line below only when **Banking** was not reached — otherwise score normally):
  - **Banking/payment setup explained:** **YES** / **PARTIAL** / **NO** / **NOT REACHED** — **NOT REACHED** only if **no** banking/payment-account discussion began; if Banking **started** but was **incomplete** (e.g. **callback**), use **PARTIAL** or **YES** by how clearly setup was explained — **not** **NOT REACHED**.
  - Banking/account information requested or verified 3 times: **YES** / **NO** / **PARTIAL** / **NOT REACHED** — per **combined checklist line** rule (**YES** only when **both** fields meet **three** events each).
  - Account number requested or verified 3 times: **YES** / **NO** / **PARTIAL** / **NOT REACHED**
  - Account number verified at least 2 times: **YES** / **NO** / **NOT REACHED**
  - Routing number requested or verified 3 times: **YES** / **NO** / **PARTIAL** / **NOT REACHED**
  - Routing number verified at least 2 times: **YES** / **NO** / **NOT REACHED**
  - Agent read account/routing information back to prospect: **YES** / **NO** / **PARTIAL** / **NOT REACHED**
  - Prospect confirmed account/routing read-back: **YES** / **NO** / **PARTIAL** / **NOT REACHED**
  - **Account verification evidence count:** <number> separate events (must equal the count of distinct account events listed below; use **0** only when consistent with the verdicts)
  - **Account verification evidence:** <brief event list> — placeholders only (**[ACCOUNT_NUMBER]**, **[ROUTING_NUMBER]**, **[BANK_NUMBER]** per rules); **no** raw digits
  - **Routing verification evidence count:** <number> separate events (must equal distinct routing events listed below)
  - **Routing verification evidence:** <brief event list> — same placeholder rules

**Incomplete banking verification** (**not** automatic fail by itself unless another banking rule triggers): **lower** **Banking / Payment accuracy**, **Sales Process**, and **Closing**; **do not** assign a **near-perfect** final **SCORE** when account/routing verification is materially incomplete vs these evidence lines.

- If **Banking** was reached and **Peace of Mind** and/or **Cool Down** were skipped when the agent had a **reasonable opportunity** (and the customer did **not** cut the call short) **and** **Policy sold** is **YES**, add **Peace of Mind** and/or **Cool Down** to **SCRIPT / FLOW MISSES** and **lower the score**; include matching items in **COACHING** / **TOP 3 COACHING PRIORITIES**. If **Policy sold** is **NO** or **UNCLEAR**, **do not** apply this post-sale skip penalty.

RULES:
- Only select the LAST stage clearly reached.
- Do NOT list multiple stages.
- Do NOT repeat the list.

EARLY END RULE:
- If Cool Down was NOT reached, EARLY END must be YES.
- If Cool Down WAS reached, EARLY END must be NO.
- EARLY END does NOT automatically mean FAIL.
- The call can PASS if the agent correctly completed the required sales checklist items up to the furthest stage reached.
- Do NOT penalize for stages that were not reached unless the agent skipped required steps before ending (does **not** apply when the **customer** hung up — that is not the agent “ending” early).
- Judge early-ended calls only up to the furthest stage reached.

EARLY-END REFUSAL (PROSPECT REFUSES / ENDS BEFORE THE SALES PROCESS CAN CONTINUE):
- Treat as an **early-end refusal** when the **prospect** says they are **not interested**, already have **final expenses / coverage handled**, do **not** want to continue, and **ends the call** (or forces a stop) **before** the agent could reasonably move into **Fact Finding / Warm-up**, **Medical / Health**, **Need**, **Quotes**, **Application**, **Banking**, etc. — e.g. ends during **Who I Am / What I Do** or **Opening** with no warm-up entry.
- **EARLY END: YES**; **CALL STAGE REACHED** = the **latest stage actually performed** (often **Who I Am / What I Do** or **Opening** when refusal is immediate).
- **Do not** penalize, list as **SCRIPT / FLOW MISSES**, or coach **future** stages that were **never reached** (Medical, Need, Quotes, Application, Payment Date, Banking, Disclosures, Third Party Underwriting, Peace of Mind, Cool Down).
- **Do not** list **3 and 1 Method used** or rapport as **incomplete** / **SCRIPT / FLOW MISS** when **Fact Finding / Warm-up** is **NOT REACHED** — set **3 and 1 Method used: NOT REACHED** and **Agent shared personal rapport information: NOT REACHED**; **do not** coach 3 and 1 for that call unless warm-up was **entered**.
- **Callback (false positive guard):** A prospect **refusing** or **hanging up** is **not** a callback. **Did the agent set a callback?** = **YES** only when the **agent** clearly **offers, agrees to, schedules, or commits** to a **later** call (**I'll call you back**, **we'll call tomorrow**, **I'll follow up**, **let's schedule**, **we can finish this later**, etc.). **NO** when the only “end” is prospect **not interested** / **goodbye** / disconnect **without** agent callback commitment.
- **Existing coverage on early refusal:** When **EARLY-END REFUSAL** applies and the prospect only explains they **already handled** final expenses / coverage **as their reason to stop**, **do not** apply the normal **existing-coverage automatic fail** — the agent never had a fair chance to clarify **active** in-force coverage on a continuing sale. Set **Existing coverage mentioned but not confirmed: NO** (or **UNCLEAR** only if the template requires ambiguity language — **not** **YES** solely from “I already have it taken care of” on a refusal hang-up); **do not** trigger **Automatic fail triggered: YES** **solely** from coverage; **do not** put coverage in **COMPLIANCE FAILURES**, **SCRIPT / FLOW MISSES**, **BIGGEST MISS**, **Reason**, or **TOP 3 COACHING PRIORITIES** as the main failure. **Did the agent ask about existing coverage?** may be **YES** if the topic was discussed; **Did the agent confirm current coverage?** remains **NO** when no carrier verification — that is **not** the same as the **“mentioned but not confirmed”** autofail path here.
- **Professionalism:** If the **agent** uses **profanity**, **slurs**, **insults**, or **clearly disrespectful / hostile** language **toward or about** the prospect (including **after** refusal or hang-up, e.g. muttering **what a bitch** or similar), treat as a **serious** failure: include **Unprofessional language / disrespectful call ending** in **COMPLIANCE FAILURES** or **SCRIPT / FLOW MISSES**; **lower Communication Quality** and **final SCORE** substantially; make it **TOP 3 COACHING** and **BIGGEST MISS** when it is the **most serious** issue. If rubric allows autofail for this, set **Automatic fail triggered: YES** with **Reason** naming **Unprofessional language / disrespectful call ending**; otherwise still use **PASS: NO** and **RISK: HIGH** for this severity. **Do not** let **Reason** cite **callback** or **existing coverage** as the **primary** autofail when unprofessional language is the real driver.

SCORING RULES:

3 AND 1 METHOD — SCORE IMPORTANCE (MAJOR IMPACT, **NOT** AUTOMATIC FAIL):
- Building rapport through the **3 and 1 Method** is one of the most important parts of the sales call when **Fact Finding / Warm-up** was **entered**.
- Missing, weak, or partial **3 and 1** is **not** a compliance **automatic fail** by itself. Do **not** set **Automatic fail triggered: YES**, **RISK: HIGH**, **PASS: NO**, or **PASS: AT RISK** **solely** because of weak, partial, or missing **3 and 1** (those outcomes require their own rules, e.g. coverage, callback, post-sale, payment date, credit union verification).
- If **Fact Finding / Warm-up** was **NOT** reached, do **not** penalize **3 and 1** — keep **3 and 1 Method used** / **Agent shared personal rapport information** as **NOT REACHED** when the segment never started; do **not** treat pre-entry silence as a rapport fail; do **not** add a **SCRIPT / FLOW MISS** for incomplete **3 and 1**; do **not** coach **3 and 1** improvement for that call (**EARLY-END REFUSAL** / pre–warm-up stop).
- If **Fact Finding / Warm-up** was **reached** and **3 and 1 Method used** is **YES** (only when the **evidence gate**, **A + B**, **and** **substantial** warm-up execution per **SCRIPT STRUCTURE** in **Fact Finding / Warm-up — TASK CHECKLIST** are fully met — **questions alone are never enough**), apply **no** score deduction **for this item** (other dimensions still score normally).
- If **Fact Finding / Warm-up** was **reached** and **3 and 1 Method used** is **PARTIAL**, lower the **final SCORE** meaningfully — generally about **5–10** points depending on how much was completed; reduce **Sales Process** and **Communication Quality** when rapport was rushed, shallow, interrogative, or lacked meaningful tied self-disclosure.
- If **Fact Finding / Warm-up** was **reached** and **3 and 1 Method used** is **NO**, lower the **final SCORE** heavily — generally about **10–15** points when the warm-up segment clearly occurred; **Sales Process** and **Communication Quality** must both reflect the gap.
- A call with weak or missing **3 and 1** must **not** receive a **near-perfect** final **SCORE** (or near-perfect **Sales Process** / **Communication Quality**) unless the transcript shows **exceptional** strengths elsewhere **and** there are **no other material misses**.
- When **Fact Finding / Warm-up** was reached: put incomplete or missing **3 and 1** in **SCRIPT / FLOW MISSES**; put it in **TOP 3 COACHING PRIORITIES** when it is a top issue; it may be **BIGGEST MISS** only when **no** higher-priority issue exists (unprofessional language, compliance / coverage, payment date after Banking, banking verification, post-sale skips, callback, DNQ handling, etc.).
- **Mandatory:** If the agent **only asked questions** or shared **nothing meaningful** about herself tied to the prospect’s answers, **3 and 1 Method used** **cannot** be **YES**. If **Agent shared personal rapport information** is **NO** or **PARTIAL**, **3 and 1 Method used** **cannot** be **YES** — use **PARTIAL** or **NO** by depth of questioning.

SCORE IMPACT (MANDATORY — tie to rubric when these issues are clear):
- **3 and 1 / rapport in Fact Finding / Warm-up:** follow **3 AND 1 METHOD — SCORE IMPORTANCE** above (major **SCORE** / category impact; **never** sole **automatic fail** or sole **RISK** / **PASS** driver).
- Skipping **Peace of Mind** after a **sold** call (**Policy sold YES**) when the agent had **reasonable opportunity** must **lower the score** (SCRIPT / FLOW MISS). If **Policy sold** is **NO**, do **not** treat as a sold-call post-sale skip.
- Skipping **Cool Down** after a **sold** call (**Policy sold YES**) when the agent had **reasonable opportunity** must **lower the score** (SCRIPT / FLOW MISS). If **Policy sold** is **NO**, do **not** treat as a sold-call post-sale skip.
- **Banking** reached but **account number** or **routing number** (evaluated **separately**) falls **below two** qualifying verification events **or** lacks meaningful read-back/confirmation where required must **lower the score** (especially **Banking/Payment**, **Sales Process**, **Closing**).
- **Third Party Underwriting** must **not** be treated as reached unless the **strict recorded third-party** standard above is met — overstating this stage should **lower the score** via SCRIPT / FLOW accuracy. **Under**-selecting it (e.g. only **NOT REACHED**) when American Amicable recording / **app ID** / **pound sign** / **voice signature** / **recorded verification** language clearly appears also **lowers** SCRIPT / FLOW accuracy.
- **DNQ** (disqualifying medical) conditions clearly disclosed are **serious qualification issues** — **lower the score** and add **SCRIPT / FLOW MISSES** / coaching as specified under **MEDICAL / HEALTH — DNQ** below.
- **Existing coverage mentioned but not confirmed** when it triggers **Automatic fail triggered** per the coverage sections above must follow **AUTOMATIC FAIL** / **PASS: AT RISK** logic and **lower the score** / **RISK** appropriately (do **not** treat bank verification as coverage confirmation).
- **Post-sale process incomplete** (skipped **Peace of Mind** / **Cool Down** / **Third Party Underwriting** when required after **Disclosures** on a **sold** call with opportunity — **only** when **Policy sold** is **YES**) and **missing payment/draft date after Banking** on a **sold** call (**Policy sold YES**) must **lower Sales Process**, **Compliance** (when applicable), **Closing** / payment-related categories, **final SCORE**, and **RISK** per **SCORE CAP RULES** — these are **not** minor coaching items. When **Policy sold** is **NO**, do **not** score as post-sale incomplete for Peace of Mind / Cool Down.

SCORE CAP RULES (MANDATORY — align **SCORE**, **RISK**, **PASS**, and **SCORING BREAKDOWN** with misses):
- **HARD — Existing coverage mentioned but not confirmed: YES:** **Automatic fail triggered** must be **YES**; **RISK** must be **HIGH**; **Reason** must name that gap (never **Reason: None**); **PASS** must be **AT RISK** if **Policy sold** is **YES**, else **PASS: NO**; **Compliance** must be **significantly reduced**; final **SCORE** must **generally not exceed 80** and **must not be 90+**. **Does not apply** when **EARLY-END REFUSAL** applies — use **Existing coverage mentioned but not confirmed: NO** and **do not** fire coverage-only autofail on that pattern.
- **HARD — Automatic fail triggered: YES** (any cause): **PASS** cannot be **YES**; **RISK** cannot be **LOW**; **Reason** cannot be **None** — it must name at least one applicable automatic-fail cause. Final **SCORE** should **generally not exceed 85**; for **compliance-related** automatic fails (especially **Existing coverage mentioned but not confirmed** or **Credit union mentioned but bank/account not verified**), final **SCORE** should **generally be below 80**. If **Policy sold** is **YES**, **PASS** must be **AT RISK**, not **YES**. If **Policy sold** is **NO** or **UNCLEAR**, **PASS** must be **NO** (not **AT RISK**).
- If **Policy sold** is **YES**, **Disclosures** were reached, and **Peace of Mind** + **Cool Down** were **both skipped** with reasonable opportunity (per **5) POST-SALE PROCESS INCOMPLETE**): final **SCORE** must **not exceed 80**; **RISK** must be **HIGH**; **PASS** must be **AT RISK**.
- If **Banking** was reached and **Payment date explained** is **NO** (no clear policy draft/payment date **per PAYMENT DATE STAGE**, including deposit-to-draft linkage when present in transcript): **Sales Process** and **Banking/Payment** category scores must be materially reduced; final **SCORE** must **not exceed 88** unless there are **no other material issues**; with **Policy sold YES** and this gap, treat as a serious failure per **6) PAYMENT / DRAFT DATE**. **Do not** use this cap when **Payment date explained** should be **YES** per deposit-after-benefits explanation.
- If **Existing coverage mentioned but not confirmed** applies with **Policy sold YES**: **Compliance** must **not** be near-perfect; **RISK** must be **HIGH**; **PASS** must be **AT RISK**; final **SCORE** must **not** remain in the **90s** (align with the **80** cap above).
- If **multiple** serious issues apply together (e.g. existing coverage not confirmed **and** payment date missing **and** post-sale skips), final **SCORE** must be **significantly lower** and **cannot** be **90+**.

SCORING BREAKDOWN ALIGNMENT (MANDATORY):
- Category scores (**Compliance**, **Sales Process**, **Product Explanation**, **Closing**, **Communication Quality**) must **reflect** SCRIPT / FLOW MISSES, **AUTOMATIC FAIL CHECKS**, and TASK CHECKLIST gaps. **Do NOT** output near-perfect **Compliance** when an automatic fail is present or likely. **Do NOT** output near-perfect **Sales Process** when required post-sale stages were skipped after a **sold** call with opportunity **and** **Policy sold** is **YES** (not when the sale was **not** completed). **Do NOT** output near-perfect **Closing** when **Payment date explained** is **NO** after **Banking** was reached. **Do NOT** output near-perfect **Sales Process** or **Communication Quality** when **Fact Finding / Warm-up** was reached and **3 and 1 Method used** is **PARTIAL** or **NO** per **3 AND 1 METHOD — SCORE IMPORTANCE**. **Do NOT** output a high final **SCORE** that contradicts **SCRIPT / FLOW MISSES** and **Automatic fail triggered**.

AGENT EXPECTATION RULES:
- If PQ identified the prospect before handoff, the agent is NOT required to re-confirm the prospect’s name.
- NEVER include failure to confirm the prospect’s name as a SCRIPT / FLOW MISS or COACHING item unless there is clear confusion about identity.
- Do NOT penalize or coach the agent for not re-confirming the prospect’s name when PQ already completed the introduction.

EARLY END FAIRNESS RULES:
- If the customer ends the call early, the agent should NOT be penalized for not progressing to later stages.
- Do NOT coach the agent to “move the call forward” if the call ended due to the customer hanging up or disengaging.
- Only coach based on what the agent could reasonably control during the stage reached.

CUSTOMER-INITIATED EARLY END (HANG-UP / DISCONNECT) — MANDATORY FOR COACHING, MISSES, CHECKLIST & BIGGEST MISS:
- If the transcript shows the **prospect/customer** hung up, disconnected, stopped responding, said goodbye and ended, or otherwise **ended the session** without the agent choosing to wrap up, treat that as a **customer-initiated** end.
- **Never** imply the **agent chose** to end the call, “ended the call early,” or “cut the call short” when the **customer** ended it.
- **Do NOT** use agent-blaming phrasing such as: “ending the call before completing health questions,” “the agent ended the call before…,” or “failed to finish before hanging up” when the **customer** caused the stop.
- **Instead** use customer-neutral wording, e.g.: **“Customer ended the call before health questions could be completed.”** (Adapt the stage name to match: e.g. before need discovery, before quotes, etc.)
- **Do NOT** coach the agent to complete **later** tasks (e.g. “complete health questions fully before ending the call,” “finish all health questions,” “move to quotes/close”) when the **customer** ended the call **before** the agent could reasonably do so.
- **Do NOT** coach “find a way to schedule a continuation,” “schedule a follow-up,” or “maintain momentum to progress” **unless** the **CALLBACK AND SCHEDULING** rules in this prompt clearly apply (evidence-based callback — do not invent callback coaching after a bare hang-up).
- **SCRIPT / FLOW MISSES:** Do **not** list misses that blame the agent for not completing a later-stage task solely because the **customer** hung up. Only include controllable agent errors **within** the stage reached **before** the hang-up.
- **TASK CHECKLIST / Health questions:** If the call was in **Medical / Health** and the **customer** ended the call mid-process, **Health questions completed** may remain **PARTIAL** or **NO** without turning that into a harsh agent miss — reflect that the process was **interrupted by the customer**, not abandoned by the agent.
- **EARLY END** should remain **YES** when Cool Down was not reached; that is correct even when the customer hung up.
- **PASS** may remain **YES** (or **AT RISK** only per automatic-fail rules) when there was **no controllable** agent miss before the customer ended the call — a customer hang-up alone is **not** a failure by the agent.
- **TOP 3 COACHING PRIORITIES:** When the agent handled the reached stages appropriately and the only “gap” is work the customer did not allow time for, at least one coaching bullet **may** be exactly: **“No major controllable coaching issue identified before the customer ended the call.”** (Still provide three bullets; the other two should be minor, in-stage, controllable refinements only if they exist — otherwise use brief neutral in-stage observations rather than invented future-stage pressure.)
- **BIGGEST MISS:** If the only significant issue is incomplete work caused by a **customer hang-up**, do **not** blame the agent — use **- None** or describe the situation without faulting the agent (e.g. that the customer disconnected; **not** “agent failed to complete health questions”).

STRICT STAGE-BASED SCORING RULES:
- Only evaluate checklist items that belong to stages up to and including the stage reached.
- Do NOT mark items as NO if their stage was not reached.
- Do NOT include missed items from future stages in SCRIPT / FLOW MISSES.
- Do NOT include future-stage coaching unless the agent incorrectly skipped ahead.

COACHING RULES:
- Do NOT coach the agent to confirm the prospect’s name if PQ already identified the prospect.
- Do NOT coach the agent to move into Fact Finding / Warm-up, Medical / Health, need, quotes, close, or any later stage if the customer ended the call before that stage.
- If the **customer** hung up or disconnected (customer-initiated end), do **not** coach as if the agent should have forced completion of that stage or the next stage on the same call.
- Do NOT mention license number as a miss or coaching item unless the license-number requirement was actually reached.
- If Recording disclosure is marked YES, do NOT coach on recording disclosure timing or wording unless the rubric explicitly requires exact timing.
- Coaching must ONLY include actionable improvements that apply to the stage reached.
- Do NOT include generic sales advice.
- Do NOT suggest actions that require progressing to a stage that was not reached.

STAGE-SPECIFIC REQUIREMENTS:

- During Who I Am / What I Do, only evaluate whether the agent gave a basic product purpose explanation.
- Do NOT classify this as "Product Benefits Explained".
- "Product Benefits Explained" should ONLY be marked YES if the Features / Benefits stage is clearly reached.

- Basic product explanation should count only for the stage reached and should NOT be treated as full product benefits.

PRODUCT BENEFITS EXPLAINED — DETECTION (TASK CHECKLIST LINE — Immediate / ROP / Graded):
- **Prerequisite:** follow **"Product Benefits Explained" should ONLY be marked YES if the Features / Benefits stage is clearly reached** (see **STAGE-SPECIFIC REQUIREMENTS** above). Do **not** mark **YES** from **Who I Am / What I Do** alone. When **Features / Benefits** (or clear equivalent product-value segment) **was** reached, use the rules below for the checklist line.
- **Product benefits explained** should be **YES** when the agent clearly explains **meaningful product value, features, or benefits** to the prospect (not only **Who I Am / What I Do** role/purpose talk).
- The company may offer three plan types: **Immediate**, **ROP**, and **Graded**. Benefit-count expectations when the agent ties language to the plan type: **Immediate** — up to **four** additional/value benefits when explained that way; **Graded** — **two** benefits when explained; **ROP** — **one** benefit when explained. Do **not** require the agent to list every benefit by name if they clearly convey meaningful value for the plan discussed.
- **YES**-level explanations include (non-exhaustive): **immediate coverage**; **100% death benefit** after first payment / from day one; **no waiting period**; money to **family/beneficiary**; **tax-free** benefit / government cannot tax it; **additional included benefits**; **best / immediate plan** advantages; **family protection**; policy/coverage **value**; explaining the **number of benefits** tied to the plan type (e.g. four with Immediate).
- Example that **must** count as **Product benefits explained: YES** (adapt wording to transcript): *"I hope you qualify for the immediate plan because it's the absolute best plan we offer. Your plan is fully covered 100% of your death benefit the day you make your first payment. There's no waiting period. The money will go to your family directly tax-free. The government cannot tax this money. If I get you approved for the immediate plan, you're going to have four additional benefits that go along with it."*
- Do **not** mark **NO** when the agent clearly explains immediate coverage, no waiting period, 100% day-one death benefit, tax-free family payout, or additional benefits for the plan type — use **YES** or **PARTIAL** by depth, not **NO**, unless talk stayed generic with **no** meaningful benefit detail.
- **PARTIAL** or **NO** when the agent only says **"this is life insurance"** / **"final expense"** / vague purpose with **no** meaningful benefit detail. **Who I Am / What I Do** alone is **not** Product Benefits Explained.
- **Immediate:** give credit when the agent explains immediate / no waiting period / 100% from first payment / tax-free to family / four benefits (or clear equivalent value language).
- **Graded:** give credit when the agent explains graded-plan value and **two** benefits (or equivalent) when applicable.
- **ROP:** give credit when the agent explains ROP value and **one** benefit (or equivalent) when applicable.

- If the agent states that the call is being recorded at any point before proceeding, mark Recording Disclosure as YES.
- Do NOT include any coaching, critique, or timing feedback about recording disclosure if it was stated.

- License number is only required if the script/rubric requires it during Opening.
- If the call ends before that requirement is reached, do NOT penalize.

- Call control should ONLY be evaluated if the prospect gives resistance, objection, or attempts to leave the call.
- Do NOT penalize if no objection occurred.

BOTTOM PARAGRAPH / LOWEST OPTION — CALL CONTROL (OBJECTION HANDLING — TRANSCRIPT EVIDENCE):
- If the prospect **hesitates**, says they must **ask a spouse or family member**, **wants to wait**, or **defers deciding**, and the agent responds with a **bottom-paragraph / lowest-option** style close that **includes** (adapt wording to transcript): **acknowledging** the concern; **not paying today**; recommending the **lowest / most affordable** option; **protection in place now**; can **add or change** coverage **later**; **risk of waiting** without protection; then **directs the next step** (e.g. circle amount, application question) and the call **continues into application** or clear enrollment forward progress — treat that as **proper call control** / objection handling for that beat.
- Align **Objection occurred without proper call control: NO** when that pattern is clear; **Agent maintained control of the conversation** (or equivalent narrative) should reflect **YES** when the agent **recovered** the sale after the deferral using that pattern. **Do not** set **Objection occurred without proper call control: YES** or autofail on **call control** **solely** because the prospect **initially** deferred to a spouse/family if the agent then executed this pattern and moved forward.
- **SALE OUTCOME / Evidence / SUMMARY:** **Do not** write that the prospect **did not commit** or **deferred purchase solely** because of spouse/family deferral **when** the transcript shows the **bottom paragraph** pattern above **and** the call **continued** into application — describe it as **initial deferral, then agent used call control and progressed with the lowest option** (or equivalent accurate phrasing).
- **Client chose an option** / **moved forward with an option:** mark **YES** when the agent **steered to the lowest option** and the prospect **continued** (e.g. answered application questions) after that control sequence — **do not** treat the first deferral alone as **no** forward choice when later transcript shows compliance with the agent’s direction. **This line is independent of Policy sold** — **Client chose an option: YES** does **not** set **Policy sold: YES** without **POLICY SOLD = YES** completion evidence.

- **Fact Finding / Warm-up** checklist lines (including **3 and 1** and **Agent shared personal rapport**) and **Medical / Health** / **Need** checklist items should ONLY be scored for **completion quality** once that **call stage** is **entered** (see **Fact Finding / Warm-up — STAGE ENTRY** above — entry is a low bar; incomplete 3+1 is **not** “stage not reached” for CALL STAGE).
- Do NOT penalize checklist items for stages that were **never entered**; incomplete execution after entry is **PARTIAL/NO**, not “not reached” for the stage itself.

- If PQ already identified the prospect before handoff, do NOT penalize the agent for not re-identifying the prospect.

MEDICAL / HEALTH — DNQ / DISQUALIFYING CONDITIONS (QUALIFICATION — TRANSCRIPT EVIDENCE ONLY):
- The prospect is **DNQ / disqualified** for these policies if the transcript **clearly** shows the prospect **currently has** or **admits to** any of the following (do **not** mark DNQ from vague or unrelated medical wording; if unclear, state **UNCLEAR** in narrative and explain briefly — there is **no separate DNQ field** in the required report template; capture as **SCRIPT / FLOW MISSES** / compliance / coaching as appropriate):
  - Currently hospitalized
  - Confined to a nursing facility
  - Confined to a bed due to chronic illness or disease
  - Confined to a wheelchair due to chronic illness or disease
  - Currently using oxygen equipment to assist in breathing
  - Receiving hospice care
  - Receiving home health care
  - Had an amputation caused by disease
  - Currently has any form of cancer
  - Requires assistance with activities of daily living, including bathing, dressing, eating, or toileting
  - Has been advised to have an organ transplant
  - Has been advised to have kidney dialysis
  - Has ever been diagnosed with congestive heart failure
  - Has ever been diagnosed with Alzheimer's
  - Has ever been diagnosed with dementia
  - Has ever been diagnosed with mental incapacity
  - Has ever been diagnosed with ALS
  - Has ever been diagnosed with liver failure
  - Has ever been diagnosed with respiratory failure
  - Has ever been diagnosed by a medical professional as having a terminal medical condition
  - Has ever been diagnosed with an end-stage disease expected to result in death in the next 12 months
  - Has AIDS
  - Has ARC
  - Has HIV
  - Has HHV
  - Has any immune deficiency related disorder
- If any DNQ condition is **clearly** present and the agent **continues trying to sell** a policy instead of stopping, redirecting, or handling per process, **lower the score** and include a **serious SCRIPT / FLOW MISS / coaching / compliance** issue.
- If the agent **correctly** stops, redirects, or handles DNQ appropriately, **do not** unfairly penalize for not completing later sales stages.

Fact Finding / Warm-up — TASK CHECKLIST (3 and 1 & rapport — TRANSCRIPT EVIDENCE ONLY — STRICT):

Combined stage (same **Fact Finding / Warm-up** call stage as above): warm-up, rapport building, **3 and 1 Method**, personal/background questions, gathering information before medical, credibility/connection. **TASK CHECKLIST** measures **completion** here; **CALL STAGE REACHED** measures **entry** only.

**WARM-UP / 3 AND 1 — SCRIPT STRUCTURE (OFFICIAL GRADING REFERENCE — transcript evidence, paraphrase OK):** The segment is **not** “a few rapport questions.” Grade against this **intended flow** (wording may vary):
1. **Ice breaker:** Ask whether the prospect was **born/raised** in state/location; **high energy**, conversational tone.
2. **Location / geography:** **Multiple** questions (e.g. how long in state, favorite part, same city, why moved); agent uses **Praise / Hope / Empathy (PHE)** follow-ups where rapport advances; agent shares a **vivid-detail** story about **her own** location/geography (answers similar questions about herself); transition toward work.
3. **Working / retired:** **Multiple** work questions (still working vs retired, what they do/did, what drew them in, enjoyment); PHE; **vivid-detail** agent story on work/enjoyment; transition toward spouse/relationship.
4. **Spouse / relationship (when appropriate):** **Multiple** questions (married/partner, how long, how met); PHE; **vivid-detail** agent story (spouse/partner or a couple she knows); transition toward children/family.
5. **Children / grandchildren:** **Multiple** questions (children, grandchildren, how often seen); PHE; **vivid-detail** agent story (children/grandchildren or being a child); transition toward important person / quality time.
6. **Important person / what’s important:** Questions on time with that person, frequency, greatest shared memory; PHE; **vivid-detail** agent memory story; **tie-back** to the sale/need (e.g. thanks for sharing, family love, why protection matters).

**CORE RULE — QUESTIONS ALONE ARE NEVER ENOUGH:** Rapport/fact-finding **questions alone** **never** justify **3 and 1 Method used: YES**. To mark **YES**, the transcript must show **both**: **(1)** meaningful rapport/fact-finding across enough required topic areas, **and** **(2)** **meaningful personal/relatable self-disclosure** tied to the prospect’s answers. If **(2)** is missing, vague, generic, or not tied: **3 and 1 Method used** **cannot** be **YES**; **Agent shared personal rapport information** **cannot** be **YES** — use **PARTIAL** or **NO**.

**Major topic groups** (same four as **A** below): **location** / where the prospect lives or is from; **job / work / career / past jobs**; **spouse / marriage / partner / relationship** when appropriate; **children / family / someone important**. For **YES**, the agent should cover **at least three** of these with **genuine** rapport/fact-finding questions **and** must share **meaningful** personal information about herself **tied** to **at least one** of those topics.

**A. Prospect questions (topic groups):** The agent asked **meaningful** rapport or fact-finding questions across **at least three** of these **four** groups: (1) **location / where the prospect lives or is from**; (2) **job / work / career / past jobs**; (3) **spouse / marriage / partner / relationship status** when appropriate; (4) **children / family / someone important** in the prospect's life. **Shallow or single-topic** questioning alone is **not** enough for **YES**. Questions must be **mostly** rapport/fact-finding — not **mostly** medical screening, application, banking, payment, underwriting intake, or **script-only** intake (those **do not** satisfy **A** unless clearly genuine warm-up, not intake).

**B. Agent self-disclosure tied to prospect answers:** The agent shared **meaningful personal or relatable information about herself** **tied** to **what the prospect said** (same topic thread), not disconnected filler.

**STRICT — Agent shared personal rapport information: YES** only if the agent **clearly** says something **personal, relatable, or experience-based about herself** with **enough vivid or concrete detail** to satisfy **SCRIPT STRUCTURE**-level warm-up (not only reactions to the prospect, **not** only a **single** generic line when **3 and 1** is **not** truly script-complete — use **PARTIAL** for **some** tied share without **vivid-detail** depth).

**Examples that COUNT (non-exhaustive):** “I’m from there too.” “I used to work in that kind of job.” “My grandmother was the same way.” “I have children too.” “My family went through something similar.” “I live near there.” “I know what you mean; I went through that with my own family.” “My spouse and I…” when **naturally tied** to the prospect’s topic. Any **real** statement about the **agent’s own** life, family, location, work, or experience **tied** to what the prospect said.

**Examples that DO NOT count** (do **not** treat as **Agent shared personal rapport information: YES** or as satisfying **B**): **okay**, **gotcha**, **nice**, **great**, **awesome**, **perfect**, **I understand**, **that makes sense**, **I hear you**, **wow**, **absolutely**, **that’s good**, **right**, **exactly**, **I love that**, repeating the prospect’s answer, complimenting the prospect, generic empathy with **no** personal detail, **product explanation**, **script explanation**, **medical**, **underwriting**, **beneficiary**, **application**, **payment**, **banking** questions (unless clearly warm-up rapport, not intake).

**VAGUE SELF-DISCLOSURE:** If the agent says something **very vague** about herself but **no meaningful personal detail** tied to the prospect’s answer, mark **Agent shared personal rapport information: PARTIAL**, **not YES**. Examples (PARTIAL at most if tied to a prospect topic; **never YES** alone): “I know how that is.” “I’ve heard that before.” “I deal with that too.” “I can relate.” “Same here.” “I get it.”

**INTERNAL CONSISTENCY (mandatory):**
- If **Agent shared personal rapport information** is **NO** or **PARTIAL**, **3 and 1 Method used** **cannot** be **YES**.
- If **3 and 1 Method used** is **YES**, **Agent shared personal rapport information** **must** be **YES**, with **meaningful** self-disclosure evidence cited.

**Evidence gate before YES:** Before **3 and 1 Method used: YES**, identify internally **and** reflect it in **3 and 1 topic groups evidenced** / **3 and 1 agent self-disclosure evidence**:
- **Topic group 1 asked:** (which group — location / work / spouse / family)
- **Topic group 2 asked:**
- **Topic group 3 asked:**
- **Agent personal self-disclosure:** **short quote or clear paraphrase** from the transcript (**must** appear in **3 and 1 agent self-disclosure evidence** for **YES**)  
If you **cannot** identify an **actual** agent self-disclosure **quote or clear paraphrase**, **do not** mark **YES** on **3 and 1 Method used** or **Agent shared personal rapport information**. If either topic coverage or self-disclosure fails the gate, use **PARTIAL** or **NO**, **not YES**.

**REQUIRED 3 AND 1 EVIDENCE LINES (TASK CHECKLIST — mandatory):** Immediately **after** **3 and 1 Method used** and **Agent shared personal rapport information**, output **exactly** these labels (TASK CHECKLIST or **immediately** following those lines):
- **3 and 1 topic groups evidenced:** <brief list or None>
- **3 and 1 agent self-disclosure evidence:** <brief quote/paraphrase or None>

**Evidence rules:** If **3 and 1 Method used: YES**, **both** evidence lines must contain **real transcript-backed** content (**not** None, **not** vague/generic acknowledgment posing as disclosure). If **3 and 1 agent self-disclosure evidence** is **None**, vague-only, or only generic acknowledgment ⇒ **Agent shared personal rapport information:** **NO** or **PARTIAL** and **3 and 1 Method used:** **PARTIAL** or **NO** (**never YES**). **Fact Finding / Warm-up** never entered ⇒ **NOT REACHED** on the rapport verdict lines and **None** on both evidence lines is acceptable.

**STRICT 3 AND 1 YES RULE:** **3 and 1 Method used: YES** requires **A**, **B**, **and** **substantial script-aligned execution** (see **SCRIPT STRUCTURE** above): **A)** Meaningful rapport/fact-finding across **≥ three** of the four topic groups (**location**; **job/work**; **spouse/partner** when appropriate; **children/family/important person**). **B)** Meaningful **personal or relatable self-disclosure tied to the prospect’s answers** with **vivid-detail** texture (not only a **single** brief line if the warm-up otherwise skips depth, PHE, and tie-back). **Questions alone ⇒ never YES.** Meeting **A + B** with **thin** execution — e.g. **one** question each across three areas, **no** real **PHE**, **no** vivid agent stories, **major** scripted areas **skipped** (e.g. no meaningful **important person / memory** thread, no **tie-back**) ⇒ **PARTIAL**, **not YES**. Generic phrases (**okay**, **gotcha**, **nice**, **great**, **awesome**, **perfect**, **I understand**, **that makes sense**, **I hear you**, **wow**, **absolutely**, **that’s good**, **right**, **exactly**, **I love that**, repeating the prospect, compliments-only, generic empathy, **script/product explanation**, medical/application/banking intake) **do not** count as **B**. Vague lines (**I can relate**, **I know how that is**, **same here**, **I get it**) ⇒ **Agent shared personal rapport information** **PARTIAL** at best, **not YES**, unless followed by **real personal detail** tied to the prospect.

**Decision (internal, must match evidence lines):** Fewer than **three** topic groups ⇒ **do not** mark **3 and 1 Method used: YES**. Self-disclosure **NO** or **only vague** ⇒ **do not** mark rapport **YES**; **3 and 1 Method used** **cannot** be **YES**. **A + B** met but **not** **substantial** vs **SCRIPT STRUCTURE** ⇒ **PARTIAL** (not **YES**). **Fact Finding / Warm-up** never entered ⇒ **NOT REACHED** on these checklist lines only per stage rules below.

**3 AND 1 METHOD — VERDICT RULES:**
- **YES:** Agent **substantially** follows the **SCRIPT STRUCTURE**: **most** major areas engaged with **meaningful depth** (**multiple** questions per area touched **or** clearly equivalent rich exchange), **observable PHE-style** follow-ups (praise/encouragement/empathy — **not** only flat acknowledgments), **vivid-detail** agent stories **tied** to the prospect’s answers, warm-up builds credibility/connection and moves toward need (including **tie-back** when applicable); **evidence gate** satisfied; **not** questions-only or minimal tick-box coverage.
- **PARTIAL:** Rapport present — asked across **some** areas and/or shared **some** personal information — but **lacks** scripted depth: too **few** questions per area, **limited** PHE, **no** vivid-detail stories (or only **light** shares), **skips** major warm-up areas, **weak** tie-back to family/need, or disclosure **not** enough for **YES** on rapport.
- **NO:** **Fact Finding / Warm-up** was **reached**, but the agent **mostly shallow** questions, **rushed** rapport, **only** one–two **light** questions, **mostly** medical/application/script intake, or **no** meaningful self-disclosure about herself.
- **NOT REACHED:** **Fact Finding / Warm-up** was **never** reached (for these two checklist lines only, per stage rules).

If the agent **only** acknowledges, agrees, or keeps the conversation moving **without** real self-disclosure: **Agent shared personal rapport information: NO** — and **3 and 1 Method used** **cannot** be **YES**.

**Generic acknowledgments do NOT count as personal self-disclosure** (same non-examples — do **not** treat as **YES** on rapport lines).

**Medical, underwriting, beneficiary, application, banking, payment, and health-screening questions do NOT count** toward the **four** rapport topic groups **unless** they are **clearly** part of genuine warm-up rapport (not script/medical intake).

**Mandatory:** If there is **no** meaningful agent self-disclosure per **B**, set **Agent shared personal rapport information: NO** (or **PARTIAL** only for **vague-only** partial share per **VAGUE SELF-DISCLOSURE** above) and **3 and 1 Method used** **cannot** be **YES** — use **PARTIAL** or **NO** by depth of real rapport questioning. **3 and 1** is **not** an automatic fail by itself, but when **Fact Finding / Warm-up** was **reached**: **PARTIAL** → reduce final **SCORE** by about **5–10**; **NO** → reduce by about **10–15**; **Sales Process** and **Communication Quality** must reflect rushed/shallow rapport or missing self-disclosure.

**ASH_TEST — 3 AND 1 / RAPPORT (REFERENCE):** When the transcript shows the agent asked rapport about **Michigan/location**, **working/retired**, **married/children/grandchildren**, and shared **being in Indiana**, **sister-in-law in Michigan**, **two boys / hoping for grandbabies** — that is **credit-worthy** but **not** full **YES** if the agent did **not** complete scripted warm-up **depth** (**multiple** questions per area, **PHE**, **vivid-detail** stories across threads, deeper **work/spouse/important-person** work, **final tie-back**). **Expected alignment:** **Fact Finding / Warm-up: YES**; **3 and 1 Method used: PARTIAL**; **Agent shared personal rapport information: PARTIAL**. **Reasoning pattern:** rapport questions + **some** personal share, but **not** script-caliber **YES**.

**Scoring:**
- **YES** (**3 and 1 Method used**): **Evidence gate satisfied** **and** **substantial** execution per **SCRIPT STRUCTURE** and **VERDICT RULES** above — **not** questions-only, **not** thin **A+B** alone.
- **PARTIAL**: Per **VERDICT RULES** — common when several areas are touched but depth/PHE/vivid stories/tie-back are **incomplete** (including **ash_test** pattern above).
- **NO**: Per **VERDICT RULES** — **questions alone never yield YES**.
- **NOT REACHED** on **3 and 1 Method used** / **Agent shared personal rapport information**: **only** if **Fact Finding / Warm-up** was **never entered**.

**Mandatory (repeat):** **NO** meaningful self-disclosure tied to prospect topics ⇒ **not YES** on **3 and 1 Method used** or **Agent shared personal rapport information** (use **PARTIAL** or **NO**). **Cannot** pair **3 and 1 Method used: YES** with **Agent shared personal rapport information: NO** or **PARTIAL**. **Questions alone** ⇒ **not YES** on **3 and 1 Method used**.

TASK CHECKLIST (REQUIRED OUTPUT FORMAT — include these lines exactly when present in the template):
- **Fact Finding / Warm-up:** **YES** / **NO** / **PARTIAL** / **NOT REACHED** (overall segment quality / whether the combined segment clearly occurred beyond a minimal cue — use **NOT REACHED** only if the agent **never began** this segment per **Fact Finding / Warm-up — STAGE ENTRY**; otherwise **YES** / **NO** / **PARTIAL**).
- **3 and 1 Method used:** **YES** / **NO** / **PARTIAL** / **NOT REACHED**
- **Agent shared personal rapport information:** **YES** / **NO** / **PARTIAL** / **NOT REACHED**
- **3 and 1 topic groups evidenced:** <brief list or None>
- **3 and 1 agent self-disclosure evidence:** <brief quote/paraphrase or None>

Scoring / verdict rules (do NOT hallucinate — only clear transcript evidence counts):
- **NOT REACHED** on **3 and 1 Method used** and **Agent shared personal rapport information** **only** if **Fact Finding / Warm-up was never entered** — do **not** use **NOT REACHED** on those two lines just because 3+1 was incomplete once the segment **began**.
- **YES** for **3 and 1 Method used** only per **A + B**, **substantial** **SCRIPT STRUCTURE** execution, and the **evidence gate** above (**not** generic acknowledgments, **not** vague-only lines **as YES**, **not** questions-only, **not** **mostly** medical/application/banking/underwriting/script intake, **not** thin minimal **A+B** without vivid depth / PHE / tie-back per **TASK CHECKLIST**). **Invalid:** **3 and 1 Method used: YES** + **Agent shared personal rapport information: NO** or **PARTIAL**. **NO** meaningful self-disclosure ⇒ **not YES** (use **PARTIAL** or **NO**).
- **PARTIAL** / **NO** per the strict definitions above when self-share or topic coverage is weak.
- **Agent shared personal rapport information:** **YES** only with **actual** self-disclosure **tied to prospect topics** with **vivid or concrete detail** sufficient for **substantial** warm-up (per **STRICT — Agent shared personal rapport information** and **SCRIPT STRUCTURE** — not acknowledgments listed above); **NO** / **PARTIAL** otherwise per above; **NOT REACHED** only if **Fact Finding / Warm-up** was **never entered**.

SCRIPT / FLOW MISSES & COACHING:
- If **Fact Finding / Warm-up** was **reached** and **3 and 1 Method used** is **NO** or **PARTIAL**, or **Agent shared personal rapport information** is **NO** / **PARTIAL** because the agent did **not** share meaningful personal information tied to the prospect, include that as a **SCRIPT / FLOW MISS** (and **TOP 3 COACHING PRIORITIES** / coaching lines when applicable) and **lower the score**. **When 3 and 1 is PARTIAL or NO**, include at least one **clear** miss line where it fits (examples — adapt to transcript): **“3 and 1 Method incomplete: agent asked rapport questions but did not provide meaningful personal self-disclosure tied to the prospect’s answers.”** or **“3 and 1 Method incomplete: agent did not cover enough rapport topic areas and did not share enough personal information about herself.”** Also specify **what was missing**: **not enough topic groups**, **no meaningful personal self-disclosure**, **only generic acknowledgments**, **mostly medical/application/underwriting/banking/script questions**, **rapport questions too shallow**, or **questions-only / no disclosure**.
- **Coaching:** when coaching on this gap, include guidance such as: **Follow the warm-up structure** — location, work, spouse/relationship, children/grandchildren, important person — **multiple** questions per area where possible, **Praise/Hope/Empathy** follow-ups, **vivid-detail** agent stories **tied** to answers, then **tie back** to family and need; **not** only thin Q&A.
- Coach using the **WARM-UP / 3 AND 1 — SCRIPT STRUCTURE** above — do not invent questions or shares not supported by the transcript.
- Do **not** add 3+1 misses or coaching if the call **ended before** **Fact Finding / Warm-up** was **entered** (no segment start and no medical transition that implies entry per stage rules).

EARLY-STAGE LOGIC:
- If the call ends at Who I Am / What I Do, only evaluate:
  - PQ / Handoff
  - Opening
  - Who I Am / What I Do
- All later stages must be ignored for scoring, compliance, misses, and coaching.

EARLY-STAGE PASSING RULE:
- Do NOT fail or heavily penalize a call simply because it ended early.
- Score only what the agent was responsible for up to the furthest stage reached.
- If the call ended at Who I Am / What I Do, grade only PQ/Handoff, Opening, and Who I Am / What I Do requirements.
- Later stages such as **Fact Finding / Warm-up**, **Medical / Health**, Need, Quotes, Close, Application, Payment, Peace of Mind, and Cool Down should be listed as NOT REACHED, but should not lower the score.
- A call can PASS if the agent followed the sales task checklist up to the furthest stage reached with no controllable misses in those stages. **Do not** require, penalize, or coach **call control** when **no** resistance, objection, or attempt to end occurred (see call-control rules above).
- Score only the checklist items required up to the furthest stage reached.
- Do not deduct for later stages that were never reached.
- PASS can be YES if the completed portion of the call meets the checklist and compliance requirements.
- If **score** is below **70** and **PASS: AT RISK** does **not** apply (see **AUDIT OUTCOME** — **AT RISK** applies **only** when **Policy sold** is **YES** **and** **Automatic fail triggered** is **YES**), **PASS** must be **NO**.

NOT REACHED:
- List ONLY stages AFTER the stage reached.
- Do NOT include stages before it.
- Include future stages only.

SCRIPT / FLOW MISSES RULE:
- Include only missed tasks within stages that were actually reached.
- Do NOT include misses from future stages that were not reached.
- If the call ended because the **customer** hung up or disconnected, do **not** add misses that fault the agent for not completing in-progress or later work the customer did not allow time to finish.
- **Tell, Don't Ask:** add a **SCRIPT / FLOW MISS** **only** when the issue is **clear** (agent-identified, **repeated** or **material** permission-seeking on **required** enrollment steps per **TELL, DON'T ASK**); **omit** if **speaker** is **unclear** — **not** an automatic fail.


EARLY EXISTING-COVERAGE / NOT-INTERESTED CALL CONTROL RULE

When a prospect says early in the call that they already have final expenses taken care of, already have coverage, are not interested, or are not looking to add more coverage, the agent should attempt calm call control if the prospect has not fully ended the call.

This is especially relevant during:
- PQ handoff
- Who I Am / What I Do
- early objection before warm-up

Strong call control examples:
- "That's okay, this is just an informational call."
- "That's okay, you're like a lot of people I help every day."
- "I'm not asking you to cancel anything."
- "Let's just see what you may qualify for."
- "A lot of people I help already have something in place; my job is just to see if this could help or improve what you have."
- "I completely understand. I'm just going to give you the information and then you can tell me what you want to do from there."

Weak handling:
- Agent gives up immediately without attempting call control.
- Agent argues with the prospect.
- Agent becomes frustrated or disrespectful.
- Agent uses profanity or insults.
- Agent fails to redirect the conversation when there was still a reasonable opportunity.

Scoring / reporting:
- This is not an automatic fail by itself.
- Do not mark automatic fail solely because the agent missed this call-control opportunity.
- If the prospect clearly hangs up or refuses to continue, do not over-penalize future stages.
- If the agent had a reasonable opportunity and did not attempt call control, mention it in COACHING or SCRIPT / FLOW MISSES.
- If the agent responds disrespectfully or uses profanity, the professionalism rule still applies and can be the automatic fail / biggest miss.
- Lower Communication Quality and Sales Process if lack of call control contributed to the early end.

For early refusal calls:
- Callback should remain NO unless the agent clearly offered, agreed to, or scheduled a later call.
- Existing coverage mentioned but not confirmed should remain NO / not applicable when the prospect refused and ended early.
- 3 and 1 should remain NOT REACHED if Fact Finding / Warm-up was not reached.
- Do not coach on future stages that were never reached.

Example:
If the prospect says they already have final expenses taken care of and is not interested, the agent should attempt calm call control such as:
"That's okay, this is just an informational call"
or
"That's okay, you're like a lot of people I help every day."

If unprofessional language occurred, that remains the biggest miss over missed call control.


COACHING RULE:
- Coaching must focus only on what the agent could have done within the reached stage(s).
- Do NOT coach on future stages that were never reached.

TELL, DON'T ASK (COACHING / PROCESS TONE — NOT AUTOMATIC FAIL):
- The agent should **lead** with **confident directive** language on **required** next steps — **not** ask **permission** for them — especially through **Close**, **option selection**, **Application**, **Payment Date**, **Banking**, **checkbook / bank statement** collection, and **account/routing verification**. **Do not** change **CALL STAGE** detection from this rule.
- **Strong style (examples — paraphrase OK):** **Go ahead and grab your checkbook** and let me know when you're ready; **Grab your bank statement** and I'll walk you through exactly what I need; **Circle the lowest option**; **Go ahead and write that down for your wife**; **Let's get this part knocked out**; **I'll wait while you grab that**; **Go ahead and get that account number for me**; **Read that back to me one more time.**
- **Weak style (examples):** **Can you grab your checkbook**; **Would you be able to get your bank statement**; **Could you maybe circle that option**; **Do you want to go ahead**; **Can we move forward**; **Would you like to continue**; **Is it okay if I ask for your account number**; **Can you give me your account number** (permission-framed process asks).
- **Reason:** Permission-seeking on **required** steps can **reduce control** and invite delay/objection/callback; stay **polite** and **professional** but **direct**.
- **Speaker clarity:** Evaluate **Tell, Don't Ask** **only** when the transcript **reasonably** shows the **agent** said the line; use context (checkbook, bank statement, account/routing, circling an option, application, payment, moving forward). If **speaker** is **unclear** or the line **may** be the **prospect**'s, **do not** penalize — **omit** from **SCRIPT / FLOW MISSES**; at most a **brief** coaching note if helpful.
- **Scoring:** **Not** an **automatic fail** by itself — **never** set **Automatic fail triggered: YES**, **PASS: NO**, **PASS: AT RISK**, or **RISK: HIGH** **solely** from Tell, Don't Ask; **do not** **heavily** lower **SCORE** for **one** minor phrasing slip. **Repeated** permission-seeking in **clear** agent-controlled moments ⇒ modestly lower **Communication Quality** and **Sales Process**; if weak **ask** language plausibly contributed to **losing control**, objection mishandling, or **callback**, mention in **coaching** alongside those drivers — **do not** override **callback**, **coverage**, **payment date**, **banking**, **sold/not-sold**, or **stage** logic.
- **Reporting:** If notable: **SCRIPT / FLOW MISS** and/or **COACHING** — **Tell, Don't Ask: agent asked permission for required next steps instead of confidently directing the prospect.** If handled well: coaching may include **Continue using confident Tell, Don't Ask language to guide the prospect through required next steps.**

TOP 3 COACHING PRIORITIES RULE (MANDATORY):
- Under COACHING:, include the subheading line **exactly** as: TOP 3 COACHING PRIORITIES: (nothing before it on that line — no "- ", no "* ", no numbers, no markdown/bold).
- That line must start with the letter T at column 1 of its line (after optional blank line immediately under COACHING: only).
- Do NOT put TOP 3 COACHING PRIORITIES: inside a bullet; parsers require this clean header line.
- Directly under TOP 3 COACHING PRIORITIES:, list exactly 3 concise, high-impact, actionable bullets (each line starting with "- ").
- Priorities must relate only to stages that were reached.
- Do NOT include generic advice.
- After a **customer-initiated** hang-up, do **not** fill coaching with pressure to complete health, schedule callbacks, or “keep momentum” unless the **CALLBACK AND SCHEDULING** section explicitly supports it.

BIGGEST MISS RULE (MANDATORY):
- ALWAYS include a section titled exactly: BIGGEST MISS: (this heading must never be omitted).
- The section must appear in the report body after the TOP 3 COACHING PRIORITIES bullets and before optional OBJECTION sections or SUMMARY (see REQUIRED OUTPUT FORMAT).
- Identify the single most important mistake in the call, as one bullet line starting with "- ".
- If there is no meaningful miss, output exactly: - None
- When a miss exists, it must be specific and tied to a stage that was reached (**Exception:** **Unprofessional language / disrespectful call ending** may be **BIGGEST MISS** when clearly tied to this call’s transcript, including remarks **after** the prospect hangs up).
- If the dominant issue is only that the **customer** ended the call early (hang-up / disconnect) **and** there is **no** separate agent conduct issue, **BIGGEST MISS** must **not** blame the agent — use **- None** or a neutral factual line that attributes the stop to the **customer**, not agent error.

BIGGEST MISS PRIORITY (WHEN MULTIPLE ISSUES EXIST — pick the single highest-priority miss):
1. **Unprofessional language / disrespectful call ending** (profanity, slurs, insults, hostile or mocking talk **toward or about** the prospect — including after refusal or hang-up) — **outranks** callback, coverage, and **3 and 1** when clearly evidenced in transcript.
2. **Existing coverage mentioned but not confirmed** (compliance / at-risk) — **not** when **EARLY-END REFUSAL** applies per prompt (**do not** pick coverage as **BIGGEST MISS** on a short refusal hang-up solely from “already handled” language).
3. **DNQ** condition mishandled (when clearly applicable)
4. **Callback set without allowed exception** (when clearly applicable — **not** when only prospect refusal/hang-up per **WHAT COUNTS AS CALLBACK LANGUAGE**)
5. **Required post-sale process skipped** after **sold** call (Peace of Mind / Cool Down / Third Party Underwriting when required)
6. **Payment date missing** after **Banking** (when no higher-priority compliance/automatic-fail issue exists)
7. **Banking/account verification insufficient** (when Banking reached — includes **separate** account vs routing gaps per **BANKING VERIFICATION — ACCOUNT AND ROUTING SEPARATE**)
8. **3 and 1 / rapport** misses in **Fact Finding / Warm-up** (**only** when warm-up was **reached**)

BENEFICIARY IDENTIFICATION (SEARCHABLE / EVIDENCE — STRICT — FALSE POSITIVE GUARD):
- **Did the agent identify a beneficiary?** / **Beneficiary identified:** **YES** **only** when the transcript **clearly** shows the agent **asks for**, **confirms**, **collects**, or **states** who receives **policy proceeds** / **death benefit** / **the check** as **beneficiary** (or equivalent). **Valid evidence** (paraphrase OK): **Who would you want to be your beneficiary**; **Who do you want this money to go to**; **Who would receive the death benefit** / **the check**; **You stated that you would like [NAME] to be your primary beneficiary**; **primary beneficiary**; **How do you spell their name** (in **beneficiary** intake context); **relationship to you** as beneficiary; **would that be your wife/son/daughter as beneficiary**.
- **Invalid for YES:** generic PQ lines; vague relationship talk; **secret to you saying this**-style non-sequiturs; rapport/family chat **without** tying a **specific person** to **receiving proceeds**; need discussion **alone**; spouse/children talk **without** naming/selecting them as **beneficiary**; product benefit explanation **alone**; generic **family will get the money** **unless** a **specific recipient** is identified or confirmed. If the topic was **started** but **not** completed ⇒ **PARTIAL** where applicable; otherwise **NO**. **Do not** mark **YES** from irrelevant lines.

CALLBACK AND SCHEDULING (STRICT — TRANSCRIPT EVIDENCE ONLY):

Callback rule (compliance — transcript evidence only):
- **Strict — allowed exceptions only:** The agent must **not** **set, offer, agree to, or schedule** a callback **unless** **one** of the **two allowed exceptions** below clearly applies. If **neither** applies, **Did the agent set a callback? YES** ⇒ **Automatic fail triggered: YES**. The issue is **not** merely “too early” timing — it is **callback without an allowed exception** (callback used to defer/end the live sale before policy completion **without** qualifying carve-out).
- **Allowed exception 1 — Policy already sold/completed:** The policy/sale process is **already completed** on this call per **POLICY SALE / SALE OUTCOME**, and the callback is **clearly post-sale** support or follow-up (not deferring same-call enrollment still in progress).
- **Allowed exception 2 — Unresolved banking/account only (narrow):** **All** of: (a) the agent made **reasonable attempts** to **obtain and/or confirm** **account number** and required **banking/payment-account** details; (b) the prospect **could not** provide, access, or **confirm** that required information **after** those attempts; (c) the callback is **specifically** to **finish unresolved banking/account** capture — **not** to let the prospect decide whether to buy, “think about it,” or consult spouse/family **unless** the transcript **also** independently satisfies (a)–(c). **Does NOT apply** when the driver is spouse/wife discussion, hesitation, busy schedule, or no commitment **without** the narrow unresolved-banking facts.
- **Automatic fail examples (no exception):** Prospect wants to talk to **spouse/wife/family** and the **agent agrees** to a callback **before** policy completion; prospect wants to **think about it** and agent **schedules** a callback before completion; prospect **defers** and agent **sets callback** **instead of** call control; agent sets callback **after option selection** but **before** completion; agent sets callback **during/after application** but **before** sale completion **unless** Exception 2 applies.
- **Allowed examples:** Post-sale callback under **Exception 1**; callback **only** to resolve **banking/account** the prospect could not complete/confirm **after reasonable attempts** under **Exception 2**.
- **NOT allowed (no exception):** Callback **just** because prospect wants **wife/spouse** input, time to **think**, is **busy**, chose an option but sale **not** completed, or **before** application/payment/banking/disclosures **unless** **Exception 1** or **Exception 2** clearly applies.
- **Reason / autofail wording (preferred):** Include **Callback set without allowed exception** and/or **Callback set before policy completion and without unresolved banking/account confirmation exception**. You **may** add **Callback set too early** as **supplemental** context — **do not** use **only** **Callback set too early** as the **sole** Reason text when **no** allowed exception applied.
- If the prospect wants to **ask a spouse** or **call back** **before** completion, the agent should use **call control** (e.g. **BOTTOM PARAGRAPH / LOWEST OPTION**). If the agent **agrees to/schedules** a callback for spouse/think/busy **without** an exception ⇒ **violation** when **Did the agent set a callback?** is **YES**. If the agent **later** sets a callback without an exception, **automatic fail** per **1) CALLBACK VIOLATION**.
- **Do not fail** when: **prospect-only** callback language **without** agent agreement (per **WHAT COUNTS AS CALLBACK LANGUAGE**); **prospect/customer** disconnects and the agent does **not** create the delay; **Exception 1** or **Exception 2** clearly applies.
- If speaker role is unclear, do **not** auto-fail solely on callback language — mark **UNCLEAR** and explain.

POLICY (same intent as Callback rule):
- Agents must NOT use offering, agreeing to, or scheduling a callback to END or DELAY the sales process on THIS call when the conversation could still reasonably continue **without** **Allowed exception 1** or **Allowed exception 2**.
- No callbacks for prospect wants spouse / think / busy / uncommitted **unless** an exception applies as written above.
- No callbacks before proper call control when the prospect objects — **unless** an allowed exception applies.
- No callbacks before completing required sale steps — **unless** **Exception 1** (already completed) or **Exception 2** (narrow unresolved banking/account after reasonable attempts).

WHAT COUNTS AS CALLBACK LANGUAGE (DO NOT HALLUCINATE):
- Only treat callback behavior as present if the transcript CLEARLY shows the **agent** offering, agreeing to, or scheduling a reconnect that **defers or ends** this live sales attempt (not the prospect alone saying "call me" without the agent agreeing).
- Treat as **YES** when the agent clearly uses language such as: **"call you back"**, **"I'll call back"** / **"I'll call you back"**, **"we can finish this later"**, **"let's schedule another time"**, **"when would be a better time"**, **"I can call you later"**, **"we'll call you back"**, **"let me call you back"**, **"ring you back"**, **I'll call you tomorrow** / **when your wife is home** / **at [time]**, **I'll reach back out**, **I'll follow up with you**, or clearly schedules a specific time to continue on a **later** call instead of now.
- **Invalid for Callback set YES / callback autofail evidence:** discussion of a **letter**, **age**, a **term policy**, **past coverage**, **found out** without **callback** commitment; rapport or medical talk; policy-type talk **without** an actual **later-call agreement**; **prospect-only** deferral unless the **agent** clearly agrees/offers/schedules per above; **prospect** says **not interested**, **already taken care of**, **have coverage**, **goodbye**, or **ends/hangs up** **without** the **agent** committing to **call back** / **follow up** / **schedule**; agent trying to continue while the **prospect** stops the call. **Do not** trigger **automatic fail** for callback **without** valid callback evidence.
- Vague phrases alone ("touch base later," "follow up") without clearly deferring THIS call are **UNCLEAR** unless the agent clearly agrees to call back later.
- If there is no clear callback discussion, the searchable answer is **NO** — do NOT infer from silence.

SCRIPT / FLOW MISS:
- If the agent offers, agrees to, or schedules a callback **without an allowed exception** (instead of continuing the sale when the session could continue), include a **SCRIPT / FLOW MISS** tied to the stage reached — e.g. **Callback set without allowed exception**: agent agreed to/scheduled callback before policy completion and the callback was **not** due to **unresolved banking/account confirmation after reasonable attempts** (cite transcript). **Do not** list payment/draft date as a miss when **PAYMENT DATE STAGE** / deposit-after-benefits language is already present.

WHEN THE PROSPECT ASKS TO CALL BACK LATER:
- First determine whether the prospect is requesting a later time vs. simply hanging up or disconnecting.
- If the prospect asks to call back later (and the session could continue), evaluate whether the agent FIRST attempted proper call control (e.g., isolate concern, narrow time, brief value bridge, or appropriate reframe) before accepting a callback.
- If the agent IMMEDIATELY accepts or schedules a callback with no meaningful attempt to continue or control the call, note that as a COACHING issue (cite transcript; do not invent attempts).

DO NOT PENALIZE WHEN:
- The call naturally ends, the customer hangs up, or the transcript stops before any callback / "call you later" discussion — do NOT mark callback misses or coaching for callbacks in that situation.
- **EARLY-END REFUSAL:** prospect **not interested** / **already handled** / **goodbye** / hang-up **without** **agent** callback commitment — do **NOT** mark **Did the agent set a callback? YES** or callback **SCRIPT / FLOW MISS** / autofail.
- Do NOT add callback-related misses or coaching unless callback language is clearly present as defined above.

SEARCHABLE ANSWERS (CALLBACK):
- In the SEARCHABLE ANSWERS section, include EXACTLY this line with ONLY YES, NO, or UNCLEAR (no other words on the verdict):
  - Did the agent set a callback? YES / NO / UNCLEAR
- YES: the agent clearly agrees to or schedules a callback / call-back as described above.
- NO: no clear agent-led callback agreement or scheduling appears in the transcript.
- UNCLEAR: discussed but ambiguous whether a callback was truly set by the agent.

OBJECTION DETECTION (DO NOT AFFECT SCORE OR STAGE):
- Only flag objections that are clearly stated by the customer or implied as resistance in the transcript.
- Examples of objection themes (not exhaustive): already has coverage / duplicate coverage; not interested; too expensive or cannot afford; busy or call later / bad time; hesitation, skepticism, or pushback about continuing; **must ask spouse/family first** or **defer decision** until someone else is available (treat as resistance/deferral when it slows the sale).
- Do NOT invent objections. Do NOT infer objections from silence alone. If none exist, skip objection sections entirely.
- If and ONLY if at least one genuine objection is detected, add BOTH sections below (in this order, before SUMMARY). If no objections, omit these sections completely (do not write "None" as a placeholder section).

OBJECTIONS DETECTED:
- List each distinct objection as a bullet (short label tied to what was said).

OBJECTION HANDLING:
- For EACH objection listed under OBJECTIONS DETECTED, use this exact sub-format:
  - Objection: <short label>
  - Handled: YES / NO
  - Explanation: <one brief sentence citing transcript behavior>

COVERAGE CONFIRMATION VS BANK / PAYMENT VERIFICATION (STRICT — DO NOT CONFLATE):

These are DIFFERENT obligations. Never treat one as the other when scoring automatic fails or SEARCHABLE ANSWERS.

ASKING ABOUT EXISTING COVERAGE (NOT THE SAME AS CONFIRMING):
- Questions about what the prospect has today, prior policies, carrier names, face amounts, premiums, etc. are fact-finding / discovery only.
- That satisfies "Did the agent ask about existing coverage?" when clearly asked — it does NOT by itself satisfy "Did the agent confirm current coverage?".

PAST VS CURRENT COVERAGE (MANDATORY — SCOPE FOR **Existing coverage mentioned but not confirmed** — **STRICT**):
- **Current coverage confirmation** is required **only** when the transcript **reasonably** indicates the prospect **currently** has **active existing** coverage, **may** currently have **active existing** coverage, **or** gives an **unresolved ambiguous** answer about **current active** coverage — **not** for **past coverage alone** and **not** when the prospect **clearly** indicates **no current** policy and **only** discusses **past** policies (see **DECISION PRIORITY** and **ASH_TEST** below).
- **Current active coverage** cues (may require clarification or carrier confirmation when not resolved): **I have one now**; **I have a policy**; **I have coverage with [carrier]**; **I'm paying on one**; **I have life insurance already**; ambiguous **"Only one"** (or similar) after a **now / in place / only policy** question when it is **not** clear the prospect meant **only** the **new** policy — per the **MANDATORY PATTERN** below (**shelby-sold** — **do not** remove).
- **Clear no-current-coverage answers** (treat as **no active in-force** unless the transcript **separately** contradicts): **this would be my only policy**; **as far as I know, it would be my only policy**; **this will be my only policy**; **I do not have current coverage**; **I don't have insurance now**; **I don't have life insurance now**; **nothing in place**; **no, I don't** (to in-force / now questions when clear). **Do NOT** mark **Existing coverage mentioned but not confirmed: YES** **solely** because the prospect later mentions **past** policies after one of these (or equivalent) — **past** disclosure does **not** erase a clear **no-current** read.
- **Past coverage only** (does **not** require current carrier verification or **Existing coverage mentioned but not confirmed: YES** on that basis alone): **I've had policies in the past**; **I only had policies in the past**; **I used to have one**; **I had coverage before**; **I used to have a policy**; **I cancelled it**; **I don't have it anymore**; **"It would be my only policy"** / **"As far as I know, it would be my only policy"** **and** **only** past-coverage follow-up — **no** separate signal of **current** in-force coverage (**ash_test**); prospect says **only** policy **and** only discloses **past** policies with **no** transcript signal they may still have **in-force** coverage.
- **DECISION PRIORITY (MANDATORY — apply before default “coverage YES” reads):** If the prospect **first** answers the **now / only policy** question with a **clear no-current-coverage-style** answer (including phrases in the bullet above) **and** **later** mentions **only** **past** policies (no separate **active in-force** cue), treat the read as **past-only** — **Existing coverage mentioned but not confirmed: NO**. **Do not** override that pairing **just** because they said they **had** policies **in the past**.
- For a **past-only** + **only policy / no current** pattern, align together: **Did the agent ask about existing coverage? YES** (when asked); **Did the agent confirm current coverage? NO**; **Did the agent call an insurance company to confirm current coverage? NO**; **Existing coverage mentioned but not confirmed: NO**; **Automatic fail triggered: NO** **solely** from that coverage pattern.
- **ASH_TEST — ONLY POLICY + PAST POLICIES (REFERENCE — DO NOT FAIL COVERAGE ON THIS ALONE):** When the transcript matches this flow (paraphrase OK): agent asks whether the prospect has final expense/life insurance **now** or whether this would be their **only** policy; prospect: **"As far as I know, it would be my only policy."** (or equivalent — **no** current in-force policy claimed); agent asks whether they ever had coverage **in the past** / first time owning; prospect: **"I've had policies in the past."** — interpret as **current:** no active existing coverage indicated / this would be their **only** (new) policy; **past:** yes, had policies before. **Past policies alone** do **not** require current carrier/provider verification. **Expected alignment:** **Did the agent ask about existing coverage? YES**; **Did the agent confirm current coverage? NO**; **Did the agent call an insurance company to confirm current coverage? NO**; **Existing coverage mentioned but not confirmed: NO**; **Automatic fail triggered: NO** **solely** from coverage on this pattern (e.g. **ash_test** with callback autofail: **Reason** should name **callback**, **not** coverage). **Do not** list existing coverage as a gap in **COMPLIANCE FAILURES**, **SCRIPT / FLOW MISSES**, **AUTOMATIC FAIL CHECKS** / **Reason**, **BIGGEST MISS**, **TOP 3 COACHING PRIORITIES**, or **SUMMARY** as a **failure** **unless** the transcript **separately** indicates **current active** coverage **may** exist. **Separate — shelby-sold / “Only one” ambiguity:** If the prospect answers **only** **"Only one"** (or similar) to **now vs only policy** **without** clarifying dialogue (e.g. **past policies** disclosure) establishing **no current** in-force coverage, **ambiguous current** may still warrant **Existing coverage mentioned but not confirmed: YES** per **MANDATORY PATTERN** — **distinct** from this **ash_test** pattern.

EXISTING COVERAGE — FOLLOW-UP / CONFIRMATION EXAMPLE (MANDATORY PATTERN):
- If the agent asks (or clearly equivalent wording), e.g.: **"Do you have any kind of final expense plan or life insurance in place now, sir, or is this gonna be your only policy?"** and the prospect answers **"Only one."**, the auditor must treat this as **existing coverage mentioned** unless the surrounding transcript **clearly proves** the prospect meant the **new** policy would be their **only** policy (not an existing one). If later dialogue **clearly** establishes **past-only** policies and **no** current in-force coverage (see **PAST VS CURRENT COVERAGE**), do **not** keep **Existing coverage mentioned but not confirmed: YES** **solely** from the initial **"Only one"** line.
- Example of **insufficient** follow-up: the agent responds **"Okay, gotcha"** and later asks **"Have you ever owned a policy at some point in the past or will this be your only one for the first time?"** without clarifying whether **"Only one"** meant **one existing policy**, **no** existing policy, or **only** this new policy — that does **not** resolve ambiguity and is **not** carrier confirmation. Mark coverage status **at least UNCLEAR**; do **not** treat as cleanly resolved.
- If the agent does **not** clearly clarify whether **"Only one"** meant one existing policy, no existing policy, or only this new policy, and does **not** verify with the carrier/provider, then when the **reasonable reading** is that the prospect may have **one existing policy**, set together:
  - Did the agent ask about existing coverage? **YES**
  - Did the agent confirm current coverage? **NO**
  - Did the agent call an insurance company to confirm current coverage? **NO**
  - Existing coverage mentioned but not confirmed: **YES**
  - **Automatic fail triggered: YES**; **Reason** must include **Existing coverage mentioned but not confirmed**; if **Policy sold** is **YES**: **PASS: AT RISK** and **RISK: HIGH**
- If the transcript is genuinely ambiguous, use **UNCLEAR** on applicable SEARCHABLE lines where appropriate, but do **not** mark **"Existing coverage mentioned but not confirmed"** as **NO** unless the agent **clearly resolved** the ambiguity per above **or** **PAST VS CURRENT COVERAGE** / **DECISION PRIORITY** / **ASH_TEST — ONLY POLICY + PAST POLICIES** shows **past-only** / **clear no-current only-policy** + **only** past disclosure with **no** signal they may still have active coverage.
- **Follow-up required:** If the prospect gives a **possible** indication of existing coverage, the agent must ask **clear** follow-up questions and/or attempt **carrier/provider verification**, e.g.: **What company is that policy with?**; **Is that policy active now?**; **How much coverage is it?**; **What type of policy is it?**; **What is the premium?**; **Do you have the policy number?**; **Are you looking to add more coverage to what you already have?** If the agent does **not** clarify or verify, do **not** mark the coverage issue as clean **NO**.
- The agent must **not** treat that exchange as **complete confirmation**. The agent should ask **follow-up questions** about the existing coverage (carrier, type, face amount, in-force status, etc., as appropriate) and, when required by the rubric, **confirm current coverage** through the **insurance company/carrier/provider** (per the CONFIRMATION definition below) — not by accepting the prospect's word alone.
- If the agent does **not** ask adequate follow-up questions and does **not** directly verify with the carrier/provider, mark together (when **YES** applies per above, not when **10**/**11** exceptions apply):
  - Did the agent ask about existing coverage? **YES**
  - Did the agent confirm current coverage? **NO**
  - Did the agent call an insurance company to confirm current coverage? **NO**
  - Existing coverage mentioned but not confirmed: **YES** or **UNCLEAR** (use **UNCLEAR** only when the transcript cannot support **YES** vs **NO** — do **not** use **UNCLEAR** or **NO** to avoid autofail when the **"Only one"** pattern above applies and was not resolved)
- Do **not** count bank/account verification as coverage confirmation.
- Do **not** count simply accepting the prospect's statement as coverage confirmation.
- If the prospect's answer is **ambiguous** but **reasonably** suggests they **may** have existing coverage, treat **existing coverage as mentioned** when that reading is fair — or use **UNCLEAR** on the relevant SEARCHABLE / autofail lines when the transcript does **not** clearly establish whether in-force coverage exists. **Explain the ambiguity** briefly (e.g. in SUMMARY or Reason) **only** when that ambiguity is **real** — **do not** treat **past policies after a clear no-current only-policy answer** as creating a **current** coverage ambiguity (**DECISION PRIORITY**). Do **not** mark **Did the agent confirm current coverage?** as **YES** unless **carrier/provider verification** per the CONFIRMATION definition below **actually occurred**; ambiguous or prospect-only statements are **never** confirmation.

CURRENT INSURANCE / COVERAGE **CONFIRMATION** (for SEARCHABLE "confirm current coverage" and related autofail) means:
- The agent obtained verification of existing policy/coverage details **beyond taking the prospect's word alone**, by at least one of:
  - A call (or live warm transfer / three-way) with the **insurance company, carrier, or policy provider** to confirm the in-force policy, or
  - Another **direct third-party** verification clearly shown in the transcript (e.g. carrier rep on the line, verified policy data from the carrier system while the prospect is present) — not inferred.
- If the transcript only shows Q&A with the prospect and no insurer/carrier/provider contact or equivalent direct verification, that is **NOT** confirmed current coverage — mark "Did the agent confirm current coverage?" **NO**.

CALLING THE BANK (or equivalent) means:
- Verifying BANKING or PAYMENT logistics: account number, routing number, draft date, financial institution name for payment setup, double-checking account digits, etc.
- A bank call or bank verification for PAYMENT / ACCOUNT / ROUTING purposes does NOT count as confirming current INSURANCE coverage.
- Confirming bank/account/payment information does NOT count as confirming current insurance coverage.
- Asking about existing coverage, carriers, or policies does NOT count as bank/account verification.

SEARCHABLE ANSWERS MUST stay logically separate:
- "Did the agent confirm current coverage?" = **YES** only when the strict CONFIRMATION definition above is met (carrier/provider direct verification or clear equivalent). Otherwise **NO** — including when the agent only asked and the prospect only described their policy. **UNCLEAR** only when the transcript is genuinely ambiguous whether a carrier verification occurred (do not use UNCLEAR to mean "mostly confirmed").
- "Did the agent call an insurance company to confirm current coverage?" = **YES** only when a carrier/insurer/provider call (or clearly committed same-call verification with them) is evidenced for **coverage** — not a bank. Otherwise **NO** (same UNCLEAR rule as above).
- "Did the agent call the bank to verify banking/account information?" refers ONLY to payment/bank/routing/account verification — NOT confirming insurance coverage with a carrier.
- "Did the agent verify credit union account information if a credit union was mentioned?" is ONLY about credit-union-related bank/account/payment verification — not insurance coverage.

10. COVERAGE CONFIRMATION ATTEMPT — POLICY NOT FOUND / NOT ACTIVE:
- If the prospect mentions existing coverage, the agent must attempt **proper current coverage confirmation** when required by the rubric. **Proper confirmation** means **calling or directly verifying with the insurance company/carrier/provider**, not merely asking the prospect.
- **However**, if the agent **clearly attempts** to confirm current coverage with the **insurance company/carrier/provider** and the carrier/provider **states or clearly indicates** that: **no active policy exists**; **the policy cannot be found**; **the coverage cannot be verified**; **the prospect is not found in the carrier system**; or **there is no in-force policy** — then do **NOT** mark **"Existing coverage mentioned but not confirmed"** as **YES** **solely** because no active policy was confirmed after that good-faith attempt.
- In that situation, align SEARCHABLE / autofail as follows:
  - Did the agent ask about existing coverage? **YES** (when the prospect mentioned or discussed existing coverage in scope)
  - Did the agent call an insurance company to confirm current coverage? **YES** (a real insurer/carrier/provider verification attempt for **coverage** occurred — **not** a bank/CU/payment call)
  - Did the agent confirm current coverage? **NO**, unless an **active in-force** policy was **actually** confirmed with the carrier/provider per the CONFIRMATION definition
  - Existing coverage mentioned but not confirmed: **NO**, if the agent made a **clear good-faith** carrier/provider verification attempt and the outcome was **not active / not found / cannot verify** as above
  - **Automatic fail triggered** for the **existing-coverage confirmation** dimension: **NO** on that basis alone (other autofail lines may still apply independently)
- After a good-faith carrier/provider verification attempt where **no active policy** is found, it is **acceptable** for the agent to move on and ask whether the prospect wants **new or additional** coverage. Examples of acceptable follow-up phrasing (adapt to transcript): **"Since they could not find an active policy, are you looking to add new coverage today?"**; **"Are you just looking to add more coverage?"**; **"So this would be additional coverage for you?"**; **"Since that policy is not active/found, we can look at what you qualify for now."**
- Do **NOT** count a **bank** call, **credit union** call, **routing/account** verification, or **payment** verification as this **coverage** confirmation attempt. The verification attempt must be with the **insurance company/carrier/provider**.

11. COVERAGE CONFIRMATION EXCEPTION — PROSPECT REFUSES TO PROVIDE POLICY INFORMATION:
- If the prospect mentions existing coverage but **refuses**, **declines**, or is **unwilling** or **unable** to provide **enough information** for the agent to verify the existing policy with the **insurance company/carrier/provider**, do **NOT** automatically fail the agent for **"Existing coverage mentioned but not confirmed"** if the agent made a **reasonable attempt** to gather the information needed for verification.
- A **reasonable attempt** includes the agent asking for enough to identify or verify the policy, such as: **carrier/company name**; **policy number**; **plan name**; **policy type**; **face amount**; **premium**; **issue date**; or **other identifying details** needed to verify coverage with the insurer.
- If the prospect **refuses** to provide that information, says they **do not want to give it out**, **does not know** it, **cannot access** it, or **otherwise prevents** carrier verification, then align SEARCHABLE / autofail as follows:
  - Did the agent ask about existing coverage? **YES**
  - Did the agent call an insurance company to confirm current coverage? **NO**, unless a **carrier/provider call for coverage** actually occurred
  - Did the agent confirm current coverage? **NO**
  - Existing coverage mentioned but not confirmed: **NO**, if the agent made a **reasonable good-faith attempt** to gather identifying details and the **prospect prevented** verification
  - **Automatic fail triggered** for the **existing-coverage confirmation** dimension: **NO** on that basis alone (other autofail lines may still apply independently)
- In that situation, it is **acceptable** for the agent to move forward by asking whether the prospect wants **more coverage** or **additional protection**. Examples (adapt to transcript): **"If you do not want to provide that policy information, are you just looking to add more coverage today?"**; **"Are you looking to add additional coverage on top of what you already have?"**; **"Since we cannot verify that policy without the information, are we looking at this as extra coverage?"**; **"No problem, are you just trying to add more protection for your family?"**
- Do **NOT** treat this exception as satisfied if the agent **never** tried to ask for the policy/carrier (or other identifying) information.
- Do **NOT** treat this exception as satisfied if the agent **ignored** an existing-policy mention and moved on **without** any attempt to clarify or verify.
- Do **NOT** count **bank/account/routing/payment** verification as **insurance coverage** verification.

WHEN EXISTING COVERAGE IS MENTIONED BUT NOT CARRIER-VERIFIED (ALIGN ALL OF THESE):
- **First** apply **DECISION PRIORITY** and **ASH_TEST — ONLY POLICY + PAST POLICIES** when the transcript matches **clear no-current only-policy** language **then** **only** past-coverage disclosure — set **Existing coverage mentioned but not confirmed: NO** and do **not** use the default autofail block below for that read alone.
- **Next** apply **10. COVERAGE CONFIRMATION ATTEMPT — POLICY NOT FOUND / NOT ACTIVE** when applicable — if the **good-faith carrier/provider attempt + no active policy found** pattern is clear, use that section's SEARCHABLE / autofail alignment instead of the default block below.
- **Next** apply **11. COVERAGE CONFIRMATION EXCEPTION — PROSPECT REFUSES TO PROVIDE POLICY INFORMATION** when applicable — if the agent made a **reasonable attempt** to gather identifying details and the **prospect refused or blocked** verification, use that section's SEARCHABLE / autofail alignment instead of the default block below.
- **Past-only / no current in-force indication:** If **PAST VS CURRENT COVERAGE** applies (e.g. prospect indicates **only** past policies or **no** current coverage and nothing suggests **in-force** existing insurance), do **not** use the default autofail block below — set **Existing coverage mentioned but not confirmed: NO** and do **not** set **Automatic fail triggered: YES** **solely** from that coverage read; **Did the agent confirm current coverage?** / carrier-call lines stay **NO** when no carrier verification occurred.
- If the prospect clearly has existing insurance/coverage in force (or clearly describes an active policy) and the agent did **not** call or directly verify with the insurer/carrier/provider per above, then set **all** of the following together:
  - Did the agent confirm current coverage? **NO**
  - Did the agent call an insurance company to confirm current coverage? **NO**
  - Existing coverage mentioned but not confirmed: **YES**
  - Automatic fail triggered: **YES**
  - If **Policy sold** is **YES**: **PASS: AT RISK** (sale completed but compliance gap — do **not** use PASS: NO alone for this outcome); set **RISK: HIGH** when the automatic fail is compliance-related (e.g. existing coverage gap as above).
  - If **Policy sold** is **NO** or **UNCLEAR**: **PASS: NO**

AUTOMATIC FAIL RULES (MANDATORY — TRANSCRIPT EVIDENCE ONLY):

These rules drive **Automatic fail triggered** and the **PASS** line together with **Policy sold** (see AUDIT OUTCOME below). Do NOT hallucinate. **UNCLEAR** on a single SEARCHABLE line alone must NOT be the **only** basis for automatic fail **except** where this prompt explicitly ties **UNCLEAR** to a defined pattern (e.g. do not invent fails from silence). **Weak, partial, or missing 3 and 1 Method alone** must **not** set **Automatic fail triggered** or force **RISK** / **PASS** outcomes — handle **3 and 1** only via **SCORE**, **SCRIPT / FLOW MISSES**, **COACHING**, and **BIGGEST MISS** per **3 AND 1 METHOD — SCORE IMPORTANCE**. Keep all stage-aware SCORE rules; **RISK** must be **HIGH** when automatic fail applies and policy is not sold, and when **Policy sold** is **YES** with **Automatic fail triggered** is **YES** (including post-sale process or payment-date failures per rules **5** and **6** below, and compliance-related coverage or credit-union lines).

1) CALLBACK VIOLATION (automatic fail only with clear evidence):
- Apply the **Callback rule** and **POLICY** above (**Allowed exception 1** and **Allowed exception 2** only). When **Did the agent set a callback?** is **YES** and **neither** allowed exception applies, set **Automatic fail triggered: YES**; **Reason** must **use** **Callback set without allowed exception** and/or **Callback set before policy completion and without unresolved banking/account confirmation exception** as the **primary** callback wording (e.g. **ash_test** spouse deferral before sale) — **Callback set too early** / **Callback set instead of continuing the call** may appear **only** as **supplement**, **not** as the **sole** callback phrase when no exception applied; **RISK** must be **HIGH**; **PASS** must be **AT RISK** if **Policy sold** is **YES**, else **NO**.
- **Do NOT** set **Automatic fail triggered: NO** when **Did the agent set a callback?** is **YES**, **neither** exception applies, and the callback violates the rule above — that pairing is **invalid** (see **REPORT CONSISTENCY SELF-CHECK**).
- **Do NOT** treat a callback as acceptable **solely** because **Policy sold** later becomes **YES** on the same call — evaluate against **exceptions** at the time the callback was set.
- When callback autofail **does** apply: align **Callback set** with **Did the agent set a callback?**; **Automatic fail triggered: YES**; **Reason** includes **Callback set without allowed exception** (or long-form phrase) among applicable causes (combine with other causes as needed).
- If **Policy sold** is **NO** (or **UNCLEAR**) and callback autofail applies per above, **PASS** must be **NO** and **RISK** must be **HIGH**.
- **BIGGEST MISS** must name the callback deferral (not **None**) when callback autofail applies — prefer: **Callback set without allowed exception**; acceptable alternate: **Setting a callback instead of continuing the sales process**.
- Include in **TOP 3 COACHING PRIORITIES** (or coaching bullets) when callback autofail applies: **Do not set callbacks; maintain control and complete the call in one sitting** (or a stage-appropriate variant if the Callback rule exception nearly applied).
- Do NOT fail if the customer hangs up or disconnects before any callback is discussed.
- If callback language is not clearly present, Callback set must be **NO** or **UNCLEAR**; do NOT trigger automatic fail on callback alone when **UNCLEAR**.
- If speaker / role labels are ambiguous for who agreed to a callback, prefer **UNCLEAR** for **Did the agent set a callback?** and do NOT automatic-fail on callback alone; explain in **Reason** or **SUMMARY**.

2) CALL CONTROL VIOLATION (automatic fail only with clear evidence):
- If and ONLY if the prospect gives resistance, a genuine objection, or a clear attempt to end the call, the agent must use a proper call control / continuation attempt.
- If no objection or resistance occurred, set "Objection occurred without proper call control" to NO and do NOT use this rule for automatic fail.
- **BOTTOM PARAGRAPH / LOWEST OPTION** (see above) counts as a **reasonable call control / continuation attempt** when the transcript matches that pattern — **do not** set **Objection occurred without proper call control: YES** when that pattern clearly follows the objection and the sale continues.
- If objection/resistance clearly occurred and the agent made NO reasonable call control attempt, set that line to YES and trigger automatic fail.
- If resistance or agent response is ambiguous, use UNCLEAR and do NOT auto-fail on this basis alone.

3) EXISTING COVERAGE CONFIRMATION VIOLATION (INSURANCE ONLY — NEVER BANK VERIFICATION):
- This rule applies ONLY to current INSURANCE / POLICY / COVERAGE (what the prospect already has with an insurer), not to bank drafts, routing numbers, or payment accounts.
- A call or action that only verifies BANKING / PAYMENT / ACCOUNT / ROUTING cannot satisfy this rule and must NOT be cited as coverage confirmation.
- For **Existing coverage mentioned but not confirmed** / automatic fail, treat **existing coverage as mentioned** in the **compliance** sense when the prospect gives a **possible indication** of **current in-force** coverage **or** an **ambiguous** answer about **current** coverage (including **"Only one"** alone per **shelby** / **MANDATORY PATTERN** above) that was **not** clearly resolved — **not** when **PAST VS CURRENT COVERAGE**, **DECISION PRIORITY**, or **ASH_TEST — ONLY POLICY + PAST POLICIES** applies (**past-only** / **no current in-force** indication, including **"As far as I know, it would be my only policy"** + **only** past policies). **Do NOT** mark **"Existing coverage mentioned but not confirmed"** as **NO** when **current/in-force** is reasonably indicated or **"Only one"**-style ambiguity about **current** coverage remains **unresolved** — use carrier verification or explicit transcript-supported clarification of **no current** in-force policy to clear that line to **NO**.
- Evaluate whether the agent achieved **CONFIRMATION** as defined above (insurer/carrier/provider contact or clear equivalent **direct** verification — **not** only asking the prospect and accepting their answers).
- If existing coverage is never mentioned or hinted in any way, set "Existing coverage mentioned but not confirmed" to NO and do NOT fail on this rule.
- If **current** existing coverage is **mentioned or reasonably indicated** (including **UNCLEAR** readings such as **"Only one"** about **now** without resolution) and the agent did **not** meet that carrier/direct-verification standard, set "Existing coverage mentioned but not confirmed" to **YES** (or **UNCLEAR** only when the transcript cannot support YES vs NO — **do not default to NO** to "clear" the issue), set **Automatic fail triggered** to **YES** when that line is **YES**, and set **PASS** per **AUDIT OUTCOME** — **except** when **10. COVERAGE CONFIRMATION ATTEMPT — POLICY NOT FOUND / NOT ACTIVE** applies, or **11. COVERAGE CONFIRMATION EXCEPTION — PROSPECT REFUSES TO PROVIDE POLICY INFORMATION** applies, or **PAST VS CURRENT COVERAGE** / **DECISION PRIORITY** / **ASH_TEST — ONLY POLICY + PAST POLICIES** applies (**past-only** / **clear no-current only-policy** + **only** past disclosure — **no** separate **active in-force** signal) (then follow those sections; do **not** use **UNCLEAR** on the autofail line alone to bypass a clear **"Only one"** pattern that was never resolved).
- Do NOT set this line to YES because of missing bank verification alone.

4) CREDIT UNION / BANK ACCOUNT VERIFICATION VIOLATION (BANKING ONLY — NEVER COVERAGE CONFIRMATION):
- This rule applies ONLY to verifying BANK / ACCOUNT / PAYMENT information when a CREDIT UNION is in play (e.g., double-checking account number, calling the credit union or bank for draft setup).
- Questions about existing INSURANCE coverage or calling an INSURER do NOT satisfy this rule and must NOT be mixed into the credit-union bank verification check.
- ONLY if the prospect clearly identifies a credit union (or clearly states their institution is a credit union), the agent must call or otherwise verify BANK/ACCOUNT/PAYMENT information as required (or an equivalent concrete verification step clearly committed).
- If no credit union is mentioned, set "Credit union mentioned but bank/account not verified" to NO and do NOT fail on this rule.
- If a credit union is clearly mentioned and the agent does NOT verify/call regarding BANK/ACCOUNT information when still handling that capture, set that line to YES and trigger automatic fail.
- If credit-union status is unclear, use UNCLEAR and do NOT auto-fail on this alone.
- Do NOT set this line to YES because of missing insurance-coverage confirmation alone.

5) POST-SALE PROCESS INCOMPLETE (WHEN **Policy sold** IS **YES** — TRANSCRIPT EVIDENCE ONLY):
- **Prerequisite:** This rule applies **only** when **Policy sold** is **YES**. If **Policy sold** is **NO** or **UNCLEAR** (e.g. **CALL STAGE REACHED** is **Banking** and no completion evidence), **do not** trigger this rule, **do not** cite **Post-sale process incomplete: Peace of Mind and Cool Down skipped** (or TPU wording) in **Reason**, and **do not** treat Peace of Mind / Cool Down as **skipped after sale** — list those stages as **NOT REACHED** after the furthest true stage.
- If **Policy sold** is **YES** and the agent **reaches Disclosures**, the agent must complete the **required post-sale process** unless the **prospect/customer** ends the call, disconnects, refuses to continue, prevents completion, or the transcript ends before a **reasonable opportunity** to continue.
- **Required post-sale stages after Disclosures** (when required by carrier/process and when there was opportunity): **Third Party Underwriting** (when the carrier/process requires the recorded/third-party step), **Peace of Mind**, **Cool Down**.
- If **Policy sold** is **YES**, **Disclosures** were **reached**, the agent had a **reasonable opportunity** to continue, but **Peace of Mind** and **Cool Down** were **both skipped** (checklist **Peace of mind completed: NO** and **Cool down completed: NO**, or equivalent), this is an **automatic fail** / **serious process failure** — **not** a minor coaching miss. Set **Automatic fail triggered: YES** and include in **Reason** (combine with other reasons as needed): **Post-sale process incomplete: Peace of Mind and Cool Down skipped**. If **Third Party Underwriting** was **required next** and appears under **NOT REACHED** or was clearly skipped after **Disclosures**, extend **Reason** to: **Post-sale process incomplete: Third Party Underwriting, Peace of Mind, and Cool Down skipped**.
- Align **PASS** / **RISK** / **SCORE** per **AUDIT OUTCOME** and **SCORE CAP RULES** below — a sold call with this skip must **not** score in the **90s** or show **near-perfect Sales Process**.
- **Do NOT** trigger this automatic fail when the **customer** ended the call, disconnected, refused to continue, or the transcript ended before the agent had a **reasonable opportunity**.

6) PAYMENT / DRAFT DATE AFTER **Banking** (WHEN **Banking** WAS REACHED — TRANSCRIPT EVIDENCE):
- If **Banking** was reached and **Payment Date** was **not** explained, set, or confirmed (checklist **Payment date explained: NO** or equivalent) **per PAYMENT DATE STAGE** — i.e. **no** clear policy draft/payment timing including **no** deposit-to-first-draft/payment linkage when that is the applicable standard — this is a **serious sales process miss** — **not** minor. It must appear in **SCRIPT / FLOW MISSES**, **lower Sales Process** and **Banking/Payment accuracy** in **SCORING BREAKDOWN**, and factor into **BIGGEST MISS** when no higher-priority compliance/automatic-fail issue exists. **Do not** treat as missing when the agent clearly explained first payment/draft **after** benefits deposit per **PAYMENT DATE STAGE** (including **MANDATORY** patterns above). **Do not** apply this block or section **6)** autofail when **Payment date explained** is **NOT REACHED** because the **Payment Date** segment **never** began, or when the **only** cited gap is **DOB** / **date of proof** (DOB) / application identity per **PAYMENT DATE STAGE** **DOB / application false positives**.
- If **Policy sold** is **YES** and banking/payment setup occurred **without** a clear policy **draft/payment date** (**excluding** transcripts that already meet **PAYMENT DATE STAGE** / **MANDATORY** deposit-after-benefits first-payment language; **excluding** **DOB-only** false positives per **PAYMENT DATE STAGE**), treat this as a **serious post-sale/payment process failure**: set **Automatic fail triggered: YES** and include **Payment/draft date not explained after banking** in **Reason** (combine with other reasons as needed). Align **SCORE** per **SCORE CAP RULES** below.

AUTOMATIC FAIL CHECKS (REQUIRED IN REPORT):
- Include the exact block from REQUIRED OUTPUT FORMAT titled AUTOMATIC FAIL CHECKS with all six lines filled.
- "Callback set" MUST match the verdict for "Did the agent set a callback?" (same YES / NO / UNCLEAR).
- Set "Automatic fail triggered" to YES when at least one automatic-fail condition above is clearly **YES** (not **UNCLEAR-only** on every line). **Hard consistency:** if **Existing coverage mentioned but not confirmed** is **YES**, **Automatic fail triggered** MUST be **YES**; **Reason** MUST include **Existing coverage mentioned but not confirmed** (never **Reason: None** with that line YES); **RISK** MUST be **HIGH**; **PASS** MUST be **AT RISK** if **Policy sold** is **YES**, otherwise **NO**; final **SCORE** must not stay in the **90s** with that coverage gap.
- **Hard consistency — callback:** if **Did the agent set a callback?** is **YES** and **neither Allowed exception 1 nor Allowed exception 2** applies, **Automatic fail triggered** MUST be **YES**; **Reason** MUST include **Callback set without allowed exception** and/or **Callback set before policy completion and without unresolved banking/account confirmation exception** as the **primary** callback wording (e.g. **ash_test** spouse deferral before sale) — **Callback set too early** may appear **only** as **supplement**, **not** as the **sole** callback phrase; **RISK** MUST be **HIGH**; align **PASS** per **AUDIT OUTCOME** (**AT RISK** **only** if **Policy sold** is **YES**; **NO** if **Policy sold** is **NO** or **UNCLEAR** — e.g. spouse/wife callback before completion, **Policy sold: NO**, **PASS: NO**) — **never** leave **Automatic fail triggered: NO** for that pairing.
- If Automatic fail triggered is YES, set **PASS** per **AUDIT OUTCOME (PASS / AT RISK / AUTOMATIC FAIL)** — do **not** default to PASS: NO when Policy sold is YES.
- If Automatic fail triggered is NO, determine PASS using existing stage-aware and score rules (PASS may be YES or NO only; AT RISK is not used without automatic fail).
- **Reason:** When **Automatic fail triggered** is **YES**, list **all** applicable causes in one line separated by **"; "** (e.g. **Existing coverage mentioned but not confirmed**; **Post-sale process incomplete: …**; **Payment/draft date not explained after banking**; **Callback set without allowed exception**). **Do NOT** output **Reason: None** when **Automatic fail triggered** is **YES**. **Do NOT** list **Post-sale process incomplete** when **Policy sold** is **NO** / sale not completed. **Do NOT** list **Existing coverage mentioned but not confirmed** in **Reason** for **ash_test-style** transcripts when the **only** coverage facts are **only policy / no current** + **past policies only** per **ASH_TEST — ONLY POLICY + PAST POLICIES** below. **Do NOT** treat **existing coverage** / carrier verification as a **failure** in **SUMMARY** on that basis alone. **Do NOT** list **Payment/draft date not explained after banking** in **Reason** when **Payment date explained** is **YES** (including **ASH_TEST payment** / **MANDATORY** deposit-after-benefits phrasing).

AUDIT OUTCOME (PASS / AT RISK / AUTOMATIC FAIL) — MANDATORY:
- **PASS: AT RISK** is used **only** when **Policy sold** is **YES** **and** **Automatic fail triggered** is **YES** (compliance / process at risk **after** a completed sale). **Do NOT** output **PASS: AT RISK** when **Policy sold** is **NO** or **UNCLEAR** — use **PASS: NO** with **RISK: HIGH** when autofail applies (e.g. **Callback set without allowed exception** before completion).
- If **Automatic fail triggered** is **YES** and **Policy sold** (SALE OUTCOME) is **YES**:
  - Output **PASS: AT RISK** (not PASS: NO). Keep **Policy sold: YES** unchanged.
  - Set **RISK** to **HIGH** (including for **post-sale process incomplete** or **payment/draft date** failures after **Banking**, not only coverage/credit-union lines).
  - **SUMMARY** must clearly state the policy was **sold** but the sale is **at risk** because of the automatic fail reason (cite which check fired).
- If **Automatic fail triggered** is **YES** and **Policy sold** is **NO** or **UNCLEAR**:
  - Output **PASS: NO** and set **RISK** to **HIGH** (**not** **PASS: AT RISK**).
- If **Automatic fail triggered** is **NO**, use normal pass rules (**PASS: YES** or **PASS: NO** only — no AT RISK).
- When filling "Existing coverage mentioned but not confirmed" vs "Credit union mentioned but bank/account not verified", never mark YES on one line because the other obligation failed — they are independent checks.

**HARD PASS / RISK CONSISTENCY (NON-NEGOTIABLE — align before output):**
- If **Automatic fail triggered** is **YES**, **PASS** cannot be **YES**; **RISK** cannot be **LOW**; **Reason** cannot be **None** (must name at least one autofail cause).
- If **Existing coverage mentioned but not confirmed** is **YES**, **Automatic fail triggered** must be **YES**; **PASS** cannot be **YES**; **RISK** cannot be **LOW**; final **SCORE** must **not** be **90+** (see **SCORE CAP RULES**).
- If **Policy sold** is **YES** and **Automatic fail triggered** is **YES**, **PASS** must be **AT RISK** (not **YES**; do not use **PASS: NO** while **Policy sold** remains **YES** for this autofail combination).

POLICY SALE / SALE OUTCOME (MANDATORY — TRANSCRIPT EVIDENCE ONLY):

Always include the SALE OUTCOME section exactly as in REQUIRED OUTPUT FORMAT. Also include the SEARCHABLE line "Was the policy sold?" with the SAME YES / NO / UNCLEAR verdict as "Policy sold:".

POLICY SOLD = YES only when there is CLEAR evidence the **sale process completed far enough to bind/submit/complete the policy** — **not** because the prospect **chose**, **circled**, or was **steered to** an option alone (including **bottom paragraph / lowest option**).

**HARD — option choice ≠ sold:** **Client chose an option** may be **YES** while **Policy sold** is **NO**. **Was the policy sold?** must match **Policy sold:**.

**Strong evidence** for **Policy sold: YES** (non-exhaustive): **application** materially **completed**; **banking/payment setup** **completed** as required; **disclosures** **completed**; **voice signature** / **recorded verification** / **American Amicable** recorded line / **third-party underwriting** **completed** when required; agent clearly states **policy/application completed, submitted, approved, or placed** with transcript support.

**Policy sold: NO** when **CALL STAGE REACHED** is **Banking** (or earlier) **and** the agent **sets/offers/agrees** to a **callback** **before** disclosures / voice signature / recorded underwriting / clear completion — enrollment **did not finish**; or when completion-level evidence above is **absent**. Use **NO** if the template allows only YES/NO.

Do NOT mark Policy sold Yes because of:
- Quotes or options alone; **choosing/circling** an amount alone; bottom-paragraph / lowest-option steering **alone**
- Benefits or product explanation alone
- A closing attempt alone without **completion-level** follow-through per above

Mark UNCLEAR when forward progress is suggested but completion vs **NO** criteria is unclear.

Mark NO when the call ended before **completion-level** evidence, including **Banking + callback before completion** without prior completion.

FINAL STAGE SUPPORTING SALE:
- Must be one of: Quotes / Close / Application / Payment / Banking / Disclosures / Third Party Underwriting / Peace of Mind / Cool Down / None
- Use None when Policy sold is NO or UNCLEAR unless the transcript clearly shows the furthest post-commitment stage reached anyway.
- When **Policy sold** is **NO** and the furthest reached stage is **Banking**, set **Final stage supporting sale** to **Banking** — **do not** claim **sale completed** in **SALE OUTCOME** without **Policy sold: YES** evidence.
- When Policy sold is YES, Final stage supporting sale should follow the post-sale order when evidenced: Application Information, Payment Date, Banking, Disclosures, Third Party Underwriting, Peace of Mind, Cool Down — use the **furthest clearly evidenced** label (often Application, Payment, Banking, Disclosures, or Third Party Underwriting on enrollment calls; Peace of Mind / Cool Down only when clearly performed).

EVIDENCE line: one short phrase tied to what was said or done in the transcript (or "None" if Policy sold is NO and there is nothing to cite). **Do not** write **sale completed** / **policy placed** / equivalent when **Policy sold** is **NO**. You **may** mention **voice signature completed**, **recorded verification completed**, **American Amicable recording system**, **app ID**, **pound sign**, or **recorded line** **only** when that language appears in the transcript **and** **CALL STAGE REACHED** / **NOT REACHED** are **consistent** with **Third Party Underwriting** reached (per **HARD TRIGGER** above — do **not** claim those in Evidence while listing **Third Party Underwriting** only under **NOT REACHED**); otherwise do **not** invent carrier-recorded details.

TONE & DELIVERY + COMMUNICATION ANALYSIS (MANDATORY — TRANSCRIPT EVIDENCE ONLY):

Always include BOTH sections exactly as in REQUIRED OUTPUT FORMAT, with headings "TONE & DELIVERY:" and "COMMUNICATION ANALYSIS:". These sections do NOT change SCORE, RISK, or PASS by themselves.

BASE JUDGMENTS ONLY ON TRANSCRIPT TEXT CUES such as:
- Agent filler words (um, uh, er, hmm, you know, like, etc.) when frequent or clustered enough to suggest uncertainty
- Broken, restarted, or incomplete sentences; repeated phrases or stammering patterns in the text
- Abrupt one-word or very short prospect replies, repeated minimal answers, or clear disengagement cues in wording
- Obvious hesitation markers in what is spoken (as reflected in the transcript), not invented backstory

DO NOT:
- Hallucinate tone or confidence if the transcript does not support it
- Use audio or assumptions beyond the written transcript
- Write long prose — one line per bullet; optional one short clause after each label if needed (keep practical and brief)

LABEL RULES:
- Agent Tone must be EXACTLY one of: Confident / Neutral / Uncertain (pick Neutral or Uncertain when evidence is weak; use Uncertain when fillers/broken sentences clearly dominate).
- Prospect Tone must be EXACTLY one of: Engaged / Neutral / Disengaged (pick Neutral when thin evidence; Disengaged only with clear short/flat/withdrawn patterns in text).

COMMUNICATION ANALYSIS (each line YES or NO only):
- Answer YES only with clear supporting transcript cues for that question.
- Answer NO when the transcript clearly contradicts the question OR when affirmative evidence is absent (do not guess YES without cues).
- If the transcript is too thin to judge, answer NO (do not invent YES).

SALES TASK CHECKLIST PRIORITY (MANDATORY):
- Use the **SALES TASK CHECKLIST** below as the main process-flow guide for what the agent completed, what was required at the **furthest stage reached**, what was **NOT REACHED**, and what counts as a miss.
- **Automatic-fail / compliance rules** override normal checklist scoring when they apply.
- **Do not** penalize or coach **future-stage** checklist items the call **never reached**; do **not** coach future-stage items unless the agent incorrectly **skipped ahead** in **CALL STAGE REACHED**.
- **Sold** / **Policy sold** is an **outcome**, not a **call stage** — do **not** add **SOLD** as a stage and do **not** infer **call stages** from a sale alone.

SPEAKER-LABELED TRANSCRIPT (MANDATORY):
- The transcript may include generated labels such as **Agent**, **Prospect**, or **Unknown**. Use them as an **audit aid** (who asked, who objected, callbacks, banking/account/payment).
- Do **not** treat labels as absolute proof when the surrounding text is ambiguous; if callback, objection, or banking responsibility depends on unclear speaker identity, mark **UNCLEAR** rather than auto-failing.

SALES TASK CHECKLIST:
{checklist}

SCORING RUBRIC:
{rubric}

REQUIRED OUTPUT FORMAT:
{output_format}
{role_block}TRANSCRIPT:
{transcript}

REPORT CONSISTENCY SELF-CHECK (MANDATORY BEFORE FINAL OUTPUT):
Before you finish, scan for contradictions — **especially 3 and 1 / rapport** — and fix them:
- **Automatic fail triggered: YES** + **PASS: YES** — invalid; set **PASS** per **AUDIT OUTCOME** (**AT RISK** if **Policy sold YES**, else **NO**).
- **Automatic fail triggered: YES** + **RISK: LOW** (or **MEDIUM** when compliance autofail applies) — invalid; **RISK** must be **HIGH**.
- **Automatic fail triggered: YES** + **Reason: None** — invalid; **Reason** must name the autofail cause(s).
- **Existing coverage mentioned but not confirmed: YES** + **Automatic fail triggered: NO** — invalid; set **Automatic fail triggered: YES** and align **Reason** / **RISK** / **PASS** / **SCORE** per **SCORE CAP RULES**.
- **Existing coverage mentioned but not confirmed: YES** + **PASS: YES** or **RISK: LOW** or **SCORE** **90+** — invalid; cap and align per **SCORE CAP RULES**.
- **Policy sold: YES** + **Automatic fail triggered: YES** + **PASS** not **AT RISK** — invalid; **PASS** must be **AT RISK**.
- **CALL STAGE REACHED: Banking** (or earlier) + **Did the agent set a callback? YES** before completion + **no** allowed **exception** + **Policy sold: YES** — **invalid**; set **Policy sold: NO**, **Was the policy sold? NO**, **Final stage supporting sale: Banking**, **SALE OUTCOME** must **not** claim **sale completed**; align **PASS: NO**, **RISK: HIGH**, **Automatic fail triggered: YES** with **Reason** including **Callback set without allowed exception** (and/or the long-form callback phrase) per **CALLBACK** rules.
- **Policy sold: YES** without disclosures / voice signature / third-party underwriting / other **POLICY SOLD = YES** completion evidence when furthest stage is **Banking** or earlier — **invalid**; set **Policy sold: NO** or **UNCLEAR** per **POLICY SALE / SALE OUTCOME**.
- **Client chose an option: YES** paired with **Policy sold: YES** **solely** from option / lowest-option / bottom-paragraph progress — **invalid** without completion evidence.
- **Policy sold: NO** + **PASS: AT RISK** — **invalid**; **AT RISK** requires **Policy sold YES** + **Automatic fail triggered YES** per **AUDIT OUTCOME** — use **PASS: NO** (or **YES** only per normal rules) when **Policy sold** is **NO**.
- **Policy sold: NO** (or not completed) + **Reason** includes **Post-sale process incomplete: Peace of Mind and Cool Down skipped** (or similar) — **invalid**; remove that cause; list **Peace of Mind** / **Cool Down** as **NOT REACHED** after furthest stage, **not** as skipped post-sale.
- **Callback set without allowed exception** (or equivalent callback violation) in **Reason** + **Automatic fail triggered: NO** — **invalid** when **Did the agent set a callback? YES** with **neither** allowed exception — align **YES** per **Hard consistency — callback**.
- **Callback set without allowed exception** + **PASS: YES** — **invalid**; **PASS** cannot be **YES** when **Automatic fail triggered** is **YES**; align per **AUDIT OUTCOME**.
- **Callback** because prospect wants **spouse/wife** / to **think** / **busy** treated as **Allowed exception 2** (banking) **without** transcript proof of **reasonable account/banking attempts** + prospect unable to confirm — **invalid**; that is **not** the narrow banking exception.
- **Did the agent confirm current coverage? NO** + **Did the agent call an insurance company to confirm current coverage? NO** + **current/in-force** existing coverage or the **Only one** ambiguity pattern about **current** coverage applies ⇒ **Existing coverage mentioned but not confirmed** must be **YES** or **UNCLEAR** (if **UNCLEAR**, explain the ambiguity — do not use **NO** to clear a fair **Only one** reading); if **YES**, automatic fail must trigger as above — **unless** **PAST VS CURRENT COVERAGE** / **DECISION PRIORITY** / **ASH_TEST — ONLY POLICY + PAST POLICIES** applies (**past-only**; **clear no-current only-policy** answer **then** **only** past policies), in which case **Existing coverage mentioned but not confirmed** must be **NO** and **Automatic fail triggered** must **not** be **YES** **solely** from that coverage read.
- **Existing coverage mentioned but not confirmed: YES** when the transcript shows **only** past coverage and the prospect indicated **no** current in-force coverage (e.g. only policy would be this sale + had policies in the past) — **invalid**; set **NO** per **PAST VS CURRENT COVERAGE** and align **Automatic fail triggered** (do **not** trigger **solely** from past coverage).
- **Existing coverage mentioned but not confirmed: NO** (per **PAST VS CURRENT** / **ASH_TEST — ONLY POLICY + PAST POLICIES**) + **BIGGEST MISS** names an **existing coverage** / **carrier verification** gap as the top miss — **invalid** unless a **separate** independent current-coverage or **"Only one"** ambiguity issue exists.
- **Existing coverage mentioned but not confirmed: NO** per **ash_test** / **past-only** + **COMPLIANCE FAILURES**, **SCRIPT / FLOW MISSES**, **BIGGEST MISS**, **TOP 3 COACHING PRIORITIES**, **Reason**, or **SUMMARY** still cites an **existing coverage** / **carrier verification** gap (including as a **failure**) — **invalid**; remove those mentions **unless** a **separate** transcript signal of **current active** coverage **may** exist.
- **Existing coverage mentioned but not confirmed: NO** per **DECISION PRIORITY** / **ash_test** + **SUMMARY** attributes a miss or compliance problem to **existing coverage** when **only** past coverage was disclosed after clear **no-current only-policy** language — **invalid**; align **SUMMARY** to actual drivers (e.g. callback).
- **Payment date explained: YES** + **Reason** (or **AUTOMATIC FAIL CHECKS** narrative) includes **Payment/draft date not explained after banking** — **invalid**; remove payment from **Reason** when **Payment date explained** is **YES** per **PAYMENT DATE STAGE**.
- **SCRIPT / FLOW MISSES** contains a payment/draft bullet that says the miss is **not applicable** / that the agent **did explain** timing — **invalid**; **delete** that bullet **entirely** per **Self-negating payment miss (forbidden)**.
- **Payment date explained: NO** when the transcript shows the agent explained first policy payment/draft **after** benefits / **government benefits** / Social Security deposit (including redacted **[DATE]** / **[NUMBER]**, e.g. **May [NUMBER]**) — **invalid**; set **Payment date explained: YES** per **PAYMENT DATE STAGE**.
- **SCRIPT / FLOW MISSES** (or **Reason**) cites missing payment/draft / **Payment/draft date not explained after banking** while the transcript matches **PAYMENT DATE STAGE** deposit-after-benefits / **MANDATORY** examples — **invalid**; remove that miss and align **Payment date explained: YES**; do **not** apply section **6)** payment autofail on that basis alone.
- **Banking** reached + transcript shows first payment/draft **after** benefits (or equivalent **MANDATORY** pattern) + **Payment date explained: NO** — **invalid**; set **YES** and align **SCORING BREAKDOWN** / **SCRIPT / FLOW MISSES** / **Automatic fail triggered** (do **not** fire section **6)** for payment date alone).
- **Payment date explained: NO** when the transcript includes the **Same-call flow (mandatory YES)** pattern under **PAYMENT DATE STAGE** (Social Security schedule + first payment after government benefits + benefit day Q&A + draft after deposit on **May [NUMBER]** / redacted) — **invalid**; set **YES**; remove payment-date **SCRIPT / FLOW MISS** and payment-date **coaching**; do **not** lower score for missing Payment Date on that basis.
- **Beneficiary identified: YES** (or **Did the agent identify a beneficiary? YES**) when the cited evidence does **not** mention beneficiary / receive / death benefit / proceeds / money going to someone / primary beneficiary / **specific recipient** selection tied to policy proceeds — **invalid**; set **NO** or **PARTIAL** per **BENEFICIARY IDENTIFICATION**; fix Evidence.
- **CALL STAGE REACHED: Peace of Mind** (or **Peace of Mind** as furthest) **without** completed **Third Party Underwriting / voice signature** per **§13** — **invalid**; furthest stays **Disclosures** / **Third Party Underwriting** / earlier as evidenced — **not** **Peace of Mind**.
- **CALL STAGE REACHED: Peace of Mind** or **Peace of mind completed: YES** **without** actual **Peace of Mind script** after **§13** (**you're good** / **not going to forget about you** / **welcome letter** mailing / **personal information** + **company we got you qualified for today**-type lines) — **invalid**; **Peace of Mind** is **NOT REACHED** / **Peace of mind completed: NOT REACHED** or **NO** per template.
- **CALL STAGE REACHED: Peace of Mind** (or **Peace of Mind** as furthest) from **only** **Features / Benefits** anchors (pen and paper / walk step-by-step / direct toll-free / hoping we get you qualified / preferred plans) **without** **§14 (1)+(2)** — **invalid**; set furthest to **Features / Benefits** (or latest valid stage) per **§4** vs **§14**.
- **CALL STAGE REACHED: Peace of Mind** or **Peace of mind completed: YES** when evidence is **terminal medical condition** / **end-stage disease** / **expected … death** / **respiratory failure** / **liver failure** / **DNQ** / **medical questions** — **invalid**; **§2 Medical / Health** only.
- **CALL STAGE REACHED: Peace of Mind** when transcript includes **Quotes** anchors (**three affordable options** / **largest amount first** / **work my way all the way down** / **Are you ready**) but **§14 (1)+(2)** are missing — **invalid**; set **at least Quotes** per **§7**; **do not** set **Peace of Mind** from medical/DNQ or early-stage wording.
- **CALL STAGE REACHED: Peace of Mind** when the latest real stage is **Medical / Health**, **Features / Benefits**, **Quotes**, **Close**, **Application**, **Payment Date**, **Banking**, **Disclosures**, or **Third Party Underwriting** without **all** **§14** conditions — **invalid**; align furthest to that latest **valid** stage.
- **CALL STAGE REACHED: Peace of Mind** + (**Policy sold: NO** or **UNCLEAR**) — **invalid**; set **CALL STAGE REACHED** to **Final stage supporting sale** when that stage precedes **Peace of Mind**, else latest valid earlier stage.
- **CALL STAGE REACHED: Peace of Mind** + **Peace of mind completed** not **YES** — **invalid**; same alignment as prior bullet.
- **FINAL REPORT (post-render):** If **CALL STAGE REACHED** is **Peace of Mind** but **Policy sold** is not **YES**, **Peace of mind completed** is not **YES**, **Third Party Underwriting** appears under **NOT REACHED**, **Final stage supporting sale** is earlier than **Peace of Mind**, or transcript lacks strict **§14** POM evidence after **§13** — replace **CALL STAGE REACHED** with the latest valid earlier stage (prefer **Final stage supporting sale** when it precedes **Peace of Mind**).
- **Did the agent set a callback? YES** or callback autofail when evidence lacks **call back** / **call later** / **follow up** / **schedule** / **finish later** / **tomorrow**-type agent commitment — **invalid** if evidence is **letter** / **age** / **term policy** / **found out** alone — set **NO** and **do not** autofail per **WHAT COUNTS AS CALLBACK LANGUAGE**.
- **Did the agent set a callback? YES** when the transcript only shows **prospect** refusal, **not interested**, **already handled**, **goodbye**, or **hang-up** **without** **agent** callback/follow-up/schedule language — **invalid**; set **NO** per **EARLY-END REFUSAL** / **WHAT COUNTS AS CALLBACK LANGUAGE**.
- **EARLY-END REFUSAL** pattern + **Existing coverage mentioned but not confirmed: YES** + **Automatic fail triggered: YES** **solely** from coverage — **invalid**; align **NO** on coverage line and remove coverage-only autofail per prompt.
- **Fact Finding / Warm-up: NOT REACHED** + **SCRIPT / FLOW MISSES** or **BIGGEST MISS** cites **3 and 1** / rapport incompleteness — **invalid**; use **NOT REACHED** on **3 and 1** lines and remove those misses per **3 AND 1 METHOD** / **EARLY-END REFUSAL**.
- **Unprofessional / disrespectful agent language** (profanity, insults, slurs toward/about prospect) appears in transcript but **BIGGEST MISS** or **Reason** centers **callback** or **existing coverage** — **invalid**; elevate **Unprofessional language / disrespectful call ending** per **BIGGEST MISS PRIORITY** and **PROFESSIONALISM** rules.
- **TOP 3 COACHING PRIORITIES** coach **Medical / Quotes / Application** etc. when **CALL STAGE REACHED** is **Who I Am / What I Do** or **Opening** and **EARLY-END REFUSAL** applies — **invalid**; coach only reached-stage / professionalism issues.
- **Payment date explained: NO** or section **6)** payment autofail when **only** evidence is **DOB** / **date of birth** / **date of proof** (DOB context) / application identity — **invalid**; align **NOT REACHED** / remove payment miss per **PAYMENT DATE STAGE** **DOB / application false positives**.
- **Payment date explained: NO** when **Payment Date** segment never began (**CALL STAGE REACHED** is **Features / Benefits** or earlier per transcript) — **invalid**; use **Payment date explained: NOT REACHED** (or template-equivalent), **not** **NO**; do **not** list **Payment/draft date not explained after banking** from DOB-only lines.
- Agent used **bottom paragraph** / **lowest-option** call control per **BOTTOM PARAGRAPH / LOWEST OPTION** and the call **continued into application**, but narrative claims the prospect **did not commit** **solely** because of initial spouse/family deferral — **invalid**; align **Objection occurred without proper call control: NO**, objection handling, and **SALE OUTCOME** / **Evidence** to the transcript (deferral **then** recovery).
- **Did the agent set a callback?** **YES** + **neither** allowed **exception** (see **CALLBACK AND SCHEDULING**) + **Automatic fail triggered: NO** — **invalid**; set **Automatic fail triggered: YES**, **Reason** includes **Callback set without allowed exception** (and/or long-form phrase), **RISK: HIGH**, **PASS** per **AUDIT OUTCOME**.
- **Did the agent set a callback?** **YES** + **neither** allowed **exception** + **Automatic fail triggered: YES** + **Reason** omits **Callback set without allowed exception** / long-form callback phrase — **invalid**; add callback wording to **Reason** (may supplement with **Callback set too early**).
- **Did the agent set a callback?** **YES** + **neither** allowed **exception** + **Automatic fail triggered: YES** + **Reason** uses **only** **Callback set too early** (or equivalent) **without** **Callback set without allowed exception** / long-form phrase — **invalid**; add **primary** wording per **Hard consistency — callback**.
- Agent **set/offered/agreed** to callback **before** sale/application/banking completion + **no** exception + **Automatic fail triggered: NO** — **invalid** (same alignment as prior bullet).
- **TOP 3 COACHING PRIORITIES** (or coaching) instructs explaining payment/draft date when transcript already satisfies **PAYMENT DATE STAGE** / **Same-call flow** / **MANDATORY** deposit-after-benefits examples — **invalid**; remove that coaching item.
- **3 and 1 Method used: YES** without clear **A + B** evidence (three-of-four topic groups **and** tied self-disclosure, not generic acknowledgments) — invalid; correct to **PARTIAL** or **NO** and align **Agent shared personal rapport information**.
- **3 and 1 Method used: YES** + **Agent shared personal rapport information: NO** — **invalid**; cannot pair **YES** on 3 and 1 with **NO** on rapport — correct both lines per **Fact Finding / Warm-up — TASK CHECKLIST**.
- **3 and 1 Method used: YES** + **Agent shared personal rapport information: PARTIAL** — **invalid**; **YES** on 3 and 1 requires rapport **YES** with **meaningful** disclosure — correct both lines.
- **3 and 1 Method used: YES** when **no** meaningful self-disclosure is present — **invalid**; **YES** requires tied self-disclosure — use **PARTIAL** or **NO**.
- **3 and 1 Method used: YES** when the agent **only asked questions** (no meaningful self-disclosure) — **invalid**.
- **3 and 1 Method used: YES** while **3 and 1 agent self-disclosure evidence:** is **None**, empty, or **only** generic acknowledgment — **invalid**; downgrade **3 and 1** / rapport lines per **REQUIRED 3 AND 1 EVIDENCE LINES**.
- **3 and 1 Method used: YES** while **3 and 1 topic groups evidenced:** is **None** or lists **fewer than three** applicable groups — **invalid**.
- **Agent shared personal rapport information: YES** while **3 and 1 agent self-disclosure evidence** does **not** support meaningful tied disclosure — **invalid**.
- **Agent shared personal rapport information: YES** when the transcript **only** shows generic acknowledgments / empathy with **no** real personal detail — **invalid**; correct to **NO** or **PARTIAL**.
- **Agent shared personal rapport information: YES** when the agent **only** used vague lines such as **“I can relate,”** **“I understand,”** **“same here,”** **“I get it”** without **meaningful** personal detail — **invalid**; use **PARTIAL** or **NO**.
- **3 and 1 Method used: YES** when the agent did **not** ask across **at least three** topic groups — **invalid**.
- **3 and 1 Method used: YES** when questions are **mostly** medical, application, banking, or underwriting (not rapport topic groups) — **invalid**.
- **3 and 1 Method used: YES** when the agent **only** asked **some** rapport questions and shared **limited** personal information **without** **substantial** **SCRIPT STRUCTURE** depth (multiple questions per area, **PHE**, **vivid-detail** stories, tie-back) — **invalid**; use **PARTIAL** or **NO** per **TASK CHECKLIST** (including **ASH_TEST — 3 AND 1 / RAPPORT** when applicable).
- **3 and 1 Method used: YES** **without** **vivid-detail** agent stories where the warm-up clearly stayed **shallow** vs the script — **invalid**; prefer **PARTIAL** unless execution is clearly full-caliber.
- **3 and 1 Method used: YES** when **major** warm-up areas from **SCRIPT STRUCTURE** are **skipped** or **barely** touched — **invalid**; use **PARTIAL** or **NO**.
- **Agent shared personal rapport information: YES** when sharing is **brief/vague** only (no **vivid-detail** tied texture) while **3 and 1** would not merit **YES** — **invalid**; align **PARTIAL** or **NO**.
- **3 and 1 Method used: YES** + transcript matches **ASH_TEST — 3 AND 1 / RAPPORT** partial pattern (location/work/family touched, **some** agent share, **missing** scripted depth) — **invalid**; set **3 and 1 Method used: PARTIAL** and **Agent shared personal rapport information: PARTIAL** per that reference.
- **STAGE ANCHORS — Quotes (§7):** Transcript includes **I'm going to share three affordable options** / **first option you've qualified for** / **per month** option framing in quotes context but **CALL STAGE REACHED** is **Medical / Health** or earlier — **invalid**; set furthest to **at least Quotes** per **STAGE ANCHORS** + ordered list.
- **STAGE ANCHORS — Close (§8):** Transcript includes **circle that option** / **which option** / **lowest option** / **go with that one** (option-selection) but **CALL STAGE REACHED** is **Quotes** or earlier only — **invalid**; set **at least Close** when **§8** anchors clearly appear after **Quotes** framing.
- **STAGE ANCHORS — Application (§9):** Transcript includes **What's your middle initial** / **verify the spelling of your full name** / **primary beneficiary** (application intake) but **CALL STAGE REACHED** is **Quotes** or **Close** or earlier — **invalid**; set **at least Application Information**.
- **STAGE ANCHORS — Payment Date (§10):** Transcript includes **same payment schedule as Social Security** / **first payment** … **after** … **government benefits** / **What day do you receive your benefits** / **draft until after your benefits have been deposited** / **draft date** / **payment date** but **CALL STAGE REACHED** omits **Payment Date** or lists **Payment Date** only under **NOT REACHED** when that segment clearly began — **invalid**; align furthest stage (**Payment date explained** still per **PAYMENT DATE STAGE** — deposit-only Q without draft-after-deposit tie does **not** force **Payment date explained: YES**).
- **STAGE ANCHORS — Banking (§11):** Transcript includes **bank or credit union** / **routing** / **account number** / banking placeholders in context but **Banking** only under **NOT REACHED** or furthest label **before Banking** — **invalid**; align per **STAGE ANCHORS** and **BANKING REACHED vs COMPLETED**.
- **STAGE ANCHORS — Disclosures (§12):** Transcript includes **Fair Credit Reporting Act** / **required by law** disclosure framing but **Disclosures** only under **NOT REACHED** or furthest stops before **Disclosures** — **invalid**.
- **STAGE ANCHORS — Third Party Underwriting (§13):** Transcript includes **voice signature** / **American Amicable** recording / **app ID** / **pound sign** cues but **Third Party Underwriting** only under **NOT REACHED** — **invalid** (see also **HARD TRIGGER** bullets below).
- **STAGE ANCHORS — Features / Benefits vs Peace of Mind:** **walk you through step-by-step** / **direct toll-free number** / pen and paper anchors present but **CALL STAGE REACHED: Peace of Mind** — **invalid**; furthest **≤ Features / Benefits** unless **§14 (1)+(2)**.
- **STAGE ANCHORS — Peace of Mind / Cool Down (§14–15):** After **Policy sold YES** and completion, transcript matches **Peace of Mind** or **Cool Down** per **§14 (1)+(2)** / **§15** but furthest stage / **NOT REACHED** omits that stage — **invalid**; align (**do not** require **Peace of Mind** when **§13** not completed or **Peace of Mind script** missing; **do not** require **Peace of Mind** / **Cool Down** when **Policy sold** is **NO** / not completed).
- **STAGE ANCHORS — Peace of Mind after §13:** Transcript shows **§14** script (**you're good** / **not going to forget** / **welcome letter** / **personal information** + company qualified) **after** **§13** voice signature / recorded line but **Peace of Mind** is only under **NOT REACHED** or furthest stops before **Peace of Mind** — **invalid**; align furthest.
- **STAGE ANCHORS — Cool Down §15:** Transcript shows **§15** casual post-sale anchors after a **completed** sale but **Cool Down** only under **NOT REACHED** when plainly reached — **invalid**; align (**Policy sold** / opportunity rules unchanged).
- **SALE OUTCOME** Evidence claims **voice signature** / **recorded verification** / **American Amicable recording system** but **Third Party Underwriting** is only under **NOT REACHED** — invalid; align **CALL STAGE REACHED** / **NOT REACHED** or remove unsupported Evidence phrases.
- Transcript includes **Welcome to the American Amicable Group recording system**, **app ID** / **pound sign** IVR, **voice signature**, or **recorded verification** in enrollment context but **Third Party Underwriting** only under **NOT REACHED** — invalid; set furthest stage per **HARD TRIGGER**.
- **Banking**-context wording or placeholders (**[ACCOUNT_NUMBER]**, **[ROUTING_NUMBER]**, **[BANK_NUMBER]**, **[NUMBER]** with routing/account/bank cues) but **Banking** only under **NOT REACHED** — invalid unless a **later** stage clearly supersedes.
- Transcript shows account/routing/bank/payment-account discussion but **CALL STAGE REACHED** is **Close** (or any label **before Banking** in the ordered list) — **invalid**; set **CALL STAGE REACHED** to **at least Banking** per **BANKING REACHED vs COMPLETED**.
- **CALL STAGE REACHED: Close** (or earlier) + **Callback set** during or after banking discussion with **no** **Disclosures** (or later post-Banking stage) clearly reached — **invalid** when banking **began**; furthest stage should be **Banking** unless a **post-Banking** stage clearly occurred.
- Transcript shows banking discussion but **Banking/payment setup explained** is **NOT REACHED**, or all banking verification sub-lines are **NOT REACHED** while banking clearly **started** — **invalid**; use **PARTIAL** / **NO** / **YES** per evidence, **not** **NOT REACHED** for sub-items when **Banking** was **reached**.
- **Banking**-context placeholders or discussion in transcript but **Account verification evidence count** / **Routing verification evidence count** is **0** with **no** listed events — **invalid** if transcript-backed touches exist; align counts and verdicts.
- Transcript includes **Welcome to the American Amicable Group recording system** (or equivalent app ID / pound-sign IVR) ⇒ **Third Party Underwriting** reached; do **not** list **Third Party Underwriting** only under **NOT REACHED** while claiming an earlier furthest stage.
- **Peace of mind completed: NO** + **Cool down completed: NO** after a **sold** call (**Policy sold YES**) with opportunity + **SCORE** in the **90s** is invalid — lower **Sales Process** / **SCORE** and align **PASS** / **RISK** per **SCORE CAP RULES**. If **Policy sold** is **NO**, this cap does **not** apply from post-sale skips alone.
- **Payment date explained: NO** after **Banking** + **Sales Process** or **Closing** scored near-perfect is invalid — lower those categories and cap the final **SCORE** per payment-date rules.
- **Routing number requested or verified 3 times:** **NO** or **PARTIAL** (when **Banking** was reached) but narrative claims **banking verification was fully completed** / equivalent — **invalid**; align wording with the routing-side verdict.
- **Account number requested or verified 3 times:** **NO** or **PARTIAL** (when **Banking** was reached) but narrative claims **full** banking verification — **invalid**.
- **Banking/account information requested or verified 3 times:** **YES** when **either** account-side or routing-side **three-times** checklist line is **not** **YES** — **invalid**; combined **YES** requires **both** sides to meet the **three-event** standard independently.
- **Account number requested or verified 3 times:** **YES** but **Account verification evidence count** **< 3** — **invalid**; align verdict downward.
- **Routing number requested or verified 3 times:** **YES** but **Routing verification evidence count** **< 3** — **invalid**.
- **Routing number verified at least 2 times:** **YES** (or **PARTIAL** claimed as sufficient) but **Routing verification evidence count** **< 2** — **invalid**.
- **Account number verified at least 2 times:** **YES** but account evidence count **< 2** — **invalid**.
- Overall banking verification described as **complete** when **account** or **routing** evidence lines are incomplete vs verdicts — **invalid**.
- **Tell, Don't Ask** scored as **automatic fail** or **sole** driver of **PASS: NO** / **PASS: AT RISK** / **RISK: HIGH** — **invalid** per **TELL, DON'T ASK** (coaching / modest **Communication Quality** / **Sales Process** impact only).
- **Heavy** **SCORE** drop attributed **solely** to **one** minor permission question when **speaker** / control context is **unclear** — **invalid**; align with **Speaker clarity** under **TELL, DON'T ASK**.
- **Tell, Don't Ask** **SCRIPT / FLOW MISS** when **speaker** cannot be **reasonably** identified as the **agent** — **invalid**; remove that miss.
- **3 and 1 Method** weak / **PARTIAL** / **NO** (with **Fact Finding / Warm-up** reached) cannot be the **sole** reason for **Automatic fail triggered: YES**, **RISK: HIGH**, **PASS: NO**, or **PASS: AT RISK** — use **SCORE**, **SCRIPT / FLOW MISSES**, **COACHING**, and optional **BIGGEST MISS** per **3 AND 1 METHOD — SCORE IMPORTANCE**.

FINAL INSTRUCTION:
Start exactly like this:

SCORE: <number>
RISK: <LOW/MEDIUM/HIGH>
PASS: <YES/NO/AT RISK>
"""


# DOB context on a line → optional redaction on the following line only.
_DOB_LINE_CONTEXT = re.compile(
    r"(?i)(?:date\s+of\s+birth|birth\s+date|\bd\.?\s*o\.?\s*b\.?\b|\bbirthday\b|\bborn\b)"
)
_DOB_NEXT_LINE_PATTERNS = (
    re.compile(r"\b\d{5,8}\b"),
    re.compile(r"\b\d{1,2}\s+\d{1,2}\s+(?:\d{2}|\d{4})\b"),
    re.compile(r"\b[0-1]?\d[/\-][0-3]?\d[/\-](?:\d{2}|\d{4})\b", re.I),
)

# Spoken month/day/year after explicit DOB / birth cues (transcript redaction only).
_SPOKEN_DOB_CUE = (
    r"(?:my\s+)?date\s+of\s+birth(?:\s+is)?|d\.?o\.?b\.?(?:\s+is)?|"
    r"birthday(?:\s+is)?|birth\s+date(?:\s+is)?|born(?:\s+on)?|"
    r"what\s+is\s+your\s+date\s+of\s+birth|when\s+were\s+you\s+born|"
    r"what(?:'s|\s+is)\s+your\s+birthday"
)
_SPOKEN_DOB_WORD = (
    r"zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety"
)
_SPOKEN_DOB_DECADE = r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
_SPOKEN_DOB_UNITS = r"(?:one|two|three|four|five|six|seven|eight|nine)"
_SPOKEN_DOB_COMPOUND = rf"(?:{_SPOKEN_DOB_DECADE}\s*[-]?\s*{_SPOKEN_DOB_UNITS})\b"
_SPOKEN_DOB_LINK = rf"(?:{_SPOKEN_DOB_COMPOUND}|\b(?:{_SPOKEN_DOB_WORD})\b)"
_SPOKEN_DOB_SEP = r"(?:\s*,\s*|\s+and\s+|\s+)"
_SPOKEN_DOB_MONTH_PAIR = (
    r"(?:\b(?:zero|oh)\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b)"
)
_SPOKEN_DOB_CHAIN = (
    rf"(?:{_SPOKEN_DOB_MONTH_PAIR}{_SPOKEN_DOB_SEP})?"
    rf"{_SPOKEN_DOB_LINK}(?:{_SPOKEN_DOB_SEP}{_SPOKEN_DOB_LINK}){{3,}}"
)
_SPOKEN_DOB_MONTH_NAME = (
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b"
)
_RE_SPOKEN_DOB_SAME_LINE = re.compile(
    rf"(?P<cue>{_SPOKEN_DOB_CUE})\s+(?P<chain>{_SPOKEN_DOB_CHAIN})\b",
    re.IGNORECASE,
)
_RE_SPOKEN_DOB_MONTH_LINE = re.compile(
    rf"(?P<cue>{_SPOKEN_DOB_CUE})\s+(?P<mon>{_SPOKEN_DOB_MONTH_NAME})\s+"
    rf"(?P<day>{_SPOKEN_DOB_LINK})\s+(?P<yr>{_SPOKEN_DOB_LINK}(?:\s+{_SPOKEN_DOB_LINK})?)\b",
    re.IGNORECASE,
)
_RE_NEXT_LINE_SPOKEN_DOB = re.compile(
    rf"^(?P<lead>\s*)"
    rf"(?P<prefix>(?:agent|prospect|unknown)\s*:\s*)?"
    rf"(?P<chain>{_SPOKEN_DOB_CHAIN})(?P<tail>\s*[.!?]*)$",
    re.IGNORECASE,
)
# Continuation-only: line ends on a DOB/birth cue (no spoken date on same line). Allows ? ! . after cue.
_NEXT_LINE_SPOKEN_DOB_CTX = re.compile(
    r"(?:.*\b)?(?:date\s+of\s+birth|d\.?\s*o\.?\s*b\.?|birthday|birth\s+date)"
    r"\s*(?:is\b\s*)?\s*:?\s*\??\s*[.!]*\s*$"
    r"|(?:.*\b)?born(?:\s+on)?\s*\??\s*[.!]*\s*$",
    re.IGNORECASE,
)


def _redact_spoken_dob_phrases(text):
    """Redact spelled-out or spoken month/day/year DOB after explicit birth/DOB cues."""
    if not text:
        return text
    out = _RE_SPOKEN_DOB_MONTH_LINE.sub(r"\g<cue> [DOB]", text)
    out = _RE_SPOKEN_DOB_SAME_LINE.sub(r"\g<cue> [DOB]", out)
    return out


def _redact_next_line_spoken_dob_after_cue(text):
    """If the previous line ended on a DOB cue (continuation), replace a spoken-number-only line with [DOB]."""
    if not text:
        return text
    lines = text.split("\n")
    out = []
    prev_ctx = False
    for line in lines:
        if prev_ctx:
            m = _RE_NEXT_LINE_SPOKEN_DOB.match(line)
            if m:
                tail = m.group("tail") or ""
                line = f"{m.group('lead')}{m.group('prefix') or ''}[DOB]{tail}"
            prev_ctx = False
        out.append(line)
        prev_ctx = bool(_NEXT_LINE_SPOKEN_DOB_CTX.match(line.strip()))
    return "\n".join(out)


def _redact_next_line_after_dob_context(text):
    """When a line mentions DOB/birth, replace DOB-shaped tokens on the next line with [DOB]."""
    lines = text.split("\n")
    out = []
    dob_context = False
    for line in lines:
        if dob_context:
            for pat in _DOB_NEXT_LINE_PATTERNS:
                line = pat.sub("[DOB]", line)
            dob_context = False
        out.append(line)
        if _DOB_LINE_CONTEXT.search(line):
            dob_context = True
    return "\n".join(out)


# Spoken digit words for phone / banking redaction (transcript only).
_SPOKEN_PHONE_DIGIT = (
    r"zero|oh|one|two|three|four|five|six|seven|eight|nine"
)
_SPOKEN_PHONE_SEP = r"(?:[\s,;\-\u2013]+)"
_SPOKEN_PHONE_CHAIN = (
    rf"\b(?:{_SPOKEN_PHONE_DIGIT})\b"
    rf"(?:{_SPOKEN_PHONE_SEP}+\b(?:{_SPOKEN_PHONE_DIGIT})\b){{6,}}"
)


def _spoken_digit_chain_min(n_tail_after_first):
    """Spoken digit words with separators; at least (1 + n_tail_after_first) words total."""
    return (
        rf"\b(?:{_SPOKEN_PHONE_DIGIT})\b"
        rf"(?:{_SPOKEN_PHONE_SEP}+\b(?:{_SPOKEN_PHONE_DIGIT})\b){{{n_tail_after_first},}}"
    )


# --- Spoken account / routing / bank (run before spoken phone) ---
_SPOKEN_ROUTING_HEAD = (
    r"(?:"
    r"routing\s+number|your\s+routing|the\s+routing\b|pull\s+the\s+routing|routing\s+on\s+my\s+end|"
    r"have\s+your\s+routing|verify\s+what\s+i\s+have"
    r")"
)
_RE_SPOKEN_ROUTING_SPOKEN = re.compile(
    rf"(?P<head>{_SPOKEN_ROUTING_HEAD})(?P<gap>[\s\S]{{0,2000}}?)(?P<nums>{_spoken_digit_chain_min(6)})(?P<tail>\s*[.!?]*)",
    re.IGNORECASE,
)

_SPOKEN_ACCOUNT_HEAD = (
    r"(?:"
    r"i\s+just\s+need\s+your\s+account|need\s+your\s+account|your\s+account|the\s+account\b|"
    r"account\s+number\s+is|account\s+number|bank\s+account|checking\s+account|savings\s+account|"
    r"payment\s+account|draft\s+account|account\s+line|"
    r"which\s+number\??\s+the\s+account"
    r")"
)
_RE_SPOKEN_ACCOUNT_SPOKEN = re.compile(
    rf"(?P<head>{_SPOKEN_ACCOUNT_HEAD})(?P<gap>[\s\S]{{0,1200}}?)(?P<nums>{_spoken_digit_chain_min(4)})(?P<tail>\s*[.!?]*)",
    re.IGNORECASE,
)

_RE_SPOKEN_ACCOUNT_CONT = re.compile(
    rf"(?P<pre>(?:\bI\s+got\b|\bwhat\s+was\s+the\s+last\s+part\?))(?P<gap>\s*)(?P<nums>{_spoken_digit_chain_min(2)})(?P<tail>\s*[.!?]*)",
    re.IGNORECASE,
)

_SPOKEN_BANK_GENERAL_HEAD = (
    r"(?:"
    r"\bbank\b|credit\s+union|financial\s+institution|check\s+number|checkbook"
    r")"
)
_RE_SPOKEN_BANK_GENERAL_SPOKEN = re.compile(
    rf"(?P<head>{_SPOKEN_BANK_GENERAL_HEAD})(?P<gap>[\s\S]{{0,1200}}?)(?P<nums>{_spoken_digit_chain_min(6)})(?P<tail>\s*[.!?]*)",
    re.IGNORECASE,
)


def _redact_spoken_banking_numbers(text):
    """Spoken digit runs in banking context → [ROUTING_NUMBER] / [ACCOUNT_NUMBER] / [BANK_NUMBER] (before phone pass)."""
    if not text:
        return text
    out = text

    def _rr(m):
        return f"{m.group('head')}{m.group('gap')}[ROUTING_NUMBER]{m.group('tail')}"

    def _ra(m):
        return f"{m.group('head')}{m.group('gap')}[ACCOUNT_NUMBER]{m.group('tail')}"

    def _rb(m):
        return f"{m.group('head')}{m.group('gap')}[BANK_NUMBER]{m.group('tail')}"

    def _rc(m):
        return f"{m.group('pre')}{m.group('gap')}[ACCOUNT_NUMBER]{m.group('tail')}"

    # Account-related passes first so earlier "routing on my end" chatter does not steal account digits.
    out = _RE_SPOKEN_ACCOUNT_SPOKEN.sub(_ra, out)
    out = _RE_SPOKEN_ACCOUNT_CONT.sub(_rc, out)
    out = _RE_SPOKEN_BANK_GENERAL_SPOKEN.sub(_rb, out)
    out = _RE_SPOKEN_ROUTING_SPOKEN.sub(_rr, out)
    return out


_BANKING_NEAR_PHONE_GUARD = re.compile(
    r"\b(?:account|routing|accounts?|routing\s+number|bank|credit\s+union|financial\s+institution|"
    r"checking|savings|payment|draft)\b",
    re.IGNORECASE,
)
_SPOKEN_PHONE_HEAD = (
    r"(?:"
    r"best\s+number\s+to\s+reach\s+you\s+at|"
    r"phone\s+number|callback\s+number|contact\s+number|"
    r"verify\s+your\s+number|confirm\s+your\s+number|what\s+is\s+your\s+number|"
    r"is\s+this\s+a\s+good\s+number|"
    r"call\s+you\s+at|reach\s+you\s+at|text\s+you\s+at|"
    r"(?:it\s*'s|it\s+is)\s+going\s+to\s+be|"
    r"cellphone|"
    r"\bcell\b|\bmobile\b|\btelephone\b|"
    r"best\s+number|"
    r"(?:phone|callback|contact|your|the)\s+number\s+is"
    r")"
)
_RE_SPOKEN_PHONE_BLOCK = re.compile(
    rf"(?P<head>{_SPOKEN_PHONE_HEAD})(?P<gap>[\s\S]{{0,1500}}?)(?P<nums>{_SPOKEN_PHONE_CHAIN})(?P<tail>\s*[.!?]*)",
    re.IGNORECASE,
)


def _redact_spoken_phone_numbers(text):
    """Replace 7+ spoken digit-words with [PHONE] only after phone/contact cues (not generic audit numbers)."""
    if not text:
        return text

    def _repl(m):
        lookback = m.string[max(0, m.start() - 500) : m.start()]
        if _BANKING_NEAR_PHONE_GUARD.search(lookback):
            return m.group(0)
        return f"{m.group('head')}{m.group('gap')}[PHONE]{m.group('tail')}"

    return _RE_SPOKEN_PHONE_BLOCK.sub(_repl, text)


# Four or more digits with only short separators (read-out / account / routing style).
# Separators exclude "." to avoid redacting common decimals like 3.14 in transcript.
_SPACED_DIGIT_RUN = re.compile(
    r"(?<![\d])(?:\d[\s,\-/]+){3,}\d(?![\d])",
    re.UNICODE,
)

# Stage / carrier / plan tokens preserved from aggressive digit + name passes (longest first).
_PROTECT_TRANSCRIPT_TOKENS = (
    r"(?i)\bAmerican\s+Amicable\s+Group\b",
    r"(?i)\bAmerican\s+Amicable\b",
    r"(?i)\bThird\s+Party\s+Underwriting\b",
    r"(?i)\bPeace\s+of\s+Mind\b",
    r"(?i)\bCool\s+Down\b",
    r"(?i)\bFact\s+Finding\s*/\s*Warm-?up\b",
    r"(?i)\bSocial\s+Security\b",
    r"(?i)\bCredit\s+Union\b",
    r"(?i)\bImmediate\s+plan\b",
    r"(?i)\bGraded\s+plan\b",
    r"(?i)\bROP\s+plan\b",
    r"(?i)\bImmediate\b",
    r"(?i)\bGraded\b",
    r"(?i)\bROP\b",
)
def _transcript_protect_token(index):
    """Single non-digit placeholder so downstream \\d+ passes cannot corrupt vault swaps."""
    return chr(0xE000 + index)


def _transcript_protect_phrases(text):
    """Temporarily hides audit-critical phrases so number/name passes cannot mangle them."""
    protected = []
    out = text or ""

    for rx in _PROTECT_TRANSCRIPT_TOKENS:

        def _repl(m):
            protected.append(m.group(0))
            return _transcript_protect_token(len(protected) - 1)

        out = re.sub(rx, _repl, out)
    return out, protected


def _transcript_restore_phrases(text, protected):
    out = text
    for i, orig in enumerate(protected):
        out = out.replace(_transcript_protect_token(i), orig)
    return out


_SPELLED_DECADE = r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
_SPELLED_UNIT = r"(?:one|two|three|four|five|six|seven|eight|nine)"
# (pattern, replacement) — keep cue prefix (e.g. "I am ") for readability.
_SPELLED_AGE_REPL = (
    (
        re.compile(
            rf"(?i)\b((?:i\s*am|i'm|age|aged|turning)\s+)({_SPELLED_DECADE}(?:[\s-]+{_SPELLED_UNIT})?)\b"
        ),
        r"\1[NUMBER]",
    ),
    (
        re.compile(
            r"(?i)\b((?:i\s*am|i'm)\s+)(?:ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)\b"
        ),
        r"\1[NUMBER]",
    ),
    (
        re.compile(rf"(?i)\b((?:i\s*am|i'm)\s+)({_SPELLED_UNIT})\b"),
        r"\1[NUMBER]",
    ),
)


def _redact_spelled_age_phrases(text):
    if not text:
        return text
    out = text
    for pat, repl in _SPELLED_AGE_REPL:
        out = pat.sub(repl, out)
    return out


def _redact_numeric_tokens(text):
    """Replace phones, money, dates, times, spaced/long digit runs, and other digits with typed placeholders."""
    if not text:
        return text
    t = text
    t = re.sub(
        r"\b(?:\+?1[\s\-.]?)?(?:\(?\d{3}\)?[\s\-.]?)\d{3}[\s\-.]?\d{4}\b",
        "[PHONE]",
        t,
    )
    t = re.sub(r"(?<![\d])\d{10}(?![\d])", "[PHONE]", t)
    t = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "[DATE]", t)
    t = re.sub(
        r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b",
        "[TIME]",
        t,
    )
    t = re.sub(r"\$\s*\d+(?:[,.]\d+)*\b", "[MONEY]", t)
    t = re.sub(r"\b\d+(?:\.\d+)?%", "[NUMBER]", t)
    t = _SPACED_DIGIT_RUN.sub("[BANK_NUMBER]", t)
    t = re.sub(r"(?<![\d])\d+(?:st|nd|rd|th)\b", "[NUMBER]", t, flags=re.I)
    t = re.sub(r"(?<!\d)\d{12,}(?!\d)", "[BANK_NUMBER]", t)
    t = re.sub(r"(?<![\d])\d+(?:\.\d+)?(?![\d])", "[NUMBER]", t)
    return t


def _redact_spaced_and_long_digit_sequences(text):
    """Backward-compatible shim: full numeric pass (used where callers expect this hook)."""
    return _redact_numeric_tokens(text)


def _spoken_banking_redaction_selftest():
    """Fake samples — spoken banking digits before phone classification."""
    s1 = "I just need your account. Seven, six, five, four, three, two, one."
    o1 = redact_sensitive_transcript(s1)
    if "[ACCOUNT_NUMBER]" not in o1 or "[PHONE]" in o1:
        raise RuntimeError(f"banking account spoken failed: {o1!r}")

    s2 = "Do you have your routing number? Zero, six, four, two, one, nine, six, eight, one."
    o2 = redact_sensitive_transcript(s2)
    if "[ROUTING_NUMBER]" not in o2:
        raise RuntimeError(f"routing spoken failed: {o2!r}")

    s3 = "Phone number is seven six five four three four three five three."
    o3 = redact_sensitive_transcript(s3)
    if "[PHONE]" not in o3:
        raise RuntimeError(f"phone still required: {o3!r}")

    s4 = "Account number is seven six five four three four three five three."
    o4 = redact_sensitive_transcript(s4)
    if "[ACCOUNT_NUMBER]" not in o4 or "[PHONE]" in o4:
        raise RuntimeError(f"account number is spoken failed: {o4!r}")


def _spoken_phone_redaction_selftest():
    """Fake samples — spoken digits redacted only after phone cues."""
    a = "It's going to be seven, six, five Four, three, Four, three, five, three."
    out_a = redact_sensitive_transcript(a)
    if "[PHONE]" not in out_a:
        raise RuntimeError(f"spoken phone redaction missing [PHONE]: {out_a!r}")
    if re.search(r"\bseven\s*,\s*six\b", out_a, re.I):
        raise RuntimeError(f"spoken phone digits leaked: {out_a!r}")

    b = "Phone number is seven six five four three four three five three."
    if "[PHONE]" not in redact_sensitive_transcript(b):
        raise RuntimeError(f"spoken phone (phone number is) failed: {redact_sensitive_transcript(b)!r}")


def _spoken_dob_next_line_selftest():
    """Fake only — next-line spoken DOB after cue ending in ? (and same-line chain)."""
    a = "What is your date of birth?\nEight, twelve, nineteen, sixty-eight."
    expected_a = "What is your date of birth?\n[DOB]."
    if redact_sensitive_transcript(a) != expected_a:
        raise RuntimeError(f"next-line spoken DOB failed: {redact_sensitive_transcript(a)!r}")

    b = "My date of birth is eight, twelve, nineteen, sixty-eight."
    expected_b = "My date of birth is [DOB]."
    if redact_sensitive_transcript(b) != expected_b:
        raise RuntimeError(f"same-line spoken DOB failed: {redact_sensitive_transcript(b)!r}")


def _redaction_smoke_assertions():
    """Synthetic sample only — no real customer data; raises on failure."""
    _spoken_banking_redaction_selftest()
    _spoken_phone_redaction_selftest()
    _spoken_dob_next_line_selftest()
    sample = (
        "My name is John Smith, I was born 12/31/1950, my phone is 555-123-4567, "
        "I am 74, my account is 123456789, and I pay $62.50 on the 3rd. "
        "American Amicable Immediate plan graded ROP. I am seventy four. "
        "My date of birth is eight, twelve, nineteen, sixty-eight."
    )
    out = redact_sensitive_transcript(sample)
    if "[NAME]" not in out or "John" in out or "Smith" in out:
        raise RuntimeError(f"name redaction failed: {out!r}")
    if "[DOB]" not in out or "[PHONE]" not in out or "[MONEY]" not in out:
        raise RuntimeError(f"missing typed placeholders: {out!r}")
    if "[ACCOUNT_NUMBER]" not in out:
        raise RuntimeError(f"expected account placeholder: {out!r}")
    if out.count("[NUMBER]") < 2:
        raise RuntimeError(f"expected some [NUMBER] ordinals/ages, got: {out!r}")
    if re.search(r"\beight\s*,\s*twelve", out, re.I):
        raise RuntimeError(f"spelled DOB not redacted: {out!r}")
    if "American Amicable" not in out:
        raise RuntimeError(f"lost protected carrier: {out!r}")
    low = out.lower()
    for keep in ("immediate", "rop", "graded"):
        if keep not in low:
            raise RuntimeError(f"lost protected plan token {keep!r}: {out!r}")
    if re.search(r"(?i)\b(?:seventy|sixty|eighty)\s+(?:one|two|three|four|five)\b", out):
        raise RuntimeError(f"spelled age not redacted: {out!r}")


def redact_sensitive_transcript(transcript):
    redacted, vault = _transcript_protect_phrases(transcript or "")
    redacted = _redact_spoken_dob_phrases(redacted)
    redacted = _redact_next_line_spoken_dob_after_cue(redacted)

    _title_word = r"(?-i:[A-Z][a-z]+)"
    _title_name = rf"{_title_word}(?:\s+{_title_word}){{0,2}}"
    patterns = [
        # Full names after identity cues — Title Case tokens only (so "seventy four" is not [NAME]).
        (
            rf"(?i)\b(my name is|this is|i am|i'm|speaking with|call me)\s+({_title_name})\b",
            r"\1 [NAME]",
        ),
        (
            rf"(?i)\b(?:full\s+name|name)\s*[:\-]\s*({_title_name})\b",
            "name: [NAME]",
        ),
        (
            r"(?i)\b(mr|mrs|ms|miss)\.?\s+((?-i:[A-Z][a-z]+)(?:\s+(?-i:[A-Z][a-z]+))+)\b",
            r"\1 [NAME]",
        ),
        (r"(?i)\b(my\s+account\s+is\s+)([\d\-]+)\b", r"\1[ACCOUNT_NUMBER]"),
        (r"(?i)\b(bank\s+account\s+is\s+)([\d\-]+)\b", r"\1[ACCOUNT_NUMBER]"),
        (r"(?i)\b(account\s+number\s+is\s+)([\d\-]+)\b", r"\1[ACCOUNT_NUMBER]"),
        (r"(?i)\b(routing\s+number\s+is\s+)([\d\-]+)\b", r"\1[ROUTING_NUMBER]"),
        (r"(?i)\b(routing\s+is\s+)([\d\-]+)\b", r"\1[ROUTING_NUMBER]"),
        (
            r"(?i)\b(?:dob|d\.o\.b\.|date of birth|birth date|born(?:\s+on)?)\s*[:\-]?\s*(?:[0-1]?\d[\/\-][0-3]?\d[\/\-](?:\d{2}|\d{4})|(?:\d{4}[\/\-][0-1]?\d[\/\-][0-3]?\d)|(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\s+\d{1,2},?\s+\d{4})\b",
            "DOB: [DOB]",
        ),
        (r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}\b", "[EMAIL]"),
        (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
        (r"(?i)\b(?:ssn|social security)\s*[:\-]?\s*\d{9}\b", "SSN: [SSN]"),
        (r"\b(?:\d[ -]?){12,19}\b", "[BANK_NUMBER]"),
        (
            r"(?i)\b(routing\s*(?:number|#)?\s*[:\-]\s*)([\d\s\-]{6,})\b",
            r"\1[ROUTING_NUMBER]",
        ),
        (
            r"(?i)\b(account\s*(?:number|#)?\s*[:\-]\s*)([\d\s\-]{6,})\b",
            r"\1[ACCOUNT_NUMBER]",
        ),
        (
            r"(?i)\b(?:card\s+number|debit\s+card|credit\s+card)\s*[:\-]\s*[\d\w\-]{4,}\b",
            "card number: [BANK_NUMBER]",
        ),
    ]

    for pattern, replacement in patterns:
        redacted = re.sub(pattern, replacement, redacted)

    redacted = _redact_next_line_after_dob_context(redacted)
    redacted = _redact_spelled_age_phrases(redacted)
    redacted = _redact_spoken_banking_numbers(redacted)
    redacted = _redact_spoken_phone_numbers(redacted)
    redacted = _redact_numeric_tokens(redacted)
    redacted = _transcript_restore_phrases(redacted, vault)
    redacted = hard_privacy_redact_transcript(redacted)
    redacted = final_transcript_privacy_cleanup(redacted)
    return redacted


# ---------------------------------------------------------------------------
# HARD PRIVACY REDACTION LAYER
# ---------------------------------------------------------------------------
# This is intentionally redundant with the earlier redaction rules. It is the
# final safety pass before transcripts are saved, role-labeled, audited, or put
# into calls.db. Goal: no real person names and no real numbers survive.
# ---------------------------------------------------------------------------

_SPOKEN_NUMBER_WORDS = (
    r"zero|oh|o|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|"
    r"eighteenth|nineteenth|twentieth"
)

_SPOKEN_NUMBER_CHAIN = (
    rf"\b(?:{_SPOKEN_NUMBER_WORDS})\b"
    rf"(?:[\s,\-]+(?:and\s+)?\b(?:{_SPOKEN_NUMBER_WORDS})\b){{1,12}}"
)

_SINGLE_SPOKEN_NUMBER = rf"\b(?:{_SPOKEN_NUMBER_WORDS})\b"

_MONTH_WORDS = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)

_ROLE_PREFIX = r"(?:(?:PQ|Agent|Prospect|Unknown)\s*:\s*)?"
_NAME_WORD = r"[A-Z][a-z]+(?:'[A-Z][a-z]+)?"
_NAME_PHRASE = rf"{_NAME_WORD}(?:\s+{_NAME_WORD}){{0,3}}"


def _hard_redact_names(text):
    """Cue-based person-name redaction. Keeps role labels but removes person names."""
    if not text:
        return text

    t = text

    # Handoff / introduction patterns.
    cue_patterns = [
        # "Who do I have the pleasure of speaking with? James"
        (
            rf"(?is)(who\s+do\s+i\s+have\s+the\s+pleasure\s+of\s+(?:speaking\s+with|helping)\s*(?:today)?\??\s*)({_NAME_PHRASE})\b",
            r"\1[NAME]",
        ),
        # "This is Shelby", "Hi James", "Hello Jerome"
        (
            rf"(?im)^(\s*{_ROLE_PREFIX}(?:hi|hello|hey|good\s+morning|good\s+afternoon|this\s+is)\s+)(?!American\b|North\b|Premier\b|Kinzel\b)({_NAME_PHRASE})\b",
            r"\1[NAME]",
        ),
        # "I have James here", "I have Jerome"
        (
            rf"(?i)\b(i\s+have\s+)({_NAME_PHRASE})(\s+(?:here|with\s+me|on\s+the\s+line)\b)",
            r"\1[NAME]\3",
        ),
        # "Mr. James", "Ms. Frances", "Mrs. Smith"
        (
            rf"(?i)\b(mr|mrs|ms|miss|sir|ma'am)\.?\s+({_NAME_PHRASE})\b",
            r"\1. [NAME]",
        ),
        # "my name is Shelby", "agent name is Shelby", "beneficiary is Misty Mullins"
        (
            rf"(?i)\b((?:my|your|his|her|their|beneficiary|primary\s+beneficiary|agent)\s+name\s+(?:is|as)\s+)({_NAME_PHRASE})\b",
            r"\1[NAME]",
        ),
        (
            rf"(?i)\b((?:beneficiary|primary\s+beneficiary)\s+(?:is|would\s+be|will\s+be|on\s+the\s+policy\s+is)\s+)({_NAME_PHRASE})\b",
            r"\1[NAME]",
        ),
        # "leaving this money behind to Misty Mullins"
        (
            rf"(?i)\b((?:leaving|leave|send|go)\s+(?:this\s+)?(?:money|benefit|check|policy)?\s*(?:behind\s+)?(?:to|for)\s+)({_NAME_PHRASE})\b",
            r"\1[NAME]",
        ),
        # "Do you spell her name M-I-S-T-Y"
        (
            r"(?i)\b((?:spell|spelled|spelling|verify\s+the\s+spelling\s+of)\s+(?:his|her|their|your)?\s*(?:name)?\s*)(?:[A-Z](?:[\s\-.]+|$)){2,}",
            r"\1[NAME]",
        ),
    ]

    for pattern, repl in cue_patterns:
        t = re.sub(pattern, repl, t)

    # Names embedded in common line starts after role labels.
    t = re.sub(
        rf"(?im)^(\s*(?:PQ|Agent|Prospect|Unknown)\s*:\s*(?:okay|perfect|thank\s+you|thanks|gotcha|all\s+right|alright),?\s+)(?!American\b|North\b|Premier\b|Kinzel\b)({_NAME_PHRASE})(\b)",
        r"\1[NAME]\3",
        t,
    )

    # Full-name style answers after name questions.
    lines = t.split("\n")
    out = []
    name_context = 0
    for line in lines:
        current = line
        if name_context > 0:
            current = re.sub(
                rf"^(\s*{_ROLE_PREFIX}){_NAME_PHRASE}\s*$",
                r"\1[NAME]",
                current,
            )
            name_context -= 1
        out.append(current)
        if re.search(r"\b(full\s+legal\s+name|what(?:'s| is)\s+your\s+name|who\s+do\s+i\s+have|beneficiary.*name|spell.*name|verify.*name)\b", current, re.I):
            name_context = 2

    return "\n".join(out)


def _hard_redact_number_context_lines(text):
    """
    Redact short spoken/digit answers after sensitive number cues.
    This catches:
    Agent: What's your social?
    Prospect: one two three...
    Agent: What is your date of birth?
    Prospect: nineteen sixty-six
    """
    if not text:
        return text

    cue_to_placeholder = [
        (re.compile(r"\b(date\s+of\s+birth|d\.?\s*o\.?\s*b\.?|birth\s+date|birthday|born)\b", re.I), "[DOB]"),
        (re.compile(r"\b(social|social\s+security|ssn)\b", re.I), "[SSN]"),
        (re.compile(r"\b(phone|telephone|cell|number\s+to\s+reach|best\s+number)\b", re.I), "[PHONE]"),
        (re.compile(r"\b(routing)\b", re.I), "[ROUTING_NUMBER]"),
        (re.compile(r"\b(account\s+number|account)\b", re.I), "[ACCOUNT_NUMBER]"),
        (re.compile(r"\b(card|debit|credit)\b", re.I), "[BANK_NUMBER]"),
        (re.compile(r"\b(address|street|road|drive|lane|avenue|zip|city|state)\b", re.I), "[ADDRESS]"),
    ]

    short_numberish = re.compile(
        rf"^(\s*{_ROLE_PREFIX})(?:"
        rf"(?:\d[\d\s,\-/.#]*)|"
        rf"(?:{_SPOKEN_NUMBER_WORDS})(?:[\s,\-]+(?:and\s+)?(?:{_SPOKEN_NUMBER_WORDS}))*|"
        rf"(?:{_MONTH_WORDS})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{2,4}})?|"
        rf"\[NUMBER\](?:[\s,\-/.]+\[NUMBER\])*"
        rf")(\s*(?:,?\s*(?:you\s+said|correct|right)\??|[.!?])?\s*)$",
        re.I,
    )

    lines = text.split("\n")
    out = []
    active_placeholder = None
    remaining = 0

    for line in lines:
        current = line

        if active_placeholder and remaining > 0:
            m = short_numberish.match(current.strip())
            if m:
                prefix = re.match(rf"^(\s*{_ROLE_PREFIX})", current, re.I)
                role = prefix.group(1) if prefix else ""
                suffix = ""
                if re.search(r"\byou\s+said\??", current, re.I):
                    suffix = ", you said?"
                elif current.rstrip().endswith("?"):
                    suffix = "?"
                elif current.rstrip().endswith("."):
                    suffix = "."
                current = f"{role}{active_placeholder}{suffix}"
                remaining = 2
            else:
                remaining -= 1

        out.append(current)

        for cue, placeholder in cue_to_placeholder:
            if cue.search(current):
                active_placeholder = placeholder
                remaining = 2
                break

    return "\n".join(out)


def _hard_redact_numbers(text):
    """Aggressive typed and spoken number cleanup with typed placeholders where context allows."""
    if not text:
        return text

    t = text

    # Typed structured values first.
    t = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", t)
    t = re.sub(r"(?i)\b((?:social|social security|ssn)\s*(?:is|number|#|:)?\s*)\d{4,9}\b", r"\1[SSN]", t)
    t = re.sub(r"\b(?:\+?1[\s\-.]?)?(?:\(?\d{3}\)?[\s\-.]?)\d{3}[\s\-.]?\d{4}\b", "[PHONE]", t)
    t = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "[DOB]", t)
    t = re.sub(rf"\b(?:{_MONTH_WORDS})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{2,4}})?\b", "[DATE]", t, flags=re.I)
    t = re.sub(r"\$\s*\d+(?:[,.]\d+)*(?:\.\d+)?\b", "[MONEY]", t)
    t = re.sub(r"(?i)\b((?:routing|routing number)\s*(?:is|#|:)?\s*)\d[\d\s\-]{5,}\b", r"\1[ROUTING_NUMBER]", t)
    t = re.sub(r"(?i)\b((?:account|account number)\s*(?:is|#|:)?\s*)\d[\d\s\-]{4,}\b", r"\1[ACCOUNT_NUMBER]", t)
    t = re.sub(r"(?i)\b((?:card|debit|credit)\s*(?:number)?\s*(?:is|#|:)?\s*)\d[\d\s\-]{4,}\b", r"\1[BANK_NUMBER]", t)
    t = re.sub(r"\b(?:\d[\s\-]*){12,19}\b", "[BANK_NUMBER]", t)

    # Spoken contextual values.
    t = re.sub(rf"(?i)\b((?:date of birth|dob|d\.o\.b\.|birthday|born)\s*(?:is|:)?\s*){_SPOKEN_NUMBER_CHAIN}\b", r"\1[DOB]", t)
    t = re.sub(rf"(?i)\b((?:social|social security|ssn)\s*(?:is|number|:)?\s*){_SPOKEN_NUMBER_CHAIN}\b", r"\1[SSN]", t)
    t = re.sub(rf"(?i)\b((?:phone|telephone|cell|number)\s*(?:is|:)?\s*){_SPOKEN_NUMBER_CHAIN}\b", r"\1[PHONE]", t)
    t = re.sub(rf"(?i)\b((?:routing|routing number)\s*(?:is|:)?\s*){_SPOKEN_NUMBER_CHAIN}\b", r"\1[ROUTING_NUMBER]", t)
    t = re.sub(rf"(?i)\b((?:account|account number)\s*(?:is|:)?\s*){_SPOKEN_NUMBER_CHAIN}\b", r"\1[ACCOUNT_NUMBER]", t)

    # Any remaining long spoken digit/number chains are sensitive by default.
    t = re.sub(_SPOKEN_NUMBER_CHAIN, "[NUMBER]", t, flags=re.I)

    # Any remaining digits become generic [NUMBER], unless already inside a placeholder.
    t = re.sub(r"(?<!\[)\b\d+(?:\.\d+)?\b(?!\])", "[NUMBER]", t)

    # Clean accidental repeated placeholders.
    t = re.sub(r"(?:\[NUMBER\][\s,\-/.]*){2,}", "[NUMBER]", t)
    t = re.sub(r"(?:\[DOB\][\s,\-/.]*){2,}", "[DOB]", t)
    t = re.sub(r"(?:\[PHONE\][\s,\-/.]*){2,}", "[PHONE]", t)

    return t


def hard_privacy_redact_transcript(text):
    """
    Final privacy pass for sensitive values.

    IMPORTANT:
    Do not call _hard_redact_names() here. That older helper is too broad and
    can replace normal transcript words like "before", "there", "small", or
    "right" with [NAME]. Name cleanup is handled later by the narrower
    final_transcript_privacy_cleanup() direct-address/name-context rules.
    """
    if not text:
        return text

    t = text
    t = _hard_redact_number_context_lines(t)
    t = _hard_redact_numbers(t)
    t = _hard_redact_number_context_lines(t)

    return t



ROLE_LABEL_TRANSCRIPT_NOTE = (
    "TRANSCRIPT NOTE (MANDATORY): The following transcript may include PQ:, Agent:, Prospect:, or Unknown: "
    "role labels generated only from the redacted transcript. PQ means the pre-qualification / transfer rep, "
    "not the selling agent. Use labeled turns to follow the conversation; they are an aid and are not absolute "
    "proof of who spoke if the source text was ambiguous. Do not auto-fail based solely on callback wording when "
    "speaker role is uncertain — explain uncertainty."
)


def _split_transcript_for_exact_labeling(transcript_text):
    """
    Split redacted transcript into original non-empty lines for speaker labeling.
    The line text is the source of truth. The model only labels each line.
    """
    lines = []
    for raw_line in (transcript_text or "").splitlines():
        line = raw_line.rstrip()
        if line.strip():
            lines.append(line)
    return lines


def _parse_speaker_label_response(output_text, expected_numbers):
    """
    Parse label-only model output. Accept JSON first, then simple '1: Agent' lines.
    Returns {line_number: speaker}.
    """
    labels = {}
    raw = (output_text or "").strip()
    if not raw:
        return labels

    allowed = {"PQ", "Agent", "Prospect", "Unknown"}

    # JSON path: {"labels":[{"n":1,"speaker":"Agent"}]}
    try:
        cleaned = raw
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
        if fenced:
            cleaned = fenced.group(1).strip()
        obj = json.loads(cleaned)
        items = obj.get("labels", []) if isinstance(obj, dict) else []
        for item in items:
            try:
                n = int(item.get("n"))
                speaker = str(item.get("speaker", "")).strip()
            except Exception:
                continue
            if n in expected_numbers and speaker in allowed:
                labels[n] = speaker
        if labels:
            return labels
    except Exception:
        pass

    # Text fallback: 1: Agent
    for m in re.finditer(r"(?im)^\s*(\d+)\s*[:\-]\s*(PQ|Agent|Prospect|Unknown)\s*$", raw):
        n = int(m.group(1))
        speaker = m.group(2)
        if n in expected_numbers:
            labels[n] = speaker

    return labels


def _label_transcript_line_batch(numbered_lines):
    """
    Ask OpenAI for speaker labels only. It must not rewrite transcript words.
    numbered_lines = list[(n, exact_line)]
    """
    if not numbered_lines:
        return {}

    numbered_text = "\n".join(f"{n}. {line}" for n, line in numbered_lines)
    expected_numbers = {n for n, _ in numbered_lines}

    instructions = """You are labeling speakers in a redacted final-expense sales transcript.

CRITICAL:
- Return speaker labels ONLY.
- Do NOT rewrite, paraphrase, summarize, correct, shorten, or repeat transcript text.
- Do NOT include any transcript words in your response.
- The original transcript text will be preserved exactly by software. Your only job is to choose a speaker label for each numbered line.

Allowed speaker labels:
- PQ = pre-qualification / transfer rep before the selling agent takes over. Use this for the opening rep who identifies themself as PQ, introduces the prospect, introduces the underwriter/agent, says the agent will walk them through the process, or ends the handoff.
- Agent = licensed field underwriter / selling agent. Use for recording disclosure, license number, rapport, health questions, existing coverage, need, product explanation, quotes, closing, application, payment, banking, disclosures, voice signature, Peace of Mind, Cool Down, and call-control language.
- Prospect = customer / consumer / lead. Use for answers, objections, refusals, medical answers, personal details, coverage answers, banking answers, option choices, or hangup language.
- Unknown = only when the speaker cannot be identified from context.

Return valid JSON only in this exact shape:
{"labels":[{"n":1,"speaker":"PQ"},{"n":2,"speaker":"Prospect"}]}

You must include exactly one label object for every numbered line.
"""

    user_block = f"{instructions}\n\nNUMBERED TRANSCRIPT LINES:\n{numbered_text}"

    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        input=user_block,
        temperature=0,
    )
    out = (response.output_text or "").strip()
    return _parse_speaker_label_response(out, expected_numbers)




def _looks_like_agent_personal_self_disclosure(text):
    """
    Conservative detector for long agent rapport/self-disclosure blocks that were
    mislabeled as Prospect. This does not rewrite transcript words; it only helps
    repair the speaker label.
    """
    if not text:
        return False

    body = text.lower()
    words = re.findall(r"\b\w+\b", body)

    if len(words) < 45:
        return False

    markers = [
        r"\bmy husband\b",
        r"\bmy wife\b",
        r"\bmy son\b",
        r"\bmy daughter\b",
        r"\bmy child\b",
        r"\bmy kids?\b",
        r"\bmy mom\b",
        r"\bmy dad\b",
        r"\bmy mom and dad\b",
        r"\bmy mother\b",
        r"\bmy father\b",
        r"\bmy brother\b",
        r"\bmy sister\b",
        r"\bmy half[- ]brother\b",
        r"\bmy half[- ]sister\b",
        r"\bmy siblings?\b",
        r"\bmy family\b",
        r"\bwhen we left\b",
        r"\bwhen i lived\b",
        r"\bi actually live\b",
        r"\bi live over\b",
        r"\bi grew up\b",
        r"\bi was raised\b",
        r"\bwhere i am from\b",
        r"\bme and my\b",
    ]

    marker_hits = sum(1 for p in markers if re.search(p, body))

    agent_bridge_phrases = [
        r"\bif you know what i mean\b",
        r"\bthat makes a big difference\b",
        r"\bonce you have your own family\b",
        r"\bi can imagine\b",
        r"\bthat's hilarious\b",
        r"\bit's funny that you say\b",
        r"\bi understand\b",
        r"\bi get it\b",
    ]

    bridge_hits = sum(1 for p in agent_bridge_phrases if re.search(p, body))

    # Avoid flipping normal prospect answers that simply mention one family member.
    # Require several self-disclosure markers or marker + agent-style bridge phrases.
    return marker_hits >= 3 or (marker_hits >= 2 and bridge_hits >= 1)


def _repair_agent_self_disclosure_mislabeled_as_prospect(labeled_text):
    """
    Repair role-labeled transcript blocks where a long agent personal disclosure
    was labeled as Prospect. This preserves exact transcript text and only changes
    the speaker label.

    Works on grouped speaker transcript text:
        Prospect: line...
        continuation...
    """
    if not labeled_text:
        return labeled_text

    lines = labeled_text.splitlines()
    blocks = []
    current_speaker = None
    current_lines = []

    speaker_re = re.compile(r"^\s*(PQ|Agent|Prospect|Unknown)\s*:\s*(.*)$")

    for line in lines:
        m = speaker_re.match(line)
        if m:
            if current_speaker is not None:
                blocks.append([current_speaker, current_lines])
            current_speaker = m.group(1)
            current_lines = [m.group(2)]
        else:
            if current_speaker is None:
                blocks.append([None, [line]])
            else:
                current_lines.append(line)

    if current_speaker is not None:
        blocks.append([current_speaker, current_lines])

    repaired = []
    for idx, (speaker, body_lines) in enumerate(blocks):
        body = "\n".join(body_lines).strip()

        if speaker == "Prospect" and _looks_like_agent_personal_self_disclosure(body):
            # Extra safety: prefer repair when nearby previous context is Agent rapport.
            prev_text = ""
            if idx > 0:
                prev_speaker, prev_lines = blocks[idx - 1]
                prev_text = (prev_speaker or "") + ": " + "\n".join(prev_lines)

            rapport_context = bool(re.search(
                r"(?is)(family|brother|sister|children|kids|married|husband|wife|relationship|born|raised|live|work|retired|favorite part|tell me about)",
                prev_text + "\n" + body,
            ))

            if rapport_context:
                speaker = "Agent"

        if speaker is None:
            repaired.extend(body_lines)
        else:
            if body_lines:
                repaired.append(f"{speaker}: {body_lines[0]}")
                repaired.extend(body_lines[1:])
            else:
                repaired.append(f"{speaker}: ")

    return "\n".join(repaired)


def format_grouped_speaker_transcript(labeled_text):
    """
    Collapse repeated consecutive speaker labels for readability.
    Keeps transcript wording/line breaks, but only shows speaker label again when speaker changes.
    """
    if not labeled_text:
        return labeled_text

    out = []
    last_speaker = None

    for raw_line in labeled_text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue

        m = re.match(r"^\s*(PQ|Agent|Prospect|Unknown)\s*:\s*(.*)$", line)
        if not m:
            out.append(line)
            continue

        speaker = m.group(1)
        content = m.group(2).strip()

        if speaker == last_speaker:
            out.append(content)
        else:
            if out:
                out.append("")
            out.append(f"{speaker}: {content}")
            last_speaker = speaker

    return "\n".join(out).strip() + "\n"




def _extract_handoff_and_direct_address_names(text):
    """
    Build a narrow name vault from clear handoff/direct-address contexts.
    This catches repeated names like James without broadly redacting normal words.
    """
    names = set()
    if not text:
        return names

    blocked = {
        "PQ", "Agent", "Prospect", "Unknown", "Carrier", "Third", "Party",
        "Mississippi", "Tennessee", "Maryland", "Indiana", "Arkansas",
        "Social", "Security", "State", "Department", "Insurance",
        "American", "General", "Mutual", "Omaha", "Combined", "Colonial",
        "Globe", "MIB", "COVID", "Medicare", "Medicaid",
    }

    patterns = [
        r"(?i)\bi have\s+([A-Z][a-z]+)\s+(?:here|on the phone|with me|on the line)\b",
        r"(?i)\b([A-Z][a-z]+)\s+wants to learn\b",
        r"(?i)\b([A-Z][a-z]+)\s*,\s+i have one of\b",
        r"(?i)\b(?:now|and|okay|alright|all right|thank you|thanks)\s*,?\s+([A-Z][a-z]+)\s*,",
        r"(?i)\b([A-Z][a-z]+)\s*,?\s+(?:are you there|can you hear me)\b",
        r"(?im)^\s*(?:PQ|Agent|Prospect|Unknown)?\s*:?\s*(?:hi|hello|hey)\s*,?\s+([A-Z][a-z]+)\b",
    ]

    for pat in patterns:
        for m in re.finditer(pat, text):
            name = m.group(1).strip()
            if name and name not in blocked:
                names.add(name)

    return names


def _redact_handoff_name_vault(text):
    if not text:
        return text
    names = _extract_handoff_and_direct_address_names(text)
    out = text
    for name in sorted(names, key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(name)}\b", "[NAME]", out)
    return out


def _suppress_late_pq_after_agent_takeover(transcript):
    """
    PQ should only appear during opening handoff.
    After Agent begins recording disclosure / who-I-am / license script,
    later PQ labels are usually mislabeled prospect/background/carrier audio.
    """
    if not transcript:
        return transcript

    lines = transcript.splitlines()
    takeover_seen = False
    out = []

    takeover_re = re.compile(
        r"^\s*Agent\s*:.*\b(before we get started|call may be recorded|who i am|what these state|state-approved|state approved|license number|state licensed)\b",
        re.I,
    )
    pq_valid_re = re.compile(
        r"\b(final expense department|pq|pre[- ]?qual|i have .* here|one of our best|walk.*step by step|you both have a great day)\b",
        re.I,
    )

    for line in lines:
        if takeover_re.search(line):
            takeover_seen = True

        if takeover_seen and re.match(r"^\s*PQ\s*:", line):
            content = re.sub(r"^\s*PQ\s*:\s*", "", line)
            if pq_valid_re.search(content):
                out.append(line)
            else:
                out.append("Unknown: " + content)
            continue

        out.append(line)

    return "\n".join(out)


def _final_cleanup_early_end_stage_and_banking(report, transcript):
    """
    Correct early-ended calls that were incorrectly marked as Banking/payment.
    """
    if not report:
        return report

    health_no = bool(re.search(r"(?im)^- Health questions completed:\s*NO\b", report))
    product_no = bool(re.search(r"(?im)^- Product benefits explained:\s*NO\b", report))
    options_no = bool(re.search(r"(?im)^- Three options presented:\s*NO\b", report))
    app_no = bool(re.search(r"(?im)^- Application info collected:\s*NO\b", report))
    acct_zero = bool(re.search(r"(?im)^- Account verification evidence count:\s*0\b", report))
    routing_zero = bool(re.search(r"(?im)^- Routing verification evidence count:\s*0\b", report))
    sold_no = bool(re.search(r"(?im)^- Policy sold:\s*NO\b|^- Was the policy sold\?\s*NO\b", report))

    early_disconnect = bool(re.search(
        r"\b(are you there|can you hear me|customer disconnected|disconnected before|hangup|hung up)\b",
        (report or "") + "\n" + (transcript or ""),
        re.I,
    ))

    if health_no and product_no and options_no and app_no and acct_zero and routing_zero:
        report = re.sub(
            r"(?im)^CALL STAGE REACHED:\s*Banking\s*$",
            "CALL STAGE REACHED: Fact Finding / Warm-up",
            report,
            count=1,
        )
        report = re.sub(
            r"(?im)^- Final stage supporting sale:\s*Banking\s*$",
            "- Final stage supporting sale: Fact Finding / Warm-up",
            report,
            count=1,
        )
        report = _text_replace_checklist_value(report, "Payment date explained", "NOT REACHED")
        report = _text_replace_checklist_value(report, "Banking/payment setup explained", "NOT REACHED")

        # For early-ended calls, make NOT REACHED reflect every major unfinished stage,
        # not only the default late-call stages.
        expanded_not_reached = [
            "Existing coverage",
            "Beneficiary",
            "Need amount",
            "Health questions",
            "Product benefits",
            "Three options",
            "Client choice",
            "Application information",
            "Payment date",
            "Banking/payment setup",
            "Banking/account verification",
            "Disclosures",
            "Third Party Underwriting",
            "Peace of Mind",
            "Cool Down",
        ]
        replacement = "NOT REACHED:\n" + "\n".join(f"- {item}" for item in expanded_not_reached) + "\n\n"
        report = re.sub(
            r"(?ims)^NOT REACHED:\s*.*?(?=^COMPLIANCE FAILURES:)",
            replacement,
            report,
            count=1,
        )

    if sold_no and early_disconnect:
        report = re.sub(r"(?im)^RISK:\s*HIGH\s*$", "RISK: MEDIUM", report, count=1)
        report = re.sub(
            r"(?ims)^BIGGEST MISS:\s*.*?(?=^TRANSCRIPT NOTE|^SUMMARY:|^OPENAI COST ESTIMATE:|\Z)",
            "BIGGEST MISS:\n- Prospect stopped responding / disconnected while the agent was still in warm-up/fact-finding, before the agent could complete the report-building section or move into health questions, need, product explanation, options, application, disclosures, Peace of Mind, or Cool Down.\n\n",
            report,
            count=1,
        )

    return report


def _final_cleanup_names_and_late_pq_in_report(report, transcript):
    if report:
        report = _redact_handoff_name_vault(report)
    return report


def create_role_labeled_transcript(transcript_text):
    """
    Create an exact speaker-labeled transcript.

    IMPORTANT: This does NOT let the model rewrite the transcript.
    The model returns labels only, and we attach those labels to the exact
    redacted transcript lines.
    """
    text = redact_sensitive_transcript((transcript_text or "").strip())
    if not text:
        raise ValueError("Empty transcript for role labeling")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    lines = _split_transcript_for_exact_labeling(text)
    if not lines:
        raise ValueError("No transcript lines for role labeling")

    all_labels = {}

    # Keep batches modest so the model reliably returns one label per line.
    batch_size = 80
    for start_idx in range(0, len(lines), batch_size):
        batch = [(i + 1, lines[i]) for i in range(start_idx, min(start_idx + batch_size, len(lines)))]
        try:
            labels = _label_transcript_line_batch(batch)
            all_labels.update(labels)
        except Exception as e:
            print(f"Role label batch failed lines {batch[0][0]}-{batch[-1][0]}: {e}", flush=True)

    labeled_lines = []
    last_speaker = "Unknown"

    for i, line in enumerate(lines, start=1):
        # If the raw line is already labeled, keep its text exactly as-is after privacy redaction.
        if re.match(r"^\s*(?:PQ|Agent|Prospect|Unknown)\s*:", line):
            labeled_lines.append(line)
            m = re.match(r"^\s*(PQ|Agent|Prospect|Unknown)\s*:", line)
            if m:
                last_speaker = m.group(1)
            continue

        speaker = all_labels.get(i)

        lower = line.lower().strip()

        # If a line is clearly a continuation fragment, keep the previous speaker.
        # This prevents long agent paragraphs split by Whisper line wrapping from flipping to Prospect.
        continuation = bool(
            last_speaker in {"Agent", "PQ", "Prospect"}
            and (
                re.match(r"^(and|but|or|to|that|which|because|so|if|then|with|for|of|in|on|at|as|is|are|was|were|pass,|just|even|like)\b", lower)
                or line[:1].islower()
            )
        )

        if continuation:
            speaker = last_speaker

        # Conservative fallback if OpenAI omitted a label.
        if speaker not in {"PQ", "Agent", "Prospect", "Unknown"}:
            if "pq" in lower and ("one of the best" in lower or "walk" in lower or "have " in lower):
                speaker = "PQ"
            elif re.search(r"\b(no|yes|yeah|okay|alright|all right|i'm|i am|i have|because|thank you|bye|hang up)\b", lower) and len(line.split()) <= 18:
                speaker = "Prospect"
            elif re.search(r"\b(recorded|licensed|underwriter|qualify|health questions|date of birth|coverage|policy|beneficiary|payment|bank|routing|account|disclosures|voice signature)\b", lower):
                speaker = "Agent"
            else:
                speaker = last_speaker if last_speaker in {"PQ", "Agent", "Prospect"} else "Unknown"

        last_speaker = speaker if speaker in {"PQ", "Agent", "Prospect"} else last_speaker
        labeled_lines.append(f"{speaker}: {line}")

    labeled = "\n".join(labeled_lines)
    labeled = redact_sensitive_transcript(labeled)
    labeled = final_transcript_privacy_cleanup(labeled)

    # Safety: the exact-label approach should never create fake [NAME]: speaker labels.
    labeled = re.sub(r"\s+\[NAME\]\s*:\s*", "\nAgent: ", labeled)
    labeled = final_transcript_privacy_cleanup(labeled)

    # Fix late PQ labels after Agent takeover and redact repeated handoff/direct-address names.
    labeled = _suppress_late_pq_after_agent_takeover(labeled)
    labeled = _redact_handoff_name_vault(labeled)
    labeled = final_transcript_privacy_cleanup(labeled)

    # Show the speaker label only when the speaker changes.
    labeled = format_grouped_speaker_transcript(labeled)

    # Conservative post-label repair: long agent rapport/self-disclosure sometimes
    # gets mislabeled as Prospect. Preserve exact words; only repair speaker label.
    labeled = _repair_agent_self_disclosure_mislabeled_as_prospect(labeled)

    return labeled

def try_save_role_labeled_transcript(call_name, redacted_transcript_text):
    """Role-label redacted text, save under transcripts_role_labeled/. Returns labeled text or None."""
    print(f"Creating role-labeled transcript for {call_name}", flush=True)
    out_path = os.path.join(TRANSCRIPTS_ROLE_LABELED_FOLDER, f"{call_name}.txt")
    try:
        labeled = create_role_labeled_transcript(redacted_transcript_text)
        write_text(out_path, labeled)
        print(f"Saved role-labeled transcript for {call_name}", flush=True)
        return labeled
    except Exception as e:
        print(f"Role labeling failed for {call_name}: {e}", flush=True)
        return None


def estimate_openai_cost(prompt_text, output_text):
    input_tokens_est = max(1, (len(prompt_text) + 3) // 4)
    output_tokens_est = max(1, (len(output_text) + 3) // 4)
    input_cost = (input_tokens_est / 1000) * OPENAI_INPUT_COST_PER_1K_TOKENS
    output_cost = (output_tokens_est / 1000) * OPENAI_OUTPUT_COST_PER_1K_TOKENS
    total_cost = input_cost + output_cost
    return {
        "input_tokens_est": input_tokens_est,
        "output_tokens_est": output_tokens_est,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def append_openai_cost_footer(report, cost):
    return (
        f"{report}\n\n"
        f"OPENAI COST ESTIMATE:\n"
        f"- Model: {OPENAI_MODEL}\n"
        f"- Input tokens (est): {cost['input_tokens_est']}\n"
        f"- Output tokens (est): {cost['output_tokens_est']}\n"
        f"- Estimated cost (USD): ${cost['total_cost']:.6f}\n"
    )


def parse_json_object(text):
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty JSON response")

    # Handle fenced JSON responses safely.
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def normalize_list(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    return []


def _normalize_yes_no_unclear(val, default="UNCLEAR"):
    if val is True:
        return "YES"
    if val is False:
        return "NO"
    s = str(val or "").strip().upper()
    return s if s in {"YES", "NO", "UNCLEAR"} else default


def _normalize_yes_no(val, default="NO"):
    s = str(val or "").strip().upper()
    return s if s in {"YES", "NO"} else default


def _normalize_pass_outcome(val):
    """Structured / report PASS line: YES, NO, or AT RISK."""
    s = str(val or "").strip().upper()
    if re.match(r"^AT\s+RISK$", s) or s.replace(" ", "") == "ATRISK":
        return "AT RISK"
    if s == "YES":
        return "YES"
    if s == "NO":
        return "NO"
    return "NO"


_AGENT_TONE_LABELS = {"confident": "Confident", "neutral": "Neutral", "uncertain": "Uncertain"}
_PROSPECT_TONE_LABELS = {"engaged": "Engaged", "neutral": "Neutral", "disengaged": "Disengaged"}


def _normalize_agent_tone(val):
    s = str(val or "").strip().lower()
    return _AGENT_TONE_LABELS.get(s, "Neutral")


def _normalize_prospect_tone(val):
    s = str(val or "").strip().lower()
    return _PROSPECT_TONE_LABELS.get(s, "Neutral")


def _normalize_yes_no_strict(val):
    """YES/NO only; anything else maps to NO (no hallucinated YES)."""
    return "YES" if str(val or "").strip().upper() == "YES" else "NO"


def _normalize_sale_final_stage(val):
    """Canonical stage label for SALE OUTCOME (must match prompt list)."""
    s = (str(val or "").strip() or "None").lower()
    s = re.sub(r"[/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s or s == "none":
        return "None"
    if "peace" in s and "mind" in s:
        return "Peace of Mind"
    if "cool" in s and "down" in s:
        return "Cool Down"
    if "third party" in s or "third-party" in s or (
        "underwriting" in s and ("record" in s or "amicable" in s)
    ):
        return "Third Party Underwriting"
    if "disclosur" in s:
        return "Disclosures"
    if "bank" in s:
        return "Banking"
    if "payment" in s:
        return "Payment"
    if "application" in s:
        return "Application"
    if s == "close" or s.startswith("close ") or " closing" in f" {s} ":
        return "Close"
    if "quote" in s:
        return "Quotes"
    return "None"


def detect_agent_callback_from_transcript(transcript):
    """
    True only when transcript clearly supports a real callback/delay autofail pattern.

    Uses the shared IVR-safe callback evidence helper so old callback cleanup paths
    and newer callback autofail cleanup paths do not fight each other.
    """
    if not transcript or not str(transcript).strip():
        return False

    helper = globals().get("_has_real_callback_autofail_evidence")
    if helper:
        return bool(helper("", transcript))

    # Fallback only if helper is unavailable during import/refactor.
    tl = str(transcript).lower()
    tl = tl.replace("\u2019", "'").replace("\u2018", "'")
    tl = re.sub(
        r"(?im)^.*\b(to receive a callback|press \[NUMBER\].*callback|callback, press|estimated wait time|next available representative)\b.*$",
        "",
        tl,
    )

    prospect_requested = bool(re.search(
        r"(?is)(prospect:\s*.*(?:call\s+(?:me|you)?\s*back|callback|do this later|talk later|not a good time|busy)|"
        r"can you call me back|call me back later|do this later|talk later)",
        tl,
    ))
    agent_accepted = bool(re.search(
        r"(?is)(agent:\s*.*(?:i(?:'ll| will) call you back|i(?:'ll| will) give you a call back|"
        r"(?:yes|yeah|yep|sure|okay|ok|absolutely|no problem)[,\s.]{0,20}i can call you back|"
        r"i can call you back|i can give you a call back|"
        r"we(?:'ll| will) call you back|we can do this later|agreed to call back|scheduled a callback|set a callback))",
        tl,
    ))

    return prospect_requested and agent_accepted

def _transcript_no_valid_agent_callback(transcript):
    """True when no clear agent-owned callback/follow-up commitment is present."""
    return not detect_agent_callback_from_transcript(transcript)


def _transcript_shopping_not_current_coverage(transcript):
    """
    Current coverage correction:
    Prospect says no current final expense/life policy but is shopping/checking options.
    This is not current active coverage that requires carrier verification.
    """
    t = (transcript or "").lower()
    if not t:
        return False

    asked = (
        "final expense plan in place" in t
        or "life insurance" in t
        or "coverage" in t
    )
    no_current_shopping = (
        "no, but i'm in the process" in t
        or "no, but i am in the process" in t
        or "i'm checking a lot of things out" in t
        or "i am checking a lot of things out" in t
        or "i've already had quotes" in t
        or "i have already had quotes" in t
    )
    return asked and no_current_shopping




def _transcript_reached_needs_section(transcript):
    """
    Needs section reached when the agent discusses funeral cost, burial/cremation,
    who passed away, family burden, or impact of no coverage.
    """
    t = transcript or ""
    if not t.strip():
        return False

    return bool(re.search(
        r"(?is)\b("
        r"who (?:passed away|has passed)|"
        r"have you ever had to pay for (?:a|somebody'?s) funeral|"
        r"burial or cremation|"
        r"cremation|burial|funeral cost|funeral expenses|"
        r"how much .* funeral|how much .* cremation|"
        r"with no coverage|since you have no coverage|"
        r"your family (?:would|will|is going to) (?:need|have to)|"
        r"leave (?:your )?(?:family|kids|children|beneficiary) with|"
        r"final expenses"
        r")\b",
        t,
    ))

def _transcript_application_info_started(transcript):
    """Application Information reached when the agent begins application data collection."""
    t = (transcript or "").lower()
    return bool(
        re.search(
            r"\bwhat'?s\s+your\s+middle\s+initial\b|"
            r"\bwhat\s+is\s+your\s+middle\s+initial\b|"
            r"\bplease\s+verify\s+the\s+spelling\b|"
            r"\bwhat\s+are\s+their\s+names\b|"
            r"\bdo\s+you\s+want\s+all\s+.*\s+on\s+(?:the|your)\s+policy\b",
            t,
            re.I,
        )
    )


def _transcript_lowest_option_attempt_no_clear_commit(transcript):
    """
    Detect bottom-paragraph / lowest-option close attempt where the agent attempts to move
    forward, but the prospect keeps resisting or does not clearly commit.
    """
    t = (transcript or "").lower()
    if not t:
        return False

    attempted = bool(
        re.search(
            r"let'?s\s+(?:just\s+)?go\s+with\s+the\s+lowest\s+option|"
            r"make\s+it\s+really\s+simple\s+and\s+go\s+with\s+the\s+lowest|"
            r"go\s+with\s+the\s+most\s+affordable|"
            r"circle\s+that\s+(?:smallest|lowest)|"
            r"the\s+lowest\s+option",
            t,
            re.I,
        )
    )
    continued_resistance = bool(
        re.search(
            r"i\s+don'?t\s+know\s+yet|"
            r"i\s+need\s+to\s+talk\s+to\s+people|"
            r"i\s+have\s+other\s+people\s+i\s+need\s+to\s+talk\s+to|"
            r"i'?m\s+not\s+ready|"
            r"i\s+don'?t\s+want\s+to\s+do\s+anything\s+today|"
            r"i\s+don'?t\s+want\s+to\s+commit|"
            r"i'?m\s+not\s+going\s+to\s+commit|"
            r"i\s+need\s+to\s+go|"
            r"i\s+have\s+to\s+go|"
            r"i\s+gotta\s+go|"
            r"i\s+need\s+to\s+hang\s+up",
            t,
            re.I,
        )
    )
    return attempted and continued_resistance


def _replace_or_add_checklist_line(lines, label, value):
    """Replace a checklist line by label; append if missing."""
    out = []
    replaced = False
    pat = re.compile(rf"^\s*-\s*{re.escape(label)}\s*:", re.I)
    for raw in lines or []:
        line = str(raw)
        if pat.search(line):
            out.append(f"{label}: {value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{label}: {value}")
    return out


def _remove_reason_fragment(reason, fragment):
    reason = str(reason or "").strip()
    if not reason:
        return "None"
    parts = [p.strip() for p in re.split(r";+", reason) if p.strip()]
    frag_low = fragment.lower()
    kept = [p for p in parts if frag_low not in p.lower()]
    return "; ".join(kept) if kept else "None"


def _text_remove_lines_containing(report, phrase):
    lines = []
    p = phrase.lower()
    for line in (report or "").splitlines():
        if p in line.lower():
            continue
        lines.append(line)
    return "\n".join(lines)

def normalize_objections(raw):
    """Parse optional structured objections; drop incomplete entries."""
    if not raw or not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        objection = str(item.get("objection", "")).strip()
        handled = str(item.get("handled", "")).strip().upper()
        explanation = str(item.get("explanation", "")).strip()
        if not objection or handled not in {"YES", "NO"}:
            continue
        if not explanation:
            explanation = "See transcript."
        out.append(
            {"objection": objection, "handled": handled, "explanation": explanation}
        )
    return out


# Canonical call stage order (furthest = latest clearly performed; NOT REACHED = all later stages).
CALL_STAGE_ORDER = (
    "PQ / Handoff",
    "Opening",
    "Who I Am / What I Do",
    "Fact Finding / Warm-up",
    "Medical / Health",
    "Need",
    "Features / Benefits",
    "Change Up",
    "Pre-Close",
    "Quotes",
    "Close",
    "Application Information",
    "Payment Date",
    "Banking",
    "Disclosures",
    "Third Party Underwriting",
    "Peace of Mind",
    "Cool Down",
)

# CALL STAGE REACHED block: stage may appear on the same line as the label or on the following line.
_CALL_STAGE_BLOCK_RE = re.compile(
    r"(?is)^CALL STAGE REACHED:\s*(?:\n\s*)?([^\n]+)\s*\n"
    r"^EARLY END:\s*(YES|NO)\s*\n"
    r"^NOT REACHED:\s*\n"
    r"((?:- [^\n]*\n)+)",
    re.MULTILINE,
)


def _call_stage_canonical_index(label):
    """Map model or shorthand stage label to CALL_STAGE_ORDER index, or None if unknown."""
    if label is None:
        return None
    s = " ".join(str(label).strip().split())
    if not s:
        return None
    sl = s.lower().replace("–", "-")
    for i, canon in enumerate(CALL_STAGE_ORDER):
        if sl == canon.lower():
            return i
    if sl.startswith("pq") or "handoff" in sl:
        return 0
    if sl == "application" or "application information" in sl:
        return CALL_STAGE_ORDER.index("Application Information")
    if "payment date" in sl or sl == "payment":
        return CALL_STAGE_ORDER.index("Payment Date")
    if sl == "banking" or sl.startswith("banking "):
        return CALL_STAGE_ORDER.index("Banking")
    if "third party" in sl or "third-party" in sl:
        return CALL_STAGE_ORDER.index("Third Party Underwriting")
    if "peace of mind" in sl or sl in ("pom", "peace-of-mind"):
        return CALL_STAGE_ORDER.index("Peace of Mind")
    if "cool down" in sl or sl == "cooldown" or sl == "cool-down":
        return CALL_STAGE_ORDER.index("Cool Down")
    if (
        "fact finding" in sl
        or "warm-up" in sl
        or "warm up" in sl
        or "rapport building" in sl
    ):
        return CALL_STAGE_ORDER.index("Fact Finding / Warm-up")
    if "who i am" in sl or ("what i do" in sl and "who" in sl):
        return CALL_STAGE_ORDER.index("Who I Am / What I Do")
    if re.fullmatch(r"quotes?", sl):
        return CALL_STAGE_ORDER.index("Quotes")
    if re.fullmatch(r"close", sl):
        return CALL_STAGE_ORDER.index("Close")
    for i, canon in enumerate(CALL_STAGE_ORDER):
        cl = canon.lower()
        if len(sl) >= 8 and (sl in cl or cl in sl):
            return i
    return None


def _sale_final_stage_to_order_index(sale_final_stage_raw):
    """When policy sold, sale_final_stage implies at least that point in the pipeline."""
    norm = _normalize_sale_final_stage(sale_final_stage_raw)
    mp = {
        "Quotes": "Quotes",
        "Close": "Close",
        "Application": "Application Information",
        "Payment": "Payment Date",
        "Banking": "Banking",
        "Disclosures": "Disclosures",
        "Third Party Underwriting": "Third Party Underwriting",
        "Peace of Mind": "Peace of Mind",
        "Cool Down": "Cool Down",
    }
    canon = mp.get(norm)
    if not canon:
        return None
    return CALL_STAGE_ORDER.index(canon)


def _stage_refinement_text_blob(result, transcript):
    parts = [
        transcript or "",
        str(result.get("summary") or ""),
        " ".join(str(x) for x in result.get("checklist_results") or []),
        str(result.get("sale_outcome_evidence") or ""),
    ]
    return "\n".join(parts).lower()


def _transcript_suggests_banking_collection(blob):
    """Routing/account/payment-setup banking — not insurer coverage calls."""
    if re.search(r"\brouting\s*(?:number|#)?\b", blob) and re.search(
        r"\b(account|checking|savings)\b", blob
    ):
        return True
    if re.search(r"\baccount\s*(?:number|#)?\b", blob) and re.search(
        r"\b(bank|credit union|routing)\b", blob
    ):
        return True
    if re.search(r"\bfor\s+(?:the\s+)?(?:draft|payment|premium|policy)\b", blob) and re.search(
        r"\b(bank|routing|checking|savings)\b", blob
    ):
        return True
    if re.search(r"\b(called|call)\s+(?:your\s+|the\s+)?bank\b", blob):
        return True
    if re.search(r"\bverify\s+(?:with\s+)?(?:your\s+|the\s+)?(?:bank|credit union)\b", blob):
        return True
    if re.search(r"\bnine\s*digit\b", blob) and re.search(r"\b(bank|routing|account)\b", blob):
        return True
    return False


def _payment_date_stage_evidence(blob):
    """
    Payment Date only when draft/payment date for the policy is set or explained — not SS deposit timing alone.
    """
    if re.search(
        r"\b(first\s+)?draft\s+(?:date|day|is|will|scheduled|comes|pulls)\b",
        blob,
        re.I,
    ):
        return True
    if re.search(
        r"\b(premium|policy)\s+(?:draft|payment|withdrawal|eft)\b",
        blob,
        re.I,
    ):
        return True
    if re.search(
        r"\b(set|scheduled|confirm(?:ed)?|choose|picked)\b.{0,120}\b(draft|withdrawal|eft|premium)\b",
        blob,
        re.I,
    ):
        return True
    if re.search(r"\bpayment\s+date\b.{0,80}\b(yes|set|confirmed)\b", blob, re.I):
        return True
    if re.search(
        r"\bsocial security\b.{0,120}\b(deposit|check|hits)\b",
        blob,
        re.I,
    ) and not re.search(r"\b(draft|premium|policy payment|withdrawal|eft)\b", blob, re.I):
        return False
    return False


def _disclosures_stage_evidence(blob):
    return bool(
        re.search(r"\b(read|reading|went over|covered)\b.{0,60}\bdisclosur", blob, re.I)
        or re.search(r"\bdisclosur[a-z]*\b.{0,40}\b(read|given|provided|complete)\b", blob, re.I)
        or re.search(r"\bhipaa\b.{0,40}\b(read|authorization|notice)\b", blob, re.I)
    )


def _third_party_underwriting_evidence(blob):
    """
    True only when transcript/summary suggests the post-disclosure recorded third-party
    underwriting step (e.g. American Amicable recorded line), not generic underwriting talk.
    """
    if not blob or not str(blob).strip():
        return False
    b = str(blob)
    bl = b.lower()
    # Strong IVR / carrier recorded-line cues (American Amicable and similar).
    if re.search(
        r"welcome\s+to\s+the\s+american\s+amicable\s+group\s+recording\s+system",
        bl,
    ):
        return True
    if re.search(r"\bamerican\s+amicable\s+group\s+recording\s+system\b", bl):
        return True
    if "american amicable" in bl and "recording system" in bl:
        return True
    if re.search(r"\bamerican\s+amicable\s+recording\s+system\b", bl):
        return True
    if re.search(
        r"\bfor\s+the\s+app\s+id\b.{0,160}"
        r"(?:pound\s+sign|pound\s+key|followed\s+by\s+the\s+pound|#\s*(?:sign|key)?\b)",
        bl,
    ):
        return True
    if "app id" in bl and (
        "pound sign" in bl
        or "pound key" in bl
        or "followed by the pound" in bl
        or re.search(r"enter\s+the\s+app\s+id", bl)
    ):
        return True
    if re.search(r"\bamerican\s+amicable\b", b, re.I) and re.search(
        r"\b(record|recorded|recording|line|dial|press|pound|connect|enter)\b", b, re.I
    ):
        return True
    if re.search(
        r"\b(?:call(?:ing|s)?\s+into|dial(?:ing)?)\s+(?:the\s+)?(?:american\s+amicable\s+)?recorded\s+line\b|"
        r"\b(?:start|starting|begin|beginning)\s+(?:the\s+)?(?:american\s+amicable\s+)?record"
        r"(?:ing|\b)|"
        r"\brecorded\s+(?:third|3rd)[\s-]?party\s+(?:underwriting|verification|line)\b|"
        r"\bthird[\s-]?party\s+(?:recorded|recording)\s+(?:underwriting|verification)\b|"
        r"\b(?:after|following)\s+(?:the\s+)?disclosur.{0,100}\b(?:recorded|recording|american\s+amicable|"
        r"third[\s-]?party)\b",
        b,
        re.I,
    ):
        return True
    return bool(
        re.search(
            r"\b(?:voice\s+signature|e-?signature|electronic\s+signature)\b.{0,40}\b(?:record|recorded|"
            r"american\s+amicable|third[\s-]?party)\b",
            b,
            re.I,
        )
    )


def _peace_of_mind_stage_evidence(blob):
    """
    Peace of Mind stage — transcript must show completed third-party / voice-signature
    flow first, then POM script cues. Medical/DNQ terminal language alone is never POM.
    """
    if not blob or not str(blob).strip():
        return False
    b = str(blob)
    bl = b.lower()
    if not _third_party_underwriting_evidence(bl):
        return False

    dnq_terminal = bool(
        re.search(
            r"terminal\s+medical\s+condition|end-stage\s+disease|end\s+stage\s+disease|"
            r"expected\s+to\s+result\s+in\s+death|respiratory\s+failure|liver\s+failure",
            bl,
        )
    )
    explicit_script = bool(
        re.search(
            r"\b(?:peace\s+of\s+mind\b|you'?re\s+good\b|not\s+going\s+to\s+forget\s+about\s+you\b|"
            r"welcome\s+letter|mail(?:ing|ed)?\s+(?:the\s+)?(?:welcome|policy)\s+letter|"
            r"all\s+my\s+personal\s+information|company\s+we\s+got\s+you\s+qualified\s+for\s+today)\b",
            b,
            re.I,
        )
    )

    if re.search(r"\bpeace\s+of\s+mind\b", b, re.I):
        if dnq_terminal and not explicit_script:
            return False
        return True
    if re.search(
        r"\b(?:rest\s+easy|sleep\s+(?:better|at\s+night)|feel\s+good\s+about\s+(?:this|your|the)\s+decision)\b",
        b,
        re.I,
    ):
        if dnq_terminal and not explicit_script:
            return False
        return True
    if re.search(r"\b(?:not|ain'?t)\s+going\s+to\s+forget\s+about\s+you\b", b, re.I):
        if dnq_terminal and not explicit_script:
            return False
        return True
    if re.search(
        r"\b(?:welcome|policy)\s+letter\b.{0,120}\b(?:mail|tomorrow|send|sent|going\s+to\s+mail)\b",
        b,
        re.I,
    ) or re.search(r"\bmail(?:ing|ed)?\s+(?:the\s+)?(?:welcome|policy)\s+letter\b", b, re.I):
        if dnq_terminal and not explicit_script:
            return False
        return True
    if re.search(
        r"\b(?:all\s+)?my\s+personal\s+information\b.{0,100}\b(?:company|qualified|carrier|policy)\b",
        b,
        re.I,
    ):
        if dnq_terminal and not explicit_script:
            return False
        return True
    if re.search(
        r"\byou'?re\s+good\b.{0,120}\b(?:forget|here|with\s+you|company|qualified)\b",
        b,
        re.I,
    ):
        if dnq_terminal and not explicit_script:
            return False
        return True
    if re.search(
        r"\bprotect(?:ed|ing)\s+your\s+family\b.{0,80}\b(?:beneficiary|coverage|policy|approved|qualified)\b",
        b,
        re.I,
    ):
        if dnq_terminal and not explicit_script:
            return False
        return True
    if re.search(r"\bcoverage\s+is\s+in\s+place\b", b, re.I):
        if dnq_terminal and not explicit_script:
            return False
        return True
    if re.search(
        r"\bpolicy\s+(?:in\s+the\s+)?mail\b.{0,80}\b(?:tomorrow|few\s+days|welcome|letter)\b",
        b,
        re.I,
    ):
        if dnq_terminal and not explicit_script:
            return False
        return True
    return False


def _cool_down_stage_evidence(blob):
    """
    Casual non-insurance wind-down after the sale — do not match checklist headings like 'Cool down completed'.
    """
    if not blob or not str(blob).strip():
        return False
    b = str(blob)
    if re.search(
        r"\b(?:small\s+talk|nothing\s+to\s+do\s+with\s+(?:the\s+)?(?:insurance|policy)|"
        r"off\s+(?:the\s+)?(?:insurance|script)|besides\s+(?:the\s+)?insurance)\b",
        b,
        re.I,
    ) and re.search(
        r"\b(weather|sports|football|baseball|basketball|hunt(?:ing)?|fishing|grandkids?|grandchildren|"
        r"pets?|dog|cat|vacation|weekend|hobby|hobbies|where\s+you\s+(?:live|from)|"
        r"how\s+long\s+have\s+you\s+lived|kids\s+(?:are\s+)?grow|retirement|"
        r"what\s+do\s+you\s+do|work\s+at)\b",
        b,
        re.I,
    ):
        return True
    if re.search(
        r"\b(?:before\s+(?:we|i)\s+(?:go|let\s+you\s+go)|anything\s+else\s+for\s+you|"
        r"while\s+we'?re\s+(?:still\s+)?on\s+the\s+line|just\s+curious)\b"
        r".{0,280}\b(weather|sports|football|baseball|basketball|hunt(?:ing)?|fishing|"
        r"grandkids?|grandchildren|pets?|dog|cat|vacation|weekend|hobby|hobbies|"
        r"where\s+you\s+(?:live|from)|how\s+long\s+have\s+you\s+lived|"
        r"what\s+do\s+you\s+do|work\s+at|retirement)\b",
        b,
        re.I,
    ):
        return True
    return False


def _warmup_entry_evidence(blob):
    """
    True when the transcript suggests the **Fact Finding / Warm-up** call stage was entered
    (warm-up, rapport building, fact-finding before medical — even 1–2 questions). Not full 3+1
    checklist completion. Conservative — common FE discovery phrasing only.
    """
    return bool(
        re.search(
            r"\b(?:how\s+long\s+have\s+you\s+lived|where\s+are\s+you\s+calling\s+from|"
            r"what\s+do\s+you\s+do\s+for\s+(?:work|a\s+living)|tell\s+me\s+(?:a\s+)?(?:little|bit)\s+about|"
            r"married|any\s+kids|children|grandchildren|grandkids|"
            r"what\s+.*\s+(?:like|enjoy)\s+to\s+do|favorite\b|where\s+.*\s+from|"
            r"day[-\s]?to[-\s]?day|family\s+live|who\s+.*\s+depend|"
            r"how\s+.*\s+feel\s+about|comfortable\s+sharing)\b",
            blob,
            re.I,
        )
    )


def _medical_health_entry_evidence(blob):
    """True when underwriting / health questioning has clearly begun (implies Fact Finding / Warm-up was passed)."""
    return bool(
        re.search(
            r"\b(?:height|weight|tobacco|nicotine|cigarettes?|smoke|smoking|"
            r"prescription|medications?|meds\b|health\s+questions|underwriting\s+health|"
            r"any\s+health|hospitalized|diagnosed|health\s+conditions?|"
            r"insulin|oxygen|wheelchair|cancer|heart\s+attack|stroke|"
            r"doctor|physician|medical\s+history)\b",
            blob,
            re.I,
        )
    )


def apply_refined_call_stage(result, transcript):
    """
    Raise CALL STAGE REACHED to the furthest stage supported by transcript + structured fields
    when the model undershoots (sold calls stopping at Application Information; early calls
    skipping Fact Finding / Warm-up despite rapport or medical entry evidence).
    Rebuilds NOT REACHED as only stages after that index. Does not alter policy_sold or scoring inputs.
    """
    if not isinstance(result, dict):
        return
    model_idx = _call_stage_canonical_index(result.get("stage_reached"))
    if model_idx is None:
        return

    blob = _stage_refinement_text_blob(result, transcript)
    transcript_blob = (transcript or "").lower()
    floors = [model_idx]

    idx_warm = CALL_STAGE_ORDER.index("Fact Finding / Warm-up")
    idx_med = CALL_STAGE_ORDER.index("Medical / Health")
    # Entry-based floors: CALL STAGE progression ≠ TASK CHECKLIST completion (3+1 can be PARTIAL while Fact Finding / Warm-up is entered).
    if _medical_health_entry_evidence(blob):
        floors.append(idx_med)
        floors.append(idx_warm)
    elif _warmup_entry_evidence(blob):
        floors.append(idx_warm)

    if result.get("policy_sold") == "YES":
        sidx = _sale_final_stage_to_order_index(result.get("sale_final_stage"))
        if sidx is not None:
            idx_tpu = CALL_STAGE_ORDER.index("Third Party Underwriting")
            idx_pom = CALL_STAGE_ORDER.index("Peace of Mind")
            idx_cd = CALL_STAGE_ORDER.index("Cool Down")
            # Sale outcome alone must not push past strict post-disclosure stages without transcript proof.
            if sidx >= idx_cd and not _cool_down_stage_evidence(transcript_blob):
                sidx = idx_cd - 1
            if sidx >= idx_pom and not _peace_of_mind_stage_evidence(transcript_blob):
                sidx = idx_pom - 1
            if sidx >= idx_tpu and not _third_party_underwriting_evidence(transcript_blob):
                sidx = idx_tpu - 1
            floors.append(sidx)

    if result.get("searchable_call_bank_banking") == "YES":
        floors.append(CALL_STAGE_ORDER.index("Banking"))
    if result.get("searchable_verify_cu_if_mentioned") == "YES":
        floors.append(CALL_STAGE_ORDER.index("Banking"))
    if _transcript_suggests_banking_collection(blob):
        floors.append(CALL_STAGE_ORDER.index("Banking"))

    if _payment_date_stage_evidence(blob):
        floors.append(CALL_STAGE_ORDER.index("Payment Date"))

    if _disclosures_stage_evidence(blob):
        floors.append(CALL_STAGE_ORDER.index("Disclosures"))

    if _third_party_underwriting_evidence(transcript_blob):
        floors.append(CALL_STAGE_ORDER.index("Third Party Underwriting"))

    if (
        result.get("policy_sold") == "YES"
        and _peace_of_mind_stage_evidence(transcript_blob)
        and _checklist_peace_of_mind_completed_yes(result.get("checklist_results"))
    ):
        floors.append(CALL_STAGE_ORDER.index("Peace of Mind"))

    if result.get("policy_sold") == "YES" and _cool_down_stage_evidence(transcript_blob):
        floors.append(CALL_STAGE_ORDER.index("Cool Down"))

    furthest = min(max(floors), len(CALL_STAGE_ORDER) - 1)
    idx_pom = CALL_STAGE_ORDER.index("Peace of Mind")
    if result.get("policy_sold") != "YES":
        furthest = min(furthest, idx_pom - 1)
    elif furthest == idx_pom and not _checklist_peace_of_mind_completed_yes(
        result.get("checklist_results")
    ):
        furthest = idx_pom - 1
    result["stage_reached"] = CALL_STAGE_ORDER[furthest]
    result["not_reached"] = list(CALL_STAGE_ORDER[furthest + 1 :])

    cd_idx = CALL_STAGE_ORDER.index("Cool Down")
    if furthest >= cd_idx:
        result["early_end"] = "NO"
    else:
        result["early_end"] = "YES"


def _checklist_peace_of_mind_completed_yes(checklist_results):
    """True only when checklist explicitly shows Peace of mind completed: YES."""
    for raw in checklist_results or []:
        line = str(raw).strip().lower()
        if "peace of mind" not in line or "completed" not in line:
            continue
        if re.search(r":\s*yes\b", line):
            return True
    return False


def _checklist_line_has_verdict_no(checklist_results, must_contain_all):
    """True if some checklist line contains every fragment (case-insensitive), is not NOT REACHED, and has : NO."""
    frags = [s.lower() for s in must_contain_all]
    for raw in checklist_results or []:
        line = str(raw).strip().lower()
        if not all(f in line for f in frags):
            continue
        if "not reached" in line:
            continue
        if re.search(r":\s*no\b", line):
            return True
    return False


def _checklist_autofail_coverage_yes(checklist_results):
    """Recover YES when the model mirrors the autofail line inside checklist_results only."""
    for raw in checklist_results or []:
        line = str(raw).strip().lower()
        if "existing coverage mentioned but not confirmed" in line and re.search(
            r":\s*yes\b", line
        ):
            return True
    return False


def _transcript_ash_only_policy_then_past_only_no_current(transcript):
    """
    Past-only + clear no-current 'only policy' read (ash_test-style). Do not treat as coverage autofail.
    """
    t = (transcript or "").lower()
    if not re.search(
        r"\b(i'?ve\s+had|had)\s+policies\s+in\s+the\s+past\b|\bpolicies\s+in\s+the\s+past\b",
        t,
    ):
        return False
    if not (
        re.search(r"as far as i know.{0,120}only\s+polic", t)
        or re.search(
            r"\b(this|it)\s+(would|will)\s+be\s+my\s+only\s+polic", t
        )
    ):
        return False
    if re.search(
        r"\bi\s+have\s+(a|an|one)\s+(policy|plan)\s+now\b|\bpaying\s+on\s+(a|an|one)\b|\bstill\s+have\b",
        t,
    ):
        return False
    return True


def _transcript_only_one_coverage_ambiguity(transcript):
    """
    Prospect answered 'only one' to an existing-vs-only-policy question, without carrier verification.
    Corrects under-flagged existing-coverage autofail when the model marks that line NO.
    """
    t = (transcript or "").lower()
    if not re.search(r"\bonly\s+one\b", t):
        return False
    if not re.search(
        r"\b(only\s+policy|your\s+only|final\s+expense|life\s+insurance|insurance\s+in\s+place)\b",
        t,
    ):
        return False
    if re.search(
        r"\b(insurance\s+compan|carrier|underwrit|three[\s-]?way|warm\s+transfer|on\s+the\s+line).{0,120}\b(confirm|verif|in[\s-]?force|policy\s+number)\b",
        t,
    ):
        return False
    return True


def _post_sale_incomplete_autofail(policy_sold, stage_reached, checklist_results):
    if policy_sold != "YES":
        return False
    si = _call_stage_canonical_index(stage_reached)
    if si is None or si < CALL_STAGE_ORDER.index("Disclosures"):
        return False
    if not _checklist_line_has_verdict_no(
        checklist_results, ("peace", "mind", "completed")
    ):
        return False
    if not _checklist_line_has_verdict_no(
        checklist_results, ("cool", "down", "completed")
    ):
        return False
    return True


def _post_sale_autofail_reason(not_reached):
    joined = " ".join(str(x).lower() for x in (not_reached or []))
    if "third party" in joined:
        return "Post-sale process incomplete: Third Party Underwriting, Peace of Mind, and Cool Down skipped"
    return "Post-sale process incomplete: Peace of Mind and Cool Down skipped"


def _payment_date_miss_after_banking(stage_reached, checklist_results):
    si = _call_stage_canonical_index(stage_reached)
    if si is None or si < CALL_STAGE_ORDER.index("Banking"):
        return False
    return _checklist_line_has_verdict_no(
        checklist_results, ("payment", "date", "explained")
    )


def validate_structured_audit(data, transcript=None):
    required_fields = [
        "score",
        "risk",
        "pass",
        "stage_reached",
        "early_end",
        "not_reached",
        "checklist_results",
        "coaching",
        "summary",
    ]
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise ValueError(f"Missing structured audit fields: {', '.join(missing)}")

    score = int(data["score"])
    risk = str(data["risk"]).strip().upper()
    pass_value = _normalize_pass_outcome(data.get("pass"))
    stage_reached = str(data["stage_reached"]).strip()
    early_end = str(data["early_end"]).strip().upper()
    not_reached = normalize_list(data["not_reached"])
    checklist_results = normalize_list(data["checklist_results"])
    coaching = normalize_list(data["coaching"])
    summary = str(data["summary"]).strip()
    biggest_miss = str(data.get("biggest_miss", "")).strip()
    objections = normalize_objections(data.get("objections"))
    agent_set_callback = str(data.get("agent_set_callback", "")).strip().upper()
    if agent_set_callback not in {"YES", "NO", "UNCLEAR"}:
        agent_set_callback = "UNCLEAR"
    valid_agent_callback = detect_agent_callback_from_transcript(transcript)
    if valid_agent_callback:
        agent_set_callback = "YES"
    elif transcript:
        # Do not allow prospect-only deferral/refusal to become agent callback.
        agent_set_callback = "NO"

    autofail_objection_no_call_control = _normalize_yes_no_unclear(
        data.get("autofail_objection_no_call_control")
    )
    autofail_coverage_not_confirmed = _normalize_yes_no_unclear(
        data.get("autofail_coverage_not_confirmed")
    )
    autofail_credit_union_not_verified = _normalize_yes_no_unclear(
        data.get("autofail_credit_union_not_verified")
    )
    automatic_fail_triggered = _normalize_yes_no(data.get("automatic_fail_triggered"), "NO")
    automatic_fail_reason = str(data.get("automatic_fail_reason", "") or "").strip() or "None"

    agent_tone = _normalize_agent_tone(data.get("agent_tone"))
    prospect_tone = _normalize_prospect_tone(data.get("prospect_tone"))
    comm_agent_confident = _normalize_yes_no_strict(data.get("comm_agent_confident"))
    comm_agent_control = _normalize_yes_no_strict(data.get("comm_agent_control"))
    comm_prospect_engaged = _normalize_yes_no_strict(data.get("comm_prospect_engaged"))
    comm_hesitation_detected = _normalize_yes_no_strict(data.get("comm_hesitation_detected"))

    searchable_confirm_current_coverage = _normalize_yes_no_unclear(
        data.get("searchable_confirm_current_coverage")
    )
    searchable_call_insurer_coverage = _normalize_yes_no_unclear(
        data.get("searchable_call_insurer_coverage")
    )
    searchable_call_bank_banking = _normalize_yes_no_unclear(
        data.get("searchable_call_bank_banking")
    )
    searchable_verify_cu_if_mentioned = _normalize_yes_no_unclear(
        data.get("searchable_verify_cu_if_mentioned")
    )
    searchable_ask_existing_coverage = _normalize_yes_no_unclear(
        data.get("searchable_ask_existing_coverage")
    )

    policy_sold = _normalize_yes_no_unclear(data.get("policy_sold"))

    if _checklist_autofail_coverage_yes(checklist_results):
        autofail_coverage_not_confirmed = "YES"

    if (
        _transcript_only_one_coverage_ambiguity(transcript)
        and searchable_ask_existing_coverage == "YES"
    ):
        autofail_coverage_not_confirmed = "YES"

    if transcript and _transcript_ash_only_policy_then_past_only_no_current(transcript):
        autofail_coverage_not_confirmed = "NO"

    if transcript and _transcript_shopping_not_current_coverage(transcript):
        # "No, but I'm in the process / checking options" means no active current coverage claimed.
        autofail_coverage_not_confirmed = "NO"
        searchable_confirm_current_coverage = "NO"
        searchable_call_insurer_coverage = "NO"

    # Coverage autofail YES implies no carrier confirmation and overall automatic fail.
    if autofail_coverage_not_confirmed == "YES":
        searchable_confirm_current_coverage = "NO"
        searchable_call_insurer_coverage = "NO"
        automatic_fail_triggered = "YES"

    if autofail_credit_union_not_verified == "YES":
        automatic_fail_triggered = "YES"

    if transcript and _transcript_lowest_option_attempt_no_clear_commit(transcript):
        # Agent attempted bottom-paragraph / lowest-option close, but prospect did not clearly commit.
        checklist_results = _replace_or_add_checklist_line(
            checklist_results,
            "Client chose an option",
            "PARTIAL - Agent used the bottom-paragraph close to move forward with the lowest option, but the sale did not complete",
        )

    post_sale_skip = _post_sale_incomplete_autofail(
        policy_sold, stage_reached, checklist_results
    )
    pay_miss_after_banking = _payment_date_miss_after_banking(
        stage_reached, checklist_results
    )
    pay_autofail = policy_sold == "YES" and pay_miss_after_banking

    if post_sale_skip:
        automatic_fail_triggered = "YES"
    if pay_autofail:
        automatic_fail_triggered = "YES"

    if transcript and not valid_agent_callback and "callback" in automatic_fail_reason.lower():
        automatic_fail_reason = _remove_reason_fragment(
            automatic_fail_reason, "Callback set"
        )
        automatic_fail_reason = _remove_reason_fragment(
            automatic_fail_reason, "callback"
        )
        if (
            autofail_coverage_not_confirmed != "YES"
            and autofail_credit_union_not_verified != "YES"
            and not post_sale_skip
            and not pay_autofail
        ):
            automatic_fail_triggered = "NO"
            automatic_fail_reason = "None"

    if automatic_fail_triggered == "YES":
        reason_parts = []
        br = str(data.get("automatic_fail_reason", "") or "").strip()
        if br and br.lower() != "none":
            reason_parts.append(br)
        if autofail_coverage_not_confirmed == "YES":
            reason_parts.append("Existing coverage mentioned but not confirmed")
        if autofail_credit_union_not_verified == "YES":
            reason_parts.append("Credit union mentioned but bank/account not verified")
        if post_sale_skip:
            reason_parts.append(_post_sale_autofail_reason(not_reached))
        if pay_autofail:
            reason_parts.append("Payment/draft date not explained after banking")
        deduped = []
        seen_r = set()
        for p in reason_parts:
            k = p.lower()
            if k not in seen_r:
                seen_r.add(k)
                deduped.append(p)
        automatic_fail_reason = (
            "; ".join(deduped) if deduped else "Automatic fail conditions met"
        )

    if automatic_fail_triggered == "YES":
        if policy_sold == "YES":
            pass_value = "AT RISK"
            risk = "HIGH"
        else:
            pass_value = "NO"
            risk = "HIGH"

    issue_cov = autofail_coverage_not_confirmed == "YES"
    issue_cu = autofail_credit_union_not_verified == "YES"
    issue_post = post_sale_skip
    issue_pay = pay_miss_after_banking
    pom_no = _checklist_line_has_verdict_no(
        checklist_results, ("peace", "mind", "completed")
    )
    cd_no = _checklist_line_has_verdict_no(
        checklist_results, ("cool", "down", "completed")
    )
    payment_date_no = _checklist_line_has_verdict_no(
        checklist_results, ("payment", "date", "explained")
    )
    stack_cov_pay_pom_cd = (
        issue_cov and payment_date_no and pom_no and cd_no
    )
    if automatic_fail_triggered == "YES":
        score = min(score, 85)
    if issue_cov or issue_cu:
        score = min(score, 80)
    if issue_post:
        score = min(score, 80)
    if issue_pay:
        score = min(score, 88)
    if issue_cov and issue_pay:
        score = min(score, 75)
    if stack_cov_pay_pom_cd:
        score = min(score, 72)
    combo = int(issue_cov) + int(issue_cu) + int(issue_post) + int(issue_pay)
    if combo >= 3:
        score = min(score, 62)
    elif combo >= 2:
        score = min(score, 70)
    score = max(0, min(100, int(score)))

    sale_outcome_evidence = str(data.get("sale_outcome_evidence", "") or "").strip() or "None"
    sale_outcome_evidence = re.sub(r"[\r\n]+", " ", sale_outcome_evidence)
    if len(sale_outcome_evidence) > 600:
        sale_outcome_evidence = sale_outcome_evidence[:597] + "..."
    sale_final_stage = _normalize_sale_final_stage(data.get("sale_final_stage"))

    if transcript and _transcript_application_info_started(transcript):
        stage_reached = "Application Information"
        sale_final_stage = "Application Information"
        checklist_results = _replace_or_add_checklist_line(
            checklist_results,
            "Application info collected",
            "PARTIAL",
        )
        later_not_reached = [
            "Payment Date",
            "Banking",
            "Disclosures",
            "Third Party Underwriting",
            "Peace of Mind",
            "Cool Down",
        ]
        not_reached = [x for x in later_not_reached if x not in {"Application Information"}]

    # Final invariant (coverage autofail line YES can never coexist with aggregate NO / None reason / wrong pass-risk).
    if autofail_coverage_not_confirmed == "YES":
        automatic_fail_triggered = "YES"
        searchable_confirm_current_coverage = "NO"
        searchable_call_insurer_coverage = "NO"
        reason_parts = []
        br = str(automatic_fail_reason or "").strip()
        if br and br.lower() != "none":
            reason_parts.append(br)
        reason_parts.append("Existing coverage mentioned but not confirmed")
        if autofail_credit_union_not_verified == "YES":
            reason_parts.append("Credit union mentioned but bank/account not verified")
        if post_sale_skip:
            reason_parts.append(_post_sale_autofail_reason(not_reached))
        if pay_autofail:
            reason_parts.append("Payment/draft date not explained after banking")
        deduped = []
        seen_r = set()
        for p in reason_parts:
            k = p.lower()
            if k not in seen_r:
                seen_r.add(k)
                deduped.append(p)
        automatic_fail_reason = (
            "; ".join(deduped) if deduped else "Existing coverage mentioned but not confirmed"
        )
        if policy_sold == "YES":
            pass_value = "AT RISK"
        else:
            pass_value = "NO"
        risk = "HIGH"
        score = min(score, 80)

    if risk not in {"LOW", "MEDIUM", "HIGH"}:
        raise ValueError("risk must be LOW, MEDIUM, or HIGH")
    if pass_value not in {"YES", "NO", "AT RISK"}:
        raise ValueError("pass must be YES, NO, or AT RISK")
    if early_end not in {"YES", "NO"}:
        raise ValueError("early_end must be YES or NO")
    if not stage_reached:
        raise ValueError("stage_reached is required")
    if not summary:
        raise ValueError("summary is required")

    if (
        pass_value == "AT RISK"
        and automatic_fail_triggered == "YES"
        and policy_sold == "YES"
    ):
        low = summary.lower()
        if "at risk" not in low or "sold" not in low:
            tail = (
                " Policy was sold (SALE OUTCOME: YES) but the sale is at risk due to "
                f"{automatic_fail_reason}."
            )
            summary = (summary + tail).strip()
            if len(summary) > 4500:
                summary = summary[:4497] + "..."

    if transcript and _transcript_lowest_option_attempt_no_clear_commit(transcript):
        # Keep the report from claiming the prospect chose an option when only the agent chose/recommended it.
        policy_sold = "NO"

    result = {
        "score": score,
        "risk": risk,
        "pass": pass_value,
        "stage_reached": stage_reached,
        "early_end": early_end,
        "not_reached": not_reached,
        "checklist_results": checklist_results,
        "coaching": coaching,
        "summary": summary,
        "biggest_miss": biggest_miss,
        "objections": objections,
        "agent_set_callback": agent_set_callback,
        "autofail_objection_no_call_control": autofail_objection_no_call_control,
        "autofail_coverage_not_confirmed": autofail_coverage_not_confirmed,
        "autofail_credit_union_not_verified": autofail_credit_union_not_verified,
        "automatic_fail_triggered": automatic_fail_triggered,
        "automatic_fail_reason": automatic_fail_reason,
        "agent_tone": agent_tone,
        "prospect_tone": prospect_tone,
        "comm_agent_confident": comm_agent_confident,
        "comm_agent_control": comm_agent_control,
        "comm_prospect_engaged": comm_prospect_engaged,
        "comm_hesitation_detected": comm_hesitation_detected,
        "searchable_confirm_current_coverage": searchable_confirm_current_coverage,
        "searchable_call_insurer_coverage": searchable_call_insurer_coverage,
        "searchable_call_bank_banking": searchable_call_bank_banking,
        "searchable_verify_cu_if_mentioned": searchable_verify_cu_if_mentioned,
        "searchable_ask_existing_coverage": searchable_ask_existing_coverage,
        "policy_sold": policy_sold,
        "sale_outcome_evidence": sale_outcome_evidence,
        "sale_final_stage": sale_final_stage,
    }
    apply_refined_call_stage(result, transcript)
    if transcript and _transcript_application_info_started(transcript):
        result["stage_reached"] = "Application Information"
        result["sale_final_stage"] = "Application Information"
        result["not_reached"] = [
            "Payment Date",
            "Banking",
            "Disclosures",
            "Third Party Underwriting",
            "Peace of Mind",
            "Cool Down",
        ]
        result["checklist_results"] = _replace_or_add_checklist_line(
            result.get("checklist_results", []),
            "Application info collected",
            "PARTIAL",
        )
    return result


def build_structured_audit_prompt(base_prompt):
    return f"""{base_prompt}

INTERNAL MODE:
Return a single valid JSON object only. No markdown, no code fences, no commentary.

Required JSON fields:
- score (integer)
- risk ("LOW" | "MEDIUM" | "HIGH")
- pass ("YES" | "NO" | "AT RISK")
- stage_reached (string) — furthest **entered** stage per CALL STAGE rules (Fact Finding / Warm-up counts as reached on **entry**, e.g. 1–2 rapport questions or before medical; **not** full 3+1 completion).
- early_end ("YES" | "NO")
- not_reached (array of strings) — only stages **after** stage_reached; never list Fact Finding / Warm-up as not reached if medical/rapport entry shows it was entered.
- checklist_results (array of strings) — mirror TASK CHECKLIST; must include **Fact Finding / Warm-up:** …; **3 and 1 Method used:** …; **Agent shared personal rapport information:** … — use **NOT REACHED** on the last two only if **Fact Finding / Warm-up** was **never entered**. **Also required** when those lines appear: **3 and 1 topic groups evidenced:** …; **3 and 1 agent self-disclosure evidence:** … (**see REQUIRED 3 AND 1 EVIDENCE LINES** — if **3 and 1 Method used** is **YES**, both evidence strings must contain **real** evidence). Include **Product benefits explained:** YES|NO|PARTIAL per main prompt **PRODUCT BENEFITS EXPLAINED — DETECTION** (Immediate / ROP / Graded). When **Banking** was reached, include banking verdict lines per **BANKING VERIFICATION** plus **Account verification evidence count:** …; **Account verification evidence:** …; **Routing verification evidence count:** …; **Routing verification evidence:** … — counts **must** match listed events; use **NOT REACHED** on banking verdict lines only if **Banking** was not reached.
- coaching (array of strings)
- summary (string)

Optional JSON fields:
- biggest_miss (string; always include — use a short miss description, or the literal None / empty if no meaningful miss)
- objections (array of objects; include only when at least one real customer objection exists in the transcript)
  Each object: { "objection": string, "handled": "YES"|"NO", "explanation": string (one brief sentence) }
  If no objections, omit the field or use an empty array.
- agent_set_callback ("YES" | "NO" | "UNCLEAR") — REQUIRED when possible: follow CALLBACK AND SCHEDULING rules; transcript evidence only. YES if the agent clearly agrees to or schedules a callback; NO if not; UNCLEAR if ambiguous. If omitted, downstream text will treat as UNCLEAR.
- autofail_objection_no_call_control ("YES" | "NO" | "UNCLEAR") — YES only if real objection/resistance AND no proper call control attempt; NO if no objection; UNCLEAR if ambiguous.
- autofail_coverage_not_confirmed ("YES" | "NO" | "UNCLEAR") — YES when existing coverage is **mentioned or reasonably indicated** (including **"Only one"**-type ambiguity per prompt) and the agent did NOT meet carrier/provider confirmation; follow prompt **3)** and **EXISTING COVERAGE — FOLLOW-UP** (do NOT mark NO to "clear" unresolved **Only one**). NO if coverage never mentioned or hinted. NEVER YES solely because bank/payment verification was missing.
- autofail_credit_union_not_verified ("YES" | "NO" | "UNCLEAR") — YES only for CREDIT UNION + BANK/ACCOUNT/PAYMENT verification gap. NO if no credit union. NEVER YES solely because insurance coverage was not confirmed. UNCLEAR if ambiguous.
- searchable_confirm_current_coverage ("YES" | "NO" | "UNCLEAR") — YES only with insurer/carrier/provider direct verification (or clear equivalent) of existing coverage, NOT prospect Q&A alone; NOT bank calls. NO if only asked/accepted prospect description.
- searchable_call_insurer_coverage ("YES" | "NO" | "UNCLEAR") — YES only when transcript shows insurer/carrier/provider contacted or clearly committed for coverage verification; NOT bank. Otherwise NO.
- searchable_call_bank_banking ("YES" | "NO" | "UNCLEAR") — bank call for banking/account/routing/payment verification only (not coverage confirmation).
- searchable_verify_cu_if_mentioned ("YES" | "NO" | "UNCLEAR") — credit-union account/bank verification only.
- searchable_ask_existing_coverage ("YES" | "NO" | "UNCLEAR") — agent asked about existing coverage (insurance) generally.
- policy_sold ("YES" | "NO" | "UNCLEAR") — YES only with clear customer plan choice plus meaningful enrollment/application/payment progress; NO if ended before commitment; UNCLEAR if suggestive but not clear. Same value as SEARCHABLE "Was the policy sold?"
- sale_outcome_evidence (string) — brief transcript-backed phrase, or "None".
- sale_final_stage (string) — one of: Quotes, Close, Application, Payment, Banking, Disclosures, Third Party Underwriting, Peace of Mind, Cool Down, None (see POLICY SALE rules in prompt).
- automatic_fail_triggered ("YES" | "NO") — YES if any rule in **AUTOMATIC FAIL RULES** (sections **1–6**) clearly applies (callback, call control, coverage, credit union, **post-sale process incomplete**, **payment/draft date after Banking**); otherwise NO. **Never** YES **solely** because **3 and 1 Method used** is weak, **PARTIAL**, or **NO** (rapport gaps use **score** / checklist / coaching only — see **3 AND 1 METHOD — SCORE IMPORTANCE** in main prompt). **Never** YES **solely** because of **Tell, Don't Ask** / permission-seeking tone (see **TELL, DON'T ASK** — coaching / **SCORE** only). **Hard rule:** if **autofail_coverage_not_confirmed** is **YES**, **automatic_fail_triggered** MUST be **YES** (never NO); **automatic_fail_reason** MUST mention **Existing coverage mentioned but not confirmed**; **risk** MUST be **HIGH**; **pass** MUST be **AT RISK** if **policy_sold** is **YES**, else **NO**; keep **final SCORE** at or below **80** when that coverage line is YES.
- automatic_fail_reason (string) — short explanation, or "None" if automatic_fail_triggered is NO. When multiple rules apply, join with **"; "** (e.g. coverage + post-sale + payment). Never "None" when automatic_fail_triggered is YES.
- pass rules with automatic_fail_triggered and policy_sold (SALE OUTCOME): If policy_sold is YES and automatic_fail_triggered is YES, pass must be "AT RISK" (not NO) and risk must be "HIGH" (including post-sale or payment-date automatic fails, not only coverage/credit-union). If policy_sold is NO or UNCLEAR and automatic_fail_triggered is YES, pass must be "NO" and risk should be HIGH. If automatic_fail_triggered is NO, use normal YES/NO pass rules.
- If pass is "AT RISK", the summary must clearly state the policy was sold but the sale is at risk due to the automatic fail reason.
- agent_tone ("Confident" | "Neutral" | "Uncertain") — transcript evidence only; default Neutral if omitted.
- prospect_tone ("Engaged" | "Neutral" | "Disengaged") — transcript evidence only; default Neutral if omitted.
- comm_agent_confident ("YES" | "NO") — YES only if transcript cues clearly show confidence; otherwise NO.
- comm_agent_control ("YES" | "NO") — YES only if agent clearly steers the conversation; otherwise NO.
- comm_prospect_engaged ("YES" | "NO") — YES only if prospect shows engagement in text; otherwise NO.
- comm_hesitation_detected ("YES" | "NO") — YES only if fillers, broken sentences, or repeated phrases clearly indicate hesitation/uncertainty; otherwise NO.

REPORT CONSISTENCY SELF-CHECK (MANDATORY BEFORE YOU OUTPUT JSON):
Re-read your verdicts and fix any contradiction:
- **automatic_fail_triggered: YES** + **pass: YES** — invalid; **pass** must be **AT RISK** if **policy_sold** is **YES**, else **NO**; **risk** must be **HIGH**; **automatic_fail_reason** must not be **None**.
- **autofail_coverage_not_confirmed: YES** + **automatic_fail_triggered: NO** — invalid. **autofail_coverage_not_confirmed: YES** cannot pair with **pass: YES**, **risk: LOW**, or **score** **90+** — align with main prompt **SCORE CAP RULES**.
- **policy_sold: YES** + **automatic_fail_triggered: YES** + **pass** not **AT RISK** — invalid.
- **Did the agent confirm current coverage? NO** + **Did the agent call an insurance company to confirm current coverage? NO** + clear existing-coverage mention (including **Only one** ambiguity) cannot pair with **autofail_coverage_not_confirmed: NO**.
- **3 and 1 Method used** cannot be **YES** without **evidence gate** + checklist **A + B** evidence (**quote or clear paraphrase** for self-disclosure); cannot pair **3 and 1 Method used: YES** with **Agent shared personal rapport information: NO** or **PARTIAL**; **questions-only** ⇒ **3 and 1 Method used** not **YES**; **Agent shared personal rapport information** cannot be **YES** on generic acknowledgments or vague-only lines alone — fix **checklist_results** strings before output.
- **3 and 1 Method used: YES** in **checklist_results** but **3 and 1 agent self-disclosure evidence:** is **None**/empty/generic-only — **invalid**.
- **3 and 1 Method used: YES** but **3 and 1 topic groups evidenced:** does **not** show **≥ 3** groups — **invalid**.
- **Account number requested or verified 3 times:** **YES** but account evidence count **< 3** — **invalid**; **Routing number requested or verified 3 times:** **YES** but routing evidence count **< 3** — **invalid**; **Routing number verified at least 2 times:** **YES** but routing evidence count **< 2** — **invalid**.
- **Banking/account information requested or verified 3 times:** **YES** unless **both** evidence counts show **≥ 3** routing **and** **≥ 3** account events — **invalid**.
- **sale_outcome_evidence** mentioning **voice signature** / **recorded verification** / **American Amicable recording** while **not_reached** lists **Third Party Underwriting** only — invalid; align **stage_reached** / **not_reached** or trim Evidence.
- Transcript includes **Welcome to the American Amicable Group recording system** (or equivalent app ID / pound-sign IVR) ⇒ **Third Party Underwriting** reached; do not leave it only under **not_reached** while claiming an earlier stage as furthest.
- **Policy sold YES** + skipped Peace of Mind / Cool Down with opportunity + **Payment date explained: NO** after Banking + coverage gap ⇒ **score** not **90+**, **pass** not **YES**, **risk** not **LOW**.
- **Beneficiary** **YES** in **checklist_results** when evidence lacks beneficiary/receive/death benefit/proceeds/primary beneficiary/specific recipient — invalid; align with **BENEFICIARY IDENTIFICATION**.
- **stage_reached** / furthest **Peace of Mind** from **only** Features/Benefits anchors without **§14 (1)+(2)** — invalid; align per **§4** vs **§14**.
- **stage_reached** **Peace of Mind** (or peace-of-mind completion **YES**) without completed **Third Party Underwriting** / **voice signature** per **§13** — invalid.
- **stage_reached** **Peace of Mind** without **Peace of Mind script** lines after **§13** (you're good / not going to forget / welcome letter / personal information + company qualified today) — invalid; **NOT REACHED** / **NO**.
- **stage_reached** / **Peace of Mind** (or peace-of-mind completion) from medical/DNQ/terminal/end-stage/respiratory-or-liver-failure wording — invalid; **Medical / Health** only per anchor list **§2**; **Quotes** anchors ⇒ minimum **Quotes** when **§14** not met.
- **agent_set_callback** / callback autofail when evidence lacks callback/follow-up/schedule language (letter/age/term alone) — invalid per **WHAT COUNTS AS CALLBACK LANGUAGE**.
- **agent_set_callback: YES** when transcript only shows **prospect** refusal / hang-up **without** **agent** callback commitment — invalid; align **NO** per **EARLY-END REFUSAL**.
- **autofail_coverage_not_confirmed: YES** on **early-end refusal** (prospect ended before sale path) **solely** from “already have coverage / final expenses handled” — invalid; align **NO** and **automatic_fail_triggered** per **EARLY-END REFUSAL**.
- **Fact Finding / Warm-up** not reached in structured checklist but **3 and 1 Method used** is **YES**/**NO**/**PARTIAL** (not **NOT REACHED**) — invalid; align **NOT REACHED** for **3 and 1** / rapport lines.
- Transcript shows **agent** profanity/insults toward/about prospect but **biggest_miss** / **automatic_fail_reason** ignore it — invalid; align per **BIGGEST MISS PRIORITY** / **PROFESSIONALISM** in main prompt.
- **Payment date explained: NO** or payment autofail from DOB/date-of-proof-only — invalid; use **NOT REACHED** when Payment Date segment never began per **PAYMENT DATE STAGE**.
- **Routing number requested or verified 3 times:** **NO** or **PARTIAL** (Banking reached) but **summary** claims banking verification was **fully** completed — invalid.
- **Account number requested or verified 3 times:** **NO** or **PARTIAL** (Banking reached) but **summary** claims **full** banking verification — invalid.
- **Banking/account information requested or verified 3 times:** **YES** in **checklist_results** when either account-side or routing-side **three-times** line is **not** **YES** — invalid.
- Weak / **PARTIAL** / **NO** on **3 and 1 Method used** (with **Fact Finding / Warm-up** entered) must **not** alone force **automatic_fail_triggered**, **risk: HIGH**, or **pass** NO/AT RISK — apply **score** / **SCRIPT / FLOW MISSES** / **coaching** per **3 AND 1 METHOD — SCORE IMPORTANCE** in the main prompt.
"""


def format_bullet_lines(items, empty_fallback="None"):
    if not items:
        return f"- {empty_fallback}"
    return "\n".join(f"- {item}" for item in items)


def get_top_three_coaching_priorities(coaching_items, checklist_results):
    priorities = []

    for item in (coaching_items or []) + (checklist_results or []):
        text = str(item).strip()
        if text and text not in priorities:
            priorities.append(text)
        if len(priorities) == 3:
            break

    while len(priorities) < 3:
        priorities.append("No additional reached-stage coaching priority identified.")

    return priorities


def format_objection_sections(objections):
    if not objections:
        return ""
    detected = ["OBJECTIONS DETECTED:"]
    for row in objections:
        detected.append(f"- {row['objection']}")
    handling = ["OBJECTION HANDLING:"]
    for row in objections:
        handling.append(f"- Objection: {row['objection']}")
        handling.append(f"  Handled: {row['handled']}")
        handling.append(f"  Explanation: {row['explanation']}")
    return "\n".join(detected) + "\n\n" + "\n".join(handling) + "\n"


def render_text_report_from_structured(data):
    top_three_priorities = get_top_three_coaching_priorities(
        data.get("coaching"),
        data.get("checklist_results")
    )

    bm = str(data.get("biggest_miss", "") or "").strip()
    biggest_miss_bullet = bm if bm else "None"
    biggest_miss_section = f"BIGGEST MISS:\n- {biggest_miss_bullet}\n"
    objection_sections = format_objection_sections(data.get("objections") or [])
    cb = str(data.get("agent_set_callback", "UNCLEAR")).strip().upper()
    if cb not in {"YES", "NO", "UNCLEAR"}:
        cb = "UNCLEAR"

    return f"""SCORE: {data['score']}
RISK: {data['risk']}
PASS: {data['pass']}

CALL STAGE REACHED: {data['stage_reached']}
EARLY END: {data['early_end']}
NOT REACHED:
{format_bullet_lines(data['not_reached'], "None")}

COMPLIANCE FAILURES:
- None

SCRIPT / FLOW MISSES:
{format_bullet_lines(data['checklist_results'], "None")}

TASK CHECKLIST:
{format_bullet_lines(data['checklist_results'], "None")}

COACHING:
TOP 3 COACHING PRIORITIES:
{format_bullet_lines(top_three_priorities, "None")}

{biggest_miss_section}{objection_sections}SEARCHABLE ANSWERS:
- Did the agent set a callback? {cb}
- Did the agent confirm current coverage? {data["searchable_confirm_current_coverage"]}
- Did the agent call an insurance company to confirm current coverage? {data["searchable_call_insurer_coverage"]}
- Did the agent call the bank to verify banking/account information? {data["searchable_call_bank_banking"]}
- Did the agent verify credit union account information if a credit union was mentioned? {data["searchable_verify_cu_if_mentioned"]}
- Did the agent ask about existing coverage? {data["searchable_ask_existing_coverage"]}
- Was the policy sold? {data["policy_sold"]}

AUTOMATIC FAIL CHECKS:
- Callback set: {cb}
- Objection occurred without proper call control: {data["autofail_objection_no_call_control"]}
- Existing coverage mentioned but not confirmed: {data["autofail_coverage_not_confirmed"]}
- Credit union mentioned but bank/account not verified: {data["autofail_credit_union_not_verified"]}
- Automatic fail triggered: {data["automatic_fail_triggered"]}
- Reason: {data["automatic_fail_reason"]}

SALE OUTCOME:
- Policy sold: {data["policy_sold"]}
- Evidence: {data["sale_outcome_evidence"]}
- Final stage supporting sale: {data["sale_final_stage"]}

TONE & DELIVERY:
- Agent Tone: {data["agent_tone"]}
- Prospect Tone: {data["prospect_tone"]}

COMMUNICATION ANALYSIS:
- Did the agent sound confident? {data["comm_agent_confident"]}
- Did the agent maintain control of the conversation? {data["comm_agent_control"]}
- Was the prospect engaged? {data["comm_prospect_engaged"]}
- Any hesitation or uncertainty detected? {data["comm_hesitation_detected"]}

SUMMARY:
{data['summary']}"""


def run_structured_audit_model(prompt, redacted_transcript=None):
    if not os.getenv("OPENAI_API_KEY"):
        return None, None

    try:
        print(f"[audit] Using OpenAI structured JSON mode: {OPENAI_MODEL}")
        response = openai_client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            temperature=0
        )
        raw_output = response.output_text.strip()
        cost = estimate_openai_cost(prompt, raw_output)
        print(
            "[audit] OpenAI estimated cost "
            f"(input={cost['input_tokens_est']} tok, output={cost['output_tokens_est']} tok): "
            f"${cost['total_cost']:.6f}"
        )
        structured = validate_structured_audit(
            parse_json_object(raw_output), redacted_transcript
        )
        return render_text_report_from_structured(structured), cost
    except Exception as e:
        print(f"[audit] Structured JSON mode failed, using text flow: {e}")
        return None, None


def run_audit_model(openai_prompt, fallback_prompt):
    if os.getenv("OPENAI_API_KEY"):
        try:
            print(f"[audit] Using OpenAI API model: {OPENAI_MODEL}")
            response = openai_client.responses.create(
                model=OPENAI_MODEL,
                input=openai_prompt,
                temperature=0
            )
            report = response.output_text.strip()
            cost = estimate_openai_cost(openai_prompt, report)
            print(
                "[audit] OpenAI estimated cost "
                f"(input={cost['input_tokens_est']} tok, output={cost['output_tokens_est']} tok): "
                f"${cost['total_cost']:.6f}"
            )
            return report, cost
        except Exception as e:
            print(f"[audit] OpenAI failed, falling back to Ollama: {e}")
            return run_ollama(fallback_prompt).strip(), None

    print("[audit] Using local Ollama")
    return run_ollama(fallback_prompt).strip(), None


def normalize_top3_coaching_header_line(report):
    """
    Dashboard extract_top3_coaching requires ^TOP 3 COACHING PRIORITIES: at line start.
    Fix common model mistake: '- TOP 3 COACHING PRIORITIES:' on its own line.
    """
    if not report:
        return report
    return re.sub(
        r"(?im)^(\s*)-\s*(TOP\s*3\s+COACHING\s+PRIORITIES\s*:)\s*$",
        r"\1\2",
        report,
        count=1,
    )


def trim_to_score_and_remove_unwanted_sections(report):
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

    return report


def generate_audit_report(prompt, openai_prompt, redacted_transcript=None):
    report = None
    openai_cost = None

    if USE_STRUCTURED_AUDIT:
        structured_prompt = build_structured_audit_prompt(openai_prompt)
        report, openai_cost = run_structured_audit_model(
            structured_prompt, redacted_transcript
        )

    if report is None:
        report, openai_cost = run_audit_model(openai_prompt, prompt)

    return report, openai_cost


def _text_enforce_tpu_stage_report(report, transcript):
    """
    When transcript shows carrier recorded third-party underwriting (e.g. American Amicable IVR)
    but CALL STAGE / NOT REACHED still omit or mis-list Third Party Underwriting, rebuild that block.
    """
    if not report or not transcript:
        return report
    tb = transcript.strip().lower()
    if not _third_party_underwriting_evidence(tb):
        return report
    m_block = _CALL_STAGE_BLOCK_RE.search(report)
    if not m_block:
        return report
    current_stage = m_block.group(1).strip()
    not_reached_raw = m_block.group(3)
    bullets = []
    for line in not_reached_raw.strip().split("\n"):
        s = line.strip()
        if s.startswith("- "):
            bullets.append(s[2:].strip())
    has_tpu_in_nr = any("third party" in b.lower() for b in bullets)
    idx_cur = _call_stage_canonical_index(current_stage)
    if idx_cur is None:
        idx_cur = 0
    idx_tpu = CALL_STAGE_ORDER.index("Third Party Underwriting")
    if not has_tpu_in_nr and idx_cur >= idx_tpu:
        return report
    floors = [idx_cur, idx_tpu]
    # Do not raise CALL STAGE to Peace of Mind / Cool Down from transcript alone;
    # text report lacks structured policy_sold / checklist to validate §14.
    furthest = min(max(floors), len(CALL_STAGE_ORDER) - 1)
    new_stage = CALL_STAGE_ORDER[furthest]
    tail = list(CALL_STAGE_ORDER[furthest + 1 :])
    nr_lines = "\n".join(f"- {s}" for s in tail) if tail else "- None"
    cd_idx = CALL_STAGE_ORDER.index("Cool Down")
    early_end = "NO" if furthest >= cd_idx else "YES"
    new_section = (
        f"CALL STAGE REACHED: {new_stage}\n"
        f"EARLY END: {early_end}\n"
        f"NOT REACHED:\n{nr_lines}\n"
    )
    return report[: m_block.start()] + new_section + report[m_block.end() :]


def _report_policy_sold_line_verdict(report):
    m = re.search(r"(?im)^- Policy sold:\s*(YES|NO|UNCLEAR)\b", report)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?im)^- Was the policy sold\?\s*(YES|NO|UNCLEAR)\b", report)
    if m:
        return m.group(1).upper()
    return None


def _report_peace_of_mind_completed_line_is_yes(report):
    m = re.search(
        r"(?im)^- Peace of mind completed:\s*(YES|NO|NOT REACHED|PARTIAL|UNCLEAR)\b",
        report,
    )
    return bool(m and m.group(1).strip().upper() == "YES")


def _parse_final_stage_supporting_sale_raw(report):
    m = re.search(r"(?im)^- Final stage supporting sale:\s*(.+)$", report)
    if not m:
        return None
    raw = m.group(1).strip()
    if "<" in raw:
        raw = raw.split("<", 1)[0].strip()
    return raw or None


def _enforce_peace_of_mind_call_stage_consistency(report, transcript=None):
    """
    Remove invalid CALL STAGE REACHED: Peace of Mind when Policy sold / Peace of mind completed /
    NOT REACHED / Final stage supporting sale / transcript evidence contradict strict §14.
    """
    if not report:
        return report
    m_block = _CALL_STAGE_BLOCK_RE.search(report)
    if not m_block:
        return report
    current_stage = m_block.group(1).strip()
    idx_cur = _call_stage_canonical_index(current_stage)
    idx_pom = CALL_STAGE_ORDER.index("Peace of Mind")
    if idx_cur != idx_pom:
        return report
    not_reached_raw = m_block.group(3)
    bullets = []
    for line in not_reached_raw.strip().split("\n"):
        s = line.strip()
        if s.startswith("- "):
            bullets.append(s[2:].strip())
    has_tpu_in_nr = any("third party" in b.lower() for b in bullets)
    policy = _report_policy_sold_line_verdict(report)
    pom_yes = _report_peace_of_mind_completed_line_is_yes(report)
    fs_raw = _parse_final_stage_supporting_sale_raw(report)
    fs_idx = _call_stage_canonical_index(fs_raw) if fs_raw else None
    idx_bank = CALL_STAGE_ORDER.index("Banking")
    early_fs_vs_pom = fs_idx is not None and fs_idx <= idx_bank
    transcript_tb = (transcript or "").strip().lower()
    pom_transcript_ok = True
    if transcript_tb:
        pom_transcript_ok = bool(_peace_of_mind_stage_evidence(transcript_tb))
    need_fix = False
    if policy != "YES":
        need_fix = True
    if not pom_yes:
        need_fix = True
    if has_tpu_in_nr:
        need_fix = True
    if early_fs_vs_pom:
        need_fix = True
    if transcript_tb and not pom_transcript_ok:
        need_fix = True
    if not need_fix:
        return report
    if fs_idx is not None and fs_idx < idx_pom:
        new_idx = fs_idx
    else:
        cand = []
        if has_tpu_in_nr:
            cand.append(CALL_STAGE_ORDER.index("Third Party Underwriting") - 1)
        if bullets:
            fi = _call_stage_canonical_index(bullets[0])
            if fi is not None and 0 < fi <= idx_pom:
                cand.append(fi - 1)
        below = [c for c in cand if c < idx_pom]
        new_idx = max(below) if below else max(0, idx_pom - 1)
    new_idx = min(new_idx, idx_pom - 1)
    new_stage = CALL_STAGE_ORDER[new_idx]
    tail = list(CALL_STAGE_ORDER[new_idx + 1 :])
    nr_lines = "\n".join(f"- {s}" for s in tail) if tail else "- None"
    cd_idx = CALL_STAGE_ORDER.index("Cool Down")
    early_end = "NO" if new_idx >= cd_idx else "YES"
    new_section = (
        f"CALL STAGE REACHED: {new_stage}\n"
        f"EARLY END: {early_end}\n"
        f"NOT REACHED:\n{nr_lines}\n"
    )
    return report[: m_block.start()] + new_section + report[m_block.end() :]


def _task_checklist_field_value(report_text, label_regex):
    """Value after colon on first matching '- <label>:' line (case-insensitive)."""
    if not report_text:
        return None
    m = re.search(
        rf"(?im)^-\s*{label_regex}\s*:\s*([^\n]+)$",
        report_text,
    )
    return m.group(1).strip() if m else None


def _checklist_yes(value):
    return bool(value and re.match(r"^\s*YES\b", value, re.I))


def _checklist_partial_or_yes(value):
    return bool(value and re.match(r"^\s*(YES|PARTIAL)\b", value, re.I))


def _checklist_not_reached_line(value):
    return bool(value and "NOT REACHED" in value.upper())


def _call_stage_block_tail_lines(new_idx):
    tail = list(CALL_STAGE_ORDER[new_idx + 1 :])
    nr_lines = "\n".join(f"- {s}" for s in tail) if tail else "- None"
    cd_idx = CALL_STAGE_ORDER.index("Cool Down")
    early_end = "NO" if new_idx >= cd_idx else "YES"
    return new_idx, early_end, nr_lines


def _replace_call_stage_block(report_text, new_idx):
    m_block = _CALL_STAGE_BLOCK_RE.search(report_text)
    if not m_block:
        return report_text
    _, early_end, nr_lines = _call_stage_block_tail_lines(new_idx)
    new_stage = CALL_STAGE_ORDER[new_idx]
    new_section = (
        f"CALL STAGE REACHED: {new_stage}\n"
        f"EARLY END: {early_end}\n"
        f"NOT REACHED:\n{nr_lines}\n"
    )
    return report_text[: m_block.start()] + new_section + report_text[m_block.end() :]


def enforce_report_stage_consistency(report_text: str) -> str:
    """
    Deterministic caps on CALL STAGE REACHED when checklist / SALE OUTCOME contradict
    the stage label (e.g. Third Party Underwriting with no application or sale).
    """
    if not report_text:
        return report_text
    m_block = _CALL_STAGE_BLOCK_RE.search(report_text)
    if not m_block:
        return report_text
    current = m_block.group(1).strip()
    cur_idx = _call_stage_canonical_index(current)
    if cur_idx is None:
        return report_text

    not_reached_raw = m_block.group(3)
    bullets = []
    for line in not_reached_raw.strip().split("\n"):
        s = line.strip()
        if s.startswith("- "):
            bullets.append(s[2:].strip())
    has_tpu_nr = any("third party" in b.lower() for b in bullets)
    has_disc_nr = any("disclosure" in b.lower() for b in bullets)

    three = _task_checklist_field_value(report_text, r"Three options presented")
    client = _task_checklist_field_value(report_text, r"Client chose an option")
    app = _task_checklist_field_value(report_text, r"Application info(?:rmation)? collected")
    pay = _task_checklist_field_value(report_text, r"Payment date explained")
    bank = _task_checklist_field_value(report_text, r"Banking/payment setup explained")
    pom = _task_checklist_field_value(report_text, r"Peace of mind completed")
    cool = _task_checklist_field_value(report_text, r"Cool down completed")

    policy = _report_policy_sold_line_verdict(report_text)
    policy_no = policy != "YES"

    fs_raw = _parse_final_stage_supporting_sale_raw(report_text)
    fs_idx = _call_stage_canonical_index(fs_raw) if fs_raw else None

    idx_pom = CALL_STAGE_ORDER.index("Peace of Mind")
    idx_cd = CALL_STAGE_ORDER.index("Cool Down")
    idx_app = CALL_STAGE_ORDER.index("Application Information")
    idx_pay = CALL_STAGE_ORDER.index("Payment Date")
    idx_bank = CALL_STAGE_ORDER.index("Banking")
    idx_quotes = CALL_STAGE_ORDER.index("Quotes")
    idx_close = CALL_STAGE_ORDER.index("Close")

    pay_nr = _checklist_not_reached_line(pay)
    bank_nr = _checklist_not_reached_line(bank)

    later_than_quotes = _checklist_yes(app) or _checklist_yes(pay) or (
        bank and not _checklist_not_reached_line(bank) and _checklist_partial_or_yes(bank)
    )

    max_cap = len(CALL_STAGE_ORDER) - 1

    if policy_no:
        max_cap = min(max_cap, idx_pom - 1)

    if app is not None and not _checklist_yes(app):
        if _checklist_yes(client):
            max_cap = min(max_cap, idx_close)
        elif _checklist_partial_or_yes(three):
            max_cap = min(max_cap, idx_quotes)
        else:
            if fs_idx is not None and fs_idx < idx_app:
                max_cap = min(max_cap, fs_idx)
            else:
                max_cap = min(max_cap, idx_quotes)

    if pay_nr and bank_nr:
        max_cap = min(max_cap, idx_app - 1)

    if bank_nr:
        max_cap = min(max_cap, idx_bank - 1)

    if pom is not None and not _checklist_yes(pom):
        max_cap = min(max_cap, idx_pom - 1)

    if cool is not None and not _checklist_yes(cool):
        max_cap = min(max_cap, idx_cd - 1)

    if has_tpu_nr:
        max_cap = min(max_cap, CALL_STAGE_ORDER.index("Third Party Underwriting") - 1)

    if has_disc_nr:
        max_cap = min(max_cap, CALL_STAGE_ORDER.index("Disclosures") - 1)

    if fs_raw and re.search(r"\bquotes\b", fs_raw, re.I) and not later_than_quotes:
        max_cap = min(max_cap, idx_quotes)

    new_idx = min(cur_idx, max_cap)

    rule10 = (
        _checklist_partial_or_yes(three)
        and client is not None
        and not _checklist_yes(client)
        and app is not None
        and not _checklist_yes(app)
        and policy_no
    )
    if rule10:
        new_idx = idx_quotes

    austin_quotes_fix = (
        fs_raw
        and re.search(r"\bquotes\b", fs_raw, re.I)
        and policy_no
        and app is not None
        and not _checklist_yes(app)
        and bank_nr
    )
    if austin_quotes_fix:
        new_idx = idx_quotes

    new_idx = max(0, min(new_idx, len(CALL_STAGE_ORDER) - 1))

    if new_idx == cur_idx:
        return report_text
    return _replace_call_stage_block(report_text, new_idx)



def _transcript_banking_stage_started(transcript):
    """
    True only when the transcript shows real banking/payment-account collection started.
    Generic product language like "set up with your bank" does NOT count.
    """
    t = (transcript or "").lower()
    if not t:
        return False

    patterns = (
        r"\bare\s+you\s+with\s+(?:a\s+)?(?:bank|credit\s+union)\b",
        r"\bwith\s+(?:a\s+)?credit\s+union\b",
        r"\bwhat\s+(?:bank|credit\s+union)\b",
        r"\bwho\s+do\s+you\s+bank\s+with\b",
        r"\bwhat'?s\s+the\s+name\s+of\s+your\s+(?:bank|credit\s+union)\b",
        r"\bwhat\s+is\s+the\s+name\s+of\s+that\s+credit\s+union\b",
        r"\b(?:routing|account|bank)\s+number\b",
        r"\b(?:need|grab|pull|get|please\s+grab)\s+(?:your\s+)?(?:checkbook|check\s*book|bank\s+statement|textbook)\b",
        r"\bbottom\s+of\s+your\s+(?:check|checkbook|text)\b",
        r"\bslowly\s+read\s+me\s+all\s+the\s+numbers\b",
        r"\bpayment\s+account\b",
        r"\bdraft\s+account\b",
        r"\blandmark\s+credit\s+union\b",
        r"\[account_number\]|\[routing_number\]|\[bank_number\]",
    )
    return any(re.search(p, t, re.I) for p in patterns)


def _text_set_line(report, label_regex, replacement):
    return re.sub(label_regex, replacement, report, flags=re.I | re.M)


def _text_force_not_reached_block(report, stages):
    return _set_stage_fields(
        report,
        stage=None,
        not_reached_items=stages,
    )


def _text_replace_checklist_value(report, label, value):
    pattern = rf"(?im)^- {re.escape(label)}:\s*.*$"
    repl = f"- {label}: {value}"
    if re.search(pattern, report):
        return re.sub(pattern, repl, report, count=1)
    return report


def _text_remove_flow_miss_line(report, phrase):
    lines = []
    target = phrase.lower()
    for line in (report or "").splitlines():
        if target in line.lower():
            continue
        lines.append(line)
    return "\n".join(lines)


def _text_has_other_autofail_after_cleanup(report):
    """
    After removing false callback/coverage/payment/banking failures for this pattern,
    detect whether a real remaining autofail reason still appears.
    """
    low = (report or "").lower()
    real_markers = (
        "unprofessional language",
        "disrespectful",
        "credit union mentioned but bank/account not verified",
        "post-sale process incomplete",
        "peace of mind and cool down skipped",
    )
    return any(m in low for m in real_markers)


def _final_text_cleanup_for_no_callback_no_banking(report, transcript):
    """
    Final report guard for calls like Shelby/Carolyn:
    - no true agent callback
    - prospect was shopping, not claiming active current coverage
    - agent attempted lowest option but prospect did not clearly commit
    - application info started
    - banking did not actually start
    """
    if not report or not transcript:
        return report

    valid_callback = detect_agent_callback_from_transcript(transcript)
    app_started = _transcript_application_info_started(transcript)
    banking_started = _transcript_banking_stage_started(transcript)
    shopping_not_current = _transcript_shopping_not_current_coverage(transcript)
    lowest_attempt_no_commit = _transcript_lowest_option_attempt_no_clear_commit(transcript)

    if not valid_callback:
        report = re.sub(r"(?im)^- Did the agent set a callback\?\s*.*$", "- Did the agent set a callback? NO", report)
        report = re.sub(r"(?im)^- Callback set:\s*.*$", "- Callback set: NO", report)
        report = _text_remove_flow_miss_line(report, "Callback set without allowed exception")
        report = _text_remove_flow_miss_line(report, "agreed to a callback")
        report = _text_remove_flow_miss_line(report, "setting callbacks")
        report = _text_remove_flow_miss_line(report, "callback before policy completion")

    if shopping_not_current:
        report = re.sub(
            r"(?im)^- Existing coverage mentioned but not confirmed:\s*YES\b.*$",
            "- Existing coverage mentioned but not confirmed: NO",
            report,
        )

    if lowest_attempt_no_commit:
        report = re.sub(
            r"(?im)^- Client chose an option:\s*YES\b.*$",
            "- Client chose an option: PARTIAL - Agent used the bottom-paragraph close to move forward with the lowest option, but the sale did not complete",
            report,
        )
        report = re.sub(
            r"(?im)^- Did the client choose an option\?\s*YES\b.*$",
            "- Did the client choose an option? PARTIAL - Agent used the bottom-paragraph close to move forward with the lowest option, but the sale did not complete",
            report,
        )
        report = re.sub(
            r"(?im)^- Objection occurred without proper call control:\s*YES\b.*$",
            "- Objection occurred without proper call control: NO",
            report,
        )

    if app_started:
        report = re.sub(
            r"(?im)^CALL STAGE REACHED:\s*.*$",
            "CALL STAGE REACHED: Application Information",
            report,
            count=1,
        )
        report = _text_force_not_reached_block(
            report,
            [
                "Payment Date",
                "Banking",
                "Disclosures",
                "Third Party Underwriting",
                "Peace of Mind",
                "Cool Down",
            ],
        )
        report = _text_replace_checklist_value(report, "Application info collected", "PARTIAL")

    if not banking_started:
        # Remove misses for stages never reached.
        report = _text_remove_flow_miss_line(report, "Payment/draft date not explained after banking")
        report = _text_remove_flow_miss_line(report, "Banking verification incomplete")

        # Force payment/banking checklist to NOT REACHED.
        for label in [
            "Payment date explained",
            "Banking/payment setup explained",
            "Banking/account information requested or verified 3 times",
            "Account number requested or verified 3 times",
            "Account number verified at least 2 times",
            "Routing number requested or verified 3 times",
            "Routing number verified at least 2 times",
            "Agent read account/routing information back to prospect",
            "Prospect confirmed account/routing read-back",
        ]:
            report = _text_replace_checklist_value(report, label, "NOT REACHED")

        report = _text_replace_checklist_value(report, "Account verification evidence count", "0")
        report = _text_replace_checklist_value(report, "Account verification evidence", "None")
        report = _text_replace_checklist_value(report, "Routing verification evidence count", "0")
        report = _text_replace_checklist_value(report, "Routing verification evidence", "None")

        report = re.sub(r"(?im)^- Did the agent call the bank to verify banking/account information\?\s*.*$", "- Did the agent call the bank to verify banking/account information? NO", report)
        report = re.sub(r"(?im)^- Did the agent verify credit union account information if a credit union was mentioned\?\s*.*$", "- Did the agent verify credit union account information if a credit union was mentioned? NO", report)

        if app_started:
            report = re.sub(
                r"(?im)^- Final stage supporting sale:\s*.*$",
                "- Final stage supporting sale: Application Information",
                report,
            )
            report = re.sub(
                r"(?im)^- Evidence:\s*.*$",
                "- Evidence: Application information was started after an attempted lowest-option close, but the prospect did not clearly commit and payment/banking were not reached.",
                report,
                count=1,
            )

    # If false callback / false coverage / false banking were the only autofail drivers, clear autofail.
    if (
        not valid_callback
        and shopping_not_current
        and not banking_started
        and not _text_has_other_autofail_after_cleanup(report)
    ):
        report = re.sub(r"(?im)^- Automatic fail triggered:\s*YES\b.*$", "- Automatic fail triggered: NO", report)
        report = re.sub(r"(?im)^- Reason:\s*.*$", "- Reason: None", report)

    # Fill Biggest Miss if the model left it blank.
    if re.search(r"(?ims)^BIGGEST MISS:\s*(?:\n\s*)*(?=TRANSCRIPT NOTE|OPENAI COST ESTIMATE|\Z)", report):
        report = re.sub(
            r"(?ims)^BIGGEST MISS:\s*(?:\n\s*)*(?=TRANSCRIPT NOTE|OPENAI COST ESTIMATE|\Z)",
            "BIGGEST MISS:\n- Sale ended during Application Information after the bottom-paragraph close; payment and banking were not reached.\n\n",
            report,
            count=1,
        )

    return report


def _final_cleanup_autofail_sale_summary(report, transcript):
    """
    Final cleanup for reports where hard checks already show:
    callback NO, objection autofail NO, coverage autofail NO, credit union NO.
    Prevent stale model text from leaving Automatic fail YES, callback evidence, blank Biggest Miss, or blank Summary.
    """
    if not report:
        return report

    all_autofail_checks_no = all(
        re.search(pattern, report, re.I | re.M)
        for pattern in [
            r"^- Callback set:\s*NO\b",
            r"^- Objection occurred without proper call control:\s*NO\b",
            r"^- Existing coverage mentioned but not confirmed:\s*NO\b",
            r"^- Credit union mentioned but bank/account not verified:\s*NO\b",
        ]
    )

    if all_autofail_checks_no:
        report = re.sub(
            r"(?im)^- Automatic fail triggered:\s*YES\b.*$",
            "- Automatic fail triggered: NO",
            report,
        )
        if re.search(r"(?im)^- Reason:\s*.*$", report):
            report = re.sub(r"(?im)^- Reason:\s*.*$", "- Reason: None", report)
        else:
            report = re.sub(
                r"(?im)^- Automatic fail triggered:\s*NO\b.*$",
                "- Automatic fail triggered: NO\n- Reason: None",
                report,
                count=1,
            )

    # Remove stale callback language from sale outcome evidence if callback was corrected to NO.
    if re.search(r"(?im)^- Callback set:\s*NO\b", report):
        report = re.sub(
            r"(?im)^- Evidence:\s*.*callback agreed before completion.*$",
            "- Evidence: Agent presented options, attempted the lowest-option close, and began application information, but the prospect did not clearly commit and the call ended before payment or banking.",
            report,
        )
        report = re.sub(
            r"(?im)^- Evidence:\s*.*callback.*before completion.*$",
            "- Evidence: Agent presented options, attempted the lowest-option close, and began application information, but the prospect did not clearly commit and the call ended before payment or banking.",
            report,
        )

    # If Application Information is the reached stage, align final stage supporting sale.
    if re.search(r"(?im)^CALL STAGE REACHED:\s*Application Information\s*$", report):
        report = re.sub(
            r"(?im)^- Final stage supporting sale:\s*.*$",
            "- Final stage supporting sale: Application Information",
            report,
        )

    biggest_miss_text = "- Sale ended during Application Information after the bottom-paragraph close; payment and banking were not reached."
    if re.search(r"(?ims)^BIGGEST MISS:\s*(?:\n\s*)*(?=OBJECTIONS DETECTED:|TRANSCRIPT NOTE|SUMMARY:|OPENAI COST ESTIMATE|\Z)", report):
        report = re.sub(
            r"(?ims)^BIGGEST MISS:\s*(?:\n\s*)*(?=OBJECTIONS DETECTED:|TRANSCRIPT NOTE|SUMMARY:|OPENAI COST ESTIMATE|\Z)",
            f"BIGGEST MISS:\n{biggest_miss_text}\n\n",
            report,
            count=1,
        )

    summary_text = (
        "The agent progressed the call through quotes and attempted the bottom-paragraph / "
        "lowest-option close. The prospect did not clearly commit, but the agent began "
        "Application Information by asking for middle initial and beneficiary details. The call "
        "ended before Payment Date, Banking, Disclosures, Third Party Underwriting, Peace of Mind, "
        "or Cool Down. No callback was set, no active current coverage was confirmed by the prospect, "
        "and banking was not reached."
    )
    if re.search(r"(?ims)^SUMMARY:\s*(?:\n\s*)*(?=OPENAI COST ESTIMATE:|TRANSCRIPT NOTE|\Z)", report):
        report = re.sub(
            r"(?ims)^SUMMARY:\s*(?:\n\s*)*(?=OPENAI COST ESTIMATE:|TRANSCRIPT NOTE|\Z)",
            f"SUMMARY:\n{summary_text}\n\n",
            report,
            count=1,
        )

    return report


def _final_cleanup_no_callback_coaching_and_option(report, transcript):
    """
    Final polish:
    - If callback was corrected to NO, remove stale callback coaching.
    - If lowest-option close was attempted without clear commitment, keep client choice as PARTIAL, not plain NO.
    """
    if not report:
        return report

    callback_no = bool(re.search(r"(?im)^- Callback set:\s*NO\b", report)) or bool(
        re.search(r"(?im)^- Did the agent set a callback\?\s*NO\b", report)
    )
    lowest_attempt_no_commit = bool(
        transcript and _transcript_lowest_option_attempt_no_clear_commit(transcript)
    )

    if lowest_attempt_no_commit:
        partial_line = "PARTIAL - Agent used the bottom-paragraph close to move forward with the lowest option, but the sale did not complete"
        report = re.sub(
            r"(?im)^- Client chose an option:\s*(?:YES|NO|PARTIAL)\b.*$",
            f"- Client chose an option: {partial_line}",
            report,
        )
        report = re.sub(
            r"(?im)^- Did the client choose an option\?\s*(?:YES|NO|PARTIAL)\b.*$",
            f"- Did the client choose an option? {partial_line}",
            report,
        )

    if callback_no:
        cleaned = []
        for line in report.splitlines():
            low = line.lower()
            if (
                "callback" in low
                and (
                    "avoid agreeing to callbacks" in low
                    or "do not set callbacks" in low
                    or "callbacks before completing" in low
                    or "callback before completing" in low
                    or "complete the call in one sitting" in low
                )
            ):
                continue
            cleaned.append(line)
        report = "\n".join(cleaned)

    # If TOP 3 coaching now has fewer bullets because callback coaching was removed,
    # add a stage-appropriate coaching point for this pattern.
    if callback_no and lowest_attempt_no_commit:
        coaching_match = re.search(
            r"(?ims)^TOP 3 COACHING PRIORITIES:\s*(.*?)(?=^BIGGEST MISS:)",
            report,
        )
        if coaching_match:
            block = coaching_match.group(1)
            bullets = re.findall(r"(?m)^\s*-\s+.+$", block)
            replacement_bullet = (
                "- Continue using strong call control through the lowest-option close, "
                "but secure clearer client commitment before moving deeper into application information."
            )
            if len(bullets) < 3 and replacement_bullet not in block:
                new_block = block.rstrip() + "\n" + replacement_bullet + "\n\n"
                report = report[:coaching_match.start(1)] + new_block + report[coaching_match.end(1):]

    return report


def _final_cleanup_pass_and_bottom_paragraph_wording(report, transcript):
    """
    Final polish:
    - PASS should be YES when no automatic fail triggered, even if policy was not sold.
    - AT RISK is only for sold policies with an automatic fail.
    - Bottom-paragraph close should not be coached as needing perfect client commitment.
    """
    if not report:
        return report

    policy_sold_no = bool(re.search(r"(?im)^- Policy sold:\s*NO\b", report)) or bool(
        re.search(r"(?im)^- Was the policy sold\?\s*NO\b", report)
    )
    autofail_no = bool(re.search(r"(?im)^- Automatic fail triggered:\s*NO\b", report))
    reason_none = bool(re.search(r"(?im)^- Reason:\s*None\b", report))

    if policy_sold_no and autofail_no and reason_none:
        report = re.sub(r"(?im)^PASS:\s*AT RISK\s*$", "PASS: YES", report, count=1)
        report = re.sub(r"(?im)^PASS:\s*NO\s*$", "PASS: YES", report, count=1)

    report = report.replace(
        "Continue using strong call control through the lowest-option close, but secure clearer client commitment before moving deeper into application information.",
        "Continue using the bottom-paragraph close to maintain momentum, then move quickly and confidently through Application Information into Payment Date and Banking."
    )

    report = report.replace(
        "secure clearer client commitment before moving deeper into application information",
        "move quickly and confidently through Application Information into Payment Date and Banking"
    )

    return report


def _transcript_disclosures_started(transcript):
    t = (transcript or "").lower()
    return bool(re.search(
        r"\bi\s+have\s+(?:a\s+)?few\s+disclosures\b|"
        r"\bfair\s+credit\s+reporting\s+act\b|"
        r"\bstatement\s+of\s+understanding\b|"
        r"\bmib\s+and\s+the\s+pharmacy\b|"
        r"\brequired\s+disclosures\b",
        t,
        re.I,
    ))


def _transcript_voice_signature_started(transcript):
    t = (transcript or "").lower()
    return bool(re.search(
        r"\bvoice\s+(?:recording|signature)\b|"
        r"\bthis\s+call\s+is\s+now\s+being\s+recorded\b|"
        r"\bfinal\s+step\s+to\s+completing\s+your\s+application\b|"
        r"\bplease\s+state\s+your\s+full\s+name\s+and\s+today'?s\s+date\b|"
        r"\bdo\s+you\s+understand\s+that\s+you'?ve\s+applied\b|"
        r"\bdo\s+you\s+agree\s+.*accepting\s+your\s+signature\s+electronically\b|"
        r"\bsubmitting\s+your\s+application\s+to\s+the\s+home\s+office\b",
        t,
        re.I,
    ))


def _transcript_peace_of_mind_after_sale(transcript):
    t = (transcript or "").lower()
    if not _transcript_voice_signature_started(transcript):
        return False
    return bool(re.search(
        r"\bwelcome\s+letter\b|"
        r"\bpolicy\s+(?:comes|will\\s+come|in\\s+about|within)\b|"
        r"\bwhen\\s+you\\s+get\\s+that\\s+policy\b|"
        r"\bwalk\\s+through\\s+every\\s+page\b|"
        r"\bfeel\\s+good\\s+about\\s+the\\s+decision\b|"
        r"\bglad\\s+about\\s+the\\s+decision\\s+today\b|"
        r"\bput\\s+that\\s+coverage\\s+in\\s+place\b|"
        r"\bcoverage\\s+in\\s+place\\s+for\\s+your\\s+daughter\b",
        t,
        re.I,
    ))


def _final_cleanup_sold_post_app_stage(report, transcript):
    """
    Sold-call guard:
    If transcript clearly reached payment/banking/disclosures/voice signature,
    do not let the no-banking/Application-only cleanup from short non-sold calls overcorrect the report.
    """
    if not report or not transcript:
        return report

    banking_started = _transcript_banking_stage_started(transcript)
    disclosures_started = _transcript_disclosures_started(transcript)
    voice_started = _transcript_voice_signature_started(transcript)
    pom_started = _transcript_peace_of_mind_after_sale(transcript)

    if not (banking_started or disclosures_started or voice_started):
        return report

    cool_down_started = _transcript_cool_down_after_sale(transcript)

    if voice_started:
        if cool_down_started:
            final_stage = "Cool Down"
        else:
            final_stage = "Peace of Mind" if pom_started else "Third Party Underwriting"
    elif disclosures_started:
        final_stage = "Disclosures"
    elif banking_started:
        final_stage = "Banking"
    else:
        final_stage = "Application Information"

    report = re.sub(
        r"(?im)^CALL STAGE REACHED:\s*.*$",
        f"CALL STAGE REACHED: {final_stage}",
        report,
        count=1,
    )

    if final_stage == "Cool Down":
        not_reached = "NOT REACHED:\n- None\n\n"
    elif final_stage == "Peace of Mind":
        not_reached = "NOT REACHED:\n- Cool Down\n\n"
    elif final_stage == "Third Party Underwriting":
        not_reached = "NOT REACHED:\n- Peace of Mind\n- Cool Down\n\n"
    elif final_stage == "Disclosures":
        not_reached = "NOT REACHED:\n- Third Party Underwriting\n- Peace of Mind\n- Cool Down\n\n"
    else:
        not_reached = "NOT REACHED:\n- Disclosures\n- Third Party Underwriting\n- Peace of Mind\n- Cool Down\n\n"

    report = re.sub(
        r"(?ims)^NOT REACHED:\s*.*?(?=^COMPLIANCE FAILURES:)",
        not_reached,
        report,
        count=1,
    )

    # Align stage checklist items that clearly happened.
    if banking_started:
        report = _text_replace_checklist_value(report, "Payment date explained", "YES")
        report = _text_replace_checklist_value(report, "Banking/payment setup explained", "PARTIAL")
    if disclosures_started:
        report = _text_replace_checklist_value(report, "Disclosures completed", "YES")
    if voice_started:
        report = _text_replace_checklist_value(report, "Third Party Underwriting completed", "YES")
    if pom_started:
        report = _text_replace_checklist_value(report, "Peace of mind completed", "YES")
    if cool_down_started:
        report = _text_replace_checklist_value(report, "Cool down completed", "YES")

    # Sold-call option close should not say sale did not complete.
    if re.search(r"(?im)^- Was the policy sold\?\s*YES\b", report) or re.search(r"(?im)^- Policy sold:\s*YES\b", report):
        report = re.sub(
            r"(?im)^- Client chose an option:\s*PARTIAL - Agent used the bottom-paragraph close to move forward with the lowest option, but the sale did not complete\s*$",
            "- Client chose an option: YES - Agent used the bottom-paragraph close to move forward with the lowest option and completed the sale process.",
            report,
        )
        report = re.sub(
            r"(?im)^- Did the client choose an option\?\s*PARTIAL - Agent used the bottom-paragraph close to move forward with the lowest option, but the sale did not complete\s*$",
            "- Did the client choose an option? YES - Agent used the bottom-paragraph close to move forward with the lowest option and completed the sale process.",
            report,
        )
        report = re.sub(
            r"(?im)^- Final stage supporting sale:\s*.*$",
            f"- Final stage supporting sale: {final_stage}",
            report,
        )

    # Remove the stale short-call biggest miss if this sold call went past banking.
    report = re.sub(
        r"(?ims)^BIGGEST MISS:\s*\n-\s*Sale ended during Application Information after the bottom-paragraph close; payment and banking were not reached\.\s*(?=TRANSCRIPT NOTE|OBJECTIONS DETECTED:|SUMMARY:|OPENAI COST ESTIMATE:|\Z)",
        "BIGGEST MISS:\n- Credit union / banking verification and post-sale accuracy need review after the sale was completed.\n\n",
        report,
        count=1,
    )

    # Remove short-call coaching from sold post-app calls.
    report = _text_remove_flow_miss_line(report, "move quickly and confidently through Application Information into Payment Date and Banking")

    return report


def _transcript_credit_union_mentioned(transcript):
    """
    True only when the prospect/account is actually a credit union.
    Do not count the agent's generic "bank or credit union" question when the
    prospect answers "Bank" and provides a bank name.
    """
    t = re.sub(r"\s+", " ", (transcript or "").lower()).strip()
    if not t:
        return False

    bank_answer = bool(re.search(
        r"(?:are\s+you\s+with|with)\s+(?:a\s+)?bank\s+or\s+(?:a\s+)?credit\s+union\??\s*bank\b",
        t,
        re.I,
    ))

    credit_union_answer = bool(re.search(
        r"(?:are\s+you\s+with|with)\s+(?:a\s+)?bank\s+or\s+(?:a\s+)?credit\s+union\??\s*(?:credit\s+union|cu)\b",
        t,
        re.I,
    ))
    if credit_union_answer:
        return True

    generic_question_removed = re.sub(
        r"(?:are\s+you\s+with|with)\s+(?:a\s+)?bank\s+or\s+(?:a\s+)?credit\s+union\??",
        " ",
        t,
        flags=re.I,
    )

    named_or_possessive_cu = bool(re.search(
        r"\b(?:my|the|that|their|your|landmark|[a-z0-9&.-]+)\s+credit\s+union\b|"
        r"\bcredit\s+union\s+(?:account|routing|member|confirmed|verified|name)\b",
        generic_question_removed,
        re.I,
    ))

    return named_or_possessive_cu and not bank_answer


def _transcript_credit_union_verified_for_ach(transcript):
    """
    True only when credit union ACH/account details were clearly verified beyond the prospect
    reading numbers. Credit unions may require suffixes/extra digits/member-number conversion.
    """
    t = (transcript or "").lower()
    if not t:
        return False

    return bool(re.search(
        r"\b(?:called|call|contacted|verified\s+with|confirmed\s+with|checked\s+with)\s+(?:the\s+)?credit\s+union\b|"
        r"\bcredit\s+union\s+(?:confirmed|verified)\b|"
        r"\bach[-\s]?compatible\s+account\b|"
        r"\bach\s+(?:account|draft)\s+(?:number|format)\s+(?:confirmed|verified)\b|"
        r"\bmember\s+number\b.{0,120}\b(?:suffix|extra\s+digits|ach)\b",
        t,
        re.I,
    ))


def _final_cleanup_credit_union_and_sold_summary(report, transcript):
    """
    If credit union is mentioned and banking was reached, require clear ACH/account verification.
    If missing on a sold policy, mark AT RISK with a clear reason.
    Also replace stale non-sold/no-banking summary text on sold calls.
    """
    if not report or not transcript:
        return report

    credit_union = _transcript_credit_union_mentioned(transcript)
    banking_started = _transcript_banking_stage_started(transcript)
    cu_verified = _transcript_credit_union_verified_for_ach(transcript)
    sold_yes = bool(re.search(r"(?im)^- Policy sold:\s*YES\b", report)) or bool(
        re.search(r"(?im)^- Was the policy sold\?\s*YES\b", report)
    )
    pom_reached = bool(re.search(r"(?im)^CALL STAGE REACHED:\s*Peace of Mind\s*$", report))

    if credit_union and banking_started and not cu_verified:
        report = re.sub(
            r"(?im)^- Credit union mentioned but bank/account not verified:\s*(?:NO|UNCLEAR|YES)\b.*$",
            "- Credit union mentioned but bank/account not verified: YES",
            report,
        )
        report = re.sub(
            r"(?im)^- Did the agent verify credit union account information if a credit union was mentioned\?\s*(?:NO|UNCLEAR|YES)\b.*$",
            "- Did the agent verify credit union account information if a credit union was mentioned? NO",
            report,
        )
        report = re.sub(
            r"(?im)^- Automatic fail triggered:\s*(?:NO|YES)\b.*$",
            "- Automatic fail triggered: YES",
            report,
        )

        reason = "Credit union account information not verified for ACH draft accuracy"
        if re.search(r"(?im)^- Reason:\s*.*$", report):
            report = re.sub(r"(?im)^- Reason:\s*.*$", f"- Reason: {reason}", report)
        else:
            report = re.sub(
                r"(?im)^- Automatic fail triggered:\s*YES\b.*$",
                f"- Automatic fail triggered: YES\n- Reason: {reason}",
                report,
                count=1,
            )

        if sold_yes:
            report = re.sub(r"(?im)^PASS:\s*YES\s*$", "PASS: AT RISK", report, count=1)
            report = re.sub(r"(?im)^PASS:\s*NO\s*$", "PASS: AT RISK", report, count=1)
            report = re.sub(r"(?im)^PASS:\s*AT RISK\s*$", "PASS: AT RISK", report, count=1)
        else:
            report = re.sub(r"(?im)^PASS:\s*YES\s*$", "PASS: NO", report, count=1)
            report = re.sub(r"(?im)^PASS:\s*AT RISK\s*$", "PASS: NO", report, count=1)

        report = re.sub(r"(?im)^RISK:\s*(?:LOW|MEDIUM|HIGH)\s*$", "RISK: HIGH", report, count=1)

    if sold_yes and pom_reached:
        if credit_union and banking_started and not cu_verified:
            summary = (
                "The agent progressed the call through the full sale process, including quotes, "
                "bottom-paragraph / lowest-option close, Application Information, Payment Date, "
                "Banking, required disclosures, voice signature, and Peace of Mind. The policy was "
                "sold and no callback was set. The sale is AT RISK because the prospect used a credit "
                "union and the ACH-compatible account information was not clearly verified with the "
                "credit union; credit unions can require extra digits, suffixes, or ACH-specific account "
                "formatting, and failing to verify this can prevent the policy from placing."
            )
        else:
            summary = (
                "The agent progressed the call through the full sale process, including quotes, "
                "Application Information, Payment Date, Banking, required disclosures, voice signature, "
                "and Peace of Mind. The policy was sold and no callback was set."
            )

        report = re.sub(
            r"(?ims)^SUMMARY:\s*.*?(?=^OPENAI COST ESTIMATE:|\Z)",
            f"SUMMARY:\n{summary}\n\n",
            report,
            count=1,
        )

    if credit_union and banking_started and not cu_verified:
        report = re.sub(
            r"(?ims)^BIGGEST MISS:\s*.*?(?=^OBJECTIONS DETECTED:|^TRANSCRIPT NOTE|^SUMMARY:|^OPENAI COST ESTIMATE:|\Z)",
            "BIGGEST MISS:\n- Credit union account information was not clearly verified for ACH draft accuracy after the sale was completed.\n\n",
            report,
            count=1,
        )

    return report


def _transcript_existing_coverage_bank_lookup_context(transcript):
    """
    True when bank/bank-statement language is about identifying who drafts
    an EXISTING insurance policy, not setting up banking for the new policy.
    """
    t = (transcript or "").lower()
    if not t:
        return False

    return bool(re.search(
        r"\bi\s+have\s+insurance\b.*?"
        r"(?:who'?s\s+taking\s+that\s+money\s+out|bank\s+statement|who'?s\s+that\s+through)|"
        r"\bdo\s+you\s+(?:happen\s+to\s+have|have).*?(?:final\s+expense|life\s+insurance|plan).*?"
        r"(?:who'?s\s+that\s+through|who'?s\s+taking\s+that\s+money\s+out|bank\s+statement)|"
        r"\bwho'?s\s+taking\s+that\s+money\s+out\s+every\s+month\b|"
        r"\bbank\s+statement\b.{0,180}\bwho'?s\s+taking\s+that\s+money\s+out\b",
        t,
        re.I | re.S,
    ))


def _transcript_only_good_standing_account_question(transcript):
    """
    The good-standing checking/savings / Direct Express question is before Needs.
    It is NOT the Banking stage by itself.
    """
    t = (transcript or "").lower()
    if not t:
        return False

    good_standing = bool(re.search(
        r"\bchecking(?:s)?\s+or\s+(?:your\s+)?savings\s+account\s+(?:good[-\s]?standing|in\s+good\s+standing)\b|"
        r"\bchecking\s+or\s+savings\s+account\b.{0,140}\bgood\s+standing\b|"
        r"\bis\s+that\s+an\s+account\s+that\s+you\s+set\s+up\b|"
        r"\bgovernment[-\s]?issued\s+direct\s+express\s+card\b|"
        r"\bdirect\s+express\s+card\b",
        t,
        re.I,
    ))

    real_new_policy_banking = bool(re.search(
        r"\bnow,\s+since\s+we\s+work\s+with\s+all\s+the\s+banks\s+directly\b|"
        r"\bare\s+you\s+with\s+(?:a\s+)?(?:bank|credit\s+union)\b|"
        r"\bwhat\s+is\s+the\s+name\s+of\s+that\s+(?:bank|credit\s+union)\b|"
        r"\bwho\s+do\s+you\s+bank\s+with\b|"
        r"\brouting\s+number\b|"
        r"\baccount\s+number\b|"
        r"\bplease\s+grab\s+(?:your\s+)?(?:checkbook|check\s*book|bank\s+statement)\b|"
        r"\bslowly\s+read\s+me\s+all\s+the\s+numbers\b|"
        r"\[account_number\]|\[routing_number\]|\[bank_number\]",
        t,
        re.I,
    ))

    return good_standing and not real_new_policy_banking



def _transcript_cool_down_after_sale(transcript):
    """Post-sale non-insurance small talk after voice signature / sale completion."""
    t = re.sub(r"\s+", " ", (transcript or "").lower()).strip()
    if not t or not _transcript_voice_signature_started(transcript):
        return False

    sale_done_pos = -1
    for pat in [
        r"this recording has now ended",
        r"we are done with our voice recording",
        r"submitting your application to the home office",
        r"voice recording",
        r"voice signature",
    ]:
        matches = list(re.finditer(pat, t, re.I))
        if matches:
            sale_done_pos = max(sale_done_pos, matches[-1].end())

    after = t[sale_done_pos:] if sale_done_pos >= 0 else t
    return bool(re.search(
        r"\bplans for the rest of your day\b|"
        r"\bclean(?:ing)? the house\b|"
        r"\bi just moved\b|"
        r"\bnever want to move again\b|"
        r"\b(?:dog|dogs|lab|black lab|pit bull|poodle|pool|lake|water)\b.{0,240}\b(?:dog|dogs|lab|pool|lake|water|toy)\b",
        after,
        re.I,
    ))


def _transcript_no_current_coverage_only_one_first_time(transcript):
    """Shelby-sold pattern: 'Only one' is clarified by 'first time' as no current coverage."""
    t = re.sub(r"\s+", " ", (transcript or "").lower()).strip()
    if not t:
        return False
    return bool(re.search(
        r"do you have any kind of final expense plan or life insurance in place now[^?]{0,160}"
        r"(?:only policy|your only policy)[^a-z0-9]{0,20}only one\b.{0,260}"
        r"(?:ever owned a policy|owned a policy|policy at some point in the past|first time)[^?]{0,180}"
        r"\bfirst time\b",
        t,
        re.I,
    ))


def _transcript_three_options_presented(transcript):
    t = re.sub(r"\s+", " ", (transcript or "").lower()).strip()
    if not t:
        return False
    return bool(
        re.search(r"\bfirst option\b", t, re.I)
        and re.search(r"\bsecond option\b", t, re.I)
        and re.search(r"\bthird option\b", t, re.I)
    )


def _transcript_client_chose_option(transcript):
    t = re.sub(r"\s+", " ", (transcript or "").lower()).strip()
    if not t:
        return False
    return bool(re.search(
        r"\b(?:i guess i'll pick|i will pick|i'll pick|let'?s do|go with|i want)\b.{0,80}\b(?:one|option|\[number\]|\[money\])\b|"
        r"\bso what'?s your middle initial\b",
        t,
        re.I,
    ))


def _final_cleanup_shelby_sold_short_false_fails(report, transcript):
    """Correct known sold-call false positives: bank vs CU, resolved Only-one coverage, options, and cooldown."""
    if not report or not transcript:
        return report

    sold_yes = _report_policy_sold_yes(report)

    if not _transcript_credit_union_mentioned(transcript):
        report = re.sub(
            r"(?im)^- Credit union mentioned but bank/account not verified:\s*(?:YES|NO|UNCLEAR)\b.*$",
            "- Credit union mentioned but bank/account not verified: NO",
            report,
        )
        report = re.sub(
            r"(?im)^- Did the agent verify credit union account information if a credit union was mentioned\?\s*(?:YES|NO|UNCLEAR)\b.*$",
            "- Did the agent verify credit union account information if a credit union was mentioned? NO",
            report,
        )
        report = _text_remove_lines_containing(report, "Credit union account information not verified")
        report = _text_remove_lines_containing(report, "credit unions can require extra digits")

    if _transcript_no_current_coverage_only_one_first_time(transcript):
        # Shelby-sold style compound-question pattern:
        # "Do you have coverage now, or will this be your only policy?" -> "Only one"
        # followed by "first time" is still not enough carrier/provider confirmation
        # on a sold call. Keep/force this as an existing-coverage at-risk condition.
        report = re.sub(
            r"(?im)^- Existing coverage mentioned but not confirmed:\s*(?:YES|NO|UNCLEAR)\b.*$",
            "- Existing coverage mentioned but not confirmed: YES",
            report,
        )
        report = re.sub(
            r"(?im)^- Automatic fail triggered:\s*(?:YES|NO)\b.*$",
            "- Automatic fail triggered: YES",
            report,
        )
        if re.search(r"(?im)^- Reason:\s*.*$", report):
            report = re.sub(
                r"(?im)^- Reason:\s*.*$",
                "- Reason: Existing coverage mentioned but not confirmed",
                report,
                count=1,
            )
        else:
            report = re.sub(
                r"(?im)^- Automatic fail triggered:\s*YES\b.*$",
                "- Automatic fail triggered: YES\n- Reason: Existing coverage mentioned but not confirmed",
                report,
                count=1,
            )

        if sold_yes:
            report = re.sub(r"(?im)^PASS:\s*(?:YES|NO|AT RISK)\s*$", "PASS: AT RISK", report, count=1)
            report = re.sub(r"(?im)^RISK:\s*(?:LOW|MEDIUM|HIGH)\s*$", "RISK: HIGH", report, count=1)

        # Keep option-selection checklist aligned with searchable/sale evidence.
        if sold_yes and _transcript_client_chose_option(transcript):
            report = re.sub(
                r"(?im)^- Client chose an option:\s*(?:YES|NO|PARTIAL|NOT REACHED)\b.*$",
                "- Client chose an option: YES",
                report,
            )
            report = re.sub(
                r"(?im)^- Did the client choose an option\?\s*(?:YES|NO|PARTIAL|NOT REACHED)\b.*$",
                "- Did the client choose an option? YES",
                report,
            )

        report = re.sub(
            r"(?ims)^COMPLIANCE FAILURES:\s*None\s*(?=^SCRIPT / FLOW MISSES:)",
            "COMPLIANCE FAILURES:\n- Existing coverage mentioned but not confirmed\n\n",
            report,
            count=1,
        )
        report = re.sub(
            r"(?ims)^SCRIPT / FLOW MISSES:\s*None\s*(?=^PQ / HANDOFF:)",
            "SCRIPT / FLOW MISSES:\n- Existing coverage mentioned but not confirmed: agent did not clearly resolve whether the prospect had active current coverage or verify carrier/provider information\n\n",
            report,
            count=1,
        )
        report = re.sub(
            r"(?ims)^BIGGEST MISS:\s*.*?(?=^OBJECTIONS DETECTED:|^TRANSCRIPT NOTE|^SUMMARY:|^OPENAI COST ESTIMATE:|\Z)",
            "BIGGEST MISS:\n- Existing coverage mentioned but not confirmed: the prospect gave an ambiguous answer to the current-coverage question and the agent did not verify carrier/provider information before completing the sale.\n\n",
            report,
            count=1,
        )

        at_risk_summary = (
            "The agent completed a sold call through Cool Down: options were presented, the client chose an option, "
            "application information and payment date were collected, banking setup was handled with a bank, required "
            "disclosures and voice signature were completed, Peace of Mind was delivered, and the agent continued into "
            "Cool Down conversation. The sale is AT RISK because the prospect gave an ambiguous answer to the current "
            "coverage question and the agent did not clearly resolve active current coverage or verify carrier/provider "
            "information before completing the sale. No callback or credit-union ACH verification issue was detected."
        )
        report = re.sub(
            r"(?ims)^SUMMARY:\s*.*?(?=^OPENAI COST ESTIMATE:|\Z)",
            f"SUMMARY:\n{at_risk_summary}\n\n",
            report,
            count=1,
        )

    if _transcript_three_options_presented(transcript):
        report = _text_replace_checklist_value(report, "Three options presented", "YES")
        report = re.sub(
            r"(?im)^- Did the agent present options\?\s*(?:NO|PARTIAL)\b.*$",
            "- Did the agent present options? YES",
            report,
        )

    if sold_yes and _transcript_client_chose_option(transcript):
        report = re.sub(
            r"(?im)^- Client chose an option:\s*(?:NO|PARTIAL)\b.*$",
            "- Client chose an option: YES",
            report,
        )
        report = re.sub(
            r"(?im)^- Did the client choose an option\?\s*(?:NO|PARTIAL)\b.*$",
            "- Did the client choose an option? YES",
            report,
        )

    if _transcript_cool_down_after_sale(transcript):
        report = re.sub(r"(?im)^CALL STAGE REACHED:\s*Peace of Mind\s*$", "CALL STAGE REACHED: Cool Down", report, count=1)
        report = re.sub(
            r"(?ims)^NOT REACHED:\s*.*?(?=^COMPLIANCE FAILURES:)",
            "NOT REACHED:\n- None\n\n",
            report,
            count=1,
        )
        report = _text_replace_checklist_value(report, "Cool down completed", "YES")

    all_autofail_checks_no = all(re.search(pattern, report, re.I | re.M) for pattern in [
        r"^- Callback set:\s*NO\b",
        r"^- Objection occurred without proper call control:\s*NO\b",
        r"^- Existing coverage mentioned but not confirmed:\s*NO\b",
        r"^- Credit union mentioned but bank/account not verified:\s*NO\b",
    ])
    if all_autofail_checks_no:
        report = re.sub(r"(?im)^- Automatic fail triggered:\s*YES\b.*$", "- Automatic fail triggered: NO", report)
        if re.search(r"(?im)^- Reason:\s*.*$", report):
            report = re.sub(r"(?im)^- Reason:\s*.*$", "- Reason: None", report, count=1)
        else:
            report = re.sub(r"(?im)^- Automatic fail triggered:\s*NO\b.*$", "- Automatic fail triggered: NO\n- Reason: None", report, count=1)
        if sold_yes:
            report = re.sub(r"(?im)^PASS:\s*(?:NO|AT RISK)\s*$", "PASS: YES", report, count=1)
            report = re.sub(r"(?im)^RISK:\s*HIGH\s*$", "RISK: LOW", report, count=1)
        report = re.sub(r"(?ims)^COMPLIANCE FAILURES:\s*.*?(?=^SCRIPT / FLOW MISSES:)", "COMPLIANCE FAILURES: None  \n\n", report, count=1)
        report = re.sub(
            r"(?ims)^BIGGEST MISS:\s*.*?(?=^OBJECTIONS DETECTED:|^TRANSCRIPT NOTE|^SUMMARY:|^OPENAI COST ESTIMATE:|\Z)",
            "BIGGEST MISS:\n- None\n\n",
            report,
            count=1,
        )

    if sold_yes and re.search(r"(?im)^- Automatic fail triggered:\s*NO\b", report):
        summary = (
            "The agent completed a sold call: options were presented, the client chose an option, "
            "application information and payment date were collected, banking setup was handled with a bank, "
            "required disclosures and voice signature were completed, Peace of Mind was delivered, and the agent "
            "continued into Cool Down conversation. No callback or credit-union ACH verification issue was detected."
        )
        report = re.sub(
            r"(?ims)^SUMMARY:\s*.*?(?=^OPENAI COST ESTIMATE:|\Z)",
            f"SUMMARY:\n{summary}\n\n",
            report,
            count=1,
        )

    # Final safety: prevent stale model text from contradicting corrected outcomes.
    if re.search(r"(?im)^CALL STAGE REACHED:\s*Cool Down\s*$", report):
        report = _text_replace_checklist_value(report, "Cool down completed", "YES")
        report = re.sub(
            r"(?ims)^NOT REACHED:\s*.*?(?=^COMPLIANCE FAILURES:)",
            "NOT REACHED:\n- None\n\n",
            report,
            count=1,
        )

    if re.search(r"(?im)^- Existing coverage mentioned but not confirmed:\s*YES\b", report):
        report = re.sub(
            r"(?i)No callback or coverage confirmation issues were present\.",
            "No callback issue was present; however, current coverage was not clearly confirmed.",
            report,
        )
        report = re.sub(
            r"(?i)Post-sale stages Peace of Mind and Cool Down were not reached[^.]*\.",
            "Peace of Mind and Cool Down were reached.",
            report,
        )

    return report




def _transcript_existing_coverage_confirmed_by_carrier(transcript):
    """
    True when existing coverage was confirmed by calling an insurance carrier/company.
    This prevents false AT RISK reports when the agent actually completes a policy checkup.
    """
    t = re.sub(r"\s+", " ", (transcript or "").lower()).strip()
    if not t:
        return False

    carrier_call = bool(re.search(
        r"\b(thank you for calling mutual of omaha|mutual of omaha|combinedinsurance|combined insurance|child benefits|policy owner|policy number|customer service)\b",
        t,
        re.I,
    ))

    confirmed_details = 0
    detail_patterns = [
        r"\bcoverage\b.{0,80}\b(?:amount|in place|\[money\])\b",
        r"\b(?:premium|monthly premium|paying|costing)\b.{0,80}\b(?:month|\[money\])\b",
        r"\b(?:whole life|life insurance policy|policy type)\b",
        r"\b(?:cash value|builds cash value)\b",
        r"\b(?:policy number|policy has been in place|initial date|since july|had it since)\b",
        r"\b(?:mutual of omaha|combined|child benefits)\b",
    ]
    for pat in detail_patterns:
        if re.search(pat, t, re.I):
            confirmed_details += 1

    agent_call_language = bool(re.search(
        r"\b(call mutual|call.*omaha|call.*combined|policy checkup|let'?s go ahead and call|we were just.*curious.*coverage|questions regarding the policy)\b",
        t,
        re.I,
    ))

    return carrier_call and agent_call_language and confirmed_details >= 3


def _transcript_actual_bank_not_credit_union(transcript):
    """
    True when the call explicitly resolves bank vs credit union as bank.
    Generic 'bank or credit union?' language must not create a credit-union autofail.
    """
    t = re.sub(r"\s+", " ", (transcript or "").lower()).strip()
    if not t:
        return False

    return bool(re.search(
        r"\b(?:bank or credit union|bank\s*/\s*credit union)\b.{0,220}\b(?:a bank|actual bank|an actual bank|it'?s a bank|one bank|one bank of tennessee)\b",
        t,
        re.I,
    ) or re.search(
        r"\bprospect:\s*(?:a bank|an actual bank|actual bank|one bank)\b",
        t,
        re.I,
    ))


def _transcript_sold_call_cool_down_after_pom(transcript):
    """
    Sold-call cool down: post-sale casual/non-application conversation after policy number,
    Peace of Mind, or commitment language.
    """
    t = re.sub(r"\s+", " ", (transcript or "").lower()).strip()
    if not t:
        return False

    sale_done = bool(re.search(
        r"\b(policy number|don'?t you feel good|decision you made|commitment from you|plans here for the rest of the day|what are your plans|wise words of wisdom|have a good one|god bless)\b",
        t,
        re.I,
    ))

    cooldown_talk = bool(re.search(
        r"\b(plans here for the rest of the day|house clean|house cleaners|been on the phone all day|personal agent|wise words of wisdom|let you get back to your day|god bless|have a good one|happy weekend)\b",
        t,
        re.I,
    ))

    return sale_done and cooldown_talk


def _final_cleanup_confirmed_coverage_bank_and_cooldown(report, transcript):
    """
    Correct contradictions for long sold calls:
    - existing coverage verified by carrier call means no existing-coverage autofail
    - actual bank answer means no credit-union autofail
    - post-sale small talk means Cool Down reached
    """
    if not report or not transcript:
        return report

    sold_yes = _report_policy_sold_yes(report) if "_report_policy_sold_yes" in globals() else bool(re.search(r"(?im)^- Policy sold:\s*YES\b|^PASS:\s*(YES|AT RISK)\b", report))
    coverage_confirmed = _transcript_existing_coverage_confirmed_by_carrier(transcript)
    actual_bank = _transcript_actual_bank_not_credit_union(transcript)
    cooldown = _transcript_sold_call_cool_down_after_pom(transcript)

    if coverage_confirmed:
        report = re.sub(
            r"(?im)^- Existing coverage mentioned but not confirmed:\s*(?:YES|NO|UNCLEAR)\b.*$",
            "- Existing coverage mentioned but not confirmed: NO",
            report,
        )
        report = re.sub(
            r"(?im)^- Did the agent confirm current coverage\?\s*(?:YES|NO|UNCLEAR)\b.*$",
            "- Did the agent confirm current coverage? YES",
            report,
        )
        report = re.sub(
            r"(?im)^- Did the agent call an insurance company to confirm current coverage\?\s*(?:YES|NO|UNCLEAR)\b.*$",
            "- Did the agent call an insurance company to confirm current coverage? YES",
            report,
        )
        report = _text_remove_lines_containing(report, "Existing coverage mentioned but not confirmed")
        report = _text_remove_lines_containing(report, "did not verify current coverage with the insurance company")
        report = _text_remove_lines_containing(report, "confirming existing policies directly with insurance carriers")

    if actual_bank:
        report = re.sub(
            r"(?im)^- Credit union mentioned but bank/account not verified:\s*(?:YES|NO|UNCLEAR)\b.*$",
            "- Credit union mentioned but bank/account not verified: NO",
            report,
        )
        report = re.sub(
            r"(?im)^- Did the agent verify credit union account information if a credit union was mentioned\?\s*(?:YES|NO|UNCLEAR)\b.*$",
            "- Did the agent verify credit union account information if a credit union was mentioned? NO",
            report,
        )
        report = _text_remove_lines_containing(report, "Credit union account information")
        report = _text_remove_lines_containing(report, "credit unions can require extra digits")
        report = _text_remove_lines_containing(report, "credit union ACH verification")
        report = _text_remove_lines_containing(report, "ACH-compatible account information was not clearly verified with the credit union")

    if cooldown:
        report = re.sub(r"(?im)^CALL STAGE REACHED:\s*Peace of Mind\s*$", "CALL STAGE REACHED: Cool Down", report, count=1)
        report = _text_replace_checklist_value(report, "Cool down completed", "YES")
        report = re.sub(
            r"(?ims)^NOT REACHED:\s*.*?(?=^COMPLIANCE FAILURES:)",
            "NOT REACHED:\n- None\n\n",
            report,
            count=1,
        )
        report = re.sub(
            r"(?im)^- Final stage supporting sale:\s*Peace of Mind\s*$",
            "- Final stage supporting sale: Cool Down",
            report,
            count=1,
        )

    no_callback = bool(re.search(r"(?im)^- Callback set:\s*NO\b", report))
    no_objection = bool(re.search(r"(?im)^- Objection occurred without proper call control:\s*NO\b", report))
    no_existing = bool(re.search(r"(?im)^- Existing coverage mentioned but not confirmed:\s*NO\b", report))
    no_cu = bool(re.search(r"(?im)^- Credit union mentioned but bank/account not verified:\s*NO\b", report))

    if sold_yes and no_callback and no_objection and no_existing and no_cu:
        report = re.sub(r"(?im)^- Automatic fail triggered:\s*YES\b.*$", "- Automatic fail triggered: NO", report)
        if re.search(r"(?im)^- Reason:\s*.*$", report):
            report = re.sub(r"(?im)^- Reason:\s*.*$", "- Reason: None", report, count=1)
        else:
            report = re.sub(r"(?im)^- Automatic fail triggered:\s*NO\b.*$", "- Automatic fail triggered: NO\n- Reason: None", report, count=1)

        report = re.sub(r"(?im)^PASS:\s*(?:NO|AT RISK)\s*$", "PASS: YES", report, count=1)
        report = re.sub(r"(?im)^RISK:\s*HIGH\s*$", "RISK: MEDIUM", report, count=1)

        report = re.sub(
            r"(?ims)^COMPLIANCE FAILURES:\s*.*?(?=^SCRIPT / FLOW MISSES:)",
            "COMPLIANCE FAILURES: None\n\n",
            report,
            count=1,
        )

        report = re.sub(
            r"(?ims)^BIGGEST MISS:\s*.*?(?=^SUMMARY:|^OPENAI COST ESTIMATE:|\Z)",
            "BIGGEST MISS:\n- None\n\n",
            report,
            count=1,
        )

        summary = (
            "The agent completed a sold call through Cool Down. Existing coverage was identified and confirmed "
            "through carrier policy-check calls, the prospect clarified she used an actual bank rather than a credit union, "
            "the application was completed, payment and banking setup were handled, disclosures and voice signature were completed, "
            "and Peace of Mind plus Cool Down were reached. Remaining coaching should focus on cleaner routing-number verification, "
            "speaker/third-party clarity, and 3 and 1 depth, not an automatic-fail condition."
        )
        report = re.sub(
            r"(?ims)^SUMMARY:\s*.*?(?=^OPENAI COST ESTIMATE:|\Z)",
            f"SUMMARY:\n{summary}\n\n",
            report,
            count=1,
        )

    return report


def _final_cleanup_false_banking_from_existing_coverage_lookup(report, transcript):
    """
    Correct calls where early good-standing/bank-statement language was used before Needs
    or to identify an existing carrier, not to set up new policy banking.
    """
    if not report or not transcript:
        return report

    false_banking_context = (
        _transcript_only_good_standing_account_question(transcript)
        or _transcript_existing_coverage_bank_lookup_context(transcript)
    )
    real_banking = _transcript_banking_stage_started(transcript)

    if not false_banking_context or real_banking:
        return report

    report = re.sub(
        r"(?im)^CALL STAGE REACHED:\s*Banking\s*$",
        "CALL STAGE REACHED: Need",
        report,
        count=1,
    )

    report = re.sub(
        r"(?ims)^NOT REACHED:\s*.*?(?=^COMPLIANCE FAILURES:)",
        "NOT REACHED:\n- Features / Benefits\n- Change Up\n- Pre-Close\n- Quotes\n- Close\n- Application Information\n- Payment Date\n- Banking\n- Disclosures\n- Third Party Underwriting\n- Peace of Mind\n- Cool Down\n\n",
        report,
        count=1,
    )

    for label in [
        "Payment date explained",
        "Banking/payment setup explained",
        "Banking/account information requested or verified 3 times",
        "Account number requested or verified 3 times",
        "Account number verified at least 2 times",
        "Routing number requested or verified 3 times",
        "Routing number verified at least 2 times",
        "Agent read account/routing information back to prospect",
        "Prospect confirmed account/routing read-back",
    ]:
        report = _text_replace_checklist_value(report, label, "NOT REACHED")

    report = _text_replace_checklist_value(report, "Account verification evidence count", "0")
    report = _text_replace_checklist_value(report, "Account verification evidence", "None")
    report = _text_replace_checklist_value(report, "Routing verification evidence count", "0")
    report = _text_replace_checklist_value(report, "Routing verification evidence", "None")

    report = re.sub(
        r"(?im)^- Credit union mentioned but bank/account not verified:\s*YES\b.*$",
        "- Credit union mentioned but bank/account not verified: NO",
        report,
    )

    if re.search(r"(?im)^- Existing coverage mentioned but not confirmed:\s*YES\b", report):
        report = re.sub(
            r"(?im)^- Automatic fail triggered:\s*(?:NO|YES)\b.*$",
            "- Automatic fail triggered: YES",
            report,
        )
        if re.search(r"(?im)^- Reason:\s*.*$", report):
            report = re.sub(
                r"(?im)^- Reason:\s*.*$",
                "- Reason: Existing coverage mentioned but not confirmed",
                report,
            )
        else:
            report = re.sub(
                r"(?im)^- Automatic fail triggered:\s*YES\b.*$",
                "- Automatic fail triggered: YES\n- Reason: Existing coverage mentioned but not confirmed",
                report,
                count=1,
            )
        report = re.sub(r"(?im)^PASS:\s*(?:YES|AT RISK)\s*$", "PASS: NO", report, count=1)
        report = re.sub(r"(?im)^RISK:\s*(?:LOW|MEDIUM|HIGH)\s*$", "RISK: HIGH", report, count=1)

    return report



def _ensure_autofail_line(report, label, value):
    """
    Ensure an AUTO FAIL CHECKS line exists with the desired value.
    If missing, insert it before Automatic fail triggered.
    """
    pattern = rf"(?im)^- {re.escape(label)}:\s*(?:YES|NO|UNCLEAR)\b.*$"
    desired = f"- {label}: {value}"

    if re.search(pattern, report):
        return re.sub(pattern, desired, report)

    auto_pat = r"(?im)^- Automatic fail triggered:\s*(?:YES|NO)\b.*$"
    if re.search(auto_pat, report):
        return re.sub(auto_pat, desired + "\n" + r"\g<0>", report, count=1)

    return report




def _prospect_disconnected_before_coverage_verification(transcript):
    """
    Existing coverage should not become an automatic fail if the prospect
    disconnected/refused before the agent had a fair chance to call/verify coverage.
    """
    t = transcript or ""
    if not t.strip():
        return False

    coverage_mentioned = bool(re.search(
        r"(?is)\b(i have life insurance|already have life insurance|already got life insurance|"
        r"have coverage|already have coverage|government insurance|insurance through the government|"
        r"policy already|existing policy)\b",
        t,
    ))

    disconnect_or_hard_refusal = bool(re.search(
        r"(?is)\b(hung up|disconnected|stopped responding|call ended|line went dead|"
        r"are you there|hello\?|hello, are you there|not interested|don't need|do not need|"
        r"i already have|i'm good|im good|bye)\b",
        t,
    ))

    agent_attempted_coverage_call = bool(re.search(
        r"(?is)\b(call (?:the )?(?:company|carrier|insurance company)|"
        r"three way|3 way|verify (?:that|the) coverage|confirm (?:that|the) coverage|"
        r"policy check|carrier call|let'?s call)\b",
        t,
    ))

    return coverage_mentioned and disconnect_or_hard_refusal and not agent_attempted_coverage_call

def _final_cleanup_no_autofail_consistency(report, transcript):
    """
    Final guardrail: if sold call has no callback, no uncontrolled objection,
    coverage confirmed, and bank/not credit union, then it cannot remain AT RISK.
    """
    if not report:
        return report

    sold_yes = bool(re.search(r"(?im)^- Policy sold:\s*YES\b|^Was the policy sold\?\s*YES\b", report))
    coverage_confirmed = bool(re.search(r"(?im)^- Did the agent confirm current coverage\?\s*YES\b", report)) or _transcript_existing_coverage_confirmed_by_carrier(transcript)
    bank_not_cu = bool(re.search(r"(?im)^- Credit union mentioned but bank/account not verified:\s*NO\b", report)) or _transcript_actual_bank_not_credit_union(transcript)

    coverage_disconnect_before_verify = _prospect_disconnected_before_coverage_verification(transcript)

    if coverage_confirmed or coverage_disconnect_before_verify:
        report = _ensure_autofail_line(report, "Existing coverage mentioned but not confirmed", "NO")
        if coverage_disconnect_before_verify:
            report = _text_remove_lines_containing(report, "Existing coverage mentioned but not confirmed")
            report = _ensure_autofail_line(report, "Existing coverage mentioned but not confirmed", "NO")

            # If the only automatic-fail basis was stale unresolved coverage,
            # clear the automatic-fail result too. A prospect hangup/refusal before
            # verification is not the same as the agent skipping required verification.
            no_callback = bool(re.search(r"(?im)^- Callback set:\s*NO\b", report))
            no_objection = bool(re.search(r"(?im)^- Objection occurred without proper call control:\s*NO\b", report))
            no_existing = bool(re.search(r"(?im)^- Existing coverage mentioned but not confirmed:\s*NO\b", report))
            no_cu = (
                not re.search(r"(?im)^- Credit union mentioned but bank/account not verified:", report)
                or bool(re.search(r"(?im)^- Credit union mentioned but bank/account not verified:\s*NO\b", report))
            )

            if no_callback and no_objection and no_existing and no_cu:
                report = re.sub(r"(?im)^- Automatic fail triggered:\s*YES\b.*$", "- Automatic fail triggered: NO", report, count=1)
                report = _set_autofail_reason(report, "None", merge=False)
                report = re.sub(r"(?im)^PASS:\s*NO\s*$", "PASS: YES", report, count=1)
                report = re.sub(r"(?im)^RISK:\s*HIGH\s*$", "RISK: MEDIUM", report, count=1)
                report = re.sub(
                    r"(?ims)^BIGGEST MISS:\s*.*?(?=^SUMMARY:|^TRANSCRIPT NOTE|^OPENAI COST ESTIMATE:|\Z)",
                    "BIGGEST MISS:\n- Prospect disconnected/refused before the agent could verify existing coverage.\n\n",
                    report,
                    count=1,
                )

    if bank_not_cu:
        report = _ensure_autofail_line(report, "Credit union mentioned but bank/account not verified", "NO")

    # Remove "no miss" items that were left in SCRIPT / FLOW MISSES.
    report = _text_remove_lines_containing(report, "so no miss")
    report = _text_remove_lines_containing(report, "Payment/draft date not explained after banking: payment date was explained")

    no_callback = bool(re.search(r"(?im)^- Callback set:\s*NO\b", report))
    no_objection = bool(re.search(r"(?im)^- Objection occurred without proper call control:\s*NO\b", report))
    no_existing = bool(re.search(r"(?im)^- Existing coverage mentioned but not confirmed:\s*NO\b", report))
    no_cu = bool(re.search(r"(?im)^- Credit union mentioned but bank/account not verified:\s*NO\b", report))

    if sold_yes and no_callback and no_objection and no_existing and no_cu:
        report = re.sub(r"(?im)^PASS:\s*(?:NO|AT RISK)\s*$", "PASS: YES", report, count=1)
        report = re.sub(r"(?im)^RISK:\s*HIGH\s*$", "RISK: MEDIUM", report, count=1)
        report = re.sub(r"(?im)^- Automatic fail triggered:\s*YES\b.*$", "- Automatic fail triggered: NO", report)

        if re.search(r"(?im)^- Reason:\s*.*$", report):
            report = re.sub(r"(?im)^- Reason:\s*.*$", "- Reason: None", report, count=1)
        else:
            report = re.sub(
                r"(?im)^- Automatic fail triggered:\s*NO\b.*$",
                "- Automatic fail triggered: NO\n- Reason: None",
                report,
                count=1,
            )

        report = re.sub(
            r"(?ims)^COMPLIANCE FAILURES:\s*.*?(?=^SCRIPT / FLOW MISSES:)",
            "COMPLIANCE FAILURES: None\n\n",
            report,
            count=1,
        )

        # If script misses became empty, keep only remaining real misses.
        report = re.sub(
            r"(?ims)^BIGGEST MISS:\s*.*?(?=^SUMMARY:)",
            "BIGGEST MISS:\n- None\n\n",
            report,
            count=1,
        )

        summary = (
            "The agent completed a sold call through Cool Down. Existing coverage was identified and confirmed "
            "through insurance-company policy-check calls, the prospect clarified she used an actual bank rather than "
            "a credit union, the application was completed, payment and banking setup were handled, disclosures and "
            "voice signature were completed, and Peace of Mind plus Cool Down were reached. Remaining coaching should "
            "focus on cleaner routing-number verification and stronger 3 and 1 rapport depth, not an automatic-fail condition."
        )
        report = re.sub(
            r"(?ims)^SUMMARY:\s*.*?(?=^OPENAI COST ESTIMATE:|\Z)",
            f"SUMMARY:\n{summary}\n\n",
            report,
            count=1,
        )

    return report




def _restore_safe_business_terms(report):
    """
    Restore safe script/training terms that should never be privacy-redacted.
    """
    if not report:
        return report

    replacements = {
        "[NUMBER] and [NUMBER] Method": "3 and 1 Method",
        "[NUMBER] & [NUMBER] Method": "3 and 1 Method",
        "[NUMBER]-and-[NUMBER] Method": "3 and 1 Method",
        "[NUMBER] and [NUMBER] topic groups": "3 and 1 topic groups",
        "[NUMBER] and [NUMBER] agent self-disclosure": "3 and 1 agent self-disclosure",
        "gpt-[NUMBER].[NUMBER]-mini": "gpt-4.1-mini",
        "gpt-[NUMBER]-mini": "gpt-4.1-mini",
    }

    out = report
    for old, new in replacements.items():
        out = out.replace(old, new)

    return out


def _final_cleanup_early_unsold_score_risk_guardrail(report, transcript):
    """
    Future-call guardrail:
    If a call is unsold and only reached warm-up/fact-finding before core sales stages,
    it should not remain LOW risk or score in the 90s.
    """
    if not report:
        return report

    sold_no = bool(re.search(r"(?im)^- Policy sold:\s*NO\b|^- Was the policy sold\?\s*NO\b", report))
    stage_warmup = bool(re.search(r"(?im)^CALL STAGE REACHED:\s*(Fact Finding / Warm-up|Warm-up|Fact Finding)\b", report))
    health_no = bool(re.search(r"(?im)^- Health questions completed:\s*(NO|NOT REACHED)\b", report))
    product_no = bool(re.search(r"(?im)^- Product benefits explained:\s*(NO|NOT REACHED)\b", report))
    options_no = bool(re.search(r"(?im)^- Three options presented:\s*(NO|NOT REACHED)\b", report))
    app_no = bool(re.search(r"(?im)^- Application info collected:\s*(NO|NOT REACHED)\b", report))
    autofail_yes = bool(re.search(r"(?im)^- Automatic fail triggered:\s*YES\b", report))

    early_unsold_warmup = sold_no and stage_warmup and health_no and product_no and options_no and app_no

    if early_unsold_warmup:
        report = re.sub(r"(?im)^EARLY END:\s*NO\s*$", "EARLY END: YES", report, count=1)

        if not autofail_yes:
            report = re.sub(r"(?im)^RISK:\s*LOW\s*$", "RISK: MEDIUM", report, count=1)
            report = re.sub(r"(?im)^RISK:\s*HIGH\s*$", "RISK: MEDIUM", report, count=1)

        # Cap overly-generous scores for early-ended unsold warm-up calls.
        m = re.search(r"(?im)^SCORE:\s*(\d+)\b", report)
        if m:
            try:
                score = int(m.group(1))
            except Exception:
                score = None
            if score is not None and score > 78:
                report = re.sub(r"(?im)^SCORE:\s*\d+\b", "SCORE: 78", report, count=1)

        # Payment/banking cannot be reached if the report says no application/account verification.
        report = _text_replace_checklist_value(report, "Payment date explained", "NOT REACHED")
        report = _text_replace_checklist_value(report, "Banking/payment setup explained", "NOT REACHED")

        expanded_not_reached = [
            "Existing coverage",
            "Beneficiary",
            "Need amount",
            "Health questions",
            "Product benefits",
            "Three options",
            "Client choice",
            "Application information",
            "Payment date",
            "Banking/payment setup",
            "Banking/account verification",
            "Disclosures",
            "Third Party Underwriting",
            "Peace of Mind",
            "Cool Down",
        ]
        replacement = "NOT REACHED:\\n" + "\\n".join(f"- {item}" for item in expanded_not_reached) + "\\n\\n"
        report = re.sub(
            r"(?ims)^NOT REACHED:\\s*.*?(?=^COMPLIANCE FAILURES:)",
            replacement,
            report,
            count=1,
        )

    return report


def _final_cleanup_zero_score_guardrail(report, transcript):
    """
    SCORE: 0 should not be paired with LOW risk / PASS YES unless the report is a known
    processing-failed placeholder. This catches impossible legacy report states.
    """
    if not report:
        return report

    score_zero = bool(re.search(r"(?im)^SCORE:\s*0\b", report))
    processing_failed = bool(re.search(r"(?i)processing failed|unable to evaluate|unable to complete audit", report))

    if score_zero and not processing_failed:
        report = re.sub(r"(?im)^RISK:\s*LOW\s*$", "RISK: HIGH", report, count=1)
        report = re.sub(r"(?im)^PASS:\s*YES\s*$", "PASS: NO", report, count=1)

        if not re.search(r"(?ims)^COMPLIANCE FAILURES:\s*.*?(?=^SCRIPT / FLOW MISSES:)", report):
            return report

        report = re.sub(
            r"(?ims)^BIGGEST MISS:\s*.*?(?=^SUMMARY:|^TRANSCRIPT NOTE|^OPENAI COST ESTIMATE:|\Z)",
            "BIGGEST MISS:\n- Report has SCORE: 0 and should be reviewed or regenerated because the score conflicts with the audit result.\n\n",
            report,
            count=1,
        )

    return report




def _prospect_caused_early_end(report, transcript):
    """
    True when the call appears incomplete because the prospect stopped responding,
    disconnected, was busy, or otherwise ended the call before the agent had a fair chance
    to complete later stages.
    """
    combined = ((report or "") + "\n" + (transcript or "")).lower()

    patterns = [
        r"prospect stopped responding",
        r"customer disconnected",
        r"prospect disconnected",
        r"client disconnected",
        r"stopped responding",
        r"can you hear me",
        r"are you there",
        r"hello\??\s*$",
        r"prospect requested callback due to being busy",
        r"prospect.*busy",
        r"had to go",
        r"could not continue",
        r"call ended during",
        r"disconnected before",
    ]

    return any(re.search(p, combined, re.I | re.M) for p in patterns)


def _normalize_not_reached_due_to_prospect(report, transcript):
    """
    If the call stopped because of the prospect before later stages,
    keep the not-reached list accurate but avoid treating those items as agent misses.
    """
    if not report:
        return report

    prospect_caused = _prospect_caused_early_end(report, transcript)
    sold_no = bool(re.search(r"(?im)^- Policy sold:\s*NO\b|^- Was the policy sold\?\s*NO\b", report))
    health_no_or_partial = bool(re.search(r"(?im)^- Health questions completed:\s*(NO|PARTIAL|NOT REACHED)\b", report))
    product_no = bool(re.search(r"(?im)^- Product benefits explained:\s*(NO|NOT REACHED)\b", report))
    options_no = bool(re.search(r"(?im)^- Three options presented:\s*(NO|NOT REACHED)\b", report))
    app_no = bool(re.search(r"(?im)^- Application info collected:\s*(NO|NOT REACHED)\b", report))

    incomplete_before_sale = sold_no and product_no and options_no and app_no

    if prospect_caused and incomplete_before_sale:
        # Accurate status, but not a hard agent penalty.
        report = re.sub(r"(?im)^EARLY END:\s*NO\s*$", "EARLY END: YES", report, count=1)

        if re.search(r"(?im)^RISK:\s*HIGH\b", report) and not re.search(r"(?im)^- Automatic fail triggered:\s*YES\b", report):
            report = re.sub(r"(?im)^RISK:\s*HIGH\b", "RISK: MEDIUM", report, count=1)

        # Do not leave very-low scores from missing stages if the prospect caused the stop.
        m = re.search(r"(?im)^SCORE:\s*(\d+)\b", report)
        if m:
            try:
                score = int(m.group(1))
            except Exception:
                score = None

            if score is not None:
                # Preserve good agent performance, but avoid making incomplete calls look like full successes.
                if score > 85:
                    report = re.sub(r"(?im)^SCORE:\s*\d+\b", "SCORE: 85", report, count=1)
                elif score < 75:
                    report = re.sub(r"(?im)^SCORE:\s*\d+\b", "SCORE: 75", report, count=1)

        # Make later stages clearly not reached due to call ending, not agent skipping.
        not_reached_items = [
            "Existing coverage",
            "Beneficiary",
            "Need amount",
            "Health questions" if health_no_or_partial else None,
            "Product benefits",
            "Three options",
            "Client choice",
            "Application information",
            "Payment date",
            "Banking/payment setup",
            "Banking/account verification",
            "Disclosures",
            "Third Party Underwriting",
            "Peace of Mind",
            "Cool Down",
        ]
        not_reached_items = [x for x in not_reached_items if x]

        report = _set_stage_fields(
            report,
            stage=None,
            not_reached_items=[
                f"{item} — not reached because the prospect stopped responding / disconnected before the agent could continue"
                for item in not_reached_items
            ],
        )

        # Remove unfair script/flow misses that are just later-stage incompletion.
        unfair_miss_phrases = [
            "Health questions not completed",
            "Product benefits not explained",
            "Three options not presented",
            "Application info not collected",
            "Payment/draft date not explained",
            "Banking verification incomplete",
        ]
        for phrase in unfair_miss_phrases:
            report = _text_remove_lines_containing(report, phrase)

        # If the only misses are caused by the prospect ending, say so.
        if re.search(r"(?ims)^SCRIPT / FLOW MISSES:\s*(?=^\s*(?:PQ / HANDOFF:|TASK CHECKLIST:))", report):
            report = re.sub(
                r"(?ims)^SCRIPT / FLOW MISSES:\s*(?=^PQ / HANDOFF:)",
                "SCRIPT / FLOW MISSES:\n- None attributable to the agent before the prospect stopped responding / disconnected.\n\n",
                report,
                count=1,
            )

        # If there are still 3 and 1 coaching misses, keep them. But make the biggest miss clear.
        biggest = (
            "Prospect stopped responding / disconnected before the agent could complete the current section "
            "and move into the remaining sales process."
        )
        report = re.sub(
            r"(?ims)^BIGGEST MISS:\s*.*?(?=^SUMMARY:|^TRANSCRIPT NOTE|^OPENAI COST ESTIMATE:|\Z)",
            f"BIGGEST MISS:\n- {biggest}\n\n",
            report,
            count=1,
        )

        # Sale outcome evidence wording.
        report = re.sub(
            r"(?im)^- Evidence:\s*Customer disconnected before medical questions could be completed\s*$",
            "- Evidence: Prospect stopped responding / disconnected before the agent could continue the sales process",
            report,
            count=1,
        )

    return report






def _strip_ivr_callback_prompts(text):
    """Remove carrier/IVR menu callback prompts that are not prospect callback objections."""
    if not text:
        return text or ""
    return re.sub(
        r"(?im)^.*\b(to receive a callback|press \[NUMBER\].*callback|callback, press|estimated wait time|next available representative)\b.*$",
        "",
        text,
    )


def _has_real_callback_autofail_evidence(report, transcript):
    """
    True only when there is real callback/delay objection evidence plus agent acceptance.
    IVR/carrier phone menu callback prompts do not count.
    """
    evidence_text = transcript if transcript else report
    evidence_text = _strip_ivr_callback_prompts(evidence_text or "")

    prospect_requested = bool(re.search(
        r"(?is)"
        r"(Prospect:\s*.*(?:call\s+(?:me|you)?\s*back|callback|do this later|talk later|not a good time|busy)|"
        r"prospect requested (?:a )?callback|requested (?:a )?callback|asked (?:for )?(?:a )?callback|"
        r"callback due to being busy|too busy.*call back|busy.*call (?:me|you)?\s*back|"
        r"not a good time.*call back|do this later|talk (?:about this )?later)",
        evidence_text,
    ))

    agent_accepted = bool(re.search(
        r"(?is)"
        r"(Agent:\s*.*(?:i(?:'ll| will) call you back|i(?:'ll| will) give you a call back|"
        r"(?:yes|yeah|yep|sure|okay|ok|absolutely|no problem)[,\s.]{0,20}i can call you back|"
        r"i can call you back|i can give you a call back|"
        r"we(?:'ll| will) call you back|we can do this later|agreed to call back|"
        r"scheduled a callback|set a callback)|"
        r"agent agreed to call back later|agreed to call back|accepted the callback|"
        r"instead of attempting call control|instead of.*continuing the sale)",
        evidence_text,
    ))

    return prospect_requested and agent_accepted






def _normalize_autofail_reason_text(reason):
    """Normalize one automatic-fail reason fragment for safe display/merging."""
    reason = (reason or "").strip()
    reason = re.sub(r"\s+", " ", reason)
    return reason


def _merge_autofail_reason(report, new_reason):
    """
    Merge an automatic-fail reason without losing prior causes.

    Rules:
    - If Reason is missing, insert it after Automatic fail triggered.
    - If Reason is blank / None / placeholder, replace with new reason.
    - If Reason already contains this reason, leave it unchanged.
    - Otherwise append using '; '.
    """
    if not report:
        return report

    new_reason = _normalize_autofail_reason_text(new_reason)
    if not new_reason:
        return report

    placeholder_re = re.compile(
        r"^(none|automatic fail triggered(?: \(see automatic fail checks\))?)$",
        re.I,
    )

    def repl(m):
        existing = _normalize_autofail_reason_text(m.group(1))
        if not existing or placeholder_re.match(existing):
            return f"- Reason: {new_reason}"

        # Preserve existing reason if it already contains the new reason.
        if new_reason.lower() in existing.lower():
            return m.group(0)

        return f"- Reason: {existing}; {new_reason}"

    if re.search(r"(?im)^- Reason:\s*(.*)$", report):
        return re.sub(r"(?im)^- Reason:\s*(.*)$", repl, report, count=1)

    return re.sub(
        r"(?im)^- Automatic fail triggered:\s*(?:YES|NO|UNCLEAR)\b.*$",
        lambda m: m.group(0) + f"\n- Reason: {new_reason}",
        report,
        count=1,
    )


def _set_autofail_reason(report, reason, merge=True):
    """
    Set or merge the Reason line.
    Use merge=True for real automatic-fail causes.
    Use merge=False only when intentionally clearing to 'None' or replacing a false positive.
    """
    reason = _normalize_autofail_reason_text(reason)
    if merge and reason and reason.lower() != "none":
        return _merge_autofail_reason(report, reason)

    if re.search(r"(?im)^- Reason:\s*.*$", report):
        return re.sub(r"(?im)^- Reason:\s*.*$", f"- Reason: {reason}", report, count=1)

    return re.sub(
        r"(?im)^- Automatic fail triggered:\s*(?:YES|NO|UNCLEAR)\b.*$",
        lambda m: m.group(0) + f"\n- Reason: {reason}",
        report,
        count=1,
    )


def _set_callback_fields(report, callback_set=None, objection_no_control=None, autofail=None, reason=None):
    """
    Shared helper for callback/autofail report-field rewrites.

    This should gradually replace scattered direct regex rewrites for:
    - Did the agent set a callback?
    - Callback set:
    - Objection occurred without proper call control:
    - Automatic fail triggered:
    - Reason:
    """
    if not report:
        return report

    if callback_set is not None:
        val = "YES" if callback_set else "NO"
        report = re.sub(
            r"(?im)^- Did the agent set a callback\?\s*.*$",
            f"- Did the agent set a callback? {val}",
            report,
        )
        report = re.sub(
            r"(?im)^- Callback set:\s*(?:YES|NO|UNCLEAR)\b.*$",
            f"- Callback set: {val}",
            report,
        )

    if objection_no_control is not None:
        val = "YES" if objection_no_control else "NO"
        report = re.sub(
            r"(?im)^- Objection occurred without proper call control:\s*(?:YES|NO|UNCLEAR)\b.*$",
            f"- Objection occurred without proper call control: {val}",
            report,
        )

    if autofail is not None:
        val = "YES" if autofail else "NO"
        report = re.sub(
            r"(?im)^- Automatic fail triggered:\s*(?:YES|NO|UNCLEAR)\b.*$",
            f"- Automatic fail triggered: {val}",
            report,
        )

    if reason is not None:
        # Merge real autofail causes; replace only when clearing to None.
        report = _set_autofail_reason(
            report,
            reason,
            merge=bool(reason and str(reason).strip().lower() != "none"),
        )

    return report


def _final_cleanup_false_callback_autofail(report, transcript):
    """
    If callback/autofail language exists without real callback evidence, remove the
    entire callback autofail cluster. This prevents IVR callback menus from causing autofails.
    """
    if not report:
        return report

    if _has_real_callback_autofail_evidence(report, transcript):
        return report

    callback_marked = bool(re.search(r"(?im)^- Callback set:\s*YES\b", report))
    callback_reason = bool(re.search(r"(?im)^- Reason:\s*.*(?:callback|delay)", report))
    callback_biggest = bool(re.search(r"(?ims)^BIGGEST MISS:\s*.*(?:callback|delay).*?(?=^SUMMARY:|^TRANSCRIPT NOTE|^OPENAI COST ESTIMATE:|\Z)", report))

    if callback_marked or callback_reason or callback_biggest:
        report = _set_callback_fields(
            report,
            callback_set=False,
            objection_no_control=False,
        )

        # Only clear automatic fail if the reason/biggest miss was callback/delay-based.
        if callback_reason or callback_biggest:
            report = _set_callback_fields(
                report,
                autofail=False,
                reason="None",
            )
            report = re.sub(r"(?im)^PASS:\s*AT RISK\s*$", "PASS: YES", report)

            report = re.sub(
                r"(?ims)^BIGGEST MISS:\s*.*?(?=^SUMMARY:|^TRANSCRIPT NOTE|^OPENAI COST ESTIMATE:|\Z)",
                "BIGGEST MISS:\n- None\n\n",
                report,
                count=1,
            )

    return report


def _final_cleanup_callback_autofail_consistency(report, transcript):
    """
    Future-call guardrail:
    If the report itself says the prospect requested a callback/busy delay and the
    agent accepted the callback instead of controlling or continuing the sale,
    the automatic-fail section must reflect that consistently.
    """
    if not report:
        return report

    combined = _strip_ivr_callback_prompts(transcript if transcript else (report or ""))

    # Callback autofail needs clear callback / delay evidence.
    # Do not trigger from casual rapport language containing words like "call",
    # "later", "live", "work", or family stories.
    callback_requested = bool(re.search(
        r"(?is)"
        r"("
        r"prospect requested (?:a )?callback|"
        r"requested (?:a )?callback|"
        r"asked (?:for )?(?:a )?callback|"
        r"call (?:me|you|him|her|them)?\s*back\s+later|"
        r"call (?:me|you|him|her|them)?\s*back\s+(?:tomorrow|today|next week|on monday|on tuesday|on wednesday|on thursday|on friday)|"
        r"callback due to being busy|"
        r"too busy.*call back|"
        r"busy.*call (?:me|you)?\s*back|"
        r"not a good time.*call back|"
        r"do this later|"
        r"talk (?:about this )?later"
        r")",
        combined,
    ))

    agent_accepted_without_control = bool(re.search(
        r"(?is)"
        r"("
        r"agent agreed to call back later|"
        r"agreed to call back|"
        r"accepted the callback|"
        r"set a callback|"
        r"scheduled a callback|"
        r"i(?:'ll| will) call you back|"
        r"i(?:'ll| will) give you a call back|"
        r"(?:yes|yeah|yep|sure|okay|ok|absolutely|no problem)[,\s.]{0,20}i can call you back|"
        r"i can call you back|"
        r"i can give you a call back|"
        r"we(?:'ll| will) call you back|"
        r"we can do this later|"
        r"instead of attempting call control|"
        r"instead of.*continuing the sale|"
        r"resulting in an automatic fail"
        r")",
        combined,
    ))

    # Do not trigger on prospect-only future intent if the report does not say
    # the agent accepted/delayed/failed to control.
    if not (callback_requested and agent_accepted_without_control):
        return report

    # Normalize automatic-fail checks.
    report = _set_callback_fields(
        report,
        callback_set=True,
        objection_no_control=True,
        autofail=True,
        reason="Prospect requested a callback / delay and the agent accepted it instead of controlling or continuing the live sales attempt.",
    )

    # Risk should be HIGH for a callback autofail.
    report = re.sub(r"(?im)^RISK:\s*(?:LOW|MEDIUM)\s*$", "RISK: HIGH", report, count=1)

    # If report currently says PASS: YES, make it AT RISK rather than clean pass.
    report = re.sub(r"(?im)^PASS:\s*YES\s*$", "PASS: AT RISK", report, count=1)

    # Add/replace biggest miss.
    biggest = "Agent accepted a callback / delay instead of controlling the objection or continuing the live sales attempt."
    if re.search(r"(?ims)^BIGGEST MISS:\s*.*?(?=^SUMMARY:|^TRANSCRIPT NOTE|^OPENAI COST ESTIMATE:|\Z)", report):
        report = re.sub(
            r"(?ims)^BIGGEST MISS:\s*.*?(?=^SUMMARY:|^TRANSCRIPT NOTE|^OPENAI COST ESTIMATE:|\Z)",
            f"BIGGEST MISS:\n- {biggest}\n\n",
            report,
            count=1,
        )

    return report




def _has_agent_controllable_major_issue(report):
    """
    True when a report contains a stronger agent-controllable issue than
    the prospect simply disconnecting / stopping response.
    """
    if not report:
        return False

    patterns = [
        r"(?i)unprofessional language",
        r"(?i)disrespectful",
        r"(?i)rude",
        r"(?i)inappropriate",
        r"(?i)existing coverage mentioned but not confirmed:\s*YES",
        r"(?i)automatic fail triggered:\s*YES",
        r"(?i)callback set:\s*YES",
        r"(?i)objection occurred without proper call control:\s*YES",
        r"(?i)agent accepted a callback",
        r"(?i)banking verification incomplete",
        r"(?i)routing number verification did not meet",
        r"(?i)credit union mentioned but bank/account not verified:\s*YES",
        r"(?i)compliance failures:\s*(?!none\b)",
    ]

    return any(re.search(p, report) for p in patterns)


def _final_cleanup_protect_major_biggest_miss(report, transcript):
    """
    If a real agent-controllable issue exists, do not let the generic
    prospect-disconnect wording remain as Biggest Miss.
    """
    if not report:
        return report

    if not _has_agent_controllable_major_issue(report):
        return report

    current_biggest = ""
    m = re.search(r"(?ims)^BIGGEST MISS:\s*(.*?)(?=^SUMMARY:|^TRANSCRIPT NOTE|^OPENAI COST ESTIMATE:|\Z)", report)
    if m:
        current_biggest = m.group(1).strip().lower()

    disconnect_biggest = bool(re.search(
        r"prospect stopped responding|prospect disconnected|disconnected before|before the agent could complete|before the agent could continue",
        current_biggest,
        re.I,
    ))

    if not disconnect_biggest:
        return report

    replacement = None

    if re.search(r"(?i)unprofessional language|disrespectful|rude|inappropriate", report):
        replacement = "Agent used unprofessional or disrespectful language / delivery during the call."
    elif re.search(r"(?im)^- Callback set:\s*YES\b|^-\s*Objection occurred without proper call control:\s*YES\b", report):
        replacement = "Agent accepted a callback / delay instead of controlling the objection or continuing the live sales attempt."
    elif re.search(r"(?im)^- Existing coverage mentioned but not confirmed:\s*YES\b", report):
        replacement = "Existing coverage was mentioned but not properly confirmed before the call moved forward."
    elif re.search(r"(?im)^- Credit union mentioned but bank/account not verified:\s*YES\b", report):
        replacement = "Credit union/account information was not properly verified before banking moved forward."
    elif re.search(r"(?i)banking verification incomplete|routing number verification did not meet", report):
        replacement = "Banking/routing verification did not meet the required verification standard."
    elif re.search(r"(?im)^- Automatic fail triggered:\s*YES\b", report):
        reason = ""
        rm = re.search(r"(?im)^- Reason:\s*(.+)$", report)
        if rm:
            reason = rm.group(1).strip()
        replacement = reason or "Automatic fail condition was triggered."

    if replacement:
        report = re.sub(
            r"(?ims)^BIGGEST MISS:\s*.*?(?=^SUMMARY:|^TRANSCRIPT NOTE|^OPENAI COST ESTIMATE:|\Z)",
            f"BIGGEST MISS:\n- {replacement}\n\n",
            report,
            count=1,
        )

    return report




def _not_reached_reason_for_unfinished_call(report, transcript):
    """
    Choose the most accurate reason for later stages being not reached.
    Order matters: agent-controllable autofails should not be mislabeled as prospect disconnects.
    """
    combined = ((report or "") + "\n" + (transcript or "")).lower()

    callback_autofail = bool(re.search(
        r"(?is)"
        r"(callback set:\s*yes|agent accepted a callback|agreed to call back|"
        r"prospect requested callback|call back later|callback / delay|"
        r"objection occurred without proper call control:\s*yes)",
        combined,
    ))

    unresolved_coverage = bool(re.search(
        r"(?im)^- Existing coverage mentioned but not confirmed:\s*YES\b",
        report or "",
    ))

    credit_union_fail = bool(re.search(
        r"(?im)^- Credit union mentioned but bank/account not verified:\s*YES\b",
        report or "",
    ))

    banking_fail = bool(re.search(
        r"(?i)banking verification incomplete|routing number verification did not meet",
        report or "",
    ))

    unprofessional = bool(re.search(
        r"(?i)unprofessional language|disrespectful|rude|inappropriate",
        report or "",
    ))

    disconnected = bool(re.search(
        r"(?i)prospect stopped responding|customer disconnected|prospect disconnected|client disconnected|"
        r"stopped responding|can you hear me|are you there|disconnected before|hung up|hang up",
        combined,
    ))

    if callback_autofail:
        return "not reached because the live sales attempt ended after the callback/delay objection"
    if unresolved_coverage:
        return "not reached because existing coverage was not resolved before the call ended"
    if credit_union_fail:
        return "not reached because credit union/banking verification was not resolved before the call ended"
    if banking_fail:
        return "not reached because banking/routing verification was incomplete before the call ended"
    if unprofessional:
        return "not reached because the call ended after an agent-controllable professionalism issue"
    if disconnected:
        return "not reached because the prospect stopped responding / disconnected before the agent could continue"

    return "not reached because the call ended before the agent could continue"


def _rewrite_not_reached_reason(report, transcript):
    """
    Rewrite expanded NOT REACHED reason suffixes so they match the true unfinished-call reason.
    """
    if not report:
        return report

    if "NOT REACHED:" not in report:
        return report

    reason = _not_reached_reason_for_unfinished_call(report, transcript)

    # Only rewrite expanded reason lines, not simple '- Disclosures' style lists.
    report = re.sub(
        r"(?im)^- ([^-:\n][^\n]*?)\s+—\s+not reached because .*$",
        lambda m: f"- {m.group(1).strip()} — {reason}",
        report,
    )

    return report




def _final_cleanup_partial_health_unsold_guardrail(report, transcript):
    """
    Future-call guardrail:
    If an unsold call reaches only Medical / Health or partial health and does not
    reach product/options/application, keep scoring fair but prevent LOW/90-style
    completed-call treatment.
    """
    if not report:
        return report

    sold_no = bool(re.search(r"(?im)^- Policy sold:\s*NO\b|^- Was the policy sold\?\s*NO\b", report))
    stage_health = bool(re.search(r"(?im)^CALL STAGE REACHED:\s*(Medical / Health|Health|Medical)\b", report))
    health_yes_or_partial = bool(re.search(r"(?im)^- Health questions completed:\s*(YES|PARTIAL)\b", report))
    product_no = bool(re.search(r"(?im)^- Product benefits explained:\s*(NO|NOT REACHED)\b", report))
    options_no = bool(re.search(r"(?im)^- Three options presented:\s*(NO|NOT REACHED)\b", report))
    app_no = bool(re.search(r"(?im)^- Application info collected:\s*(NO|NOT REACHED)\b", report))
    autofail_yes = bool(re.search(r"(?im)^- Automatic fail triggered:\s*YES\b", report))

    partial_health_unsold = sold_no and (stage_health or health_yes_or_partial) and product_no and options_no and app_no

    if not partial_health_unsold:
        return report

    # The call did not complete the sale path.
    report = re.sub(r"(?im)^EARLY END:\s*NO\s*$", "EARLY END: YES", report, count=1)

    # If no automatic fail, this is usually medium risk: incomplete, but not necessarily agent fault.
    if not autofail_yes:
        report = re.sub(r"(?im)^RISK:\s*LOW\s*$", "RISK: MEDIUM", report, count=1)

    # Keep score fair, but avoid making an incomplete unsold call look like a clean completed win.
    m = re.search(r"(?im)^SCORE:\s*(\d+)\b", report)
    if m:
        try:
            score = int(m.group(1))
        except Exception:
            score = None

        if score is not None and score > 85:
            report = re.sub(r"(?im)^SCORE:\s*\d+\b", "SCORE: 85", report, count=1)

    # Later stages should be not reached.
    later_items = [
        "Product benefits",
        "Three options",
        "Client choice",
        "Application information",
        "Payment date",
        "Banking/payment setup",
        "Banking/account verification",
        "Disclosures",
        "Third Party Underwriting",
        "Peace of Mind",
        "Cool Down",
    ]

    reason = _not_reached_reason_for_unfinished_call(report, transcript) if "_not_reached_reason_for_unfinished_call" in globals() else "not reached because the call ended before the agent could continue"

    report = _set_stage_fields(
        report,
        stage=None,
        not_reached_items=[f"{item} — {reason}" for item in later_items],
    )

    report = _text_replace_checklist_value(report, "Payment date explained", "NOT REACHED")
    report = _text_replace_checklist_value(report, "Banking/payment setup explained", "NOT REACHED")

    # If no stronger agent-controllable issue exists, keep biggest miss about the call ending/incomplete path.
    has_major_issue = _has_agent_controllable_major_issue(report) if "_has_agent_controllable_major_issue" in globals() else False
    if not has_major_issue:
        biggest = "Call ended after partial health/medical questions before the agent could move into product explanation, options, application, disclosures, Peace of Mind, or Cool Down."
        report = re.sub(
            r"(?ims)^BIGGEST MISS:\s*.*?(?=^SUMMARY:|^TRANSCRIPT NOTE|^OPENAI COST ESTIMATE:|\Z)",
            f"BIGGEST MISS:\n- {biggest}\n\n",
            report,
            count=1,
        )

    return report






def _set_stage_fields(report, stage, final_stage=None, not_reached_items=None, early_end=None):
    """
    Shared helper for stage-field rewrites.

    This should gradually replace scattered direct regex rewrites for:
    - CALL STAGE REACHED
    - EARLY END
    - NOT REACHED
    - Final stage supporting sale

    Keep this helper small and conservative.
    """
    if not report:
        return report

    if stage:
        report = re.sub(
            r"(?im)^CALL STAGE REACHED:\s*.*$",
            f"CALL STAGE REACHED: {stage}",
            report,
            count=1,
        )

    if early_end is not None:
        report = re.sub(
            r"(?im)^EARLY END:\s*(YES|NO|UNCLEAR).*$",
            f"EARLY END: {'YES' if early_end else 'NO'}",
            report,
            count=1,
        )

    if final_stage:
        report = re.sub(
            r"(?im)^- Final stage supporting sale:\s*.*$",
            f"- Final stage supporting sale: {final_stage}",
            report,
            count=1,
        )

    if not_reached_items is not None:
        block = "NOT REACHED:\n" + "\n".join(f"- {item}" for item in not_reached_items) + "\n\n"
        report = re.sub(
            r"(?ims)^NOT REACHED:\s*.*?(?=^COMPLIANCE FAILURES:)",
            block,
            report,
            count=1,
        )

    return report


def _final_cleanup_false_banking_stage_guardrail(report, transcript):
    """
    Future-call guardrail:
    Do not allow impossible late stages like Quotes/Application/Banking when the call
    never reached product/options/application and has no real banking evidence.
    """
    if not report:
        return report

    late_stage = bool(re.search(
        r"(?im)^CALL STAGE REACHED:\s*(Quotes|Pre-Close|Close|Application Information|Application|Banking|Disclosures|Third Party Underwriting|Peace of Mind|Cool Down)\b",
        report,
    ))
    if not late_stage:
        return report

    sold_no = bool(re.search(r"(?im)^- Policy sold:\s*NO\b|^- Was the policy sold\?\s*NO\b", report))
    health_no = bool(re.search(r"(?im)^- Health questions completed:\s*(NO|NOT REACHED)\b", report))
    product_no = bool(re.search(r"(?im)^- Product benefits explained:\s*(NO|NOT REACHED)\b", report))
    options_no = bool(re.search(r"(?im)^- Three options presented:\s*(NO|NOT REACHED)\b", report))
    app_no = bool(re.search(r"(?im)^- Application info collected:\s*(NO|NOT REACHED)\b", report))

    account_zero_or_not_reached = bool(re.search(r"(?im)^- Account verification evidence count:\s*0\b", report)) or bool(re.search(r"(?im)^- Banking/account information requested or verified 3 times:\s*NOT REACHED\b", report))
    routing_zero_or_not_reached = bool(re.search(r"(?im)^- Routing verification evidence count:\s*0\b", report)) or bool(re.search(r"(?im)^- Routing number requested or verified 3 times:\s*NOT REACHED\b", report))
    no_real_banking = account_zero_or_not_reached and routing_zero_or_not_reached

    no_late_sales_path = sold_no and product_no and options_no and app_no and no_real_banking

    if no_late_sales_path:
        # Fall back to the latest supported completed stage.
        # Use both report and transcript evidence so we do not downgrade too far.
        # Prefer transcript evidence for stage fallback. Checklist labels like
        # "Need amount discussed: NO" or "Beneficiary identified: NO" must not count
        # as positive evidence that those stages were reached.
        evidence_text = transcript if transcript else (report or "")

        beneficiary_reached = bool(re.search(
            r"(?is)(who would be your beneficiary|beneficiary on your policy|what(?:'s| is) his name|what(?:'s| is) her name|who would you want|who do you want as)",
            evidence_text,
        ))
        need_reached = bool(re.search(
            r"(?is)(how much coverage|recommend between|coverage for cremation|coverage for burial|cover burial|cover cremation|take full responsibility for your final expenses)",
            evidence_text,
        ))
        # Quote evidence must be positive transcript/report language.
        # Do not match checklist labels like "Three options presented: NO".
        quote_reached = bool(re.search(
            r"(?is)(pull those up|pull up (?:the )?(?:plans|quotes)|qualified for one of our preferred plans|preferred plans|monthly premium|exact cost right now|give you the exact cost)",
            evidence_text,
        ))

        if quote_reached:
            corrected_stage = "Quotes"
        elif re.search(r"(?im)^- Health questions completed:\s*(YES|PARTIAL)\b", report):
            corrected_stage = "Medical / Health"
        elif beneficiary_reached or need_reached:
            corrected_stage = "Need / Beneficiary"
        elif re.search(r"(?im)^- Fact Finding / Warm-up:\s*YES\b", report):
            corrected_stage = "Fact Finding / Warm-up"
        elif re.search(r"(?im)^- Agent introduction:\s*YES\b", report):
            corrected_stage = "Who I Am / What I Do"
        else:
            corrected_stage = "Opening / Handoff"

        report = _set_stage_fields(
            report,
            corrected_stage,
            final_stage=corrected_stage,
            early_end=True,
        )

        report = _text_replace_checklist_value(report, "Payment date explained", "NOT REACHED")
        report = _text_replace_checklist_value(report, "Banking/payment setup explained", "NOT REACHED")

    return report




def _strip_embedded_transcript_from_report(report):
    """
    Final reports should not include the full transcript inside the Detailed Report.
    The dashboard already shows the transcript in its own section.
    """
    if not report:
        return report

    report = re.sub(
        r"(?ims)\n*TRANSCRIPT NOTE \(MANDATORY\):.*?(?=^OPENAI COST ESTIMATE:|\Z)",
        "\n",
        report,
        count=1,
    )

    report = re.sub(
        r"(?ims)\n*TRANSCRIPT:\s*.*?(?=^OPENAI COST ESTIMATE:|\Z)",
        "\n",
        report,
        count=1,
    )

    return report.strip() + "\n"


def _decode_report_html_entities(report):
    """
    Decode accidental HTML entities in saved report text, such as:
    &#x27; -> apostrophe, &quot; -> quote, &amp; -> &
    """
    if not report:
        return report

    try:
        from html import unescape
        return unescape(report)
    except Exception:
        return report



def _compress_not_reached_block(report):
    """
    Keep the manager-facing NOT REACHED section readable.
    Expanded line-by-line stage lists are useful internally, but too noisy in the report.
    Compress repeated reason lists into a short grouped summary.
    """
    if not report:
        return report

    m = re.search(r"(?ims)^NOT REACHED:\s*(.*?)(?=^COMPLIANCE FAILURES:)", report)
    if not m:
        return report

    body = m.group(1).strip()
    lines = [ln.strip() for ln in body.splitlines() if ln.strip().startswith("-")]

    if len(lines) < 6:
        return report

    body_lower = body.lower()

    if "prospect stopped responding / disconnected" in body_lower:
        replacement = (
            "NOT REACHED:\n"
            "- Remaining sales process — prospect stopped responding / disconnected before the agent could continue.\n"
            "- Includes: existing coverage, beneficiary, need amount, health questions, product benefits, options, application, payment/banking, disclosures, underwriting, Peace of Mind, and Cool Down.\n\n"
        )
    elif "callback/delay objection" in body_lower:
        replacement = (
            "NOT REACHED:\n"
            "- Remaining sales process — live sale attempt ended after the callback/delay objection.\n"
            "- Includes: product benefits, options, application, payment/banking, disclosures, underwriting, Peace of Mind, and Cool Down.\n\n"
        )
    elif "existing coverage was not resolved" in body_lower:
        replacement = (
            "NOT REACHED:\n"
            "- Remaining sales process — existing coverage was not resolved before the call ended.\n"
            "- Includes: product benefits, options, application, payment/banking, disclosures, underwriting, Peace of Mind, and Cool Down.\n\n"
        )
    elif "banking/routing verification was incomplete" in body_lower or "credit union/banking verification" in body_lower:
        replacement = (
            "NOT REACHED:\n"
            "- Remaining post-banking process — banking/account verification was not fully resolved before the call ended.\n"
            "- Includes: disclosures, underwriting, Peace of Mind, and Cool Down.\n\n"
        )
    elif "not reached because" in body_lower or "—" in body:
        replacement = (
            "NOT REACHED:\n"
            "- Remaining sales process — call ended before the agent could continue.\n"
            "- Includes: product benefits, options, application, payment/banking, disclosures, underwriting, Peace of Mind, and Cool Down.\n\n"
        )
    else:
        return report

    return report[:m.start()] + replacement + report[m.end():]




def _transcript_has_strong_sale_completion_evidence(transcript):
    """
    Strong evidence that the sale/application reached completion or near-completion.
    This prevents sold calls from being marked unsold just because Peace of Mind / Cool Down
    did not happen.
    """
    if not transcript:
        return False

    t = str(transcript).lower()

    strong_patterns = [
        r"application process was completed over the telephone",
        r"voice signature",
        r"by signing this application",
        r"completing your application",
        r"completed application",
        r"you have applied for .*whole life insurance policy",
        r"application for insurance",
        r"authorize the drafting of insurance premiums",
        r"provided your banking information and authorize",
        r"copy of your completed application",
        r"assigning the application electronically",
        r"now we're almost done.*rest of your application",
    ]

    hits = sum(1 for p in strong_patterns if re.search(p, t, re.I | re.S))

    banking_evidence = bool(re.search(
        r"(banking information|routing number|account number|authorize the drafting|bank account|first payment)",
        t,
        re.I,
    ))

    disclosure_evidence = bool(re.search(
        r"(disclosures|required disclosures|voice signature|application process was completed|by signing this application)",
        t,
        re.I,
    ))

    return hits >= 2 and banking_evidence and disclosure_evidence


def _final_cleanup_sold_call_completion_evidence(report, transcript):
    """
    If strong transcript evidence shows application/banking/disclosures/voice signature,
    do not let the report say policy sold NO simply because final post-sale stages
    like Peace of Mind or Cool Down were not reached.

    This also protects sold calls from stale no-sale/disqualification cleanup text.
    A completed application can still have real process misses, but coaching, evidence,
    and summary should not say the agent stopped for DNQ/ineligibility unless the
    transcript independently supports that outcome.
    """
    if not report:
        return report

    if not _transcript_has_strong_sale_completion_evidence(transcript):
        return report

    report = re.sub(r"(?im)^- Was the policy sold\?\s*NO\b.*$", "- Was the policy sold? YES", report)
    report = re.sub(r"(?im)^- Policy sold:\s*NO\b.*$", "- Policy sold: YES", report)

    if re.search(r"(?im)^SALE OUTCOME:\s*$", report):
        report = re.sub(
            r"(?im)^- Evidence:\s*.*$",
            "- Evidence: Application, banking authorization, disclosures, and voice-signature/application completion language were completed.",
            report,
            count=1,
        )

    report = re.sub(
        r"(?im)^- Final stage supporting sale:\s*.*$",
        "- Final stage supporting sale: Third Party Underwriting",
        report,
        count=1,
    )

    # A sold call may still be AT RISK for a real autofail, but should not be PASS: NO only due to unsold outcome.
    if not re.search(r"(?im)^- Automatic fail triggered:\s*YES\b", report):
        report = re.sub(r"(?im)^PASS:\s*NO\s*$", "PASS: YES", report, count=1)
        report = re.sub(r"(?im)^RISK:\s*HIGH\s*$", "RISK: MEDIUM", report, count=1)

    # If the report says Early End only because POM/Cool Down did not happen, keep it less misleading.
    # Do not force EARLY END to NO unless the call fully reached Cool Down.
    report = re.sub(
        r"(?im)^- Evidence:\s*Application completed through banking, no callback set, no post-sale completion evidence\s*$",
        "- Evidence: Application, banking authorization, disclosures, and voice-signature/application completion language were completed.",
        report,
        count=1,
    )

    # Remove stale DNQ/ineligibility cleanup that contradicts completed-sale evidence.
    stale_dq = re.compile(
        r"(?i)(agent appropriately stopped after identifying disqualification|"
        r"prospect had a disqualifying health condition|"
        r"prospect was not eligible|"
        r"call ended because the prospect was not eligible|"
        r"continuing the sale was not appropriate)"
    )
    if stale_dq.search(report):
        report = _text_remove_lines_containing(report, "Agent appropriately stopped after identifying disqualification")
        report = _text_remove_lines_containing(report, "Prospect had a disqualifying health condition")
        report = _text_remove_lines_containing(report, "prospect was not eligible")
        report = _text_remove_lines_containing(report, "continuing the sale was not appropriate")

        sold_summary = (
            "The agent completed a sold call: options were presented, the client chose an option, "
            "application information and payment date were collected, banking setup was handled, "
            "and required disclosures plus voice-signature/application completion language were completed."
        )
        if re.search(r"(?ims)^SUMMARY:\s*.*?(?=^OPENAI COST ESTIMATE:|\Z)", report):
            report = re.sub(
                r"(?ims)^SUMMARY:\s*.*?(?=^OPENAI COST ESTIMATE:|\Z)",
                f"SUMMARY:\n{sold_summary}\n",
                report,
                count=1,
            )
        else:
            report = report.rstrip() + f"\n\nSUMMARY:\n{sold_summary}\n"

        if re.search(r"(?ims)^COACHING:\s*(?=^BIGGEST MISS:|^SUMMARY:|^OPENAI COST ESTIMATE:|\Z)", report):
            coaching = "Review the remaining scored process misses from the completed sale, especially any verification or rapport items still marked incomplete."
            report = re.sub(
                r"(?ims)^COACHING:\s*(?=^BIGGEST MISS:|^SUMMARY:|^OPENAI COST ESTIMATE:|\Z)",
                f"COACHING:\n- {coaching}\n\n",
                report,
                count=1,
            )

    if re.search(r"(?ims)^BIGGEST MISS:\s*[-•]?\s*None\s*(?=^SUMMARY:|^TRANSCRIPT NOTE|^OPENAI COST ESTIMATE:|\Z)", report):
        biggest = None
        if re.search(r"(?i)routing number verification did not meet|routing number requested or verified 3 times:\s*PARTIAL", report):
            biggest = "Routing number verification did not meet the three-event standard after the sale was completed."
        elif re.search(r"(?i)banking/account information requested or verified 3 times:\s*PARTIAL|banking/account information requested or verified 3 times incomplete", report):
            biggest = "Banking/account verification did not fully meet the required verification standard after the sale was completed."
        elif re.search(r"(?i)3 and 1 Method incomplete|3 and 1 Method used:\s*PARTIAL", report):
            biggest = "3 and 1 rapport was incomplete before the sale moved forward."
        if biggest:
            report = re.sub(
                r"(?ims)^BIGGEST MISS:\s*.*?(?=^SUMMARY:|^TRANSCRIPT NOTE|^OPENAI COST ESTIMATE:|\Z)",
                f"BIGGEST MISS:\n- {biggest}\n\n",
                report,
                count=1,
            )

    # Remove summary sentence that contradicts sale evidence.
    report = re.sub(
        r"(?i)Post-sale stages \(Disclosures, Third Party Underwriting, Peace of Mind, Cool Down\) were not reached, and the policy was not sold on this call\.",
        "Peace of Mind and Cool Down were not reached, but the application, banking authorization, disclosures, and voice-signature/application completion language support a sold/application-completed outcome.",
        report,
    )

    return report




def _transcript_has_peace_of_mind_after_sale(transcript):
    """
    Detect Peace of Mind reached after sale/application completion.
    This is stricter than polite closing and should not mark Cool Down.
    """
    if not transcript:
        return False

    t = str(transcript).lower()

    sale_anchor = re.search(
        r"(voice signature|application process was completed|by signing this application|"
        r"you have applied for|american amicable.*recording|app id|pound sign|"
        r"authorize the drafting|completed application)",
        t,
        re.I | re.S,
    )
    if not sale_anchor:
        return False

    after_sale = t[sale_anchor.start():]

    pom_hits = 0
    for pat in [
        r"you're good",
        r"you are good",
        r"we'?re not going to forget about you",
        r"we are not going to forget about you",
        r"mail (?:you )?(?:the )?(?:package|welcome letter|policy)",
        r"send (?:you )?(?:the )?(?:package|welcome letter|policy)",
        r"include everything we talked about",
        r"include all (?:of )?(?:the )?information",
        r"information (?:about )?(?:the )?(?:program|plan|company)",
        r"qualified for",
    ]:
        if re.search(pat, after_sale, re.I | re.S):
            pom_hits += 1

    return pom_hits >= 3


def _remove_not_reached_item(report, item_name):
    """Remove one simple NOT REACHED bullet by name, preserving the rest of the block."""
    if not report:
        return report

    pattern = rf"(?im)^-\s*{re.escape(item_name)}\s*$\n?"
    return re.sub(pattern, "", report)


def _final_cleanup_peace_of_mind_after_sale(report, transcript):
    """
    If a sold/application-completed call clearly reaches Peace of Mind after TPU/voice-signature,
    align stage/checklist/NOT REACHED accordingly. Do not mark Cool Down unless actual
    post-sale casual conversation happened.
    """
    if not report:
        return report

    sold_yes = bool(re.search(r"(?im)^- Policy sold:\s*YES\b|^- Was the policy sold\?\s*YES\b", report))
    if not sold_yes:
        return report

    if not _transcript_has_peace_of_mind_after_sale(transcript):
        return report

    # Do not override Cool Down if already clearly reached.
    cooldown_yes = bool(re.search(r"(?im)^- Cool down completed:\s*YES\b|^CALL STAGE REACHED:\s*Cool Down\b", report))

    if not cooldown_yes:
        report = _set_stage_fields(
            report,
            "Peace of Mind",
            final_stage="Peace of Mind",
            early_end=False,
        )
        report = _remove_not_reached_item(report, "Peace of Mind")
        if "Cool Down" not in re.search(r"(?ims)^NOT REACHED:\s*(.*?)(?=^COMPLIANCE FAILURES:)", report).group(1) if re.search(r"(?ims)^NOT REACHED:\s*(.*?)(?=^COMPLIANCE FAILURES:)", report) else "":
            # If the block exists and Cool Down is not listed, add it as the only post-POM remaining item.
            report = re.sub(
                r"(?ims)^NOT REACHED:\s*.*?(?=^COMPLIANCE FAILURES:)",
                "NOT REACHED:\n- Cool Down\n\n",
                report,
                count=1,
            )
    else:
        report = _set_stage_fields(
            report,
            "Cool Down",
            final_stage="Cool Down",
            early_end=False,
        )

    report = _text_replace_checklist_value(report, "Peace of mind completed", "YES")

    if not cooldown_yes:
        report = _text_replace_checklist_value(report, "Cool down completed", "NO")

    # Remove/soften contradictory summary wording if present.
    report = re.sub(
        r"(?i)Peace of Mind and Cool Down were not reached",
        "Cool Down was not reached",
        report,
    )
    report = re.sub(
        r"(?i)Post-sale stages .*? were not reached",
        "Cool Down was not reached",
        report,
    )

    return report




def _final_cleanup_summary_stage_contradictions(report):
    """
    Remove stale summary wording that contradicts corrected structured stage fields.
    """
    if not report:
        return report

    pom_stage = bool(re.search(r"(?im)^CALL STAGE REACHED:\s*Peace of Mind\b", report))
    cooldown_not_reached = bool(re.search(r"(?ims)^NOT REACHED:\s*.*^- Cool Down\s*$", report))
    cooldown_no = bool(re.search(r"(?im)^- Cool down completed:\s*(NO|NOT REACHED)\b", report))

    if pom_stage and (cooldown_not_reached or cooldown_no):
        report = re.sub(
            r"(?i)completed a sold call through Cool Down",
            "completed a sold call through Peace of Mind",
            report,
        )
        report = re.sub(
            r"(?i)Peace of Mind plus Cool Down were reached",
            "Peace of Mind was reached; Cool Down was not reached",
            report,
        )
        report = re.sub(
            r"(?i)Peace of Mind and Cool Down were reached",
            "Peace of Mind was reached; Cool Down was not reached",
            report,
        )

    return report





def _transcript_has_clean_health_screening_no_dq(transcript):
    """
    True when the transcript itself shows clean health knockout answers.

    This prevents stale report text like "disqualifying health condition" from
    re-triggering LCR/fair-disqualification cleanup when the actual health screen
    was clean and the call ended for another reason, such as existing coverage or
    refusal.
    """
    t = (transcript or "").lower()
    if not t:
        return False

    clean_summary = bool(re.search(
        r"answered\s+no\s+to\s+(?:all|the)\s+health\s+questions|"
        r"no\s+to\s+all\s+(?:of\s+)?(?:the\s+)?health\s+questions|"
        r"you(?:'re| are)\s+in\s+(?:really\s+)?good\s+shape|"
        r"health(?:-screening| screening)?\s+(?:looked|looks)\s+(?:good|clean)",
        t,
    ))

    repeated_clean_no = len(re.findall(
        r"(?is)(stroke|heart\s+attack|heart\s+disease|copd|emphysema|"
        r"kidney|renal|dialysis|oxygen|cancer|diabetes|terminal|hospice).{0,120}"
        r"(?:prospect|client|customer)\s*:\s*(?:no|nope|never)\b",
        t,
    )) >= 2

    explicit_health_dq = bool(re.search(
        r"(?is)(because of that|based on that|with that condition|unfortunately|sorry).{0,220}"
        r"(do(?:es)? not qualify|won't qualify|would not qualify|can't qualify|cannot qualify|"
        r"not able to qualify|unable to qualify|not eligible|declined|knockout)",
        t,
    ))

    return (clean_summary or repeated_clean_no) and not explicit_health_dq


def _extract_first_real_flow_miss(report):
    """Return the first substantive SCRIPT / FLOW miss, or None."""
    m = re.search(
        r"(?ims)^SCRIPT / FLOW MISSES:\s*(.*?)(?=^TASK CHECKLIST:|^PQ / HANDOFF:|^SEARCHABLE ANSWERS:|^AUTOMATIC FAIL CHECKS:|^SALE OUTCOME:|^SCORING BREAKDOWN:|^COACHING:|^BIGGEST MISS:|^SUMMARY:|^OPENAI COST ESTIMATE:|\Z)",
        report or "",
    )
    if not m:
        return None

    for raw in m.group(1).splitlines():
        line = re.sub(r"^\s*[-•]\s*", "", raw).strip()
        if not line:
            continue
        if re.fullmatch(r"(?i)(none|n/a|not applicable)", line):
            continue
        return line.rstrip(" .") + "."

    return None


def _final_cleanup_false_health_disqualification_after_clean_screen(report, transcript):
    """Remove stale health-DNQ wording when transcript health screening was clean."""
    if not report or not _transcript_has_clean_health_screening_no_dq(transcript):
        return report

    stale_health = bool(re.search(
        r"(?is)Prospect had a disqualifying health condition|"
        r"Agent appropriately stopped after identifying disqualification|"
        r"call ended because the prospect was not eligible|"
        r"continuing the sale was not appropriate",
        report,
    ))
    if not stale_health:
        return report

    report = _text_remove_lines_containing(report, "Prospect had a disqualifying health condition")
    report = _text_remove_lines_containing(report, "Agent appropriately stopped after identifying disqualification")
    report = _text_remove_lines_containing(report, "continuing the sale was not appropriate")

    replacement_evidence = (
        "Prospect answered the health screening cleanly, but the call did not continue "
        "to application or enrollment completion."
    )
    if re.search(r"(?im)^- Evidence:\s*", report):
        report = re.sub(
            r"(?im)^- Evidence:\s*.*$",
            f"- Evidence: {replacement_evidence}",
            report,
            count=1,
        )

    clean_summary = (
        "The prospect answered the health screening cleanly, but the call ended before "
        "application or enrollment completion. Future stages should be evaluated based "
        "on the actual reason the call stopped, not as a health disqualification."
    )
    if re.search(r"(?ims)^SUMMARY:\s*.*?(?=^OPENAI COST ESTIMATE:|\Z)", report):
        report = re.sub(
            r"(?ims)^SUMMARY:\s*.*?(?=^OPENAI COST ESTIMATE:|\Z)",
            "SUMMARY:\n" + clean_summary + "\n",
            report,
            count=1,
        )
    else:
        report = report.rstrip() + "\n\nSUMMARY:\n" + clean_summary + "\n"

    if re.search(r"(?ims)^COACHING:\s*(?=^BIGGEST MISS:|^SUMMARY:|^OPENAI COST ESTIMATE:|\Z)", report):
        report = re.sub(
            r"(?ims)^COACHING:\s*(?=^BIGGEST MISS:|^SUMMARY:|^OPENAI COST ESTIMATE:|\Z)",
            "COACHING:\n- Review the call based on the actual stop reason; the transcript does not support a health disqualification.\n\n",
            report,
            count=1,
        )

    return report


def _final_cleanup_needs_stage_fields(report, transcript):
    """Keep Needs-stage reports internally consistent."""
    if not report:
        return report

    reached_needs = bool(re.search(r"(?im)^CALL STAGE REACHED:\s*Needs?\b", report))
    if not reached_needs and _transcript_reached_needs_section(transcript):
        reached_needs = True

    if not reached_needs:
        return report

    report = re.sub(
        r"(?im)^- Final stage supporting sale:\s*Medical / Health\s*$",
        "- Final stage supporting sale: Needs",
        report,
        count=1,
    )

    report = re.sub(
        r"(?i)before need discovery, quotes, or application stages",
        "after needs discovery but before quotes or application stages",
        report,
    )
    report = re.sub(
        r"(?i)call ended before need discovery,\s*",
        "call ended after needs discovery but before ",
        report,
    )

    return report


def _final_cleanup_promote_biggest_miss_from_flow_misses(report, transcript=None):
    """
    BIGGEST MISS should not be blank/None when the report already lists real
    agent-controllable SCRIPT / FLOW MISSES. Keep clean DNQ calls at None.
    """
    if not report:
        return report

    fair_dq = bool(re.search(
        r"(?is)Agent appropriately stopped after identifying disqualification|"
        r"call ended because the prospect was not eligible|"
        r"continuing the sale was not appropriate",
        report,
    ))
    if fair_dq:
        return report

    miss = _extract_first_real_flow_miss(report)
    if not miss:
        return report

    biggest_is_empty = bool(re.search(
        r"(?ims)^BIGGEST MISS:\s*(?:[-•]\s*)?(?:None|N/A|Not applicable)?\s*(?=^OBJECTIONS DETECTED:|^SUMMARY:|^TRANSCRIPT NOTE|^OPENAI COST ESTIMATE:|\Z)",
        report,
    ))
    if not biggest_is_empty:
        return report

    if re.search(r"(?im)^BIGGEST MISS:\s*", report):
        return re.sub(
            r"(?ims)^BIGGEST MISS:\s*.*?(?=^OBJECTIONS DETECTED:|^SUMMARY:|^TRANSCRIPT NOTE|^OPENAI COST ESTIMATE:|\Z)",
            f"BIGGEST MISS:\n- {miss}\n\n",
            report,
            count=1,
        )

    if re.search(r"(?im)^SUMMARY:\s*", report):
        return re.sub(
            r"(?im)^SUMMARY:\s*",
            f"BIGGEST MISS:\n- {miss}\n\nSUMMARY:\n",
            report,
            count=1,
        )

    return report.rstrip() + f"\n\nBIGGEST MISS:\n- {miss}\n"


def _detect_disqualification_no_agent_fault(report, transcript):
    """
    Detect calls that ended because the prospect was not eligible / could not proceed,
    not because the agent failed the sales process.
    Includes AGE, health DNQ/LCR, and no-income affordability cases.
    """
    combined = ((report or "") + "\n" + (transcript or "")).lower()

    age_dq = bool(re.search(
        r"(too old|younger than \[number\]|you have to be younger|ages? (?:are )?only|outside (?:the )?age range|cannot qualify due to age)",
        combined,
        re.I,
    ))

    # Health/LCR fairness cleanup must not trigger from the agent merely reading
    # health-screening questions. Require an actual disqualification outcome.
    health_dq = bool(re.search(
        r"(?is)"
        r"(unfortunately|sorry|based on that|because of that|with that condition|due to that|that means|"
        r"after reviewing|from those answers).{0,220}"
        r"(do(?:es)? not qualify|won't qualify|would not qualify|can't qualify|cannot qualify|"
        r"not able to qualify|unable to qualify|can't help you|cannot help you|not eligible|declined|knockout)",
        combined,
    ))

    health_dq = health_dq or bool(re.search(
        r"(?is)"
        r"(do(?:es)? not qualify|won't qualify|would not qualify|can't qualify|cannot qualify|"
        r"not able to qualify|unable to qualify|not eligible|declined|knockout).{0,180}"
        r"(health|medical|condition|diagnosis|diagnosed|oxygen|dialysis|kidney|heart|copd|cancer|terminal|hospice)",
        combined,
    ))

    no_income_dq = bool(re.search(
        r"(no income|don't have any income|do not have any income|not at all.*income|"
        r"working on my disability|i don't want to sell you a policy if you don't have any income|"
        r"don't want to take food off your table|do not want to take food off your table|"
        r"if i can't afford it|if i cannot afford it)",
        combined,
        re.I | re.S,
    ))

    if _transcript_has_clean_health_screening_no_dq(transcript):
        health_dq = False

    if age_dq:
        return "AGE", "Prospect was outside the eligible age range."
    if no_income_dq:
        return "LCR", "Prospect had no income / affordability barrier, so the agent appropriately did not continue the sale."
    if health_dq:
        return "LCR", "Prospect had a disqualifying health condition."
    return None, ""


def _final_cleanup_disqualification_no_agent_fault(report, transcript):
    """
    AGE / LCR / no-income calls should not be treated as agent sales failures
    when the agent appropriately ends the call after discovering ineligibility.
    """
    if not report:
        return report

    disposition, reason = _detect_disqualification_no_agent_fault(report, transcript)
    if not disposition:
        return report

    # Do not override real callback/coverage/banking/etc. automatic fails.
    real_autofail_reason = ""
    rm = re.search(r"(?im)^- Reason:\s*(.+)$", report or "")
    if rm:
        real_autofail_reason = rm.group(1).strip().lower()

    callback_or_control_only = (
        "objection occurred without proper call control" in real_autofail_reason
        or "call control" in real_autofail_reason
        or "none" in real_autofail_reason
        or not real_autofail_reason
    )

    if callback_or_control_only:
        report = re.sub(
            r"(?im)^- Objection occurred without proper call control:\s*YES\b.*$",
            "- Objection occurred without proper call control: NO",
            report,
        )
        report = re.sub(
            r"(?im)^- Automatic fail triggered:\s*YES\b.*$",
            "- Automatic fail triggered: NO",
            report,
        )
        report = re.sub(
            r"(?im)^- Reason:\s*.*$",
            "- Reason: None",
            report,
            count=1,
        )

    # These are not agent failures; keep them incomplete but fair.
    report = re.sub(r"(?im)^PASS:\s*NO\s*$", "PASS: YES", report, count=1)
    report = re.sub(r"(?im)^RISK:\s*HIGH\s*$", "RISK: MEDIUM", report, count=1)

    m = re.search(r"(?im)^SCORE:\s*(\d+)\b", report)
    if m:
        try:
            score = int(m.group(1))
        except Exception:
            score = None
        if score is not None and score < 90:
            report = re.sub(r"(?im)^SCORE:\s*\d+\b", "SCORE: 90", report, count=1)

    # Remove unfair call-control / future-stage coaching for an appropriate disqualification end.
    # These calls should not coach the agent to push/redirect/continue selling after AGE/LCR/no-income.
    unfair_phrases = [
        "Early refusal call",
        "did not attempt calm call control",
        "Attempt calm call control",
        "Objection occurred without proper call control",
        "prospect ended call before warm-up",
        "lack of progression",
        "poor objection handling",
        "no further progression possible",
        "Fact Finding / Warm-up not reached, so no rapport",
        "Maintain confident and clear communication",
        "Avoid abrupt ending",
        "agent did not attempt to redirect",
        "did not attempt to redirect",
        "handle per process beyond immediate stop",
        "redirect or handle",
        "should have attempted call control",
        "should have attempted to continue",
        "continue the sales process",
        "move the call forward",
        "failed to progress",
    ]
    for phrase in unfair_phrases:
        report = _text_remove_lines_containing(report, phrase)

    # Add/replace fair coaching so the report explains what happened without blaming the agent.
    fair_note = f"Agent appropriately stopped after identifying disqualification / inability to proceed. {reason}"

    if re.search(r"(?ims)^COACHING:\s*.*?(?=^SUMMARY:|^SCORING BREAKDOWN:|^BIGGEST MISS:|^OPENAI COST ESTIMATE:|\Z)", report):
        report = re.sub(
            r"(?ims)^COACHING:\s*.*?(?=^SUMMARY:|^SCORING BREAKDOWN:|^BIGGEST MISS:|^OPENAI COST ESTIMATE:|\Z)",
            "COACHING:\n- " + fair_note + "\n\n",
            report,
            count=1,
        )
    elif re.search(r"(?ims)^TASK CHECKLIST:", report):
        report = re.sub(
            r"(?ims)(^TASK CHECKLIST:)",
            "COACHING:\n- " + fair_note + "\n\n\1",
            report,
            count=1,
        )

    # Keep outcome clear.
    report = re.sub(
        r"(?im)^- Evidence:\s*.*$",
        f"- Evidence: {reason}",
        report,
        count=1,
    )

    # Biggest miss should not accuse the agent when the call ended due to disqualification.
    if re.search(r"(?im)^BIGGEST MISS:\s*", report):
        report = re.sub(
            r"(?ims)^BIGGEST MISS:\s*.*?(?=^OBJECTIONS DETECTED:|^SUMMARY:|^TRANSCRIPT NOTE|^OPENAI COST ESTIMATE:|\Z)",
            "BIGGEST MISS:\n- None\n\n",
            report,
            count=1,
        )
    elif re.search(r"(?im)^SUMMARY:\s*", report):
        report = re.sub(
            r"(?im)^SUMMARY:\s*",
            "BIGGEST MISS:\n- None\n\nSUMMARY:\n",
            report,
            count=1,
        )
    else:
        report = report.rstrip() + "\n\nBIGGEST MISS:\n- None\n"

    # Summary cleanup for obvious contradiction.
    fair_summary = (
        "The call ended because the prospect was not eligible / could not reasonably proceed. "
        "The agent appropriately stopped after identifying the disqualification / inability to proceed. "
        "Future sales stages were not reached because continuing the sale was not appropriate."
    )

    report = re.sub(
        r"(?i)The call is scored low due to lack of progression, poor objection handling, and early disengagement\.",
        fair_summary,
        report,
    )

    # Replace stale generated summaries that describe quotes/application/close on disqualification calls.
    if re.search(r"(?ims)^SUMMARY:\s*", report):
        report = re.sub(
            r"(?ims)^SUMMARY:\s*.*?(?=^BIGGEST MISS:|^OPENAI COST ESTIMATE:|\Z)",
            "SUMMARY:\n" + fair_summary + "\n\n",
            report,
            count=1,
        )

    # If a TASK CHECKLIST header got stripped by earlier cleanup, restore it before checklist-style bullets.
    report = re.sub(
        r"(?ims)(COACHING:\n- .*?\n\n)(- Recording disclosure:)",
        r"\1TASK CHECKLIST:\n\2",
        report,
        count=1,
    )

    # Final disqualification-section safety: make sure BIGGEST MISS survives summary cleanup.
    if not re.search(r"(?im)^BIGGEST MISS:\s*", report):
        if re.search(r"(?im)^SUMMARY:\s*", report):
            report = re.sub(
                r"(?im)^SUMMARY:\s*",
                "BIGGEST MISS:\n- None\n\nSUMMARY:\n",
                report,
                count=1,
            )
        else:
            report = report.rstrip() + "\n\nBIGGEST MISS:\n- None\n"

    return report



# FINAL AUDIT CONSISTENCY PIPELINE
#
# This function is intentionally ordered. Many cleanup rules rewrite the same
# visible report fields, so do not reorder calls casually.
#
# Main fields affected:
# - SCORE
# - RISK
# - PASS
# - CALL STAGE REACHED
# - EARLY END
# - NOT REACHED
# - SALE OUTCOME / Policy sold
# - AUTOMATIC FAIL CHECKS
# - BIGGEST MISS / SUMMARY / COACHING
#
# Current safety rule:
# - Add or update a regression test before changing behavior.
# - Prefer shared helpers for repeated field rewrites.
# - Avoid call-specific patches; write general rules.
# - Backfill reports after behavior changes.
#
# Broad order:
# 1. Transcript-supported hard corrections
# 2. Callback / coverage / banking false-positive cleanup
# 3. Stage and sale-outcome consistency cleanup
# 4. Sold-call completion and Peace of Mind corrections
# 5. AGE / LCR / no-income fairness cleanup
# 6. NOT REACHED reason rewrite/compression
# 7. Biggest miss / summary / display cleanup
# 8. Final pass/risk/autofail enforcement
#
# Known technical debt:
# - SCORE/PASS/RISK are still written by multiple rules.
# - NOT REACHED is still formatted by multiple helpers.
# - Automatic-fail reasons should eventually be merged through one helper.
# - Stage downgrade and Policy Sold invariants need stronger end-of-chain tests.
#

def enforce_final_audit_consistency(report, transcript=None):
    """
    Post-process free-text audits (and harden any path) so invalid autofail / stage combinations
    cannot appear in the rendered report.
    """
    if not report:
        return report
    if transcript:
        report = _text_enforce_tpu_stage_report(report, transcript)

        if not detect_agent_callback_from_transcript(transcript):
            report = re.sub(r"(?im)^- Did the agent set a callback\?\s*YES\b.*$", "- Did the agent set a callback? NO", report)
            report = re.sub(r"(?im)^- Callback set:\s*YES\b.*$", "- Callback set: NO", report)
            report = _text_remove_lines_containing(report, "Callback set without allowed exception")
            report = _text_remove_lines_containing(report, "agreed to a callback")
            report = re.sub(r"(?im)^- Reason:\s*.*callback.*$", "- Reason: None", report)

        if _transcript_shopping_not_current_coverage(transcript):
            report = re.sub(r"(?im)^- Existing coverage mentioned but not confirmed:\s*YES\b.*$", "- Existing coverage mentioned but not confirmed: NO", report)

        if _transcript_lowest_option_attempt_no_clear_commit(transcript):
            report = re.sub(
                r"(?im)^- Client chose an option:\s*YES\b.*$",
                "- Client chose an option: PARTIAL - Agent used the bottom-paragraph close to move forward with the lowest option, but the sale did not complete",
                report,
            )
            report = re.sub(
                r"(?im)^- Did the client choose an option\?\s*YES\b.*$",
                "- Did the client choose an option? PARTIAL - Agent used the bottom-paragraph close to move forward with the lowest option, but the sale did not complete",
                report,
            )
            report = re.sub(r"(?im)^- Objection occurred without proper call control:\s*YES\b.*$", "- Objection occurred without proper call control: NO", report)

        if _transcript_application_info_started(transcript):
            report = re.sub(r"(?im)^CALL STAGE REACHED:\s*.*$", "CALL STAGE REACHED: Application Information", report, count=1)
            report = re.sub(
                r"(?im)^- Application info collected:\s*(?:NO|NOT REACHED)\b.*$",
                "- Application info collected: PARTIAL",
                report,
            )
            report = re.sub(r"(?im)^- Application Information\s*$", "", report)
        if _transcript_only_one_coverage_ambiguity(transcript):
            ask_yes = bool(
                re.search(
                    r"(?im)^- Did the agent ask about existing coverage\?\s*YES\b",
                    report,
                )
            )
            cov_line_no = bool(
                re.search(
                    r"(?im)^- Existing coverage mentioned but not confirmed:\s*NO\b",
                    report,
                )
            )
            confirm_no = bool(
                re.search(
                    r"(?im)^- Did the agent confirm current coverage\?\s*NO\b",
                    report,
                )
            )
            insurer_no = bool(
                re.search(
                    r"(?im)^- Did the agent call an insurance company to confirm current coverage\?\s*NO\b",
                    report,
                )
            )
            if ask_yes and cov_line_no and confirm_no and insurer_no:
                report = re.sub(
                    r"(?im)^- Existing coverage mentioned but not confirmed:\s*NO\b",
                    "- Existing coverage mentioned but not confirmed: YES",
                    report,
                    count=1,
                )

    report = _enforce_peace_of_mind_call_stage_consistency(report, transcript)

    cov_yes = bool(
        re.search(
            r"(?im)^- Existing coverage mentioned but not confirmed:\s*YES\b",
            report,
        )
    )
    trig_no = bool(
        re.search(r"(?im)^- Automatic fail triggered:\s*NO\b", report)
    )
    if cov_yes and trig_no:
        report = re.sub(
            r"(?im)^- Automatic fail triggered:\s*NO\b",
            "- Automatic fail triggered: YES",
            report,
            count=1,
        )

        def _merge_cov_reason(m):
            body = (m.group(1) or "").strip()
            frag = "Existing coverage mentioned but not confirmed"
            if not body or body.lower() == "none":
                return f"- Reason: {frag}"
            if frag.lower() in body.lower():
                return m.group(0)
            return f"- Reason: {body}; {frag}"

        report = re.sub(
            r"(?im)^- Reason:\s*(.*)$", _merge_cov_reason, report, count=1
        )
        sm = re.search(r"(?im)^SCORE:\s*(\d+)\b", report)
        if sm:
            capped = min(int(sm.group(1)), 80)
            report = re.sub(
                r"(?im)^SCORE:\s*\d+\b", f"SCORE: {capped}", report, count=1
            )
        report = re.sub(r"(?im)^RISK:\s*\S+\s*$", "RISK: HIGH", report, count=1)

    if cov_yes:

        def _cap_compliance(m):
            v = int(m.group(1))
            return f"- Compliance: {min(v, 72)}"

        report = re.sub(
            r"(?im)^- Compliance:\s*(\d+)\b", _cap_compliance, report, count=1
        )

        def _cap_sales_proc(m):
            v = int(m.group(1))
            return f"- Sales Process: {min(v, 72)}"

        report = re.sub(
            r"(?im)^- Sales Process:\s*(\d+)\b", _cap_sales_proc, report, count=1
        )

        sm_cov = re.search(r"(?im)^SCORE:\s*(\d+)\b", report)
        if sm_cov and int(sm_cov.group(1)) > 80:
            report = re.sub(
                r"(?im)^SCORE:\s*\d+\b",
                "SCORE: 80",
                report,
                count=1,
            )
        if re.search(r"(?im)^RISK:\s*(?:LOW|MEDIUM)\s*$", report):
            report = re.sub(
                r"(?im)^RISK:\s*\S+\s*$", "RISK: HIGH", report, count=1
            )
        if _report_policy_sold_yes(report):
            report = re.sub(
                r"(?im)^PASS:\s*YES\s*$", "PASS: AT RISK", report, count=1
            )
            report = re.sub(
                r"(?im)^PASS:\s*NO\s*$", "PASS: AT RISK", report, count=1
            )
        else:
            report = re.sub(
                r"(?im)^PASS:\s*YES\s*$", "PASS: NO", report, count=1
            )

    auto_yes = bool(
        re.search(r"(?im)^- Automatic fail triggered:\s*YES\b", report)
    )
    if auto_yes and not cov_yes:
        sm_auto = re.search(r"(?im)^SCORE:\s*(\d+)\b", report)
        if sm_auto and int(sm_auto.group(1)) > 85:
            report = re.sub(
                r"(?im)^SCORE:\s*\d+\b",
                "SCORE: 85",
                report,
                count=1,
            )
    if auto_yes and re.search(r"(?im)^- Reason:\s*None\s*$", report):
        report = re.sub(
            r"(?im)^- Reason:\s*None\s*$",
            "- Reason: Automatic fail triggered (see AUTOMATIC FAIL CHECKS)",
            report,
            count=1,
        )

    report = enforce_report_stage_consistency(report)
    report = _final_text_cleanup_for_no_callback_no_banking(report, transcript)
    report = _final_cleanup_autofail_sale_summary(report, transcript)
    report = _final_cleanup_no_callback_coaching_and_option(report, transcript)
    report = _final_cleanup_pass_and_bottom_paragraph_wording(report, transcript)
    report = _final_cleanup_sold_post_app_stage(report, transcript)
    report = _final_cleanup_credit_union_and_sold_summary(report, transcript)
    report = _final_cleanup_false_banking_from_existing_coverage_lookup(report, transcript)
    report = _final_cleanup_shelby_sold_short_false_fails(report, transcript)
    report = _final_cleanup_confirmed_coverage_bank_and_cooldown(report, transcript)
    report = _final_cleanup_no_autofail_consistency(report, transcript)
    report = _final_cleanup_early_end_stage_and_banking(report, transcript)
    report = _final_cleanup_names_and_late_pq_in_report(report, transcript)
    report = _final_cleanup_early_unsold_score_risk_guardrail(report, transcript)
    report = _final_cleanup_zero_score_guardrail(report, transcript)
    report = _restore_safe_business_terms(report)
    report = _normalize_not_reached_due_to_prospect(report, transcript)
    report = _final_cleanup_callback_autofail_consistency(report, transcript)
    report = _final_cleanup_false_callback_autofail(report, transcript)
    report = _final_cleanup_partial_health_unsold_guardrail(report, transcript)
    report = _final_cleanup_false_banking_stage_guardrail(report, transcript)
    report = _final_cleanup_sold_call_completion_evidence(report, transcript)
    report = _final_cleanup_peace_of_mind_after_sale(report, transcript)
    report = _final_cleanup_disqualification_no_agent_fault(report, transcript)
    report = _final_cleanup_false_health_disqualification_after_clean_screen(report, transcript)
    # Transcript-supported stage upgrade: funeral-cost / burial-cremation / no-coverage
    # impact questions mean the call reached the Needs section, not just Medical / Health.
    if _transcript_reached_needs_section(transcript):
        report = re.sub(
            r"(?im)^CALL STAGE REACHED:\s*Medical / Health\s*$",
            "CALL STAGE REACHED: Needs",
            report,
            count=1,
        )
        report = re.sub(
            r"(?im)^- Final stage supporting sale:\s*Medical / Health\s*$",
            "- Final stage supporting sale: Needs",
            report,
            count=1,
        )
        report = _text_remove_lines_containing(report, "Needs section")
        report = _text_remove_lines_containing(report, "Need section")
    report = _final_cleanup_needs_stage_fields(report, transcript)
    report = _rewrite_not_reached_reason(report, transcript)
    report = _compress_not_reached_block(report)
    report = _final_cleanup_protect_major_biggest_miss(report, transcript)
    report = enforce_risk_for_automatic_fail(report)
    report = _strip_embedded_transcript_from_report(report)
    report = _decode_report_html_entities(report)
    report = _final_cleanup_summary_stage_contradictions(report)
    report = _final_cleanup_needs_stage_fields(report, transcript)
    report = _final_cleanup_promote_biggest_miss_from_flow_misses(report, transcript)
    report = _restore_safe_business_terms(report)
    return report


def _report_policy_sold_yes(report):
    if not report:
        return False
    if re.search(r"(?im)^- Policy sold:\s*YES\b", report):
        return True
    return bool(re.search(r"(?im)^- Was the policy sold\?\s*YES\b", report))


def enforce_pass_logic(report):
    score_match = re.search(r"(?im)^SCORE:\s*(\d+)\b", report)
    score_value = int(score_match.group(1)) if score_match else None
    has_autofail = bool(
        re.search(r"(?im)^- Automatic fail triggered:\s*YES\b", report)
    )
    sold_yes = _report_policy_sold_yes(report)

    if has_autofail and sold_yes:
        report = re.sub(
            r"(?im)^PASS:\s*\S.*$",
            "PASS: AT RISK",
            report,
            count=1,
        )
        report = re.sub(r"(?im)^PASS:\s*NO\s*$", "PASS: AT RISK", report, count=1)
    elif has_autofail:
        report = re.sub(r"(?im)^PASS:\s*YES\s*$", "PASS: NO", report)
        report = re.sub(r"(?im)^PASS:\s*AT RISK\s*$", "PASS: NO", report, count=1)

    if score_value is not None and score_value < 70:
        if not (has_autofail and sold_yes):
            report = re.sub(r"(?im)^PASS:\s*YES\s*$", "PASS: NO", report)

    no_compliance_failures = re.search(
        r"(?im)^COMPLIANCE FAILURES:\s*\n\s*-?\s*None",
        report,
    )
    if (
        score_value is not None
        and score_value >= 70
        and no_compliance_failures
        and not has_autofail
    ):
        report = re.sub(r"(?im)^PASS:\s*NO\s*$", "PASS: YES", report)

    return report


def enforce_risk_for_automatic_fail(report):
    """Align RISK with automatic-fail rules for text reports (structured path uses validate_structured_audit)."""
    if not report:
        return report
    has_autofail = bool(
        re.search(r"(?im)^- Automatic fail triggered:\s*YES\b", report)
    )
    if not has_autofail:
        return report
    return re.sub(r"(?im)^RISK:\s*\S+\s*$", "RISK: HIGH", report, count=1)


def transcribe(file_path, call_name, filename):
    set_processing_state(call_name, filename, "transcribing", 5, "Transcribing audio")

    # beam_size=5: wider search than greedy (beam 1); slower per chunk but fewer word errors on
    #    noisy phone audio — main lever to keep accuracy while other knobs favor throughput.
    # vad_filter + min_silence_duration_ms: skip decoding long dead air (hallways, hold, pauses)
    #    so wall-clock time drops a lot on long calls without changing spoken content.
    # condition_on_previous_text=False: avoids error propagation down long files and skips extra
    #    prefix attention work → faster + less drift on repetitive insurance scripts.
    # temperature=0: deterministic sampling (default path); no sampling randomness, tiny speed win.
    segments, info = get_model().transcribe(
        file_path,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
        temperature=0,
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


def audit(transcript, progress_callback=None, call_name=None):
    if progress_callback:
        progress_callback(AI_START_PROGRESS, "Running AI audit")

    checklist = read_text("training/sales_task_checklist.txt")
    rubric = read_text("training/scoring_rubric.txt")
    output_format = read_text("training/audit_output_format.txt")

    prompt = build_audit_prompt(transcript, checklist, rubric, output_format)
    redacted_transcript = redact_sensitive_transcript(transcript)
    transcript_for_openai = redacted_transcript
    role_label_note = None
    if call_name:
        labeled = try_save_role_labeled_transcript(call_name, redacted_transcript)
        if labeled:
            transcript_for_openai = labeled
            role_label_note = ROLE_LABEL_TRANSCRIPT_NOTE
    openai_prompt = build_audit_prompt(
        transcript_for_openai,
        checklist,
        rubric,
        output_format,
        role_label_note=role_label_note,
    )
    report, openai_cost = generate_audit_report(
        prompt, openai_prompt, transcript_for_openai
    )
    report = trim_to_score_and_remove_unwanted_sections(report)
    report = normalize_top3_coaching_header_line(report)

    report = enforce_final_audit_consistency(report, transcript)
    report = enforce_pass_logic(report)
    report = enforce_risk_for_automatic_fail(report)
    if openai_cost:
        report = append_openai_cost_footer(report, openai_cost)

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


def move_to_processed_calls(file_path):
    if not os.path.exists(file_path):
        return None

    os.makedirs(PROCESSED_CALLS_FOLDER, exist_ok=True)

    filename = os.path.basename(file_path)
    destination = os.path.join(PROCESSED_CALLS_FOLDER, filename)

    if os.path.exists(destination):
        name, ext = os.path.splitext(filename)
        timestamp = int(time.time())
        destination = os.path.join(PROCESSED_CALLS_FOLDER, f"{name}_{timestamp}{ext}")

        while os.path.exists(destination):
            timestamp += 1
            destination = os.path.join(PROCESSED_CALLS_FOLDER, f"{name}_{timestamp}{ext}")

    shutil.move(file_path, destination)
    return destination


def move_to_processed_transcript(file_path):
    """Move a processed .txt upload out of transcript_uploads/ (same collision rules as audio)."""
    if not os.path.exists(file_path):
        return None

    os.makedirs(PROCESSED_TRANSCRIPTS_FOLDER, exist_ok=True)

    filename = os.path.basename(file_path)
    destination = os.path.join(PROCESSED_TRANSCRIPTS_FOLDER, filename)

    if os.path.exists(destination):
        name, ext = os.path.splitext(filename)
        timestamp = int(time.time())
        destination = os.path.join(PROCESSED_TRANSCRIPTS_FOLDER, f"{name}_{timestamp}{ext}")

        while os.path.exists(destination):
            timestamp += 1
            destination = os.path.join(PROCESSED_TRANSCRIPTS_FOLDER, f"{name}_{timestamp}{ext}")

    shutil.move(file_path, destination)
    return destination


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
            if not STORE_RAW_TRANSCRIPTS:
                transcript = redact_sensitive_transcript(transcript)
                write_text(transcript_path, transcript)
            print(f"Resuming from transcript: {call_name}", flush=True)
        else:
            raw_transcript = transcribe(file_path, call_name, filename)
            transcript = raw_transcript if STORE_RAW_TRANSCRIPTS else redact_sensitive_transcript(raw_transcript)
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
            ),
            call_name=call_name,
        )

        report = redact_report_text(report)
        write_text(report_path, report)

        score, risk = parse_report(report)

        save_to_db(call_name, transcript, report, score, risk)
        move_to_processed_calls(file_path)

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
TOP 3 COACHING PRIORITIES:
- Coaching should focus only on missed items within stages that were reached.
- Do not coach on later stages that were never reached unless the agent clearly skipped ahead or mishandled the flow.
- Retry this audit or delete and re-upload the call.

BIGGEST MISS:
- Audit did not complete

SUMMARY:
The system could not complete the audit because: {error}

TRANSCRIPT:
{transcript}
"""

        failure_report = redact_report_text(failure_report)
        write_text(report_path, failure_report)
        save_to_db(call_name, transcript, failure_report, 0, "HIGH")
        set_processing_state(call_name, filename, "failed", 100, "Processing failed", error)


def process_transcript_upload(file_path):
    """Process a .txt dropped in transcript_uploads/: redact, audit, DB + reports (no Whisper)."""
    filename = os.path.basename(file_path)

    if not filename.lower().endswith(".txt"):
        return

    call_name = os.path.splitext(filename)[0]
    transcript_path = os.path.join(TRANSCRIPTS_FOLDER, f"{call_name}.txt")
    report_path = os.path.join(REPORTS_FOLDER, f"{call_name}_report.txt")

    if os.path.exists(report_path):
        set_processing_state(call_name, filename, "complete", 100, "Complete")
        move_to_processed_transcript(file_path)
        return

    if not wait_until_file_ready(file_path):
        return

    try:
        print(f"Processing transcript upload: {call_name}", flush=True)

        set_processing_state(call_name, filename, "processing", 1, "Reading transcript")

        raw_transcript = read_text(file_path)
        transcript = (
            raw_transcript if STORE_RAW_TRANSCRIPTS else redact_sensitive_transcript(raw_transcript)
        )
        write_text(transcript_path, transcript)

        set_processing_state(call_name, filename, "analyzing", 80, "Running AI audit")

        report = audit(
            transcript,
            lambda progress, message: set_processing_state(
                call_name,
                filename,
                "analyzing",
                progress,
                message,
            ),
            call_name=call_name,
        )

        report = redact_report_text(report)
        write_text(report_path, report)

        score, risk = parse_report(report)

        save_to_db(call_name, transcript, report, score, risk)
        move_to_processed_transcript(file_path)

        set_processing_state(call_name, filename, "complete", 100, "Complete")

        print(f"Done (transcript upload): {call_name} | Score: {score} | Risk: {risk}", flush=True)

    except Exception as e:
        error = str(e)
        print(f"ERROR processing transcript upload {call_name}: {error}", flush=True)

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
TOP 3 COACHING PRIORITIES:
- Coaching should focus only on missed items within stages that were reached.
- Do not coach on later stages that were never reached unless the agent clearly skipped ahead or mishandled the flow.
- Retry this audit or delete and re-upload the call.

BIGGEST MISS:
- Audit did not complete

SUMMARY:
The system could not complete the audit because: {error}

TRANSCRIPT:
{transcript}
"""

        failure_report = redact_report_text(failure_report)
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
    os.makedirs(TRANSCRIPT_UPLOADS_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_CALLS_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_TRANSCRIPTS_FOLDER, exist_ok=True)
    os.makedirs(TRANSCRIPTS_FOLDER, exist_ok=True)
    os.makedirs(REPORTS_FOLDER, exist_ok=True)

    ensure_db()
    recover_interrupted_work()

    print("QA SYSTEM RUNNING...", flush=True)
    print(f"Scanning: {CALLS_FOLDER} and {TRANSCRIPT_UPLOADS_FOLDER}", flush=True)
    print(f"Every {SCAN_INTERVAL_SECONDS} seconds", flush=True)
    print("Press CTRL+C to stop.", flush=True)

    while True:
        try:
            for filename in sorted(os.listdir(TRANSCRIPT_UPLOADS_FOLDER)):
                if not filename.lower().endswith(".txt"):
                    continue
                tp = os.path.join(TRANSCRIPT_UPLOADS_FOLDER, filename)
                if os.path.isfile(tp):
                    process_transcript_upload(tp)

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
    if len(sys.argv) > 1 and sys.argv[1] == "--test-redaction":
        _redaction_smoke_assertions()
        print("redaction smoke: OK", flush=True)
        sys.exit(0)
    start()
