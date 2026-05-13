from flask import (
    Flask,
    jsonify,
    make_response,
    redirect,
    send_file,
    render_template_string,
    request,
    session,
    url_for,
)
from html import escape
import json
import sqlite3
import os
import re
from pathlib import Path
import time
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from urllib.parse import quote
from db_migrations import migrate_database

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI as _OpenAIClient
except ImportError:
    _OpenAIClient = None

app = Flask(__name__)
app.secret_key = os.getenv("AI_AUDITOR_SECRET_KEY", "dev-change-this-secret")
app.permanent_session_lifetime = timedelta(days=30)

DASHBOARD_USERNAME = os.getenv("AI_AUDITOR_USERNAME", "admin")
DASHBOARD_PASSWORD = os.getenv("AI_AUDITOR_PASSWORD", "change-me")

UPLOAD_FOLDER = "calls"
PROCESSED_CALLS_FOLDER = "processed_calls"
TRANSCRIPT_UPLOAD_FOLDER = "transcript_uploads"
TRANSCRIPTS_FOLDER = "transcripts"
TRANSCRIPTS_ROLE_LABELED_FOLDER = "transcripts_role_labeled"
REPORTS_FOLDER = "reports"
DB_FILE = "calls.db"
PORT = 5050
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_QA_MODEL = os.getenv("OPENAI_QA_MODEL", OPENAI_MODEL)

GOLDEN_CASES_FILE = "golden_cases/golden18.json"


def golden_call_names():
    """Names/prefixes for test fixture calls that should not appear in normal dashboard workflows."""
    try:
        with open(GOLDEN_CASES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set()

    names = set()
    for case in data.get("cases", []):
        match = str(case.get("match") or "").strip()
        if match:
            names.add(match)
    return names


def is_golden_call_name(call_name):
    call_name = str(call_name or "").strip()
    if not call_name:
        return False

    for golden in golden_call_names():
        if call_name == golden or call_name.startswith(golden):
            return True
    return False



TEST_FIXTURE_SOURCE_FILES = [
    "tests/test_audit_guardrails.py",
    "tests/check_golden_reports.py",
    "tests/test_dashboard_guardrails.py",
]


def test_fixture_call_names():
    """
    Discover call fixture basenames referenced by tests so dashboard delete/list
    flows cannot accidentally remove files the regression suite depends on.
    """
    names = set()

    for rel_path in TEST_FIXTURE_SOURCE_FILES:
        path = Path(rel_path)
        if not path.exists():
            continue

        text = path.read_text(errors="ignore")

        # Common fixture patterns:
        #   name = "sold_clean_call"
        #   run_disposition_case("lcr_cancer", "LCR")
        #   Path("transcripts/health_questions_poor_call_control.txt")
        #   Path("reports/foo_report.txt")
        names.update(re.findall(r'(?m)^\s*name\s*=\s*"([^"]+)"', text))
        names.update(re.findall(r'run_disposition_case\("([^"]+)"', text))
        names.update(re.findall(r'Path\("transcripts/([^"]+)\.txt"\)', text))
        names.update(re.findall(r'Path\("reports/([^"]+)_report\.txt"\)', text))

    return sorted(names)




def is_protected_call_name(call_name):
    if is_golden_call_name(call_name):
        return True
    call_name = call_name or ""
    return any(call_name == name or call_name.startswith(name) for name in test_fixture_call_names())


def protect_golden_call_redirect(call_name, target="/"):
    if is_protected_call_name(call_name):
        return redirect(f"{target}?error=Golden%20test%20fixture%20calls%20are%20protected.")
    return None

ASK_CONTEXT_CHAR_LIMIT = 14000
AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a")
STATUS_LABELS = {
    "queued": "Queued",
    "processing": "Queued",
    "retry": "Queued",
    "transcribing": "Transcribing",
    "analyzing": "Analyzing",
    "complete": "Complete",
    "failed": "Failed",
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_CALLS_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPT_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPTS_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPTS_ROLE_LABELED_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)


def get_downloads_folder():
    path = os.path.expanduser("~/Downloads")
    os.makedirs(path, exist_ok=True)
    return path


def ensure_upload_times_table():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS upload_times (
        filename TEXT PRIMARY KEY,
        uploaded_time INTEGER NOT NULL
    )
    """)
    conn.commit()
    conn.close()


def ensure_processing_state_table():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
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

    c.execute("PRAGMA table_info(processing_state)")
    columns = {row[1] for row in c.fetchall()}

    if "progress" not in columns:
        c.execute("ALTER TABLE processing_state ADD COLUMN progress INTEGER NOT NULL DEFAULT 0")

    if "message" not in columns:
        c.execute("ALTER TABLE processing_state ADD COLUMN message TEXT")

    conn.commit()
    conn.close()


def ensure_calls_table_schema():
    migrate_database(DB_FILE)


CALL_ROW_FIELDS = {
    "id": 0,
    "call_name": 1,
    "transcript": 2,
    "report": 3,
    "score": 4,
    "risk": 5,
    "timestamp": 6,
    "auto_disposition": 7,
    "manual_disposition": 8,
    "final_disposition": 9,
    "disposition_reason": 10,
    "duration_seconds": 11,
}


def call_field(call, field_name, default=""):
    """
    Read a calls-table field by name.

    Keeps dashboard code from depending on raw tuple indexes everywhere while
    remaining compatible with current sqlite tuple rows.
    """
    if call is None:
        return default

    try:
        value = call[field_name]
        return value if value is not None else default
    except Exception:
        pass

    idx = CALL_ROW_FIELDS.get(field_name)
    if idx is None:
        return default

    try:
        value = call[idx]
        return value if value is not None else default
    except Exception:
        return default


def call_row_id(call):
    return call_field(call, "id", None)


def call_row_name(call):
    return call_field(call, "call_name", "")


def call_row_report(call):
    return call_field(call, "report", "")


def call_row_score(call):
    return call_field(call, "score", None)


def call_row_timestamp(call):
    return call_field(call, "timestamp", None)


def get_stable_file_time(file_path):
    stat = os.stat(file_path)
    return int(getattr(stat, "st_birthtime", stat.st_mtime))


def get_upload_times():
    ensure_upload_times_table()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT filename, uploaded_time FROM upload_times")
    rows = c.fetchall()
    conn.close()

    return {filename: int(uploaded_time) for filename, uploaded_time in rows}


def remember_upload_time(filename, uploaded_time):
    ensure_upload_times_table()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    INSERT OR REPLACE INTO upload_times (filename, uploaded_time)
    VALUES (?, ?)
    """, (filename, int(uploaded_time)))
    conn.commit()
    conn.close()


def forget_upload_times(filenames):
    filenames = list(filenames)
    if not filenames:
        return

    ensure_upload_times_table()

    placeholders = ",".join(["?"] * len(filenames))
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(f"DELETE FROM upload_times WHERE filename IN ({placeholders})", filenames)
    conn.commit()
    conn.close()


def get_processing_states():
    ensure_processing_state_table()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        SELECT call_name, status, progress, message, error
        FROM processing_state
    """)
    rows = c.fetchall()
    conn.close()

    return {
        call_name: {
            "status": status,
            "progress": max(0, min(100, int(progress or 0))),
            "message": message,
            "error": error,
        }
        for call_name, status, progress, message, error in rows
    }


def remember_processing_state(call_name, filename, status, progress=0, message=None):
    ensure_processing_state_table()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT attempts FROM processing_state WHERE call_name=?", (call_name,))
    row = c.fetchone()

    if row:
        c.execute("""
            UPDATE processing_state
            SET filename=?,
                status=?,
                progress=?,
                message=?,
                error=NULL,
                updated_at=CURRENT_TIMESTAMP
            WHERE call_name=?
        """, (filename, status, int(progress), message, call_name))
    else:
        c.execute("""
            INSERT INTO processing_state
                (call_name, filename, status, progress, message)
            VALUES (?, ?, ?, ?, ?)
        """, (call_name, filename, status, int(progress), message))

    conn.commit()
    conn.close()


def forget_processing_states(call_names):
    call_names = list(call_names)
    if not call_names:
        return

    ensure_processing_state_table()

    placeholders = ",".join(["?"] * len(call_names))
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(f"DELETE FROM processing_state WHERE call_name IN ({placeholders})", call_names)
    conn.commit()
    conn.close()


def display_processing_state(call_name, transcript_exists, state):
    if state:
        raw_status = str(state["status"] or "queued").lower()
        status = STATUS_LABELS.get(raw_status, "Queued")
        progress = state["progress"]
        message = state["message"] or status
        return status, progress, message

    if transcript_exists:
        return "Analyzing", 75, "Transcription complete"

    return "Queued", 0, "Waiting for watcher"


def get_calls(include_golden=False):
    ensure_calls_table_schema()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM calls ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()

    if include_golden:
        return rows

    return [row for row in rows if not is_protected_call_name(call_row_name(row))]


def get_call(call_id):
    ensure_calls_table_schema()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM calls WHERE id=?", (call_id,))
    row = c.fetchone()
    conn.close()
    return row



def load_agent_map():
    """Load Vicidial agent number -> agent name mappings from training/agent_map.txt."""
    mapping = {}
    map_path = os.path.join("training", "agent_map.txt")
    if not os.path.exists(map_path):
        return mapping

    try:
        with open(map_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                number, name = line.split("=", 1)
                number = number.strip()
                name = name.strip()
                if number and name:
                    mapping[number] = name
    except Exception as e:
        print(f"Could not load agent map: {e}", flush=True)

    return mapping


def detect_agent_number_from_call_name(call_name):
    """
    Detect Vicidial agent number from filenames like:
    PJEK_AZ_20260428-120331_175_4802331452-all.mp3

    The agent number is the numeric field immediately after the timestamp.
    """
    name = os.path.basename(str(call_name or ""))

    match = re.search(r"\d{8}-\d{6}_(\d{2,6})(?:_|-|$)", name)
    if match:
        return match.group(1)

    # Fallback: split on underscores and look for the token after a timestamp token.
    parts = name.split("_")
    for idx, part in enumerate(parts[:-1]):
        if re.fullmatch(r"\d{8}-\d{6}", part):
            nxt = parts[idx + 1]
            if re.fullmatch(r"\d{2,6}", nxt):
                return nxt

    return ""


def detect_agent_name_from_call_name(call_name):
    agent_number = detect_agent_number_from_call_name(call_name)
    if not agent_number:
        return "Unknown"

    agent_map = load_agent_map()
    return agent_map.get(agent_number, f"Unknown {agent_number}")




def all_known_agent_names():
    """
    Agents to show in the dashboard sidebar even when they have 0 completed calls.
    Uses agent_map.json values, plus any agents inferred from calls.
    """
    agent_map = load_agent_map()
    names = set()
    for name in agent_map.values():
        if name:
            names.add(str(name).strip())
    return sorted(n for n in names if n)


def build_agent_sidebar_html(calls):
    """
    Sidebar buttons for filtering completed audits by inferred agent.
    Shows all known agents even when they currently have 0 completed calls.
    """
    counts = {}
    for c in calls:
        agent = detect_agent_name_from_call_name(c[1])
        counts[agent] = counts.get(agent, 0) + 1

    names = set(all_known_agent_names())
    names.update(counts.keys())

    parts = [
        '<button type="button" class="agent-filter-button active" data-agent="all">',
        f'<span>All Calls</span><span class="agent-count">{len(calls)}</span>',
        '</button>',
    ]

    for agent in sorted(names):
        agent_esc = escape(agent)
        count = counts.get(agent, 0)
        zero_cls = " zero" if count == 0 else ""
        parts.append(
            f'<button type="button" class="agent-filter-button{zero_cls}" data-agent="{agent_esc}">'
            f'<span>{agent_esc}</span><span class="agent-count">{count}</span>'
            f'</button>'
        )

    return "\n".join(parts)


def build_agent_filter_options_html(calls):
    names = set(all_known_agent_names())
    names.update(detect_agent_name_from_call_name(c[1]) for c in calls)
    options = ['<option value="all">All agents</option>']
    for name in sorted(names):
        safe = escape(name)
        options.append(f'<option value="{safe}">{safe}</option>')
    return "\n".join(options)

def render_report_html(report_text):
    """
    Render report text safely while coloring speaker labels anywhere they appear,
    especially in the embedded TRANSCRIPT section of a report.
    """
    if report_text is None:
        report_text = ""

    html_lines = []
    for raw_line in report_text.splitlines():
        line = raw_line.rstrip()

        if not line:
            html_lines.append("")
            continue

        m = re.match(r"^\s*(PQ|Agent|Prospect|Unknown)\s*:\s*(.*)$", line)
        if m:
            speaker = m.group(1)
            content = escape(m.group(2))
            cls = {
                "PQ": "speaker-pq",
                "Agent": "speaker-agent",
                "Prospect": "speaker-prospect",
                "Unknown": "speaker-unknown",
            }.get(speaker, "speaker-unknown")
            html_lines.append(f'<span class="speaker-label {cls}">{speaker}:</span> {content}')
        else:
            html_lines.append(escape(line))

    return "\n".join(html_lines)


def render_transcript_html(transcript_text):
    """
    Render transcript text with styled speaker labels.
    Safe: escapes all content before adding label markup.
    """
    if transcript_text is None:
        transcript_text = ""

    html_lines = []
    for raw_line in transcript_text.splitlines():
        line = raw_line.rstrip()

        if not line:
            html_lines.append("")
            continue

        m = re.match(r"^\s*(PQ|Agent|Prospect|Unknown)\s*:\s*(.*)$", line)
        if m:
            speaker = m.group(1)
            content = escape(m.group(2))
            cls = {
                "PQ": "speaker-pq",
                "Agent": "speaker-agent",
                "Prospect": "speaker-prospect",
                "Unknown": "speaker-unknown",
            }.get(speaker, "speaker-unknown")
            html_lines.append(f'<span class="speaker-label {cls}">{speaker}:</span> {content}')
        else:
            html_lines.append(escape(line))

    return "\n".join(html_lines)




def get_call_audio_path(call_name):
    """Find the original audio for a processed or currently queued call."""
    if not call_name:
        return None

    for folder in [PROCESSED_CALLS_FOLDER, UPLOAD_FOLDER]:
        for ext in AUDIO_EXTENSIONS:
            p = os.path.join(folder, f"{call_name}{ext}")
            if os.path.isfile(p):
                return p

    # Fallback for rare collision-renamed processed files.
    safe_prefix = f"{call_name}_"
    for folder in [PROCESSED_CALLS_FOLDER, UPLOAD_FOLDER]:
        if not os.path.isdir(folder):
            continue
        for filename in os.listdir(folder):
            lower = filename.lower()
            if not lower.endswith(AUDIO_EXTENSIONS):
                continue
            base, _ = os.path.splitext(filename)
            if base == call_name or base.startswith(safe_prefix):
                p = os.path.join(folder, filename)
                if os.path.isfile(p):
                    return p

    return None


def audio_mime_type(path):
    ext = os.path.splitext(path or "")[1].lower()
    if ext == ".mp3":
        return "audio/mpeg"
    if ext == ".wav":
        return "audio/wav"
    if ext == ".m4a":
        return "audio/mp4"
    return "application/octet-stream"


def get_saved_transcript_path(call_name):
    """Prefer role-labeled transcript when available; fall back to raw redacted transcript."""
    role_labeled_path = os.path.join(TRANSCRIPTS_ROLE_LABELED_FOLDER, f"{call_name}.txt")
    raw_path = os.path.join(TRANSCRIPTS_FOLDER, f"{call_name}.txt")

    if os.path.isfile(role_labeled_path):
        return role_labeled_path

    if os.path.isfile(raw_path):
        return raw_path

    return None


def get_saved_transcript_kind(call_name):
    path = get_saved_transcript_path(call_name)
    if not path:
        return "missing"
    if os.path.basename(os.path.dirname(path)) == TRANSCRIPTS_ROLE_LABELED_FOLDER:
        return "role-labeled"
    return "raw"


def get_saved_transcript(call_name):
    transcript_path = get_saved_transcript_path(call_name)
    if not transcript_path:
        return None
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def get_saved_report(call_name):
    report_path = os.path.join(REPORTS_FOLDER, f"{call_name}_report.txt")
    if not os.path.exists(report_path):
        return None
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def _ask_qa_strict_rules_block():
    return "\n\n".join(
        [
            "STRICT OUTPUT FORMAT (exact labels, in this order — no other prefix lines before Answer:):",
            "Answer: YES or NO or UNCLEAR",
            "Evidence: <short exact quote from TRANSCRIPT or REPORT below, or the word None>",
            "Explanation: <brief explanation; apply strict definitions below>",
            "",
            "GENERAL RULES:",
            "- Evidence must directly support the Answer under the strict definition for that topic.",
            "- Prefer TRANSCRIPT over REPORT for what happened on the call; use REPORT only as secondary context.",
            "- If no valid evidence exists: Evidence: None",
            "- Do not invent evidence or use outside knowledge.",
            "- Never echo full account/routing numbers, SSNs, DOBs, or other sensitive data; say redacted banking/account information.",
            "",
            "EVIDENCE DISCIPLINE (invalid as cited proof for the named topic):",
            "- Medical questions ≠ coverage confirmation.",
            "- PQ talking about the company ≠ coverage confirmation.",
            "- Bank verification ≠ insurance coverage confirmation.",
            "- Polite goodbye ≠ Cool Down.",
            "- Generic friendliness ≠ call control without an objection.",
            "",
            "1) 3 AND 1 METHOD (Fact Finding / Warm-up):",
            "YES only if transcript shows meaningful rapport/fact-finding across major areas (location, work/past work, spouse/relationship, children/family/someone important) AND personal/relatable self-disclosure tied to the prospect's answers.",
            "If only questions, no tied self-share: Answer NO (you may note mixed/partial only inside Explanation).",
            "If mixed: Answer UNCLEAR; explain gaps.",
            "Invalid as sole proof: medical/underwriting only, product explanation, generic politeness, agent saying great/okay without personal sharing.",
            "",
            "2) CURRENT COVERAGE CONFIRMATION:",
            "Means verification with insurance company, carrier, policy provider, or equivalent third-party source.",
            "Valid: carrier call/rep on line, direct policy lookup with carrier, in-force status from carrier, carrier states no policy/inactive.",
            "Invalid: PQ/company intro, who I am, medical/health, product pitch, prospect-only Q&A about coverage, application details, banking/routing/payment, bank/CU calls, Social Security deposit timing alone, payment date discussion alone.",
            "If only prospect was asked/answered without carrier verification: Answer NO; quote the exchange if useful; explain no carrier/provider verification.",
            "",
            "3) EXISTING COVERAGE MENTIONED BUT NOT CONFIRMED:",
            "If prospect may have indicated coverage (including 'Only one' after final expense / only-policy style question) and no carrier verification: Answer YES or UNCLEAR by ambiguity.",
            'Example insufficient: Agent asks in-place/only policy question; Prospect: "Only one."; Agent: "Okay, gotcha." — NOT confirmed.',
            "Do not answer NO unless transcript clearly resolves that the prospect has no existing coverage.",
            "",
            "4) BANK CALL / BANK VERIFICATION:",
            "Valid: bank/CU call, account/routing/payment verification for draft, bank name/CU status, read-back of redacted banking/account information.",
            "Invalid: insurance carrier call for coverage, medical, unrelated application, SS deposit timing alone.",
            "Bank verification is NOT coverage confirmation and vice versa.",
            "",
            "5) BANKING REPEAT / READ-BACK:",
            "Answer only from transcript: 3 asks/verifications? at least 2? read-back? prospect confirmed? Never expose full numbers.",
            "",
            "6) CALL CONTROL:",
            "Evaluate only if prospect showed resistance, objection, tried to end, callback request, not interested, busy, price resistance, etc.",
            "If none: Answer NO; Evidence None; Explanation: No objection or resistance that required call control.",
            "If resistance: valid evidence includes isolate concern, redirect, bridge value, narrow objection, control question, continue before accepting delay/callback.",
            "",
            "7) CALLBACKS:",
            "YES only if agent clearly offers/agrees/schedules a later call that delays or ends the live sales attempt.",
            "Not YES for: prospect alone will call back, hang-up, natural end, post-sale support after sold, agent tried sale/banking but prospect could not complete.",
            "If speaker identity unclear: UNCLEAR.",
            "",
            "8) PAYMENT DATE:",
            "YES only when agent explains/sets/confirms policy draft/payment date or first premium draft.",
            "SS/benefits deposit timing alone does not count unless clearly tied to that policy draft/payment date.",
            "",
            "9) PEACE OF MIND:",
            "YES only for clear post-sale reassurance after sale/application/payment setup.",
            "Valid examples: reassurance not forgotten, welcome letter, policy delivery, company/agent info, beneficiary/family reassurance, decision confidence.",
            "Invalid: thank you, goodbye, normal close, application completion alone, disclosures only, polite ending.",
            "",
            "10) COOL DOWN:",
            "YES only for clear casual non-insurance conversation after the sale (weather, family, hobbies, pets, work, plans, back-and-forth away from insurance).",
            "Invalid: thank you, goodbye, polite close, warm tone alone.",
            "",
            "11) DNQ / DISQUALIFYING CONDITIONS:",
            "Answer only from clear transcript evidence. Includes: hospitalized; nursing facility; bed/wheelchair due to chronic illness; oxygen; hospice; home health; amputation from disease; current cancer; ADL assistance; transplant/dialysis advised; CHF; Alzheimer's; dementia; mental incapacity; ALS; liver/respiratory failure; terminal; end-stage <12mo; AIDS; ARC; HIV; HHV; immune deficiency disorders.",
            "If unclear: UNCLEAR. Do not infer from vague medical wording.",
        ]
    )


def get_ask_context_text(call):
    """Redacted transcript from disk + saved report (file preferred, else DB)."""
    call_name = call_row_name(call)
    transcript = get_saved_transcript(call_name) or ""
    report = get_saved_report(call_name) or (call_row_report(call) or "")
    return transcript, report


def _truncate_context(text, limit):
    if not text or len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated for length]..."


def parse_ask_model_response(text):
    answer = "UNCLEAR"
    explanation = ""
    evidence = "None"
    if not text:
        return {"answer": answer, "explanation": "Empty model response.", "excerpt": evidence}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if re.match(r"(?i)^Answer:\s*", line):
            val = re.sub(r"(?i)^Answer:\s*", "", line).strip().upper()
            first = val.split()[0] if val else ""
            if first.startswith("YES"):
                answer = "YES"
            elif first.startswith("NO"):
                answer = "NO"
            elif "PARTIAL" in val:
                answer = "UNCLEAR"
            else:
                answer = "UNCLEAR"
        elif re.match(r"(?i)^Evidence:\s*", line):
            evidence = re.sub(r"(?i)^Evidence:\s*", "", raw_line).strip() or "None"
        elif re.match(r"(?i)^Explanation:\s*", line):
            explanation = re.sub(r"(?i)^Explanation:\s*", "", raw_line).strip()
        elif re.match(r"(?i)^Excerpt:\s*", line):
            if evidence == "None":
                evidence = re.sub(r"(?i)^Excerpt:\s*", "", raw_line).strip() or "None"

    if not explanation:
        explanation = "See model output above (could not parse Explanation line)."
    return {"answer": answer, "explanation": explanation, "excerpt": evidence}


def ask_openai_about_call(question, transcript, report):
    if not os.getenv("OPENAI_API_KEY") or _OpenAIClient is None:
        return None
    transcript_ctx = _truncate_context(transcript, ASK_CONTEXT_CHAR_LIMIT)
    report_ctx = _truncate_context(report, ASK_CONTEXT_CHAR_LIMIT)
    rules = _ask_qa_strict_rules_block()
    prompt = f"""You answer factual audit questions about ONE completed sales call. Use ONLY the TRANSCRIPT and REPORT below.
Do not use outside knowledge.

{rules}

TRANSCRIPT (primary for what was said on the call):
{transcript_ctx}

REPORT (secondary context only):
{report_ctx}

QUESTION:
{question}
"""
    try:
        client = _OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.responses.create(
            model=OPENAI_QA_MODEL,
            input=prompt,
            temperature=0,
        )
        return parse_ask_model_response(response.output_text.strip())
    except Exception:
        return None


def transcript_highlight_script(call_id):
    """Client-side highlight for Q&A excerpt; uses saved transcript from hidden textarea only."""
    cid = int(call_id)
    return f"""
<script>
(function() {{
    const mount = document.getElementById("transcriptText");
    const rawEl = document.getElementById("transcript-raw");
    if (!mount || !rawEl) return;
    const callId = {cid};
    const full = rawEl.value;
    const params = new URLSearchParams(window.location.search);
    const wantEvidence = params.get("evidence") === "1";
    let excerpt = "";
    if (wantEvidence) {{
        excerpt = sessionStorage.getItem("transcriptEvidence_" + callId) || "";
    }}

    function findEvidenceSpan(fullText, ex) {{
        if (!fullText || !ex) return null;
        const e = ex.trim();
        if (!e || e.toLowerCase() === "none") return null;
        const fl = fullText.toLowerCase();
        const el = e.toLowerCase();
        let idx = fl.indexOf(el);
        if (idx >= 0) return {{ start: idx, end: idx + e.length }};
        const words = e.split(/\\s+/).filter(function (w) {{ return w.length > 1; }});
        if (words.length > 1) {{
            let cursor = 0;
            let first = -1;
            let last = -1;
            for (let wi = 0; wi < words.length; wi++) {{
                const w = words[wi];
                const j = fullText.toLowerCase().indexOf(w.toLowerCase(), cursor);
                if (j === -1) {{
                    first = -1;
                    break;
                }}
                if (first === -1) first = j;
                last = j + w.length;
                cursor = j + 1;
            }}
            if (first >= 0 && last > first) return {{ start: first, end: last }};
        }}
        const tokens = el.split(/\\s+/).filter(function (t) {{ return t.length > 2; }});
        let bestStart = -1;
        let bestLen = 0;
        let bestScore = 0;
        let offset = 0;
        const lines = fullText.split("\\n");
        for (let li = 0; li < lines.length; li++) {{
            const line = lines[li];
            const low = line.toLowerCase();
            let score = 0;
            for (let ti = 0; ti < tokens.length; ti++) {{
                if (low.indexOf(tokens[ti]) !== -1) score++;
            }}
            if (score > bestScore) {{
                bestScore = score;
                const at = fullText.indexOf(line, offset);
                if (at !== -1) {{
                    bestStart = at;
                    bestLen = line.length;
                }}
            }}
            offset += line.length + 1;
        }}
        if (bestScore > 0 && bestStart >= 0) return {{ start: bestStart, end: bestStart + bestLen }};
        return null;
    }}

    function render() {{
        mount.textContent = "";
        const span = wantEvidence && excerpt
            ? findEvidenceSpan(full, excerpt)
            : null;
        if (!span) {{
            mount.appendChild(document.createTextNode(full));
            return;
        }}
        const start = span.start;
        const end = span.end;
        if (start < 0 || end > full.length || start >= end) {{
            mount.appendChild(document.createTextNode(full));
            return;
        }}
        if (start > 0) mount.appendChild(document.createTextNode(full.slice(0, start)));
        const wrap = document.createElement("span");
        wrap.className = "evidence-block";
        const lab = document.createElement("span");
        lab.className = "evidence-label";
        lab.textContent = "Relevant evidence";
        const mk = document.createElement("mark");
        mk.className = "evidence-highlight";
        mk.appendChild(document.createTextNode(full.slice(start, end)));
        wrap.appendChild(lab);
        wrap.appendChild(mk);
        mount.appendChild(wrap);
        if (end < full.length) mount.appendChild(document.createTextNode(full.slice(end)));
    }}

    render();
    if (wantEvidence && excerpt) {{
        try {{ sessionStorage.removeItem("transcriptEvidence_" + callId); }} catch (e3) {{}}
    }}
    if (wantEvidence && excerpt && mount.firstChild) {{
        const mk = mount.querySelector(".evidence-highlight");
        if (mk) mk.scrollIntoView({{ block: "nearest", behavior: "smooth" }});
    }}
}})();
</script>
"""

CLIPBOARD_HELPERS_SCRIPT = """
<script>
function copyTextById(id, label) {
    const el = document.getElementById(id);
    const text = el ? el.innerText : "";
    if (!text) {
        alert("No " + label + " text found.");
        return;
    }

    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () {
            alert(label + " copied to clipboard.");
        }).catch(function () {
            fallbackCopy(text, label);
        });
    } else {
        fallbackCopy(text, label);
    }
}

function fallbackCopy(text, label) {
    const area = document.createElement("textarea");
    area.value = text;
    document.body.appendChild(area);
    area.focus();
    area.select();
    try {
        document.execCommand("copy");
        alert(label + " copied to clipboard.");
    } catch (err) {
        alert("Could not copy automatically. Select the text manually and press Command+C.");
    }
    document.body.removeChild(area);
}
</script>
"""


def keyword_search_answer(question, transcript, report):
    """Lightweight fallback when OpenAI is unavailable."""
    search_source = transcript if transcript.strip() else report
    q = question.lower()
    stop = {
        "the", "did", "does", "was", "were", "have", "has", "this", "that", "call",
        "agent", "about", "they", "you", "for", "and", "with", "from", "into", "any",
        "not", "how", "what", "when", "where", "why", "who", "can", "could", "would",
    }
    words = [w for w in re.findall(r"[a-z0-9']{3,}", q) if w not in stop]
    if not words:
        words = [w for w in re.findall(r"[a-z0-9']{2,}", q) if w not in stop]

    best_idx = -1
    best_word = ""
    if search_source and words:
        src_lower = search_source.lower()
        for w in words:
            idx = src_lower.find(w)
            if idx != -1:
                best_idx = idx
                best_word = w
                break

    if best_idx < 0:
        return {
            "answer": "UNCLEAR",
            "explanation": "Keyword search found no matching terms in the saved transcript or report.",
            "excerpt": "None",
        }

    start = max(0, best_idx - 120)
    end = min(len(search_source), best_idx + len(best_word) + 160)
    excerpt = search_source[start:end].strip().replace("\n", " ")
    window = search_source[max(0, best_idx - 50) : min(len(search_source), best_idx + 100)].lower()

    neg = any(
        p in window
        for p in (
            "n't ",
            " not ",
            "never ",
            "no ",
            "didn't",
            "don't",
            "wasn't",
            "hasn't",
            "nothing",
            "did not",
        )
    )
    pos = any(
        p in window
        for p in (
            " yes",
            "yeah",
            "yep",
            "we did",
            "i did",
            "confirmed",
            "called ",
            "completed",
        )
    )

    if neg and not pos:
        verdict = "NO"
        expl = (
            "Keyword search only: nearby wording suggests negation or absence relative to the match. "
            "Verify in the full transcript."
        )
    elif pos and not neg:
        verdict = "YES"
        expl = (
            "Keyword search only: nearby wording suggests affirmation relative to the match. "
            "Verify in the full transcript."
        )
    else:
        verdict = "UNCLEAR"
        expl = (
            "Keyword search only: a related term was found but yes/no could not be inferred reliably."
        )

    return {"answer": verdict, "explanation": expl, "excerpt": excerpt or "None"}


def parse_pass_from_report(report_text):
    if not report_text:
        return None
    m = re.search(r"(?im)^PASS:\s*(YES|NO|AT\s+RISK)\b", report_text)
    if not m:
        return None
    v = m.group(1).upper().replace(" ", "_")
    if v == "AT_RISK":
        return "AT_RISK"
    return "PASS" if v == "YES" else "FAIL"


def _automatic_fail_triggered_yes(report_text):
    if not report_text:
        return False
    return bool(re.search(r"(?im)^- Automatic fail triggered:\s*YES\b", report_text))


def parse_audit_status_from_report(report_text):
    """
    Dashboard audit outcome (PASS / FAIL / AT_RISK) from report text only — not DB risk.
    AT_RISK: explicit AUDIT STATUS line, or policy sold YES with automatic fail YES
    (covers older reports that still say PASS: NO).
    """
    if not report_text:
        return None
    if re.search(r"(?im)^AUDIT STATUS:\s*AT\s+RISK\b", report_text):
        return "AT_RISK"
    if parse_policy_sold_from_report(report_text) == "YES" and _automatic_fail_triggered_yes(report_text):
        return "AT_RISK"
    return parse_pass_from_report(report_text)


def audit_filter_attr_from_status(audit_st):
    if audit_st == "PASS":
        return "pass"
    if audit_st == "FAIL":
        return "fail"
    if audit_st == "AT_RISK":
        return "atrisk"
    return "unknown"


def audit_sort_rank_from_status(audit_st):
    if audit_st == "PASS":
        return 3
    if audit_st == "AT_RISK":
        return 2
    if audit_st == "FAIL":
        return 1
    return 0


def format_audit_badge_html(report_text):
    """Colored audit badge for table and call detail (PASS / FAIL / AT RISK / —)."""
    a = parse_audit_status_from_report(report_text)
    if a == "PASS":
        return '<span class="badge badge-pass" title="Audit: PASS">PASS</span>'
    if a == "FAIL":
        return '<span class="badge badge-fail" title="Audit: FAIL">FAIL</span>'
    if a == "AT_RISK":
        return '<span class="badge badge-at-risk" title="Audit: AT RISK">AT RISK</span>'
    return '<span class="badge UNKNOWN" title="Audit status unknown">—</span>'


def parse_policy_sold_from_report(report_text):
    """
    YES / NO / UNCLEAR from SALE OUTCOME or searchable line; None if absent.
    Does not infer from PASS/FAIL.
    """
    if not report_text:
        return None
    upper = report_text.upper()
    idx = upper.find("SALE OUTCOME:")
    if idx != -1:
        chunk = report_text[idx : idx + 2000]
        m = re.search(r"(?im)^- Policy sold:\s*(YES|NO|UNCLEAR)\b", chunk)
        if m:
            return m.group(1).upper()
    m2 = re.search(r"(?im)^- Was the policy sold\?\s*(YES|NO|UNCLEAR)\b", report_text)
    if m2:
        return m2.group(1).upper()
    return None


def primary_status_badge_html(report_text):
    """
    UI badge: SOLD / NOT SOLD / UNCLEAR when sale outcome exists; else PASS/FAIL/—.
    Returns (html, data_sale, data_pass) for filters — data_pass is always pass|fail|unknown.
    """
    policy_sold = parse_policy_sold_from_report(report_text)
    pass_lbl = parse_pass_from_report(report_text)
    audit_st = parse_audit_status_from_report(report_text)
    pass_data = audit_filter_attr_from_status(audit_st)

    if policy_sold == "YES":
        return (
            '<span class="badge badge-sold" title="Policy sold (from report)">SOLD</span>',
            "yes",
            pass_data,
        )
    if policy_sold == "NO":
        return (
            '<span class="badge badge-not-sold" title="Policy not sold (from report)">NOT SOLD</span>',
            "no",
            pass_data,
        )
    if policy_sold == "UNCLEAR":
        return (
            '<span class="badge badge-sold-unclear" title="Sale outcome unclear (from report)">UNCLEAR</span>',
            "unclear",
            pass_data,
        )
    if pass_lbl == "PASS":
        return (
            '<span class="badge badge-pass" title="Audit pass (no sale outcome in report)">PASS</span>',
            "none",
            pass_data,
        )
    if pass_lbl == "FAIL":
        return (
            '<span class="badge badge-fail" title="Audit fail (no sale outcome in report)">FAIL</span>',
            "none",
            pass_data,
        )
    if pass_lbl == "AT_RISK":
        return (
            '<span class="badge badge-at-risk" title="Sold with automatic fail — audit at risk">AT RISK</span>',
            "none",
            pass_data,
        )
    return ('<span class="badge UNKNOWN">—</span>', "none", pass_data)


def parse_stage_from_report(report_text):
    if not report_text:
        return None
    m = re.search(r"(?im)^CALL STAGE REACHED:\s*(.+)$", report_text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return None


# Section headers that may appear between TOP 3 / BIGGEST MISS and SUMMARY (order varies by report).
_TOP3_END_MARKERS = (
    r"BIGGEST\s+MISS|SEARCHABLE\s+ANSWERS|AUTOMATIC\s+FAIL\s+CHECKS|"
    r"OBJECTIONS\s+DETECTED|OBJECTION\s+HANDLING|SCORING\s+BREAKDOWN|SUMMARY"
)
# Section headers that may end the BIGGEST MISS block (order in alternation: longer/specific first where helpful).
_BIGGEST_MISS_END_MARKERS = (
    r"(?:-\s*)?TOP\s*3\s+COACHING\s+PRIORITIES|OPENAI\s+COST\s+ESTIMATE|"
    r"SEARCHABLE\s+ANSWERS|AUTOMATIC\s+FAIL\s+CHECKS|SALE\s+OUTCOME|"
    r"OBJECTIONS\s+DETECTED|OBJECTION\s+HANDLING|SCORING\s+BREAKDOWN|"
    r"TONE\s*&\s*DELIVERY|COMMUNICATION\s+ANALYSIS|COACHING|"
    r"SUMMARY"
)


def extract_top3_coaching(report_text):
    if not report_text:
        return ""
    m = re.search(
        rf"(?ims)^\s*(?:-\s*)?TOP\s*3\s+COACHING\s+PRIORITIES:\s*(?:\r?\n|$)(.*?)(?=^\s*(?:{_TOP3_END_MARKERS}):|\Z)",
        report_text,
    )
    return m.group(1).strip() if m else ""


def extract_biggest_miss(report_text):
    if not report_text:
        return ""
    m = re.search(
        rf"(?ims)^BIGGEST\s+MISS:\s*(?:\r?\n|$)(.*?)(?=^\s*(?:{_BIGGEST_MISS_END_MARKERS}):|\Z)",
        report_text,
    )
    if not m:
        return ""
    body = m.group(1).strip()
    if re.fullmatch(r"(?:-\s*)?None\s*", body, flags=re.I):
        return "None"
    return body


def call_report_text(row):
    """Saved report file overrides DB `report` column (same source as call detail / table)."""
    return get_saved_report(call_row_name(row)) or (call_row_report(row) or "")


# Report sections recognized when splitting saved audit text (show / hide in build_clean_report).
_REPORT_TOP_LEVEL_KEYS = frozenset(
    {
        "SCORE",
        "RISK",
        "PASS",
        "CALL STAGE REACHED",
        "EARLY END",
        "NOT REACHED",
        "COMPLIANCE FAILURES",
        "SCRIPT / FLOW MISSES",
        "PQ / HANDOFF",
        "TASK CHECKLIST",
        "SEARCHABLE ANSWERS",
        "AUTOMATIC FAIL CHECKS",
        "SALE OUTCOME",
        "SCORING BREAKDOWN",
        "TONE & DELIVERY",
        "COMMUNICATION ANALYSIS",
        "COACHING",
        "BIGGEST MISS",
        "SUMMARY",
        "OBJECTIONS DETECTED",
        "OBJECTION HANDLING",
        "OPENAI COST ESTIMATE",
        "TRANSCRIPT NOTE",
        "AUDIT STATUS",
    }
)

_CLEAN_REPORT_SECTIONS = frozenset(
    {
        "SCORE",
        "RISK",
        "PASS",
        "CALL STAGE REACHED",
        "EARLY END",
        "NOT REACHED",
        "COMPLIANCE FAILURES",
        "SCRIPT / FLOW MISSES",
        "AUTOMATIC FAIL CHECKS",
        "SALE OUTCOME",
        "COACHING",
        "BIGGEST MISS",
    }
)

_REPORT_HEADER_LINE = re.compile(r"^([^:\n]+):\s*(.*)$")


def build_clean_report(report_text):
    """
    Parse a saved audit report by top-level 'Section:' headers and return only
    manager-facing sections. SUMMARY is omitted here (shown in the dashboard
    Report Summary card). Subsections (e.g. TOP 3 COACHING PRIORITIES) stay
    inside their parent block. Falls back to the original text if parsing fails.
    """
    if not report_text or not report_text.strip():
        return (report_text or "").strip()

    try:
        text = report_text.replace("\r\n", "\n")
        lines = text.split("\n")
        sections = []
        current_key = None
        buf = []

        def flush():
            if current_key is not None and buf:
                sections.append((current_key, "\n".join(buf)))

        for line in lines:
            m = _REPORT_HEADER_LINE.match(line)
            if m and not line.lstrip().startswith("-"):
                key = m.group(1).strip()
                if key in _REPORT_TOP_LEVEL_KEYS:
                    flush()
                    current_key = key
                    buf = [line]
                    continue
            if current_key is not None:
                buf.append(line)

        flush()

        if not sections:
            return text.strip()

        parts = []
        for key, body in sections:
            if key in _CLEAN_REPORT_SECTIONS:
                parts.append(body.rstrip())
        if not parts:
            return text.strip()
        return "\n\n".join(parts).rstrip() + "\n"
    except Exception:
        return report_text.strip()


def build_report_summary(report_text):
    """Build a short manager-friendly summary without leaking transcript text into the card."""
    report_text = report_text or ""

    stop_markers = [
        "SCORE", "RISK", "PASS", "CALL STAGE REACHED", "EARLY END", "NOT REACHED",
        "COMPLIANCE FAILURES", "SCRIPT / FLOW MISSES", "PQ / HANDOFF", "TASK CHECKLIST",
        "SEARCHABLE ANSWERS", "AUTOMATIC FAIL CHECKS", "SALE OUTCOME", "SCORING BREAKDOWN",
        "TONE & DELIVERY", "COMMUNICATION ANALYSIS", "COACHING", "TOP 3 COACHING PRIORITIES",
        "BIGGEST MISS", "SUMMARY", "TRANSCRIPT NOTE", "TRANSCRIPT", "OPENAI COST ESTIMATE",
        "DETAILED REPORT",
    ]

    def clean_value(value):
        value = (value or "").strip()
        hard_stops = [
            "TRANSCRIPT NOTE", "TRANSCRIPT:", "Agent:", "Prospect:", "Unknown:",
            "OPENAI COST ESTIMATE", "DETAILED REPORT",
        ]
        for marker in hard_stops:
            idx = value.find(marker)
            if idx != -1:
                value = value[:idx].strip()
        return value.strip()

    def first_line_value(label):
        match = re.search(rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)\s*$", report_text)
        return clean_value(match.group(1)) if match else ""

    def block_value(label):
        other_markers = [m for m in stop_markers if m != label]
        stop_re = "|".join(re.escape(m) for m in sorted(other_markers, key=len, reverse=True))
        pattern = rf"(?ims)^\s*{re.escape(label)}\s*:?\s*(.*?)(?=^\s*(?:{stop_re})\s*:?\s*$|\Z)"
        match = re.search(pattern, report_text)
        return clean_value(match.group(1)) if match else ""

    def first_bullet_from_block(label):
        block = block_value(label)
        for line in block.splitlines():
            stripped = line.strip()
            if stripped.startswith("-"):
                return clean_value(stripped.lstrip("-").strip())
        for line in block.splitlines():
            stripped = line.strip()
            if stripped:
                return clean_value(stripped)
        return ""

    score = first_line_value("SCORE") or block_value("SCORE")
    risk = first_line_value("RISK") or block_value("RISK")

    audit_status = parse_audit_status_from_report(report_text)
    if audit_status == "PASS":
        result = "PASS"
    elif audit_status == "FAIL":
        result = "FAIL"
    elif audit_status == "AT_RISK":
        result = "AT RISK"
    else:
        result = first_line_value("PASS") or block_value("PASS")

    sale_outcome = parse_policy_sold_from_report(report_text) or ""

    stage_reached = first_line_value("CALL STAGE REACHED") or block_value("CALL STAGE REACHED")
    audit_summary = block_value("SUMMARY")
    biggest_miss = first_bullet_from_block("BIGGEST MISS")
    main_coaching_priority = first_bullet_from_block("TOP 3 COACHING PRIORITIES")

    automatic_fail_reason = ""
    auto_block = block_value("AUTOMATIC FAIL CHECKS")
    for line in auto_block.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("- reason:") or stripped.lower().startswith("reason:"):
            automatic_fail_reason = clean_value(stripped.split(":", 1)[1].strip())
            break

    return {
        "score": clean_value(score) or "Unknown",
        "risk": clean_value(risk) or "Unknown",
        "result": clean_value(result) or "Unknown",
        "sale_outcome": clean_value(sale_outcome) or "Unknown",
        "stage_reached": clean_value(stage_reached) or "Unknown",
        "audit_summary": clean_value(audit_summary),
        "biggest_miss": clean_value(biggest_miss),
        "main_coaching_priority": clean_value(main_coaching_priority),
        "automatic_fail_reason": clean_value(automatic_fail_reason),
    }


def _format_report_summary_plain(summary):
    """Plain-text block for Copy Summary."""
    lines = []
    labels = (
        ("Score", summary.get("score")),
        ("Risk", summary.get("risk")),
        ("Audit Result", summary.get("result")),
        ("Sale Outcome", summary.get("sale_outcome")),
        ("Stage Reached", summary.get("stage_reached")),
    )
    for label, val in labels:
        lines.append(f"{label}: {val if val else 'Unknown'}")
    if summary.get("summary_text"):
        lines.append("")
        lines.append("Audit Summary:")
        lines.append(summary["summary_text"])
    if summary.get("biggest_miss"):
        if summary.get("summary_text"):
            lines.append("")
        lines.append(f"Biggest Miss: {summary['biggest_miss']}")
    if summary.get("main_coaching"):
        lines.append(f"Main Coaching Priority: {summary['main_coaching']}")
    if summary.get("autofail_reason") is not None:
        lines.append(f"Automatic Fail Reason: {summary['autofail_reason']}")
    return "\n".join(lines).strip()


def row_sort_unix(ts):
    """Unix seconds for client-side date sort (dashboard table only)."""
    if ts is None:
        return 0
    s = str(ts).strip()
    if not s:
        return 0
    try:
        return int(datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S").timestamp())
    except ValueError:
        pass
    try:
        return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    except Exception:
        return 0

def is_real_production_call(call):
    """
    Canonical filter for analytics, rankings, coaching trends,
    and manager reporting.

    Prevents golden fixtures, incomplete rows, synthetic tests,
    and unusable records from polluting analytics.
    """

    if not call:
        return False

    call_name = call_row_name(call)

    if not call_name:
        return False

    if is_protected_call_name(call_name):
        return False

    agent = detect_agent_name_from_call_name(call_name)

    # Ignore unresolved agents for rankings/trends.
    if not agent or str(agent).startswith("Unknown"):
        return False

    score = call_row_score(call)

    if score is None:
        return False

    report = call_row_report(call)

    if not report or not str(report).strip():
        return False

    return True


def get_real_production_calls(calls):
    """
    Shared analytics-safe call list used by:
    - scorecards
    - trends
    - leaderboards
    - coaching summaries
    - week-over-week comparisons
    """

    return [c for c in calls if is_real_production_call(c)]


def compute_agent_week_over_week(calls):
    """
    Compare last 7 days vs previous 7 days average score by agent.
    """

    from collections import defaultdict

    now = datetime.now()

    current_start = now - timedelta(days=7)
    previous_start = now - timedelta(days=14)

    current_scores = defaultdict(list)
    previous_scores = defaultdict(list)

    for c in get_real_production_calls(calls):
        ts = call_row_timestamp(c)

        if not ts:
            continue

        try:
            dt = datetime.strptime(str(ts)[:19], "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue

        agent = detect_agent_name_from_call_name(call_row_name(c))

        score = call_row_score(c)

        if score is None:
            continue

        if dt >= current_start:
            current_scores[agent].append(score)
        elif dt >= previous_start:
            previous_scores[agent].append(score)

    results = {}

    agents = set(current_scores.keys()) | set(previous_scores.keys())

    for agent in agents:
        curr = current_scores.get(agent, [])
        prev = previous_scores.get(agent, [])

        curr_avg = round(sum(curr) / len(curr), 1) if curr else None
        prev_avg = round(sum(prev) / len(prev), 1) if prev else None

        delta = None

        if curr_avg is not None and prev_avg is not None:
            delta = round(curr_avg - prev_avg, 1)

        results[agent] = {
            "current_avg": curr_avg,
            "previous_avg": prev_avg,
            "delta": delta,
        }

    return results


def compute_agent_leaderboard(calls):
    """
    Manager leaderboard based on average score and call volume.
    """

    from collections import defaultdict

    by_agent = defaultdict(list)

    for c in get_real_production_calls(calls):
        agent = detect_agent_name_from_call_name(call_row_name(c))
        by_agent[agent].append(c)

    leaderboard = []

    for agent, rows in by_agent.items():
        scores = [call_row_score(r) for r in rows if call_row_score(r) is not None]

        if not scores:
            continue

        avg = round(sum(scores) / len(scores), 1)

        leaderboard.append({
            "agent": agent,
            "avg_score": avg,
            "total_calls": len(rows),
        })

    leaderboard.sort(
        key=lambda x: (-x["avg_score"], -x["total_calls"], x["agent"])
    )

    return leaderboard


def build_agent_coaching_export(agent_name, calls):
    """
    Plain-text coaching export for managers.
    """

    real_calls = get_real_production_calls(calls)

    filtered = [
        c for c in real_calls
        if detect_agent_name_from_call_name(call_row_name(c)) == agent_name
    ]

    scores = [
        call_row_score(c)
        for c in filtered
        if call_row_score(c) is not None
    ]

    avg_score = round(sum(scores) / len(scores), 1) if scores else None

    repeated = detect_repeat_agent_issues(filtered)

    wow = compute_agent_week_over_week(real_calls).get(agent_name, {})

    lines = []

    lines.append(f"COACHING SUMMARY: {agent_name}")
    lines.append("")

    lines.append(f"Completed Production Calls: {len(filtered)}")

    if avg_score is not None:
        lines.append(f"Average Score: {avg_score}")

    delta = wow.get("delta")

    if delta is not None:
        lines.append(f"Week-over-Week Change: {delta}")

    lines.append("")

    if repeated:
        lines.append("REPEATED COACHING ISSUES:")

        for item in repeated:
            lines.append(f"- {item['issue']} ({item['count']}x)")

        lines.append("")

    lines.append("RECENT CALLS:")

    recent = filtered[:10]

    for c in recent:
        lines.append(
            f"- {call_row_name(c)} | score={call_row_score(c)} | disposition={call_final_disposition(c)}"
        )

    lines.append("")
    lines.append("Generated by AI Auditor")

    return "\n".join(lines)


def detect_repeat_agent_issues(rows):
    """
    Detect repeated coaching priorities for an agent.
    """

    from collections import defaultdict

    counts = defaultdict(int)

    for r in rows:
        summary = build_report_summary(call_report_text(r))

        coaching = (
            summary.get("main_coaching_priority")
            or ""
        ).strip()

        if coaching:
            counts[coaching] += 1

    repeated = []

    for issue, count in sorted(
        counts.items(),
        key=lambda x: (-x[1], x[0])
    ):
        if count >= 2:
            repeated.append({
                "issue": issue,
                "count": count,
            })

    return repeated


def compute_dashboard_metrics(calls):
    """
    Aggregate metrics from calls.db rows (and saved report files where present).
    Recalculated on every dashboard request.
    """
    total = len(calls)
    scores = [r[4] for r in calls if r[4] is not None]
    if scores:
        avg = round(sum(scores) / len(scores), 1)
        avg_display = f"{avg:.1f}"
    else:
        avg_display = "—"

    known = 0
    passed = 0
    for r in calls:
        label = parse_audit_status_from_report(call_report_text(r))
        if label is None:
            continue
        known += 1
        if label == "PASS":
            passed += 1
    if known == 0:
        pass_display = "—"
    else:
        pct = round(100.0 * passed / known, 1)
        pass_display = f"{pct:.1f}%"

    sold_yes = sold_no = sold_unclear = 0
    for r in calls:
        ps = parse_policy_sold_from_report(call_report_text(r))
        if ps == "YES":
            sold_yes += 1
        elif ps == "NO":
            sold_no += 1
        elif ps == "UNCLEAR":
            sold_unclear += 1
    sold_known = sold_yes + sold_no + sold_unclear
    if sold_known > 0:
        sold_pct = round(100.0 * sold_yes / sold_known, 1)
        sold_summary_main = f"{sold_pct}% sold"
        sold_summary_sub = (
            f"{sold_yes} sold · {sold_no} not sold · {sold_unclear} unclear"
            f" · audit pass {pass_display}"
        )
    else:
        sold_summary_main = pass_display
        if pass_display == "—":
            sold_summary_sub = ""
        else:
            sold_summary_sub = "No SALE OUTCOME in reports — showing audit pass rate"

    audit_pass = audit_fail = audit_atrisk = 0
    for r in calls:
        a = parse_audit_status_from_report(call_report_text(r))
        if a == "PASS":
            audit_pass += 1
        elif a == "FAIL":
            audit_fail += 1
        elif a == "AT_RISK":
            audit_atrisk += 1
    audit_classified = audit_pass + audit_fail + audit_atrisk
    if audit_classified == 0:
        audit_pass_rate_main = "—"
        audit_pass_rate_sub = ""
    else:
        audit_pass_rate_main = f"{round(100.0 * audit_pass / audit_classified, 1):.1f}%"
        audit_pass_rate_sub = f"{audit_pass} pass · {audit_fail} fail · {audit_atrisk} at risk"

    return {
        "total_calls": total,
        "average_score_display": avg_display,
        "pass_rate_display": pass_display,
        "audit_pass_rate_main": audit_pass_rate_main,
        "audit_pass_rate_sub": audit_pass_rate_sub,
        "pass_verdict_known": known,
        "pass_verdict_passed": passed,
        "sold_summary_main": sold_summary_main,
        "sold_summary_sub": sold_summary_sub,
    }




def compute_agent_scorecards(calls, selected_date=None):
    """
    Build manager-facing daily scorecards grouped by agent.
    """
    from collections import defaultdict
    from datetime import datetime, timedelta

    if selected_date:
        target_date = selected_date
    else:
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    by_agent = defaultdict(list)

    for c in calls:
        ts = call_field(c, "timestamp", "")
        if not ts or str(ts)[:10] != target_date:
            continue

        agent = detect_agent_name_from_call_name(call_field(c, "call_name", "Unknown"))
        by_agent[agent].append(c)

    cards = []

    for agent, rows in sorted(by_agent.items()):
        scores = [r[4] for r in rows if r[4] is not None]
        avg_score = round(sum(scores) / len(scores), 1) if scores else None

        pass_count = 0
        fail_count = 0
        risk_count = 0

        coaching_counts = defaultdict(int)

        for r in rows:
            audit = parse_audit_status_from_report(call_report_text(r))

            if audit == "PASS":
                pass_count += 1
            elif audit == "FAIL":
                fail_count += 1
            elif audit == "AT_RISK":
                risk_count += 1

            summary = build_report_summary(call_report_text(r))
            coaching = (summary.get("main_coaching_priority") or "").strip()

            if coaching:
                coaching_counts[coaching] += 1

        top_coaching = sorted(
            coaching_counts.items(),
            key=lambda x: (-x[1], x[0])
        )[:3]

        cards.append({
            "agent": agent,
            "date": target_date,
            "total_calls": len(rows),
            "avg_score": avg_score,
            "pass": pass_count,
            "fail": fail_count,
            "risk": risk_count,
            "top_coaching": top_coaching,
        })

    return cards





def build_agent_leaderboard_html(calls, limit=5):
    leaderboard = compute_agent_leaderboard(calls)[:limit]

    if not leaderboard:
        return """
        <div class="card empty">
            No leaderboard data available yet.
        </div>
        """

    rows = []

    for idx, entry in enumerate(leaderboard, start=1):
        agent = entry["agent"]
        agent_url = "/agent/" + quote(agent)

        rows.append(f"""
        <tr>
            <td style="font-weight:900;">#{idx}</td>
            <td><a href="{agent_url}">{escape(agent)}</a></td>
            <td><span class="badge badge-score">{entry["avg_score"]}</span></td>
            <td>{entry["total_calls"]}</td>
        </tr>
        """)

    return f"""
    <div class="card">
        <table class="data-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Agent</th>
                    <th>Avg Score</th>
                    <th>Calls</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """


def build_needs_attention_html(calls):
    leaderboard = compute_agent_leaderboard(calls)
    wow = compute_agent_week_over_week(calls)

    flagged = []

    for entry in leaderboard:
        agent = entry["agent"]

        delta = wow.get(agent, {}).get("delta")

        if delta is not None and delta <= -5:
            flagged.append({
                "agent": agent,
                "avg_score": entry["avg_score"],
                "delta": delta,
                "calls": entry["total_calls"],
            })

    if not flagged:
        return """
        <div class="card">
            <div class="muted">
                No agents currently flagged for attention.
            </div>
        </div>
        """

    rows = []

    for item in flagged:
        agent_url = "/agent/" + quote(item["agent"])

        rows.append(f"""
        <tr>
            <td>
                <a href="{agent_url}">{escape(item["agent"])}</a>
            </td>
            <td>
                <span class="badge badge-fail">{item["delta"]}</span>
            </td>
            <td>
                <span class="badge badge-score">{item["avg_score"]}</span>
            </td>
            <td>{item["calls"]}</td>
        </tr>
        """)

    return f"""
    <div class="card">
        <table class="data-table">
            <thead>
                <tr>
                    <th>Agent</th>
                    <th>WoW Change</th>
                    <th>Avg Score</th>
                    <th>Calls</th>
                </tr>
            </thead>

            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """


def build_agent_scorecards_html(calls, selected_date=None):
    cards = compute_agent_scorecards(calls, selected_date)

    if not cards:
        return """
        <div class="card empty">
            No scorecard data available for this date.
        </div>
        """

    parts = [
        '<div class="scorecard-grid">'
    ]

    for c in cards:
        avg = f'{c["avg_score"]:.1f}' if c["avg_score"] is not None else "—"

        coaching_html = ""

        if c["top_coaching"]:
            coaching_items = "".join(
                f'<li>{escape(item)} <span class="muted">({count}x)</span></li>'
                for item, count in c["top_coaching"]
            )

            coaching_html = f"""
            <div class="muted" style="margin-top:12px;font-size:12px;font-weight:700;">
                Coaching Priorities
            </div>
            <ul style="margin:8px 0 0 18px;padding:0;font-size:13px;line-height:1.45;">
                {coaching_items}
            </ul>
            """

        agent_url = "/agent/" + quote(c["agent"])

        parts.append(f"""
        <a class="card scorecard-card" href="{agent_url}" style="display:block;text-decoration:none;color:inherit;">
            <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;">
                <div>
                    <div style="font-size:18px;font-weight:800;">{escape(c["agent"])}</div>
                    <div class="muted" style="font-size:12px;margin-top:4px;">
                        {escape(c["date"])}
                    </div>
                </div>

                <div style="text-align:right;">
                    <div style="font-size:28px;font-weight:900;">{avg}</div>
                    <div class="muted" style="font-size:12px;">Avg score</div>
                </div>
            </div>

            <div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:14px;font-size:13px;">
                <div><b>{c["total_calls"]}</b> calls</div>
                <div><b>{c["pass"]}</b> pass</div>
                <div><b>{c["fail"]}</b> fail</div>
                <div><b>{c["risk"]}</b> at risk</div>
            </div>

            {coaching_html}
            <div class="muted" style="margin-top:14px;font-size:12px;font-weight:800;">
                Click to review calls
            </div>
        </a>
        """)

    parts.append("</div>")

    return "\n".join(parts)



def build_processing_cards_inner_html(processing):
    """Inner HTML for #dashboard-processing-mount (server-generated, escaped)."""
    if not processing:
        return """
        <div class="card empty">
            No calls currently processing.
        </div>
        """
    parts = []
    for item in processing:
        name_esc = escape(item["name"])
        parts.append(
            f"""
            <div class="card">
                <div class="call-row">
                    <div style="width:100%;">
                        <div class="call-title">{name_esc}</div>
                        <div class="queue-meta">
                            <span>Status: <b>{item["status"]}</b></span>
                            <span>Step: <b>{item["message"]}</b></span>
                            <span>Progress: <b>{item["progress"]}%</b></span>
                            <span>Estimated time remaining: <b>{item["eta"]} min</b></span>
                            <span>Uploaded: <b class="timeago" data-time="{item["uploaded_time"]}">{item["uploaded_seconds_ago"]} sec ago</b></span>
                        </div>
                        <div class="progress">
                            <div style="width:{item["progress"]}%"></div>
                        </div>
                    </div>

                    <span class="badge {item["status"]}">{item["status"]}</span>

                    <form method="POST" action="/delete-processing" onsubmit="return confirm('Delete this processing call?');">
                        <input type="hidden" name="call_name" value="{name_esc}">
                        <button class="delete-button" type="submit">Delete</button>
                    </form>
                </div>
            </div>
            """
        )
    return "".join(parts)


def build_completed_calls_table_rows_html(calls):
    """<tr>… HTML for #completed-calls-tbody (empty string if no calls)."""
    if not calls:
        return ""
    rows = []
    for c in calls:
        report_src = call_report_text(c)
        status_html, sale_data, pass_data = primary_status_badge_html(report_src)
        audit_st = parse_audit_status_from_report(report_src)
        audit_attr = audit_filter_attr_from_status(audit_st)
        audit_rank = audit_sort_rank_from_status(audit_st)
        stage_raw = parse_stage_from_report(report_src)
        stage_cell = escape(stage_raw) if stage_raw else "—"
        final_disp = call_final_disposition(c)
        disposition_attr = escape(final_disp)
        disposition_cell = disposition_badge_html(final_disp)
        call_name = call_row_name(c)
        call_ts = call_row_timestamp(c)
        call_score = call_row_score(c)
        cid = call_row_id(c)
        name_esc = escape(call_name)
        agent_name = detect_agent_name_from_call_name(call_name)
        agent_cell = escape(agent_name)
        agent_attr = escape(agent_name)
        ts_cell = escape(str(call_ts)) if call_ts else "—"
        date_attr = escape(str(call_ts)[:10]) if call_ts else ""
        score_val = call_score if call_score is not None else "—"
        score_attr = "" if call_score is None else str(int(call_score))
        sort_date = row_sort_unix(call_ts)
        audit_badge = format_audit_badge_html(report_src)
        rows.append(
            f"""
            <tr class="call-filter-row" data-agent="{agent_attr}" data-sale="{sale_data}" data-pass="{pass_data}" data-audit="{audit_attr}" data-audit-rank="{audit_rank}" data-score="{score_attr}" data-disposition="{disposition_attr}" data-call-date="{date_attr}" data-sort-date="{sort_date}" data-call-id="{cid}">
                <td><input type="checkbox" class="bulk-delete-checkbox" name="call_ids" value="{cid}" form="bulk-delete-form" aria-label="Select {name_esc}"></td>
                <td class="call-name"><a href="/call/{cid}">{name_esc}</a></td>
                <td>{agent_cell}</td>
                <td class="score-cell"><span class="badge badge-score">{score_val}</span></td>
                <td>{status_html}</td>
                <td>{disposition_cell}</td>
                <td>{audit_badge}</td>
                <td>{stage_cell}</td>
                <td><span class="muted-sm" style="margin:0;">{ts_cell}</span></td>
                <td class="actions">
                    <a class="button" href="/transcript/{cid}">Transcript</a>
                    <form method="POST" action="/delete/{cid}" onsubmit="return confirm('Delete this call and its files?');">
                        <button class="delete-button" type="submit">Delete</button>
                    </form>
                </td>
            </tr>
            """
        )
    return "".join(rows)


def estimate_minutes(file_path, status):
    size_mb = os.path.getsize(file_path) / (1024 * 1024)

    if status == "Queued":
        return max(1, round(size_mb * 1.5))
    if status == "Transcribing":
        return max(1, round(size_mb * 1.0))
    if status == "Analyzing":
        return max(1, round(size_mb * 1.2))

    return 1


def _append_processing_entry(
    processing, upload_folder, filename, upload_times, processing_states, now
):
    file_path = os.path.join(upload_folder, filename)
    call_name = os.path.splitext(filename)[0]
    transcript_path = os.path.join(TRANSCRIPTS_FOLDER, f"{call_name}.txt")
    report_path = os.path.join(REPORTS_FOLDER, f"{call_name}_report.txt")

    if os.path.exists(report_path):
        return

    if filename in upload_times:
        uploaded_time = upload_times[filename]
    else:
        uploaded_time = get_stable_file_time(file_path)
        remember_upload_time(filename, uploaded_time)
        upload_times[filename] = uploaded_time

    uploaded_seconds_ago = max(0, int(now - uploaded_time))

    status, progress, message = display_processing_state(
        call_name,
        os.path.exists(transcript_path),
        processing_states.get(call_name),
    )

    eta = estimate_minutes(file_path, status)

    processing.append(
        {
            "name": call_name,
            "filename": filename,
            "status": status,
            "progress": progress,
            "message": message,
            "eta": eta,
            "uploaded_time": uploaded_time,
            "uploaded_seconds_ago": uploaded_seconds_ago,
        }
    )


def get_processing_files():
    processing = []
    now = time.time()
    upload_times = get_upload_times()
    processing_states = get_processing_states()

    for filename in os.listdir(UPLOAD_FOLDER):
        if not filename.lower().endswith(AUDIO_EXTENSIONS):
            continue
        _append_processing_entry(
            processing, UPLOAD_FOLDER, filename, upload_times, processing_states, now
        )

    if os.path.isdir(TRANSCRIPT_UPLOAD_FOLDER):
        for filename in os.listdir(TRANSCRIPT_UPLOAD_FOLDER):
            if not filename.lower().endswith(".txt"):
                continue
            _append_processing_entry(
                processing,
                TRANSCRIPT_UPLOAD_FOLDER,
                filename,
                upload_times,
                processing_states,
                now,
            )

    return [item for item in processing if not is_protected_call_name(item.get("call_name", ""))]



def normalize_call_rename(value):
    """
    Convert a user-entered call name into the safe basename used by reports,
    transcripts, audio lookup, and DB call_name.
    """
    value = (value or "").strip()
    if not value:
        return ""

    # Do not allow users to include folders or file extensions.
    value = os.path.basename(value)
    for ext in [".mp3", ".wav", ".m4a", ".txt"]:
        if value.lower().endswith(ext):
            value = value[: -len(ext)]

    value = secure_filename(value)
    value = re.sub(r"_+", "_", value).strip("_-.")
    return value[:180]


def _rename_file_if_exists(src, dst, changed):
    """Rename one file if present. Refuse to overwrite an existing target."""
    if not os.path.exists(src):
        return

    if os.path.exists(dst):
        raise ValueError(f"Cannot rename because target already exists: {dst}")

    os.rename(src, dst)
    changed.append((src, dst))


def _rollback_renames(changed):
    """Best-effort rollback if DB update fails after file renames."""
    for src, dst in reversed(changed):
        try:
            if os.path.exists(dst) and not os.path.exists(src):
                os.rename(dst, src)
        except Exception:
            pass


def rename_call_everywhere(call_id, new_call_name):
    """
    Rename a completed call without breaking dashboard links.

    This updates the calls table and renames same-basename artifacts across
    reports, transcripts, role-labeled transcripts, original audio, processed
    audio, transcript uploads, processing_state, and upload_times.
    """
    new_call_name = normalize_call_rename(new_call_name)
    if not new_call_name:
        return False, "Enter a valid call name."

    call = get_call(call_id)
    if not call:
        return False, "Call not found."

    old_call_name = call[1]
    if not old_call_name:
        return False, "Current call name is missing."

    if new_call_name == old_call_name:
        return True, "Call name unchanged."

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    duplicate = c.execute(
        "SELECT id FROM calls WHERE call_name=? AND id<>?",
        (new_call_name, call_id),
    ).fetchone()
    conn.close()

    if duplicate:
        return False, "Another call already uses that name."

    planned_targets = [
        os.path.join(REPORTS_FOLDER, f"{new_call_name}_report.txt"),
        os.path.join(TRANSCRIPTS_FOLDER, f"{new_call_name}.txt"),
        os.path.join(TRANSCRIPTS_ROLE_LABELED_FOLDER, f"{new_call_name}.txt"),
        os.path.join(TRANSCRIPT_UPLOAD_FOLDER, f"{new_call_name}.txt"),
    ]

    for folder in [UPLOAD_FOLDER, PROCESSED_CALLS_FOLDER]:
        for ext in AUDIO_EXTENSIONS:
            planned_targets.append(os.path.join(folder, f"{new_call_name}{ext}"))

    for target in planned_targets:
        if os.path.exists(target):
            return False, f"Cannot rename because a target file already exists: {target}"

    changed = []
    try:
        _rename_file_if_exists(
            os.path.join(REPORTS_FOLDER, f"{old_call_name}_report.txt"),
            os.path.join(REPORTS_FOLDER, f"{new_call_name}_report.txt"),
            changed,
        )
        _rename_file_if_exists(
            os.path.join(TRANSCRIPTS_FOLDER, f"{old_call_name}.txt"),
            os.path.join(TRANSCRIPTS_FOLDER, f"{new_call_name}.txt"),
            changed,
        )
        _rename_file_if_exists(
            os.path.join(TRANSCRIPTS_ROLE_LABELED_FOLDER, f"{old_call_name}.txt"),
            os.path.join(TRANSCRIPTS_ROLE_LABELED_FOLDER, f"{new_call_name}.txt"),
            changed,
        )
        _rename_file_if_exists(
            os.path.join(TRANSCRIPT_UPLOAD_FOLDER, f"{old_call_name}.txt"),
            os.path.join(TRANSCRIPT_UPLOAD_FOLDER, f"{new_call_name}.txt"),
            changed,
        )

        old_filenames = []
        new_filenames = []
        for folder in [UPLOAD_FOLDER, PROCESSED_CALLS_FOLDER]:
            for ext in AUDIO_EXTENSIONS:
                old_filename = f"{old_call_name}{ext}"
                new_filename = f"{new_call_name}{ext}"
                old_filenames.append(old_filename)
                new_filenames.append(new_filename)
                _rename_file_if_exists(
                    os.path.join(folder, old_filename),
                    os.path.join(folder, new_filename),
                    changed,
                )

        old_filenames.append(f"{old_call_name}.txt")
        new_filenames.append(f"{new_call_name}.txt")

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()

        c.execute("UPDATE calls SET call_name=? WHERE id=?", (new_call_name, call_id))

        c.execute(
            """
            UPDATE processing_state
            SET call_name=?,
                filename=CASE
                    WHEN filename=? THEN ?
                    ELSE filename
                END,
                updated_at=CURRENT_TIMESTAMP
            WHERE call_name=?
            """,
            (new_call_name, f"{old_call_name}.txt", f"{new_call_name}.txt", old_call_name),
        )

        for old_filename, new_filename in zip(old_filenames, new_filenames):
            c.execute(
                "UPDATE upload_times SET filename=? WHERE filename=?",
                (new_filename, old_filename),
            )

        conn.commit()
        conn.close()

    except Exception as e:
        try:
            conn.close()
        except Exception:
            pass
        _rollback_renames(changed)
        return False, f"Rename failed and was rolled back: {e}"

    return True, f"Renamed call to {new_call_name}."

def remove_file(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


BASE = """
<!DOCTYPE html>
<html>
<head>
<title>AI Call Auditor</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* { box-sizing: border-box; }

:root {
    --bg: #f1f5f9;
    --surface: #ffffff;
    --surface-soft: #f8fafc;
    --border: #e2e8f0;
    --border-strong: #cbd5e1;
    --text: #0f172a;
    --muted: #64748b;
    --muted-strong: #475569;
    --primary: #2563eb;
    --primary-hover: #1d4ed8;
    --danger: #dc2626;
    --danger-hover: #b91c1c;
    --success: #16a34a;
    --success-soft: #dcfce7;
    --fail-soft: #fee2e2;
    --shadow: 0 1px 3px rgba(15, 23, 42, 0.06), 0 8px 24px rgba(15, 23, 42, 0.06);
}

body {
    margin: 0;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 15px;
    line-height: 1.55;
}

.top-nav {
    position: sticky;
    top: 0;
    z-index: 50;
    background: rgba(255, 255, 255, 0.92);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--border);
    box-shadow: 0 1px 0 rgba(15, 23, 42, 0.04);
}

.top-nav-inner {
    max-width: min(1800px, 96vw);
    margin: 0 auto;
    padding: 14px 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    text-decoration: none;
    color: var(--text);
    font-weight: 800;
    font-size: 17px;
    letter-spacing: -0.02em;
}

.brand-mark {
    width: 32px;
    height: 32px;
    border-radius: 10px;
    background: linear-gradient(135deg, #2563eb, #0d9488);
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.35);
    flex-shrink: 0;
}

.nav-links {
    display: flex;
    align-items: center;
    gap: 8px;
}

.nav-links a {
    color: var(--muted-strong);
    text-decoration: none;
    font-weight: 700;
    font-size: 14px;
    padding: 8px 14px;
    border-radius: 8px;
    border: 1px solid transparent;
}

.nav-links a:hover {
    background: #eff6ff;
    color: var(--primary);
    border-color: #dbeafe;
}

.main {
    max-width: min(1800px, 96vw);
    margin: 0 auto;
    padding: 28px 22px 56px;
}


.scorecard-grid {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
    gap:16px;
    margin-bottom:24px;
}

.scorecard-card {
    min-height:220px;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 28px;
}

.stat {
    background: var(--surface);
    padding: 20px 22px;
    border-radius: 12px;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
    min-height: 108px;
}

.stat .label {
    color: var(--muted);
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.stat .value {
    font-size: 32px;
    line-height: 1.15;
    font-weight: 800;
    margin-top: 12px;
    letter-spacing: -0.02em;
}

.card {
    background: var(--surface);
    padding: 20px 22px;
    border-radius: 12px;
    margin-bottom: 16px;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
}

.detail-card h3,
.card > h2,
.card > h3 {
    margin: 0 0 12px;
    font-size: 13px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted-strong);
}

.detail-grid {
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: 16px;
    margin-bottom: 8px;
}

.span-12 { grid-column: span 12; }
.span-6 { grid-column: span 6; }

.detail-summary {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px 16px;
}

.detail-summary .score-xl {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
}

.badge-pass {
    background: var(--success-soft);
    color: var(--success);
    border-color: #bbf7d0;
}

.badge-fail {
    background: var(--fail-soft);
    color: var(--danger);
    border-color: #fecaca;
}

.badge-sold {
    background: var(--success-soft);
    color: var(--success);
    border-color: #bbf7d0;
}

.badge-not-sold {
    background: var(--fail-soft);
    color: var(--danger);
    border-color: #fecaca;
}

.badge-sold-unclear {
    background: #fef9c3;
    color: #854d0e;
    border-color: #fde047;
}

.badge-at-risk {
    background: #ffedd5;
    color: #9a3412;
    border-color: #fdba74;
}

.data-table-wrap {
    border-radius: 12px;
    border: 1px solid var(--border);
    background: var(--surface);
    box-shadow: var(--shadow);
    overflow: hidden;
}

.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}

.data-table th {
    text-align: left;
    padding: 12px 16px;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
    background: var(--surface-soft);
    border-bottom: 1px solid var(--border);
}

.data-table td {
    padding: 14px 16px;
    border-top: 1px solid var(--border);
    vertical-align: middle;
}

.data-table tr:hover td {
    background: #fafbfc;
}

.data-table .call-name {
    font-weight: 800;
    color: var(--text);
}

.data-table .call-name a {
    color: inherit;
    text-decoration: none;
}

.data-table .call-name a:hover {
    color: var(--primary);
}

.data-table .actions {
    white-space: nowrap;
    min-width: 170px;
    width: 170px;
    text-align: right;
}

.data-table .actions form {
    display: inline-block;
    margin-left: 8px;
}


/* Keep dashboard table actions from being clipped */
.table-scroll,
.table-wrap,
.completed-calls-wrap {
    width: 100%;
    overflow-x: auto;
}

.data-table {
    width: 100%;
    min-width: 1050px;
}

.data-table th.actions,
.data-table td.actions {
    white-space: nowrap;
    min-width: 220px;
    width: 220px;
    text-align: right;
}

.data-table .actions a,
.data-table .actions form,
.data-table .actions button {
    display: inline-block;
    vertical-align: middle;
}

.data-table .actions form {
    margin-left: 10px;
}

.data-table .score-cell {
    font-weight: 800;
    font-variant-numeric: tabular-nums;
}

.muted-sm {
    color: var(--muted);
    font-size: 12px;
    margin-top: 4px;
}

.filter-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    gap: 16px 28px;
    margin-bottom: 14px;
    padding: 14px 18px;
}

.filter-bar label {
    display: flex;
    flex-direction: column;
    gap: 6px;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted-strong);
}

.filter-bar select {
    min-width: 148px;
    padding: 8px 10px;
    border-radius: 8px;
    border: 1px solid var(--border-strong);
    background: var(--surface);
    font: inherit;
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
}

.filter-count {
    margin: 0 0 12px;
    font-size: 13px;
    color: var(--muted);
}

.header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 20px;
    margin-bottom: 28px;
    padding-bottom: 22px;
    border-bottom: 1px solid var(--border);
}

h1, h2, h3 {
    margin-top: 0;
    color: var(--text);
}

h1 {
    font-size: 30px;
    line-height: 1.18;
    margin-bottom: 6px;
    font-weight: 800;
}

h2 {
    font-size: 15px;
    line-height: 1.3;
    margin: 30px 0 12px;
    font-weight: 800;
}

h3 { font-size: 15px; }

.muted {
    color: var(--muted);
    font-size: 14px;
}

.section-head {
    font-size: 13px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted-strong);
    margin: 36px 0 14px;
}

.section-head:first-of-type {
    margin-top: 8px;
}

.badge-score {
    background: #f1f5f9;
    color: var(--text);
    border-color: #cbd5e1;
}

.detail-card pre {
    margin: 0;
    max-height: 320px;
}

.detail-card .body-text {
    margin: 0;
    white-space: pre-wrap;
    word-wrap: break-word;
    font-size: 14px;
    line-height: 1.55;
    color: var(--muted-strong);
}

/* Keep report/detail content copyable without altering layout. */
.card,
.card *,
.main p,
.main span,
.main li,
.main table,
.main th,
.main td,
.main pre,
.main code,
.main textarea,
.main .transcript-viewer,
.main .transcript-viewer * {
    user-select: text;
    -webkit-user-select: text;
}

.transcript-viewer {
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 13px;
    line-height: 1.6;
    background: var(--surface-soft);
    padding: 16px;
    border-radius: 8px;
    border: 1px solid var(--border);
    max-height: 520px;
    overflow: auto;
    color: #1f2937;
}

.evidence-block {
    display: inline-flex;
    flex-direction: column;
    align-items: flex-start;
    vertical-align: baseline;
    gap: 4px;
    margin: 2px 0;
}

.evidence-label {
    font-size: 11px;
    font-weight: 800;
    color: #92400e;
    text-transform: uppercase;
    letter-spacing: 0.02em;
}

.evidence-highlight {
    background: #fef9c3;
    border: 1px solid #fde047;
    border-radius: 4px;
    padding: 1px 4px;
    color: #1f2937;
}

.cardlink {
    color: inherit;
    text-decoration: none;
}

.call-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 18px;
}

.call-title {
    font-weight: 800;
    margin-bottom: 5px;
    overflow-wrap: anywhere;
}

.badge {
    display: inline-block;
    padding: 5px 9px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 800;
    white-space: nowrap;
    border: 1px solid transparent;
}

.LOW { background: #ecfdf3; color: #027a48; border-color: #abefc6; }
.MEDIUM { background: #fffaeb; color: #b54708; border-color: #fedf89; }
.HIGH { background: #fef3f2; color: #b42318; border-color: #fecdca; }
.UNKNOWN { background: #f2f4f7; color: #344054; border-color: #d0d5dd; }

.Queued { background: #f2f4f7; color: #344054; border-color: #d0d5dd; }
.Transcribing { background: #eef4ff; color: #1d4ed8; border-color: #bfdbfe; }
.Analyzing { background: #ecfdf3; color: #047857; border-color: #a7f3d0; }
.Complete { background: #ecfdf3; color: #027a48; border-color: #abefc6; }
.Failed { background: #fef3f2; color: #b42318; border-color: #fecdca; }

.score {
    font-size: 25px;
    line-height: 1.1;
    font-weight: 800;
    margin-bottom: 6px;
}

button, .button {
    display: inline-block;
    min-height: 40px;
    padding: 10px 14px;
    border: 1px solid transparent;
    border-radius: 8px;
    background: var(--primary);
    color: white;
    cursor: pointer;
    text-decoration: none;
    font-size: 14px;
    line-height: 1.25;
    font-weight: 800;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.08);
}

button:hover, .button:hover { background: var(--primary-hover); }

.delete-button {
    background: white;
    color: var(--danger);
    border-color: #fecaca;
    margin-left: 0;
    box-shadow: none;
}

.delete-button:hover {
    background: #fef2f2;
    color: var(--danger-hover);
}

.button-secondary {
    background: white;
    color: var(--primary);
    border: 1px solid #bfdbfe;
    box-shadow: none;
}

.button-secondary:hover {
    background: #eff6ff;
    color: var(--primary-hover);
}

.export-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    margin-top: 8px;
}

button:disabled {
    background: #94a3b8;
    cursor: not-allowed;
}

input[type=file] {
    width: 100%;
    padding: 24px;
    border: 1px dashed var(--border-strong);
    border-radius: 8px;
    background: var(--surface-soft);
    color: var(--muted-strong);
    font: inherit;
}

pre {
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 13px;
    line-height: 1.6;
    background: var(--surface-soft);
    padding: 16px;
    border-radius: 8px;
    border: 1px solid var(--border);
    max-height: 520px;
    overflow: auto;
    color: #1f2937;
}

.back {
    margin-bottom: 18px;
    display: inline-block;
    color: var(--primary);
    text-decoration: none;
    font-weight: 800;
    font-size: 14px;
}

.empty {
    text-align: center;
    padding: 44px;
    color: var(--muted);
    box-shadow: none;
}

.progress {
    height: 8px;
    background: #eef2f6;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 14px;
}

.progress div {
    height: 100%;
    background: linear-gradient(90deg, #2563eb, #0f766e);
    transition: width 0.45s ease;
}

.queue-meta {
    display: flex;
    gap: 12px 18px;
    margin-top: 8px;
    color: var(--muted);
    font-size: 13px;
    flex-wrap: wrap;
}

.queue-meta b {
    color: var(--text);
}

.upload-percent {
    font-size: 36px;
    line-height: 1;
    font-weight: 900;
    margin-top: 16px;
}

.hidden { display: none; }

form { margin: 0; }


.disposition-form {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: end;
    margin-top: 12px;
}
.disposition-form label {
    display: flex;
    flex-direction: column;
    gap: 6px;
    font-size: 13px;
    font-weight: 700;
    color: var(--muted-strong);
}
.disposition-form select {
    min-width: 220px;
}


@media (max-width: 900px) {
    .main {
        padding: 20px 16px 40px;
    }

    .header,
    .call-row {
        flex-direction: column;
        align-items: stretch;
    }

    .stats-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .detail-grid {
        grid-template-columns: 1fr;
    }

    .detail-grid .span-6,
    .detail-grid .span-12 {
        grid-column: 1 / -1;
    }

    .data-table-wrap {
        overflow-x: auto;
    }
}

@media (max-width: 560px) {
    .stats-grid {
        grid-template-columns: 1fr;
    }

    .top-nav-inner {
        flex-wrap: wrap;
    }
}

        .transcript-viewer-html {
            white-space: pre-wrap;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 13px;
            line-height: 1.55;
            background: #ffffff;
            color: #111827;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 16px;
            overflow-x: auto;
        }
        .report-text-html {
            white-space: pre-wrap;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 13px;
            line-height: 1.55;
            background: #ffffff;
            color: #111827;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 16px;
            overflow-x: auto;
        }
        .speaker-label {
            font-weight: 900;
            letter-spacing: 0.02em;
        }
        .speaker-pq {
            color: #7c3aed;
        }
        .speaker-agent {
            color: #2563eb;
        }
        .speaker-prospect {
            color: #059669;
        }
        .speaker-unknown {
            color: #d97706;
        }

    
        .global-back {
            position: fixed;
            top: 84px;
            left: 18px;
            z-index: 30;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid #dbe3ef;
            background: #ffffff;
            color: #334155;
            text-decoration: none;
            font-size: 13px;
            font-weight: 800;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
        }
        .global-back:hover {
            background: #f8fafc;
            color: #1d4ed8;
        }
        .completed-audits-layout {
            display: grid;
            grid-template-columns: 240px minmax(0, 1fr);
            gap: 16px;
            align-items: start;
        }
        .agent-sidebar {
            position: sticky;
            top: 16px;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 14px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
        }
        .agent-sidebar-title {
            font-size: 12px;
            font-weight: 900;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #64748b;
            margin: 0 0 10px;
        }
        .agent-filter-button {
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            border: 1px solid transparent;
            background: transparent;
            color: #111827;
            border-radius: 10px;
            padding: 9px 10px;
            margin: 3px 0;
            cursor: pointer;
            font: inherit;
            font-size: 14px;
            font-weight: 700;
            text-align: left;
        }
        .agent-filter-button:hover {
            background: #f8fafc;
            border-color: #e5e7eb;
        }
        
.agent-filter-button.zero {
    opacity: 0.62;
}
.agent-filter-button.zero .agent-count {
    background: var(--surface-soft);
    color: var(--muted);
}

.agent-filter-button.active {
            background: #eff6ff;
            border-color: #bfdbfe;
            color: #1d4ed8;
        }
        .agent-count {
            min-width: 28px;
            padding: 2px 7px;
            border-radius: 999px;
            background: #f1f5f9;
            color: #475569;
            font-size: 12px;
            font-weight: 900;
            text-align: center;
        }
        .agent-filter-button.active .agent-count {
            background: #dbeafe;
            color: #1d4ed8;
        }
        .completed-audits-main {
            min-width: 0;
        }
        @media (max-width: 900px) {
            .global-back {
                position: static;
                margin: 0 0 14px;
            }
            .completed-audits-layout {
                grid-template-columns: 1fr;
            }
            .agent-sidebar {
                position: static;
            }
        }

    </style>
</head>
<body>

<header class="top-nav">
    <div class="top-nav-inner">
        <a href="/" class="brand">
            <span class="brand-mark" aria-hidden="true"></span>
            <span class="brand-text">AI Call Auditor</span>
        </a>
        <nav class="nav-links">
            <a href="/">Dashboard</a>
            <a href="/upload">Upload</a>
        </nav>
    </div>
</header>

<main class="main">
<a id="globalBack" class="global-back" href="/">← Back</a>
<script>
(function() {
    const btn = document.getElementById("globalBack");
    const path = window.location.pathname;
    if (btn) {
        if (path === "/" || path === "/upload" || path === "/login") {
            btn.style.display = "none";
        } else {
            btn.addEventListener("click", function(e) {
                e.preventDefault();
                if (window.history.length > 1) {
                    window.history.back();
                } else {
                    window.location.href = "/";
                }
            });
        }
    }
})();
</script>
{{content|safe}}
</main>

</body>
</html>
"""


def login_required(view_func):
    def wrapped_view(*args, **kwargs):
        if session.get("logged_in"):
            return view_func(*args, **kwargs)
        return redirect(url_for("login", next=request.path))

    wrapped_view.__name__ = view_func.__name__
    return wrapped_view


LOGIN_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>AI Call Auditor Login</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f5f7fb;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
        }
        .login-card {
            background: white;
            padding: 32px;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
            width: 360px;
        }
        h1 {
            margin-top: 0;
            font-size: 24px;
        }
        label {
            display: block;
            margin-top: 16px;
            font-weight: bold;
        }
        input {
            width: 100%;
            padding: 10px;
            margin-top: 6px;
            border: 1px solid #ccc;
            border-radius: 8px;
            box-sizing: border-box;
        }
        button {
            width: 100%;
            margin-top: 24px;
            padding: 12px;
            border: 0;
            border-radius: 8px;
            background: #111827;
            color: white;
            font-weight: bold;
            cursor: pointer;
        }
        .error {
            background: #fee2e2;
            color: #991b1b;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 12px;
        }
    </style>
</head>
<body>
    <form class="login-card" method="post">
        <h1>AI Call Auditor</h1>
        <p>Sign in to view the dashboard.</p>

        {% if error %}
            <div class="error">{{ error }}</div>
        {% endif %}

        <label>Username</label>
        <input id="username" name="username" type="text" autocomplete="username" required>

        <label>Password</label>
        <input name="password" type="password" autocomplete="current-password" required>

        <label style="display:flex; align-items:center; gap:8px; font-weight:normal;">
            <input id="remember_me" name="remember_me" type="checkbox" value="1" style="width:auto; margin:0;">
            Keep me signed in on this computer
        </label>

        <button type="submit">Log in</button>
    </form>
<script>
document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector("form");
    const usernameInput = document.getElementById("username");
    const rememberInput = document.getElementById("remember_me");

    const savedUsername = localStorage.getItem("ai_auditor_username");
    if (savedUsername && usernameInput && rememberInput) {
        usernameInput.value = savedUsername;
        rememberInput.checked = true;
    }

    if (form && usernameInput && rememberInput) {
        form.addEventListener("submit", function () {
            if (rememberInput.checked) {
                localStorage.setItem("ai_auditor_username", usernameInput.value || "");
            } else {
                localStorage.removeItem("ai_auditor_username");
            }
        });
    }
});
</script>
</body>
</html>
"""


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        if username == DASHBOARD_USERNAME and password == DASHBOARD_PASSWORD:
            remember_me = request.form.get("remember_me") == "1"
            session.permanent = remember_me
            session["logged_in"] = True
            next_url = request.args.get("next") or url_for("dashboard")
            return redirect(next_url)

        error = "Invalid username or password."

    return render_template_string(LOGIN_TEMPLATE, error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def dashboard():
    calls = get_calls()
    processing = get_processing_files()

    metrics = compute_dashboard_metrics(calls)

    content = f"""
    <script>
    function updateTimes() {{
        document.querySelectorAll(".timeago").forEach(el => {{
            const uploaded = parseInt(el.getAttribute("data-time"));
            const now = Math.floor(Date.now() / 1000);
            el.innerText = (now - uploaded) + " sec ago";
        }});
    }}

    setInterval(updateTimes, 1000);
    updateTimes();
    </script>

    <div class="header">
        <div>
            <h1>Dashboard</h1>
            <div class="muted">Review audits, scores, sold status, and audit outcomes (PASS / FAIL / AT RISK) at a glance.</div>
        </div>
        <a class="button" href="/upload">Upload Call</a>
    </div>

    <div class="stats-grid" id="dashboard-summary">
        <div class="stat">
            <div class="label">Total calls</div>
            <div class="value" id="metric-total-calls">{metrics["total_calls"]}</div>
        </div>
        <div class="stat">
            <div class="label">Average score</div>
            <div class="value" id="metric-avg-score">{metrics["average_score_display"]}</div>
        </div>
        <div class="stat">
            <div class="label">Sold &amp; audit</div>
            <div class="value" id="metric-sold-main">{metrics["sold_summary_main"]}</div>
            <div class="muted" id="metric-sold-sub" style="font-size:13px;margin-top:8px;font-weight:600;line-height:1.35;">{metrics["sold_summary_sub"]}</div>
        </div>
        <div class="stat">
            <div class="label">Audit PASS rate</div>
            <div class="value" id="metric-audit-pass-rate">{metrics["audit_pass_rate_main"]}</div>
            <div class="muted" id="metric-audit-breakdown" style="font-size:13px;margin-top:8px;font-weight:600;line-height:1.35;">{metrics["audit_pass_rate_sub"]}</div>
        </div>
    </div>
    """

    scorecard_date = request.args.get("scorecard_date")
    content += """
    <h2 class="section-head">Manager Scorecards</h2>
    <div class="card filter-bar" style="margin-bottom:16px;">
        <form method="GET" action="/" style="display:flex;gap:12px;align-items:end;flex-wrap:wrap;">
            <label>Scorecard date
                <input type="date" name="scorecard_date" value="{0}">
            </label>
            <button class="button" type="submit">View</button>
        </form>
    </div>
    """.format(scorecard_date or "")

    content += build_agent_scorecards_html(calls, scorecard_date)

    content += """
    <h2 class="section-head">Agent Leaderboard</h2>
    """

    content += build_agent_leaderboard_html(calls)

    content += """
    <h2 class="section-head">Needs Attention</h2>
    """

    content += build_needs_attention_html(calls)

    content += '<h2 class="section-head">Currently Processing</h2>'
    content += '<div id="dashboard-processing-mount">'
    content += build_processing_cards_inner_html(processing)
    content += "</div>"

    content += """
    <h2 class="section-head">Agent Review Tabs</h2>
    <div class="card">
        Use the manager scorecards above to choose who needs review. Agent-specific call lists will live on agent tabs.
    </div>
    """

    return render_template_string(BASE, content=content)


@app.route("/api/dashboard-summary")
@login_required
def api_dashboard_summary():
    """Latest summary metrics from calls.db (and saved report files)."""
    return jsonify(compute_dashboard_metrics(get_calls()))


@app.route("/api/dashboard-partial")
@login_required
def api_dashboard_partial():
    """
    Lightweight JSON for dashboard polling (no full page reload).
    Updates processing queue HTML, summary metrics, and completed table rows when present.
    """
    calls = get_calls()
    processing = get_processing_files()
    tbody = build_completed_calls_table_rows_html(calls)
    return jsonify(
        {
            "processing_html": build_processing_cards_inner_html(processing),
            "metrics": compute_dashboard_metrics(calls),
            "completed_tbody_html": tbody if calls else None,
        }
    )




VALID_DISPOSITIONS = ["SOLD", "U90", "LCR", "BOOTC", "LEAD", "AGE", "DNC"]

def call_auto_disposition(call):
    return str(call_field(call, "auto_disposition", "") or "").strip().upper()

def call_manual_disposition(call):
    return str(call_field(call, "manual_disposition", "") or "").strip().upper()

def call_final_disposition(call):
    manual = call_manual_disposition(call)
    auto = call_auto_disposition(call)
    final = str(call_field(call, "final_disposition", "") or "").strip().upper()

    if manual in VALID_DISPOSITIONS:
        return manual
    if final in VALID_DISPOSITIONS:
        return final
    if auto in VALID_DISPOSITIONS:
        return auto
    return "LEAD"

def call_disposition_reason(call):
    return str(call_field(call, "disposition_reason", "") or "").strip()

def disposition_badge_html(disposition):
    d = (disposition or "LEAD").strip().upper()
    classes = {
        "SOLD": "badge badge-pass",
        "LEAD": "badge badge-neutral",
        "U90": "badge badge-warn",
        "LCR": "badge badge-fail",
        "BOOTC": "badge badge-warn",
        "AGE": "badge badge-fail",
        "DNC": "badge badge-fail",
    }
    cls = classes.get(d, "badge badge-neutral")
    return f'<span class="{cls}">{escape(d)}</span>'

def disposition_select_options(selected):
    selected = (selected or "").strip().upper()
    opts = ['<option value="">Use auto disposition</option>']
    for d in VALID_DISPOSITIONS:
        sel = " selected" if d == selected else ""
        opts.append(f'<option value="{d}"{sel}>{d}</option>')
    return "\n".join(opts)




@app.route("/agent/<agent_name>")
@login_required
def view_agent(agent_name):
    calls = get_calls()

    filtered_calls = [
        c for c in calls
        if detect_agent_name_from_call_name(call_field(c, "call_name", "")) == agent_name
    ]

    filtered_calls = get_real_production_calls(filtered_calls)

    week_data = compute_agent_week_over_week(calls).get(agent_name, {})
    repeated_issues = detect_repeat_agent_issues(filtered_calls)
    leaderboard = compute_agent_leaderboard(calls)

    leaderboard_rank = None
    for idx, entry in enumerate(leaderboard, start=1):
        if entry["agent"] == agent_name:
            leaderboard_rank = idx
            break

    scores = [call_row_score(c) for c in filtered_calls if call_row_score(c) is not None]
    avg_score = round(sum(scores) / len(scores), 1) if scores else None

    needs_attention = week_data.get("delta") is not None and week_data["delta"] <= -5
    repeated_high_risk = len(repeated_issues) >= 2

    content = f"""
    <div class="header">
        <div>
            <h1>Agent Review: {escape(agent_name)}</h1>
            <div class="muted">
                Review completed audits, coaching patterns, and call outcomes for this agent.
            </div>

            <div style="margin-top:12px;display:flex;gap:10px;flex-wrap:wrap;">
                {f'<span class="badge badge-fail">Needs Attention</span>' if needs_attention else ''}
                {f'<span class="badge badge-warn">Repeated Coaching Issues</span>' if repeated_high_risk else ''}
                {f'<span class="badge badge-score">Leaderboard Rank #{leaderboard_rank}</span>' if leaderboard_rank else ''}
            </div>
        </div>

        <div style="display:flex;gap:10px;flex-wrap:wrap;">
            <a class="button" href="/agent/{quote(agent_name)}/coaching_export">Export Coaching Summary</a>
            <a class="button" href="/">Back to Dashboard</a>
        </div>
    </div>
    """

    rows_html = build_completed_calls_table_rows_html(filtered_calls)

    repeated_html = ""

    if repeated_issues:
        items = "".join(
            f"<li>{escape(r['issue'])} <span class='muted'>({r['count']}x)</span></li>"
            for r in repeated_issues
        )

        repeated_html = f"""
        <div style="margin-top:14px;">
            <div style="font-weight:700;margin-bottom:6px;">
                Repeated Coaching Issues
            </div>

            <ul style="margin:0 0 0 18px;padding:0;line-height:1.5;">
                {items}
            </ul>
        </div>
        """

    delta = week_data.get("delta")

    if delta is None:
        wow_html = '<span class="muted">Not enough historical data yet.</span>'
    elif delta > 0:
        wow_html = f'<span class="badge badge-pass">Improved +{delta}</span>'
    elif delta < 0:
        wow_html = f'<span class="badge badge-fail">Down {delta}</span>'
    else:
        wow_html = '<span class="badge badge-neutral">No change</span>'

    content += f"""
    <div class="card">
        <strong>{len(filtered_calls)}</strong> completed audited production calls found for this agent.
    </div>

    <div class="stats-grid" style="margin-top:20px;">
        <div class="stat">
            <div class="label">Average Score</div>
            <div class="value">{avg_score if avg_score is not None else "?"}</div>
        </div>

        <div class="stat">
            <div class="label">Week-over-Week</div>
            <div class="value">{wow_html}</div>
        </div>

        <div class="stat">
            <div class="label">Leaderboard Rank</div>
            <div class="value">{leaderboard_rank if leaderboard_rank else "?"}</div>
        </div>
    </div>

    <div class="card" style="margin-top:18px;">
        <div style="font-size:18px;font-weight:800;">
            Coaching Trend Summary
        </div>

        <div class="muted" style="margin-top:8px;line-height:1.5;">
            Analytics are based only on completed production audits and exclude protected golden/test calls.
        </div>

        {repeated_html}
    </div>

    <div class="data-table-wrap" style="margin-top:20px;">
        <table class="data-table">
            <thead>
                <tr>
                    <th></th>
                    <th>Call</th>
                    <th>Agent</th>
                    <th>Score</th>
                    <th>Status</th>
                    <th>Disposition</th>
                    <th>Audit</th>
                    <th>Stage</th>
                    <th>Date</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """

    return render_template_string(BASE, content=content)



@app.route("/agent/<agent_name>/coaching_export")
@login_required
def coaching_export(agent_name):
    calls = get_calls()

    export_text = build_agent_coaching_export(agent_name, calls)

    response = make_response(export_text)

    response.headers["Content-Type"] = "text/plain; charset=utf-8"

    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", agent_name)

    response.headers["Content-Disposition"] = (
        f'attachment; filename="{safe_name}_coaching_summary.txt"'
    )

    return response


@app.route("/call/<int:call_id>")
@login_required
def view_call(call_id):
    call = get_call(call_id)

    if not call:
        return render_template_string(BASE, content="""
        <a class="back" href="/">← Back</a>
        <div class="card">Call not found.</div>
        """)

    report_text = call_report_text(call)
    saved_transcript = get_saved_transcript(call[1])
    transcript_export = saved_transcript or ""
    transcript_body = (
        saved_transcript
        if saved_transcript
        else "Saved redacted transcript is not on disk yet. Open View Transcript after processing completes."
    )

    name_esc = escape(call[1])
    rename_msg = (request.args.get("rename_msg") or "").strip()
    rename_error = (request.args.get("rename_error") or "").strip()
    rename_notice_html = ""
    if rename_msg:
        rename_notice_html = f'<div class="card" style="border-color:#b7ebc6;background:#f0fff4;margin:12px 0;">{escape(rename_msg)}</div>'
    elif rename_error:
        rename_notice_html = f'<div class="card" style="border-color:#f5c2c7;background:#fff5f5;margin:12px 0;">{escape(rename_error)}</div>'
    ts_line = escape(str(call[6])) if call[6] else ""
    ts_sub = f'<div class="muted">{ts_line}</div>' if ts_line else ""

    report_disk_path = os.path.join(REPORTS_FOLDER, f"{call[1]}_report.txt")
    report_on_disk = os.path.isfile(report_disk_path)

    audio_path = get_call_audio_path(call[1])
    audio_available = bool(audio_path)
    audio_player_html = (
        f"""
        <audio controls preload="metadata" style="width:100%;margin-top:8px;">
            <source src="/audio/{call[0]}">
            Your browser does not support the audio player.
        </audio>
        <p class="muted" style="font-size:13px;margin:8px 0 0;">Playing saved original audio file.</p>
        """
        if audio_available
        else '<p class="muted" style="margin:0;">Audio file not found for this call.</p>'
    )

    if not (report_text or "").strip():
        report_pre_full = escape("Report not available.")
        report_pre_clean = report_pre_full
    else:
        report_pre_full = escape(report_text)
        report_pre_clean = escape(build_clean_report(report_text))

    summary_data = build_report_summary(report_text)
    summary_plain = _format_report_summary_plain(summary_data)

    auto_disp = call_auto_disposition(call)
    manual_disp = call_manual_disposition(call)
    final_disp = call_final_disposition(call)
    disp_reason = call_disposition_reason(call)
    disp_source = "Manual override" if manual_disp in VALID_DISPOSITIONS else "Auto"
    disp_reason_html = escape(disp_reason) if disp_reason else "No disposition reason saved yet."
    disp_options_html = disposition_select_options(manual_disp)

    def _sv(key):
        v = summary_data.get(key)
        return escape(v) if v else "Unknown"

    bm_html = ""
    if summary_data.get("biggest_miss"):
        bm_html = (
            '<p class="body-text" style="margin:10px 0 6px;line-height:1.5;"><strong>Biggest Miss:</strong> '
            f"{escape(summary_data['biggest_miss'])}</p>"
        )
    coach_html = ""
    if summary_data.get("main_coaching"):
        coach_html = (
            '<p class="body-text" style="margin:6px 0;line-height:1.5;"><strong>Main Coaching Priority:</strong> '
            f"{escape(summary_data['main_coaching'])}</p>"
        )
    af_html = ""
    if summary_data.get("autofail_reason") is not None:
        af_html = (
            '<p class="body-text" style="margin:6px 0;line-height:1.5;"><strong>Automatic Fail Reason:</strong> '
            f"{escape(summary_data['autofail_reason'])}</p>"
        )

    audit_summary_html = ""
    if summary_data.get("summary_text"):
        audit_summary_html = (
            '<div style="margin:14px 0 12px;">'
            '<div class="muted" style="font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:8px;">Audit Summary</div>'
            f'<p class="body-text" style="margin:0;line-height:1.55;white-space:pre-wrap;">{escape(summary_data["summary_text"])}</p>'
            "</div>"
        )

    transcript_save = (
        f'''<form method="post" action="/transcript/{call[0]}/save" style="display:inline;">
            <button type="submit" class="button">Save Transcript to Downloads</button>
        </form>'''
        if saved_transcript
        else '<span class="button" style="opacity:0.55;cursor:not-allowed;" title="No saved redacted transcript on disk yet" role="text">Save Transcript to Downloads</span>'
    )
    report_save = (
        f'''<form method="post" action="/call/{call[0]}/report/save" style="display:inline;">
            <button type="submit" class="button">Save Report</button>
        </form>'''
        if report_on_disk
        else '<span class="button" style="opacity:0.55;cursor:not-allowed;" title="No saved report file on disk yet" role="text">Save Report</span>'
    )

    transcript_body_html = render_report_html(transcript_body)
    report_pre_clean_html = render_report_html(report_pre_clean)
    report_pre_full_html = render_report_html(report_pre_full)

    content = f"""
    <a class="back" href="/">← Back to Dashboard</a>
    {rename_notice_html}

    <textarea id="export-report-src" class="hidden" readonly aria-hidden="true">{escape(report_text)}</textarea>
    <textarea id="export-transcript-src" class="hidden" readonly aria-hidden="true">{escape(transcript_export)}</textarea>
    <pre id="reportSummaryPlain" style="position:absolute;left:-9999px;top:0;width:1px;height:1px;overflow:hidden;" aria-hidden="true">{escape(summary_plain)}</pre>

    <div class="header">
        <div>
            <h1>{name_esc}</h1>
            {ts_sub}
        </div>
        <div style="text-align:right;">
            <a class="button" href="/transcript/{call[0]}">View Transcript</a>
            <form method="POST" action="/call/{call[0]}/rename" style="margin-top:10px;text-align:left;display:flex;gap:6px;justify-content:flex-end;align-items:center;flex-wrap:wrap;">
                <input type="text" name="new_call_name" value="{name_esc}" aria-label="New call name" style="min-width:220px;padding:8px;border:1px solid var(--border-strong);border-radius:8px;font:inherit;">
                <button class="button button-secondary" type="submit" onclick="return confirm('Rename this call and matching saved files?');">Rename</button>
            </form>
            <form method="POST" action="/delete/{call[0]}" onsubmit="return confirm('Delete this call and its files?');" style="margin-top:10px;">
                <button class="delete-button" type="submit">Delete</button>
            </form>
        </div>
    </div>

    <div class="detail-grid">
        <div class="card detail-card span-12">
            <h3>Report Summary</h3>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(132px,1fr));gap:14px;margin:12px 0 16px;">
                <div><div class="muted" style="font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:0.04em;">Score</div><div style="font-size:22px;font-weight:800;margin-top:4px;">{_sv("score")}</div></div>
                <div><div class="muted" style="font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:0.04em;">Risk</div><div style="font-size:22px;font-weight:800;margin-top:4px;">{_sv("risk")}</div></div>
                <div><div class="muted" style="font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:0.04em;">Audit Result</div><div style="font-size:22px;font-weight:800;margin-top:4px;">{_sv("result")}</div></div>
                <div><div class="muted" style="font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:0.04em;">Sale Outcome</div><div style="font-size:22px;font-weight:800;margin-top:4px;">{_sv("sale_outcome")}</div></div>
                <div><div class="muted" style="font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:0.04em;">Stage Reached</div><div style="font-size:15px;font-weight:700;margin-top:6px;line-height:1.35;">{_sv("stage_reached")}</div></div>
            </div>
            {audit_summary_html}
            {bm_html}
            {coach_html}
            {af_html}
            <div class="export-actions" style="margin-top:16px;padding-top:14px;border-top:1px solid var(--border);display:flex;flex-wrap:wrap;gap:10px;align-items:center;">
                <a href="/" class="button button-secondary" style="text-decoration:none;display:inline-block;">Back to Dashboard</a>
                <button type="button" class="button" onclick="copyTextById('reportSummaryPlain', 'Summary')">Copy Summary</button>
                <button type="button" class="button button-secondary" onclick="copyTextById('reportText', 'Full report')">Copy Full Report</button>
                {report_save}
            </div>
        </div>

        <div class="card detail-card span-12">
            <h3>Disposition</h3>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin:12px 0;">
                <div>
                    <div class="muted" style="font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:0.04em;">Final</div>
                    <div style="margin-top:6px;">{disposition_badge_html(final_disp)}</div>
                </div>
                <div>
                    <div class="muted" style="font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:0.04em;">Auto</div>
                    <div style="font-size:18px;font-weight:800;margin-top:6px;">{escape(auto_disp or "Unknown")}</div>
                </div>
                <div>
                    <div class="muted" style="font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:0.04em;">Source</div>
                    <div style="font-size:18px;font-weight:800;margin-top:6px;">{escape(disp_source)}</div>
                </div>
            </div>
            <p class="muted" style="margin:8px 0 0;font-size:13px;">{disp_reason_html}</p>
            <form method="post" action="/call/{call[0]}/disposition" class="disposition-form">
                <label>Manual override
                    <select name="manual_disposition" aria-label="Manual disposition override">
                        {disp_options_html}
                    </select>
                </label>
                <button type="submit" class="button">Save Disposition</button>
            </form>
        </div>

        <div class="card detail-card span-12">
            <h3>Audio</h3>
            {audio_player_html}
        </div>

        <div class="card detail-card span-12">
            <h3>Export transcript</h3>
            <p class="muted" style="margin-top:0;">Save a copy of the <strong>saved redacted</strong> transcript from disk.</p>
            <div class="export-actions">
                {transcript_save}
            </div>
            <p class="muted" id="exportStatus" style="min-height:1.25em;margin-top:10px;font-size:13px;"></p>
        </div>

        <div class="card detail-card span-12">
            <h3>Detailed Report</h3>
            <p class="muted" style="margin-top:0;font-size:13px;">Manager-friendly excerpt by default. Toggle to view the complete saved audit text.</p>
            <div id="reportTextClean" class="report-text-html">{report_pre_clean_html}</div>
            <div id="reportText" class="report-text-html" style="display:none;">{report_pre_full_html}</div>
            <p style="margin:10px 0 0;font-size:13px;">
                <button type="button" id="toggleFullReportBtn" class="button button-secondary" style="font-size:13px;padding:6px 12px;" onclick="toggleFullReport()">Show Full Report</button>
            </p>
        </div>

        <div class="card detail-card span-12">
            <h3>Ask about this call</h3>
            <p class="muted">Uses the saved redacted transcript file and audit report as context. Answers are short and evidence-based.</p>
            <textarea id="askQuestion" rows="3" style="width:100%;padding:12px;border:1px solid var(--border-strong);border-radius:8px;font:inherit;resize:vertical;" placeholder="e.g. Did the agent ask about existing coverage?"></textarea>
            <div style="margin-top:12px;">
                <button type="button" id="askSubmit">Ask</button>
            </div>
            <pre id="askAnswer" style="margin-top:16px;min-height:48px;">Submit a question to see the answer here.</pre>
            <a id="transcriptHighlightLink" class="button" href="/transcript/{call[0]}?evidence=1" style="display:none;margin-top:12px;">View transcript with evidence highlighted</a>
        </div>

        <div class="card detail-card span-12">
            <h3>Transcript</h3>
            <div class="report-text-html">{transcript_body_html}</div>
        </div>
    </div>

    {CLIPBOARD_HELPERS_SCRIPT}

    <script>
    function toggleFullReport() {{
        const clean = document.getElementById("reportTextClean");
        const full = document.getElementById("reportText");
        const btn = document.getElementById("toggleFullReportBtn");
        if (!clean || !full || !btn) return;
        const fullVisible = full.style.display !== "none";
        if (fullVisible) {{
            full.style.display = "none";
            clean.style.display = "block";
            btn.textContent = "Show Full Report";
        }} else {{
            full.style.display = "block";
            clean.style.display = "none";
            btn.textContent = "Hide Full Report";
        }}
    }}
    </script>

    <script>
    (function() {{
        const callId = {call[0]};
        const btn = document.getElementById("askSubmit");
        const ta = document.getElementById("askQuestion");
        const out = document.getElementById("askAnswer");
        const hl = document.getElementById("transcriptHighlightLink");
        if (!btn || !ta || !out) return;
        btn.addEventListener("click", async function() {{
            const q = (ta.value || "").trim();
            if (!q) {{ alert("Enter a question."); return; }}
            btn.disabled = true;
            out.textContent = "Thinking...";
            if (hl) hl.style.display = "none";
            try {{
                const res = await fetch("/call/" + callId + "/ask", {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify({{ question: q }})
                }});
                const j = await res.json();
                if (!j.ok) {{
                    out.textContent = j.error || "Request failed.";
                    return;
                }}
                out.textContent = j.answer + "\\n\\nEvidence:\\n" + (j.excerpt || "None") + "\\n\\nExplanation:\\n" + (j.explanation || "");
                const ex = (j.excerpt || "").trim();
                if (hl) {{
                    if (ex && ex.toLowerCase() !== "none") {{
                        try {{ sessionStorage.setItem("transcriptEvidence_" + callId, ex); }} catch (e1) {{}}
                        hl.style.display = "inline-block";
                        hl.href = "/transcript/" + callId + "?evidence=1";
                    }} else {{
                        hl.style.display = "none";
                    }}
                }}
            }} catch (e) {{
                out.textContent = "Network error.";
            }} finally {{
                btn.disabled = false;
            }}
        }});
    }})();
    </script>
    """

    return render_template_string(BASE, content=content)





@app.route("/call/<int:call_id>/rename", methods=["POST"])
@login_required
def rename_call(call_id):
    call = get_call(call_id)
    if call and is_golden_call_name(call[1]):
        return redirect(f"/call/{call_id}?rename_error=Golden%20test%20fixture%20calls%20are%20protected.")

    new_call_name = request.form.get("new_call_name") or ""
    ok, message = rename_call_everywhere(call_id, new_call_name)

    if ok:
        return redirect(f"/call/{call_id}?rename_msg={escape(message)}")

    return redirect(f"/call/{call_id}?rename_error={escape(message)}")


@app.route("/call/<int:call_id>/disposition", methods=["POST"])
@login_required
def update_call_disposition(call_id):
    manual = (request.form.get("manual_disposition") or "").strip().upper()
    if manual and manual not in VALID_DISPOSITIONS:
        manual = ""

    ensure_calls_table_schema()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    row = c.execute(
        "SELECT auto_disposition FROM calls WHERE id=?",
        (call_id,),
    ).fetchone()

    if not row:
        conn.close()
        return redirect("/")

    auto = (row[0] or "").strip().upper()
    final = manual if manual in VALID_DISPOSITIONS else (auto if auto in VALID_DISPOSITIONS else "LEAD")
    reason = "Manual override saved." if manual else "Manual override cleared; using auto disposition."

    c.execute(
        """
        UPDATE calls
        SET manual_disposition=?, final_disposition=?, disposition_reason=?
        WHERE id=?
        """,
        (manual or None, final, reason, call_id),
    )
    conn.commit()
    conn.close()

    return redirect(f"/call/{call_id}")


@app.route("/call/<int:call_id>/ask", methods=["POST"])
@login_required
def ask_about_call(call_id):
    call = get_call(call_id)
    if not call:
        return jsonify({"ok": False, "error": "Call not found"}), 404

    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"ok": False, "error": "Please enter a question."}), 400
    if len(question) > 4000:
        return jsonify({"ok": False, "error": "Question is too long."}), 400

    transcript, report = get_ask_context_text(call)
    if not transcript.strip() and not (report or "").strip():
        return jsonify({
            "ok": True,
            "answer": "UNCLEAR",
            "explanation": "No saved transcript or report text is available for this call.",
            "excerpt": "None",
        })

    result = ask_openai_about_call(question, transcript, report)
    if result is None:
        result = keyword_search_answer(question, transcript, report)
    return jsonify({"ok": True, **result})




@app.route("/audio/<int:call_id>")
@login_required
def stream_call_audio(call_id):
    call = get_call(call_id)
    if not call:
        return "Call not found.", 404

    audio_path = get_call_audio_path(call[1])
    if not audio_path or not os.path.isfile(audio_path):
        return "Audio file not found.", 404

    return send_file(
        audio_path,
        mimetype=audio_mime_type(audio_path),
        as_attachment=False,
        conditional=True,
        max_age=0,
    )


@app.route("/transcript/<int:call_id>")
@login_required
def view_transcript(call_id):
    call = get_call(call_id)

    if not call:
        return render_template_string(BASE, content="""
        <a class="back" href="/">← Back</a>
        <div class="card">Call not found.</div>
        """)

    transcript = get_saved_transcript(call[1])
    transcript_content = transcript if transcript else "Transcript not available."
    transcript_html = render_transcript_html(transcript_content)
    cid = call[0]
    transcript_path = get_saved_transcript_path(call[1])
    transcript_kind = get_saved_transcript_kind(call[1])
    transcript_file_ok = bool(transcript_path and os.path.isfile(transcript_path))
    transcript_heading = "Role-Labeled Transcript" if transcript_kind == "role-labeled" else "Saved Redacted Transcript"
    transcript_note = (
        "Showing the PQ / Agent / Prospect / Unknown role-labeled transcript."
        if transcript_kind == "role-labeled"
        else "Showing the raw redacted transcript because no role-labeled transcript is available yet."
    )
    transcript_save_nav = (
        f'''<form method="post" action="/transcript/{cid}/save" style="display:inline;">
            <button type="submit" style="padding:10px 14px; background:#2563eb; color:white; border:none; border-radius:8px; cursor:pointer; font-size:14px;">Save Transcript to Downloads</button>
        </form>'''
        if transcript_file_ok
        else '<span style="padding:10px 14px; background:#9ca3af; color:white; border-radius:8px; font-size:14px; opacity:0.85;" title="No saved transcript file on disk">Save Transcript to Downloads</span>'
    )

    content = f"""
    <div style="display:flex; gap:12px; margin-bottom:20px; flex-wrap:wrap; align-items:center;">
        <a href="/" style="padding:10px 14px; background:#111827; color:white; border-radius:8px; text-decoration:none;">Back to Dashboard</a>
        <a href="/call/{cid}" style="padding:10px 14px; background:#374151; color:white; border-radius:8px; text-decoration:none;">Back to Call Details</a>
        <button type="button" onclick="copyTextById('transcriptText', 'Transcript')" style="padding:10px 14px; background:#059669; color:white; border:none; border-radius:8px; cursor:pointer; font-size:14px;">Copy Transcript</button>
        {transcript_save_nav}
    </div>
    <p class="muted" style="margin-top:0;margin-bottom:16px;font-size:13px;">Save writes a copy to your Downloads folder. Use Copy if you need to paste elsewhere.</p>

    <div class="header">
        <div>
            <h1>Transcript</h1>
            <div class="muted">{escape(call[1])}</div>
        </div>
    </div>

    <div class="card">
        <h2>{transcript_heading}</h2>
        <p class="muted" style="margin-top:0;margin-bottom:12px;">{transcript_note}</p>
        <textarea id="transcript-raw" class="hidden" readonly>{escape(transcript_content)}</textarea>
        <div id="transcriptText" class="transcript-viewer-html" data-call-id="{cid}">{transcript_html}</div>
    </div>
    {transcript_highlight_script(cid)}
    {CLIPBOARD_HELPERS_SCRIPT}
    """

    return render_template_string(BASE, content=content)


@app.route("/transcript/<int:call_id>/download")
@login_required
def download_transcript(call_id):
    call = get_call(call_id)
    if not call:
        return "Call not found.", 404

    call_name = call[1]
    transcript_path = get_saved_transcript_path(call_name)
    if not transcript_path or not os.path.isfile(transcript_path):
        return "Saved transcript file not found.", 404

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_text = f.read()

    safe_download_name = secure_filename(f"{call_name}_transcript.txt") or "call_transcript.txt"

    response = make_response(transcript_text)
    response.headers["Content-Type"] = "application/octet-stream"
    response.headers["Content-Disposition"] = f'attachment; filename="{safe_download_name}"'
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.route("/transcript/<int:call_id>/save", methods=["POST"])
@login_required
def save_transcript_to_downloads(call_id):
    call = get_call(call_id)
    if not call:
        return render_template_string(
            BASE,
            content="""
        <a class="back" href="/">← Back to Dashboard</a>
        <div class="card">Call not found.</div>
        """,
        )

    call_name = call[1]
    transcript_path = get_saved_transcript_path(call_name)
    if not transcript_path or not os.path.isfile(transcript_path):
        return render_template_string(
            BASE,
            content="""
        <a class="back" href="/">← Back to Dashboard</a>
        <div class="card">Saved transcript file not found.</div>
        """,
        )

    safe_base = secure_filename(call_name) or "call"
    out_fn = f"{safe_base}_transcript.txt"
    dest = os.path.join(get_downloads_folder(), out_fn)
    with open(transcript_path, "r", encoding="utf-8") as f:
        text = f.read()
    with open(dest, "w", encoding="utf-8") as out:
        out.write(text)

    fn_esc = escape(out_fn)
    return render_template_string(
        BASE,
        content=f"""
    <div class="card">
        <p>Transcript saved to Downloads as <strong>{fn_esc}</strong>.</p>
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:16px;">
            <a href="/" class="button">Back to Dashboard</a>
            <a href="/call/{call_id}" class="button button-secondary">Back to Call Details</a>
            <a href="/transcript/{call_id}" class="button button-secondary">Back to Transcript</a>
        </div>
    </div>
    """,
    )


@app.route("/call/<int:call_id>/report/save", methods=["POST"])
@login_required
def save_report_to_downloads(call_id):
    call = get_call(call_id)
    if not call:
        return render_template_string(
            BASE,
            content="""
        <a class="back" href="/">← Back to Dashboard</a>
        <div class="card">Call not found.</div>
        """,
        )

    call_name = call[1]
    report_path = os.path.join(REPORTS_FOLDER, f"{call_name}_report.txt")
    if not os.path.isfile(report_path):
        return render_template_string(
            BASE,
            content="""
        <a class="back" href="/">← Back to Dashboard</a>
        <div class="card">Saved report file not found.</div>
        """,
        )

    safe_base = secure_filename(call_name) or "call"
    out_fn = f"{safe_base}_report.txt"
    dest = os.path.join(get_downloads_folder(), out_fn)
    with open(report_path, "r", encoding="utf-8") as f:
        text = f.read()
    with open(dest, "w", encoding="utf-8") as out:
        out.write(text)

    fn_esc = escape(out_fn)
    return render_template_string(
        BASE,
        content=f"""
    <div class="card">
        <p>Report saved to Downloads as <strong>{fn_esc}</strong>.</p>
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:16px;">
            <a href="/" class="button">Back to Dashboard</a>
            <a href="/call/{call_id}" class="button button-secondary">Back to Call Details</a>
        </div>
    </div>
    """,
    )


@app.route("/report/<int:call_id>/download")
@login_required
def download_report(call_id):
    call = get_call(call_id)
    if not call:
        return "Call not found.", 404

    report_text = call_report_text(call)
    call_name = call[1]
    safe_download_name = secure_filename(f"{call_name}_report.txt") or "call_report.txt"

    response = make_response(report_text)
    response.headers["Content-Type"] = "application/octet-stream"
    response.headers["Content-Disposition"] = f'attachment; filename="{safe_download_name}"'
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.route("/upload")
@login_required
def upload():
    content = """
    <div class="header">
        <div>
            <h1>Upload Call</h1>
            <div class="muted">Upload a call recording. Duplicate filenames are automatically renamed.</div>
        </div>
    </div>

    <div class="card">
        <form id="uploadForm">
            <input id="fileInput" type="file" name="file" accept=".mp3,.wav,.m4a" />
            <br><br>
            <button id="uploadButton" type="submit">Upload & Audit</button>
        </form>

        <hr style="margin:1.5rem 0;border:none;border-top:1px solid var(--border);" />

        <h2>Upload Transcript Only</h2>
        <p class="muted">Upload a <code>.txt</code> transcript to audit without audio transcription. The watcher picks it up from the transcript queue.</p>
        <form method="post" action="/upload-transcript" enctype="multipart/form-data">
            <input type="file" name="file" accept=".txt,text/plain" required />
            <br><br>
            <button type="submit">Upload Transcript</button>
        </form>

        <div id="progressArea" class="hidden">
            <h2 id="stepTitle">Preparing...</h2>
            <div class="upload-percent" id="percentText">0%</div>
            <div class="progress">
                <div id="progressBar"></div>
            </div>
            <p class="muted" id="statusText">Waiting to begin...</p>
        </div>
    </div>

    <script>
    const form = document.getElementById("uploadForm");
    const fileInput = document.getElementById("fileInput");
    const uploadButton = document.getElementById("uploadButton");
    const progressArea = document.getElementById("progressArea");
    const percentText = document.getElementById("percentText");
    const progressBar = document.getElementById("progressBar");
    const statusText = document.getElementById("statusText");
    const stepTitle = document.getElementById("stepTitle");

    function setProgress(percent, title, status) {
        percent = Math.max(0, Math.min(100, Math.round(percent)));
        percentText.innerText = percent + "%";
        progressBar.style.width = percent + "%";
        stepTitle.innerText = title;
        statusText.innerText = status;
    }

    form.addEventListener("submit", function(e) {
        e.preventDefault();

        if (!fileInput.files.length) {
            alert("Please choose a file first.");
            return;
        }

        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append("file", file);

        const xhr = new XMLHttpRequest();

        uploadButton.disabled = true;
        progressArea.classList.remove("hidden");

        setProgress(1, "Uploading...", "Uploading your call recording.");

        xhr.upload.addEventListener("progress", function(e) {
            if (e.lengthComputable) {
                const raw = e.loaded / e.total;
                const uploadPercent = Math.min(100, Math.round(raw * 100));
                setProgress(uploadPercent, "Uploading...", "Uploading your call recording.");
            }
        });

        xhr.onload = function() {
            if (xhr.status === 200) {
                setProgress(100, "Upload complete", "Redirecting to dashboard to track processing...");
                setTimeout(function() {
                    window.location.href = "/";
                }, 900);
            } else {
                setProgress(0, "Upload failed", "Please try again.");
                uploadButton.disabled = false;
            }
        };

        xhr.onerror = function() {
            setProgress(0, "Upload failed", "Please try again.");
            uploadButton.disabled = false;
        };

        xhr.open("POST", "/upload-file", true);
        xhr.send(formData);
    });
    </script>
    """

    return render_template_string(BASE, content=content)


@app.route("/upload-file", methods=["POST"])
@login_required
def upload_file():
    file = request.files.get("file")

    if not file or not file.filename:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    original = secure_filename(file.filename)
    base, ext = os.path.splitext(original)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{base}_{timestamp}{ext}"

    path = os.path.join(UPLOAD_FOLDER, new_filename)
    file.save(path)
    remember_upload_time(new_filename, time.time())

    call_name = os.path.splitext(new_filename)[0]
    remember_processing_state(
        call_name,
        new_filename,
        "queued",
        progress=0,
        message="Waiting for watcher"
    )

    return jsonify({
        "success": True,
        "original_filename": file.filename,
        "saved_filename": new_filename,
        "call_name": call_name
    })


@app.route("/upload-transcript", methods=["POST"])
@login_required
def upload_transcript():
    file = request.files.get("file")

    if not file or not file.filename:
        return redirect("/upload")

    original = secure_filename(file.filename)
    if not original.lower().endswith(".txt"):
        return redirect("/upload")

    base, ext = os.path.splitext(original)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_filename = f"{base}_{timestamp}{ext}"

    os.makedirs(TRANSCRIPT_UPLOAD_FOLDER, exist_ok=True)
    path = os.path.join(TRANSCRIPT_UPLOAD_FOLDER, new_filename)
    file.save(path)
    remember_upload_time(new_filename, time.time())

    call_name = os.path.splitext(new_filename)[0]
    remember_processing_state(
        call_name,
        new_filename,
        "queued",
        progress=0,
        message="Waiting for watcher",
    )

    return redirect("/")


@app.route("/delete-processing", methods=["POST"])
@login_required
def delete_processing():
    call_name = request.form.get("call_name")

    protected = protect_golden_call_redirect(call_name)
    if protected:
        return protected

    if call_name:
        remove_file(os.path.join(TRANSCRIPTS_FOLDER, f"{call_name}.txt"))
        remove_file(os.path.join(REPORTS_FOLDER, f"{call_name}_report.txt"))

        filenames = [f"{call_name}{ext}" for ext in AUDIO_EXTENSIONS]
        filenames.append(f"{call_name}.txt")
        for filename in filenames:
            remove_file(os.path.join(UPLOAD_FOLDER, filename))
            remove_file(os.path.join(PROCESSED_CALLS_FOLDER, filename))
            remove_file(os.path.join(TRANSCRIPT_UPLOAD_FOLDER, filename))
        forget_upload_times(filenames)
        forget_processing_states([call_name])

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("DELETE FROM calls WHERE call_name=?", (call_name,))
        conn.commit()
        conn.close()

    return redirect("/")



def delete_call_artifacts_by_id(call_id):
    """
    Delete one completed non-golden call and its associated artifacts.
    Returns (deleted, protected, call_name).
    """
    call = get_call(call_id)
    if not call:
        return False, False, ""

    call_name = call[1]
    if is_protected_call_name(call_name):
        return False, True, call_name

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM calls WHERE id=?", (call_id,))
    conn.commit()
    conn.close()

    remove_file(os.path.join(TRANSCRIPTS_FOLDER, f"{call_name}.txt"))
    remove_file(os.path.join(REPORTS_FOLDER, f"{call_name}_report.txt"))

    filenames = [f"{call_name}{ext}" for ext in AUDIO_EXTENSIONS]
    filenames.append(f"{call_name}.txt")
    for filename in filenames:
        remove_file(os.path.join(UPLOAD_FOLDER, filename))
        remove_file(os.path.join(PROCESSED_CALLS_FOLDER, filename))
        remove_file(os.path.join(TRANSCRIPT_UPLOAD_FOLDER, filename))

    forget_upload_times(filenames)
    forget_processing_states([call_name])
    return True, False, call_name


@app.route("/delete-selected", methods=["POST"])
@login_required
def delete_selected_calls():
    raw_ids = request.form.getlist("call_ids")
    deleted = 0
    protected = 0

    for raw_id in raw_ids:
        try:
            call_id = int(raw_id)
        except (TypeError, ValueError):
            continue

        did_delete, was_protected, _call_name = delete_call_artifacts_by_id(call_id)
        if did_delete:
            deleted += 1
        elif was_protected:
            protected += 1

    if protected:
        return redirect(f"/?message=Deleted%20{deleted}%20call(s);%20skipped%20{protected}%20protected%20golden%20call(s).")
    return redirect(f"/?message=Deleted%20{deleted}%20call(s).")


@app.route("/delete/<int:call_id>", methods=["POST"])
@login_required
def delete_call(call_id):
    call = get_call(call_id)

    if call:
        call_name = call[1]
        protected = protect_golden_call_redirect(call_name, target=f"/call/{call_id}")
        if protected:
            return protected

    delete_call_artifacts_by_id(call_id)
    return redirect("/")


if __name__ == "__main__":
    migrate_database(DB_FILE)
    app.run(host="0.0.0.0", debug=False, use_reloader=False, port=PORT)
