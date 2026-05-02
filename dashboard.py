from flask import (
    Flask,
    jsonify,
    make_response,
    redirect,
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
import time
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename

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
TRANSCRIPT_UPLOAD_FOLDER = "transcript_uploads"
TRANSCRIPTS_FOLDER = "transcripts"
TRANSCRIPTS_ROLE_LABELED_FOLDER = "transcripts_role_labeled"
REPORTS_FOLDER = "reports"
DB_FILE = "calls.db"
PORT = 5050
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_QA_MODEL = os.getenv("OPENAI_QA_MODEL", OPENAI_MODEL)
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


def get_calls():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM calls ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return rows


def get_call(call_id):
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


def build_agent_filter_options_html(calls):
    names = sorted({detect_agent_name_from_call_name(c[1]) for c in calls})
    options = ['<option value="all">All agents</option>']
    for name in names:
        safe = escape(name)
        options.append(f'<option value="{safe}">{safe}</option>')
    return "\n".join(options)

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
    call_name = call[1]
    transcript = get_saved_transcript(call_name) or ""
    report = get_saved_report(call_name) or (call[3] or "")
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
    return get_saved_report(row[1]) or (row[3] or "")


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
    result = first_line_value("PASS") or block_value("PASS")
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
        ("Result", summary.get("result")),
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
        name_esc = escape(c[1])
        agent_name = detect_agent_name_from_call_name(c[1])
        agent_cell = escape(agent_name)
        agent_attr = escape(agent_name)
        ts_cell = escape(str(c[6])) if c[6] else "—"
        score_val = c[4] if c[4] is not None else "—"
        score_attr = "" if c[4] is None else str(int(c[4]))
        cid = c[0]
        sort_date = row_sort_unix(c[6])
        audit_badge = format_audit_badge_html(report_src)
        rows.append(
            f"""
            <tr class="call-filter-row" data-agent="{agent_attr}" data-sale="{sale_data}" data-pass="{pass_data}" data-audit="{audit_attr}" data-audit-rank="{audit_rank}" data-score="{score_attr}" data-sort-date="{sort_date}" data-call-id="{cid}">
                <td class="call-name"><a href="/call/{cid}">{name_esc}</a></td>
                <td>{agent_cell}</td>
                <td class="score-cell"><span class="badge badge-score">{score_val}</span></td>
                <td>{status_html}</td>
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

    return processing


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
    max-width: 1200px;
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
    max-width: 1200px;
    margin: 0 auto;
    padding: 28px 22px 56px;
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
            background: #0f172a;
            color: #e5e7eb;
            border-radius: 12px;
            padding: 16px;
            overflow-x: auto;
        }
        .speaker-label {
            font-weight: 800;
            letter-spacing: 0.02em;
        }
        .speaker-pq {
            color: #a78bfa;
        }
        .speaker-agent {
            color: #60a5fa;
        }
        .speaker-prospect {
            color: #34d399;
        }
        .speaker-unknown {
            color: #fbbf24;
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

    content += '<h2 class="section-head">Currently Processing</h2>'
    content += '<div id="dashboard-processing-mount">'
    content += build_processing_cards_inner_html(processing)
    content += "</div>"

    content += '<h2 class="section-head">Completed Audits</h2>'

    if not calls:
        content += """
        <div class="card empty">
            No completed audits yet. Upload a call to get started.
        </div>
        """
    else:
        rows_html = build_completed_calls_table_rows_html(calls)
        agent_filter_options_html = build_agent_filter_options_html(calls)
        n_calls = len(calls)
        content += f"""
        <p class="filter-count" id="filter-count" aria-live="polite"></p>
        <div class="card filter-bar" id="completed-filters">
            <label>Agent
                <select id="filter-agent" aria-label="Filter by agent">
                    {agent_filter_options_html}
                </select>
            </label>
            <label>Sold status
                <select id="filter-sold-status" aria-label="Filter by sold status or audit pass fail">
                    <option value="all">All</option>
                    <option value="sold">Sold</option>
                    <option value="notsold">Not sold</option>
                    <option value="unclear">Unclear</option>
                    <option value="pass">Audit pass (no sale block)</option>
                    <option value="fail">Audit fail (no sale block)</option>
                    <option value="atrisk">Audit at risk (PASS line)</option>
                </select>
            </label>
            <label>Audit
                <select id="filter-audit" aria-label="Filter by audit outcome">
                    <option value="all">All</option>
                    <option value="pass">PASS</option>
                    <option value="fail">FAIL</option>
                    <option value="atrisk">AT RISK</option>
                </select>
            </label>
            <label>Score range
                <select id="filter-score" aria-label="Filter by score range">
                    <option value="all">Any</option>
                    <option value="0-59">0 – 59</option>
                    <option value="60-74">60 – 74</option>
                    <option value="75-84">75 – 84</option>
                    <option value="85-94">85 – 94</option>
                    <option value="95-100">95 – 100</option>
                </select>
            </label>
            <label>Sort by
                <select id="sort-by" aria-label="Sort completed calls">
                    <option value="date-desc" selected>Date (newest first)</option>
                    <option value="score-desc">Score (high → low)</option>
                    <option value="score-asc">Score (low → high)</option>
                    <option value="audit-pass-first">Audit (PASS first)</option>
                </select>
            </label>
        </div>
        <div class="data-table-wrap">
            <div class="completed-calls-wrap"><table class="data-table" id="completed-calls-table">
                <thead>
                    <tr>
                        <th>Call</th>
                        <th>Agent</th>
                        <th>Score</th>
                        <th>Sold status</th>
                        <th>Audit</th>
                        <th>Stage</th>
                        <th>Date / time</th>
                        <th class="actions">Actions</th>
                    </tr>
                </thead>
                <tbody id="completed-calls-tbody">
                    {rows_html}
                </tbody>
            </table></div>
        </div>
        <script>
        (function() {{
            function parseRange(v) {{
                if (!v || v === "all") return null;
                var p = v.split("-");
                return {{ lo: parseInt(p[0], 10), hi: parseInt(p[1], 10) }};
            }}
            function sortRows() {{
                var tbody = document.getElementById("completed-calls-tbody");
                var sortEl = document.getElementById("sort-by");
                if (!tbody || !sortEl) return;
                var mode = sortEl.value;
                var rows = Array.prototype.slice.call(tbody.querySelectorAll(".call-filter-row"));
                rows.sort(function(a, b) {{
                    if (mode === "date-desc") {{
                        var ta = parseInt(a.getAttribute("data-sort-date"), 10) || 0;
                        var tb = parseInt(b.getAttribute("data-sort-date"), 10) || 0;
                        var d = tb - ta;
                        if (d !== 0) return d;
                        return (parseInt(b.getAttribute("data-call-id"), 10) || 0) - (parseInt(a.getAttribute("data-call-id"), 10) || 0);
                    }}
                    if (mode === "score-desc") {{
                        var sa = a.getAttribute("data-score");
                        var sb = b.getAttribute("data-score");
                        var na = sa ? parseInt(sa, 10) : -1;
                        var nb = sb ? parseInt(sb, 10) : -1;
                        if (isNaN(na)) na = -1;
                        if (isNaN(nb)) nb = -1;
                        var d = nb - na;
                        if (d !== 0) return d;
                        return (parseInt(b.getAttribute("data-call-id"), 10) || 0) - (parseInt(a.getAttribute("data-call-id"), 10) || 0);
                    }}
                    if (mode === "score-asc") {{
                        var sa = a.getAttribute("data-score");
                        var sb = b.getAttribute("data-score");
                        var na = sa ? parseInt(sa, 10) : 10000;
                        var nb = sb ? parseInt(sb, 10) : 10000;
                        if (isNaN(na)) na = 10000;
                        if (isNaN(nb)) nb = 10000;
                        var d = na - nb;
                        if (d !== 0) return d;
                        return (parseInt(b.getAttribute("data-call-id"), 10) || 0) - (parseInt(a.getAttribute("data-call-id"), 10) || 0);
                    }}
                    if (mode === "audit-pass-first") {{
                        var ra = parseInt(a.getAttribute("data-audit-rank"), 10) || 0;
                        var rb = parseInt(b.getAttribute("data-audit-rank"), 10) || 0;
                        var d2 = rb - ra;
                        if (d2 !== 0) return d2;
                        return (parseInt(b.getAttribute("data-call-id"), 10) || 0) - (parseInt(a.getAttribute("data-call-id"), 10) || 0);
                    }}
                    return 0;
                }});
                rows.forEach(function(tr) {{ tbody.appendChild(tr); }});
            }}
            function applyFilters() {{
                var agentF = document.getElementById("filter-agent");
                var statusF = document.getElementById("filter-sold-status");
                var auditF = document.getElementById("filter-audit");
                var scoreF = document.getElementById("filter-score");
                var countEl = document.getElementById("filter-count");
                if (!agentF || !statusF || !auditF || !scoreF) return;
                var agentV = agentF.value;
                var pv = statusF.value;
                var av = auditF.value;
                var sv = scoreF.value;
                var range = parseRange(sv);
                var visible = 0;
                document.querySelectorAll(".call-filter-row").forEach(function(tr) {{
                    var ok = true;
                    var agent = tr.getAttribute("data-agent") || "Unknown";
                    var sale = tr.getAttribute("data-sale") || "none";
                    var au = tr.getAttribute("data-audit") || "unknown";
                    if (agentV !== "all" && agent !== agentV) ok = false;
                    if (pv === "sold" && sale !== "yes") ok = false;
                    if (pv === "notsold" && sale !== "no") ok = false;
                    if (pv === "unclear" && sale !== "unclear") ok = false;
                    if (pv === "pass" && (sale !== "none" || au !== "pass")) ok = false;
                    if (pv === "fail" && (sale !== "none" || au !== "fail")) ok = false;
                    if (pv === "atrisk" && au !== "atrisk") ok = false;
                    if (av === "pass" && au !== "pass") ok = false;
                    if (av === "fail" && au !== "fail") ok = false;
                    if (av === "atrisk" && au !== "atrisk") ok = false;
                    var sc = tr.getAttribute("data-score");
                    if (range) {{
                        if (!sc) ok = false;
                        else {{
                            var n = parseInt(sc, 10);
                            if (isNaN(n) || n < range.lo || n > range.hi) ok = false;
                        }}
                    }}
                    tr.style.display = ok ? "" : "none";
                    if (ok) visible++;
                }});
                var rowNodes = document.querySelectorAll(".call-filter-row");
                var denom = rowNodes.length;
                if (countEl) {{
                    if (denom === 0) {{
                        countEl.textContent = "";
                    }} else if (visible === denom) {{
                        countEl.textContent = "Showing all " + denom + " call" + (denom === 1 ? "" : "s");
                    }} else {{
                        countEl.textContent = "Showing " + visible + " of " + denom + " calls";
                    }}
                }}
            }}
            function refresh() {{
                sortRows();
                applyFilters();
            }}
            ["filter-agent", "filter-sold-status", "filter-audit", "filter-score", "sort-by"].forEach(function(id) {{
                var el = document.getElementById(id);
                if (el) el.addEventListener("change", refresh);
            }});
            refresh();
            window.__dashboardReapplyFilters = refresh;
        }})();
        </script>
        """

    content += """
    <script>
    (function() {
        function applyMetrics(m) {
            var map = {
                "metric-total-calls": m.total_calls,
                "metric-avg-score": m.average_score_display,
                "metric-sold-main": m.sold_summary_main,
                "metric-audit-pass-rate": m.audit_pass_rate_main
            };
            for (var id in map) {
                var el = document.getElementById(id);
                if (el) el.textContent = String(map[id]);
            }
            var sub = document.getElementById("metric-sold-sub");
            if (sub) sub.textContent = m.sold_summary_sub != null ? String(m.sold_summary_sub) : "";
            var ab = document.getElementById("metric-audit-breakdown");
            if (ab) ab.textContent = m.audit_pass_rate_sub != null ? String(m.audit_pass_rate_sub) : "";
        }
        function pollDashboard() {
            fetch("/api/dashboard-partial")
                .then(function(r) { return r.json(); })
                .then(function(d) {
                    if (d.metrics) applyMetrics(d.metrics);
                    var pm = document.getElementById("dashboard-processing-mount");
                    if (pm && d.processing_html != null) {
                        pm.innerHTML = d.processing_html;
                    }
                    var tb = document.getElementById("completed-calls-tbody");
                    if (tb && d.completed_tbody_html != null) {
                        tb.innerHTML = d.completed_tbody_html;
                        if (window.__dashboardReapplyFilters) {
                            window.__dashboardReapplyFilters();
                        }
                    }
                    document.querySelectorAll(".timeago").forEach(function(el) {
                        var uploaded = parseInt(el.getAttribute("data-time"), 10);
                        if (!isNaN(uploaded)) {
                            var now = Math.floor(Date.now() / 1000);
                            el.textContent = (now - uploaded) + " sec ago";
                        }
                    });
                })
                .catch(function() {});
        }
        function startPolling() {
            pollDashboard();
            setInterval(pollDashboard, 5000);
        }
        if (document.readyState === "loading") {
            document.addEventListener("DOMContentLoaded", startPolling);
        } else {
            startPolling();
        }
    })();
    </script>
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
    ts_line = escape(str(call[6])) if call[6] else ""
    ts_sub = f'<div class="muted">{ts_line}</div>' if ts_line else ""

    report_disk_path = os.path.join(REPORTS_FOLDER, f"{call[1]}_report.txt")
    report_on_disk = os.path.isfile(report_disk_path)

    if not (report_text or "").strip():
        report_pre_full = escape("Report not available.")
        report_pre_clean = report_pre_full
    else:
        report_pre_full = escape(report_text)
        report_pre_clean = escape(build_clean_report(report_text))

    summary_data = build_report_summary(report_text)
    summary_plain = _format_report_summary_plain(summary_data)

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

    content = f"""
    <a class="back" href="/">← Back to Dashboard</a>

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
                <div><div class="muted" style="font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:0.04em;">Result</div><div style="font-size:22px;font-weight:800;margin-top:4px;">{_sv("result")}</div></div>
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
            <h3>Export transcript</h3>
            <p class="muted" style="margin-top:0;">Save a copy of the <strong>saved redacted</strong> transcript from disk.</p>
            <div class="export-actions">
                {transcript_save}
            </div>
            <p class="muted" id="exportStatus" style="min-height:1.25em;margin-top:10px;font-size:13px;"></p>
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
            <pre>{escape(transcript_body)}</pre>
        </div>

        <div class="card detail-card span-12">
            <h3>Detailed Report</h3>
            <p class="muted" style="margin-top:0;font-size:13px;">Manager-friendly excerpt by default. Toggle to view the complete saved audit text.</p>
            <pre id="reportTextClean" class="transcript-viewer">{report_pre_clean}</pre>
            <pre id="reportText" class="transcript-viewer" style="display:none;">{report_pre_full}</pre>
            <p style="margin:10px 0 0;font-size:13px;">
                <button type="button" id="toggleFullReportBtn" class="button button-secondary" style="font-size:13px;padding:6px 12px;" onclick="toggleFullReport()">Show Full Report</button>
            </p>
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

    if call_name:
        remove_file(os.path.join(TRANSCRIPTS_FOLDER, f"{call_name}.txt"))
        remove_file(os.path.join(REPORTS_FOLDER, f"{call_name}_report.txt"))

        filenames = [f"{call_name}{ext}" for ext in AUDIO_EXTENSIONS]
        filenames.append(f"{call_name}.txt")
        for filename in filenames:
            remove_file(os.path.join(UPLOAD_FOLDER, filename))
            remove_file(os.path.join(TRANSCRIPT_UPLOAD_FOLDER, filename))
        forget_upload_times(filenames)
        forget_processing_states([call_name])

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("DELETE FROM calls WHERE call_name=?", (call_name,))
        conn.commit()
        conn.close()

    return redirect("/")


@app.route("/delete/<int:call_id>", methods=["POST"])
@login_required
def delete_call(call_id):
    call = get_call(call_id)

    if call:
        call_name = call[1]

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
            remove_file(os.path.join(TRANSCRIPT_UPLOAD_FOLDER, filename))
        forget_upload_times(filenames)
        forget_processing_states([call_name])

    return redirect("/")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, use_reloader=False, port=PORT)
