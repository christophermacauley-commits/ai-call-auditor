from flask import Flask, render_template_string, request, jsonify, redirect
import sqlite3
import os
import time
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "calls"
TRANSCRIPTS_FOLDER = "transcripts"
REPORTS_FOLDER = "reports"
DB_FILE = "calls.db"
PORT = 5050
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
os.makedirs(TRANSCRIPTS_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)


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


def risk_class(risk):
    if not risk:
        return "UNKNOWN"
    return str(risk).upper()


def estimate_minutes(file_path, status):
    size_mb = os.path.getsize(file_path) / (1024 * 1024)

    if status == "Queued":
        return max(1, round(size_mb * 1.5))
    if status == "Transcribing":
        return max(1, round(size_mb * 1.0))
    if status == "Analyzing":
        return max(1, round(size_mb * 1.2))

    return 1


def get_processing_files():
    processing = []
    now = time.time()
    upload_times = get_upload_times()
    processing_states = get_processing_states()

    for filename in os.listdir(UPLOAD_FOLDER):
        if not filename.lower().endswith(AUDIO_EXTENSIONS):
            continue

        file_path = os.path.join(UPLOAD_FOLDER, filename)
        call_name = os.path.splitext(filename)[0]
        transcript_path = os.path.join(TRANSCRIPTS_FOLDER, f"{call_name}.txt")
        report_path = os.path.join(REPORTS_FOLDER, f"{call_name}_report.txt")

        if os.path.exists(report_path):
            continue

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
            processing_states.get(call_name)
        )

        eta = estimate_minutes(file_path, status)

        processing.append({
            "name": call_name,
            "filename": filename,
            "status": status,
            "progress": progress,
            "message": message,
            "eta": eta,
            "uploaded_time": uploaded_time,
            "uploaded_seconds_ago": uploaded_seconds_ago
        })

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
    --bg: #f6f7f9;
    --surface: #ffffff;
    --surface-soft: #f9fafb;
    --border: #e5e7eb;
    --border-strong: #d1d5db;
    --text: #111827;
    --muted: #667085;
    --muted-strong: #475467;
    --primary: #2563eb;
    --primary-hover: #1d4ed8;
    --danger: #dc2626;
    --danger-hover: #b91c1c;
    --shadow: 0 1px 2px rgba(16, 24, 40, 0.06), 0 12px 28px rgba(16, 24, 40, 0.06);
}

body {
    margin: 0;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 15px;
    line-height: 1.5;
}

.sidebar {
    position: fixed;
    width: 264px;
    height: 100vh;
    background: var(--surface);
    color: var(--text);
    padding: 28px 20px;
    border-right: 1px solid var(--border);
}

.logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 18px;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 4px;
}

.logo::before {
    content: "";
    width: 30px;
    height: 30px;
    border-radius: 8px;
    background: linear-gradient(135deg, #2563eb, #0f766e);
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.38);
    flex: 0 0 auto;
}

.sublogo {
    color: var(--muted);
    font-size: 12px;
    font-weight: 600;
    margin: 0 0 32px 40px;
}

.sidebar a {
    display: flex;
    align-items: center;
    min-height: 42px;
    color: var(--muted-strong);
    text-decoration: none;
    padding: 10px 12px;
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 14px;
    font-weight: 700;
    border: 1px solid transparent;
}

.sidebar a:hover {
    background: #eef4ff;
    color: #1d4ed8;
    border-color: #dbeafe;
}

.main {
    margin-left: 264px;
    padding: 40px;
    max-width: 1280px;
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

.grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 28px;
}

.stat {
    background: var(--surface);
    padding: 20px;
    border-radius: 8px;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
    min-height: 120px;
}

.stat .label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
}

.stat .value {
    font-size: 34px;
    line-height: 1.1;
    font-weight: 800;
    margin-top: 14px;
}

.card {
    background: var(--surface);
    padding: 18px;
    border-radius: 8px;
    margin-bottom: 12px;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
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
    .sidebar {
        position: static;
        width: 100%;
        height: auto;
        padding: 18px;
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px 12px;
        align-items: center;
    }

    .logo,
    .sublogo {
        grid-column: 1 / -1;
    }

    .sublogo {
        margin: 0 0 4px 40px;
    }

    .main {
        margin-left: 0;
        padding: 24px 18px;
    }

    .header,
    .call-row {
        flex-direction: column;
        align-items: stretch;
    }

    .grid {
        grid-template-columns: 1fr;
    }

    .score,
    .call-row > div[style*="text-align:right"] {
        text-align: left;
    }
}
</style>
</head>
<body>

<div class="sidebar">
    <div class="logo">AI Call Auditor</div>
    <div class="sublogo">Life Insurance QA System</div>
    <a href="/">Dashboard</a>
    <a href="/upload">Upload Call</a>
</div>

<div class="main">
{{content|safe}}
</div>

</body>
</html>
"""


@app.route("/")
def dashboard():
    calls = get_calls()
    processing = get_processing_files()

    total = len(calls)
    scores = [c[4] for c in calls if c[4] is not None]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0
    high_risk = len([c for c in calls if risk_class(c[5]) == "HIGH"])

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

    setTimeout(function() {{
        window.location.href = "/";
    }}, 5000);
    </script>

    <div class="header">
        <div>
            <h1>Dashboard</h1>
            <div class="muted">Review call audits, scores, risk levels, and processing status.</div>
        </div>
        <a class="button" href="/upload">Upload Call</a>
    </div>

    <div class="grid">
        <div class="stat">
            <div class="label">Completed Calls</div>
            <div class="value">{total}</div>
        </div>
        <div class="stat">
            <div class="label">Average Score</div>
            <div class="value">{avg_score}</div>
        </div>
        <div class="stat">
            <div class="label">High Risk Calls</div>
            <div class="value">{high_risk}</div>
        </div>
    </div>
    """

    content += "<h2>Currently Processing</h2>"

    if processing:
        for item in processing:
            content += f"""
            <div class="card">
                <div class="call-row">
                    <div style="width:100%;">
                        <div class="call-title">{item["name"]}</div>
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
                        <input type="hidden" name="call_name" value="{item["name"]}">
                        <button class="delete-button" type="submit">Delete</button>
                    </form>
                </div>
            </div>
            """
    else:
        content += """
        <div class="card empty">
            No calls currently processing.
        </div>
        """

    content += "<h2>Completed Audits</h2>"

    if not calls:
        content += """
        <div class="card empty">
            No completed audits yet. Upload a call to get started.
        </div>
        """
    else:
        for c in calls:
            rid = risk_class(c[5])
            content += f"""
            <div class="card call-row">
                <a class="cardlink" href="/call/{c[0]}" style="flex:1;">
                    <div>
                        <div class="call-title">{c[1]}</div>
                        <div class="muted">{c[6]}</div>
                    </div>
                </a>

                <div style="text-align:right;">
                    <div class="score">{c[4] if c[4] is not None else "—"}</div>
                    <span class="badge {rid}">{rid}</span>
                    <form method="POST" action="/delete/{c[0]}" onsubmit="return confirm('Delete this call and its files?');" style="margin-top:10px;">
                        <button class="delete-button" type="submit">Delete</button>
                    </form>
                </div>
            </div>
            """

    return render_template_string(BASE, content=content)


@app.route("/call/<int:call_id>")
def view_call(call_id):
    call = get_call(call_id)

    if not call:
        return render_template_string(BASE, content="""
        <a class="back" href="/">← Back</a>
        <div class="card">Call not found.</div>
        """)

    rid = risk_class(call[5])

    content = f"""
    <a class="back" href="/">← Back to Dashboard</a>

    <div class="header">
        <div>
            <h1>{call[1]}</h1>
            <div class="muted">{call[6]}</div>
        </div>
        <div style="text-align:right;">
            <div class="score">{call[4] if call[4] is not None else "—"}</div>
            <span class="badge {rid}">{rid}</span>
            <form method="POST" action="/delete/{call[0]}" onsubmit="return confirm('Delete this call and its files?');" style="margin-top:10px;">
                <button class="delete-button" type="submit">Delete</button>
            </form>
        </div>
    </div>

    <div class="card">
        <h2>AI Audit Report</h2>
        <pre>{call[3]}</pre>
    </div>

    <div class="card">
        <h2>Transcript</h2>
        <pre>{call[2]}</pre>
    </div>
    """

    return render_template_string(BASE, content=content)


@app.route("/upload")
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


@app.route("/delete-processing", methods=["POST"])
def delete_processing():
    call_name = request.form.get("call_name")

    if call_name:
        remove_file(os.path.join(TRANSCRIPTS_FOLDER, f"{call_name}.txt"))
        remove_file(os.path.join(REPORTS_FOLDER, f"{call_name}_report.txt"))

        filenames = [f"{call_name}{ext}" for ext in AUDIO_EXTENSIONS]
        for filename in filenames:
            remove_file(os.path.join(UPLOAD_FOLDER, filename))
        forget_upload_times(filenames)
        forget_processing_states([call_name])

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("DELETE FROM calls WHERE call_name=?", (call_name,))
        conn.commit()
        conn.close()

    return redirect("/")


@app.route("/delete/<int:call_id>", methods=["POST"])
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
        for filename in filenames:
            remove_file(os.path.join(UPLOAD_FOLDER, filename))
        forget_upload_times(filenames)
        forget_processing_states([call_name])

    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=PORT)
