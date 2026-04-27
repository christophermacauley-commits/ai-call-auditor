import os
import subprocess
import threading
import time
import webview
from webview.menu import Menu, MenuAction

PROJECT = os.path.expanduser("~/Applications/ai-auditor")
PYTHON = os.path.join(PROJECT, "venv/bin/python")
URL = "http://127.0.0.1:5050"

RESTARTING_HTML = """
<!doctype html>
<html>
<head>
    <title>Restarting AI Auditor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f5f7fb;
            color: #111827;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
        }
        .card {
            background: white;
            padding: 32px;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
            text-align: center;
            width: 420px;
        }
        h1 {
            margin-top: 0;
            font-size: 24px;
        }
        p {
            color: #4b5563;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>Restarting AI Auditor...</h1>
        <p>Please wait while dashboard and watcher restart.</p>
    </div>
</body>
</html>
"""

os.chdir(PROJECT)
os.makedirs("logs", exist_ok=True)

watcher = None
dashboard = None
_watcher_log = None
_dashboard_log = None
_window = None


def stop_auditor_services():
    global watcher, dashboard, _watcher_log, _dashboard_log
    for proc in (watcher, dashboard):
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
    for proc in (watcher, dashboard):
        if proc is None:
            continue
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
    for logf in (_watcher_log, _dashboard_log):
        if logf is not None:
            try:
                logf.close()
            except Exception:
                pass
    watcher = dashboard = None
    _watcher_log = _dashboard_log = None


def start_auditor_services():
    global watcher, dashboard, _watcher_log, _dashboard_log
    _watcher_log = open("logs/watcher.log", "a")
    watcher = subprocess.Popen(
        [PYTHON, "watcher.py"],
        stdout=_watcher_log,
        stderr=_watcher_log,
    )
    _dashboard_log = open("logs/dashboard.log", "a")
    dashboard = subprocess.Popen(
        [PYTHON, "dashboard.py"],
        stdout=_dashboard_log,
        stderr=_dashboard_log,
    )


def restart_auditor_services():
    def _run():
        print("Restarting AI Auditor services...")
        w = _window
        if w is not None:
            try:
                w.load_html(RESTARTING_HTML)
            except Exception:
                pass
        stop_auditor_services()
        start_auditor_services()
        time.sleep(3)
        print("AI Auditor services restarted.")
        if w is not None:
            print("Reloading AI Auditor window...")
            try:
                w.load_url(URL)
            except Exception:
                try:
                    w.evaluate_js("window.location.href = '%s';" % URL)
                except Exception as e:
                    print(f"Could not reload AI Auditor window: {e}")

    threading.Thread(target=_run, daemon=True).start()


start_auditor_services()
time.sleep(2)


def cleanup():
    stop_auditor_services()


_window = webview.create_window(
    "AI Call Auditor",
    URL,
    width=1200,
    height=850,
    resizable=True,
    fullscreen=False,
    frameless=False,
    easy_drag=True,
    menu=[
        Menu(
            "Auditor",
            [
                MenuAction("Restart Auditor", restart_auditor_services),
            ],
        ),
    ],
)

try:
    webview.start()
finally:
    cleanup()
