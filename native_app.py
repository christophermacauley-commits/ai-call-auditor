import subprocess
import time
import webview
import os

PROJECT = os.path.expanduser("~/Applications/ai-auditor")
PYTHON = os.path.join(PROJECT, "env311/bin/python")
URL = "http://127.0.0.1:5050"

os.chdir(PROJECT)
os.makedirs("logs", exist_ok=True)

# kill old processes so they don’t stack
subprocess.run(["pkill", "-f", "watcher.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(["pkill", "-f", "dashboard.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

watcher = subprocess.Popen(
    [PYTHON, "watcher.py"],
    stdout=open("logs/watcher.log", "a"),
    stderr=open("logs/watcher.log", "a")
)

dashboard = subprocess.Popen(
    [PYTHON, "dashboard.py"],
    stdout=open("logs/dashboard.log", "a"),
    stderr=open("logs/dashboard.log", "a")
)

time.sleep(2)

def cleanup():
    for proc in [watcher, dashboard]:
        try:
            proc.terminate()
        except Exception:
            pass

webview.create_window(
    "AI Call Auditor",
    URL,
    width=1200,
    height=850,
    resizable=True,
    fullscreen=False,
    frameless=False,
    easy_drag=True
)

try:
    webview.start()
finally:
    cleanup()
