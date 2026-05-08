Set-Location "C:\AI-Auditor\ai-auditor"

$env:Path = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;" + $env:Path

.\venv\Scripts\python.exe native_app.py