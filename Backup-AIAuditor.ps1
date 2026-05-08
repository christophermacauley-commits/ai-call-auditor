$ErrorActionPreference = "Stop"

$Source = "C:\AI-Auditor\ai-auditor"
$BackupRoot = "C:\AI-Auditor\backups"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$BackupDir = Join-Path $BackupRoot "ai-auditor-backup-$Timestamp"

New-Item -ItemType Directory -Force $BackupDir | Out-Null

$Items = @(
    "calls.db",
    ".env",
    "reports",
    "transcripts",
    "processed_calls",
    "processed_transcripts",
    "transcripts_role_labeled",
    "logs"
)

foreach ($Item in $Items) {
    $Path = Join-Path $Source $Item
    if (Test-Path $Path) {
        Copy-Item $Path $BackupDir -Recurse -Force
        Write-Host "Backed up $Item"
    } else {
        Write-Host "Skipped missing $Item"
    }
}

Write-Host ""
Write-Host "Backup complete:"
Write-Host $BackupDir