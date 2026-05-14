param(
    [Parameter(Mandatory=$true)]
    [string]$Date,

    [int]$BatchLimit = 10,

    [int]$MinSeconds = 0,

    [string]$FilenamePrefix = ""
)

$Root = "C:\AI-Auditor\ai-auditor"
$IncomingDir = Join-Path $Root "incoming_calls\$Date"
$CallsDir = Join-Path $Root "calls"
$ProcessedDir = Join-Path $Root "processed_calls"
$ReportsDir = Join-Path $Root "reports"
$ExportsDir = Join-Path $Root "vicidial_exports"

New-Item -ItemType Directory -Force -Path $CallsDir | Out-Null

if (-not (Test-Path $IncomingDir)) {
    Write-Host "Incoming folder not found: $IncomingDir"
    exit 0
}

$currentQueue = @(Get-ChildItem $CallsDir -Filter "*.mp3" -File -ErrorAction SilentlyContinue).Count
$space = $BatchLimit - $currentQueue

if ($space -le 0) {
    Write-Host "Queue already has $currentQueue calls. No new calls added."
    exit 0
}

$durationByFile = @{}

if ($MinSeconds -gt 0 -and (Test-Path $ExportsDir)) {
    $csvs = Get-ChildItem $ExportsDir -Filter "${Date}_*_user_stats_*.csv" -File -ErrorAction SilentlyContinue

    foreach ($csv in $csvs) {
        $rows = Import-Csv $csv.FullName -Header Blank,Num,Lead,DateTime,Seconds,RecId,Filename,Location |
            Where-Object { $_.Filename -and $_.Filename -ne "FILENAME" -and $_.Seconds -match '^\d+$' }

        foreach ($row in $rows) {
            $mp3Name = "$($row.Filename)-all.mp3"
            $durationByFile[$mp3Name] = [int]$row.Seconds
        }
    }
}

$queued = 0
$skippedPrefix = 0
$skippedShort = 0
$skippedDone = 0
$skippedExisting = 0

$mp3s = Get-ChildItem $IncomingDir -Filter "*.mp3" -File -ErrorAction SilentlyContinue | Sort-Object Name

foreach ($mp3 in $mp3s) {
    if ($queued -ge $space) {
        break
    }

    if ($FilenamePrefix -and -not $mp3.Name.StartsWith($FilenamePrefix)) {
        $skippedPrefix += 1
        continue
    }

    if ($MinSeconds -gt 0) {
        if (-not $durationByFile.ContainsKey($mp3.Name) -or $durationByFile[$mp3.Name] -lt $MinSeconds) {
            $skippedShort += 1
            continue
        }
    }

    $callName = [System.IO.Path]::GetFileNameWithoutExtension($mp3.Name)
    $dest = Join-Path $CallsDir $mp3.Name
    $processed = Join-Path $ProcessedDir $mp3.Name
    $report = Join-Path $ReportsDir "$($callName)_report.txt"

    if ((Test-Path $dest) -or (Test-Path $processed) -or (Test-Path $report)) {
        $skippedDone += 1
        continue
    }

    Copy-Item $mp3.FullName $dest -Force
    Write-Host "QUEUED for audit: $($mp3.Name)"
    $queued += 1
}

Write-Host "DONE queue_next_batch date=$Date queued=$queued current_before=$currentQueue batch_limit=$BatchLimit min_seconds=$MinSeconds prefix=$FilenamePrefix skipped_prefix=$skippedPrefix skipped_short=$skippedShort skipped_done=$skippedDone"
