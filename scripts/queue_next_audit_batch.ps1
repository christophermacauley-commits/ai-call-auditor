param(
    [Parameter(Mandatory=$true)]
    [string]$Date,

    [int]$BatchLimit = 10
)

$Root = "C:\AI-Auditor\ai-auditor"
$IncomingDir = Join-Path $Root "incoming_calls\$Date"
$CallsDir = Join-Path $Root "calls"

New-Item -ItemType Directory -Force -Path $CallsDir | Out-Null

$currentQueue = @(Get-ChildItem $CallsDir -Filter "*.mp3" -File -ErrorAction SilentlyContinue).Count
$space = $BatchLimit - $currentQueue

if ($space -le 0) {
    Write-Host "Queue already has $currentQueue calls. No new calls added."
    exit 0
}

$queued = 0
$mp3s = Get-ChildItem $IncomingDir -Filter "*.mp3" -File -ErrorAction SilentlyContinue | Sort-Object Name

foreach ($mp3 in $mp3s) {
    if ($queued -ge $space) {
        break
    }

    $dest = Join-Path $CallsDir $mp3.Name

    if (Test-Path $dest) {
        continue
    }

    Copy-Item $mp3.FullName $dest -Force
    Write-Host "QUEUED for audit: $($mp3.Name)"
    $queued += 1
}

Write-Host "DONE queue_next_batch date=$Date queued=$queued current_before=$currentQueue batch_limit=$BatchLimit"
