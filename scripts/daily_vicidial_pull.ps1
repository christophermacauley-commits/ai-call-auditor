param(
    [Parameter(Mandatory=$true)]
    [string]$Date,

    [int]$BatchLimit = 10
)

$Root = "C:\AI-Auditor\ai-auditor"
$AgentMap = Join-Path $Root "training\agent_map.txt"
$ExportDir = Join-Path $Root "vicidial_exports"
$IncomingDir = Join-Path $Root "incoming_calls\$Date"
$CallsDir = Join-Path $Root "calls"
$DownloadsDir = $ExportDir

New-Item -ItemType Directory -Force -Path $ExportDir | Out-Null
New-Item -ItemType Directory -Force -Path $IncomingDir | Out-Null
New-Item -ItemType Directory -Force -Path $CallsDir | Out-Null

$agents = Get-Content $AgentMap | Where-Object { $_ -match '^\s*\d+=' }

foreach ($line in $agents) {
    $agent = ($line -split '=', 2)[0].Trim()
    $name = ($line -split '=', 2)[1].Trim()

    Write-Host "EXPORT agent=$agent name=$name"

    $before = Get-Date

    powershell -ExecutionPolicy Bypass -File "$Root\scripts\download_vicidial_export_for_agent.ps1" -Date $Date -Agent $agent

    Start-Sleep -Seconds 4

    $newest = Get-ChildItem "$DownloadsDir\user_stats_*.csv" -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -ge $before.AddSeconds(-2) } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $newest) {
        Write-Host "NO NEW CSV FOUND for agent=$agent"
        continue
    }

    $safeName = $name -replace '[^a-zA-Z0-9_-]', '_'
    $destCsv = Join-Path $ExportDir "${Date}_${agent}_${safeName}_$($newest.Name)"
    Copy-Item $newest.FullName $destCsv -Force

    powershell -ExecutionPolicy Bypass -File "$Root\scripts\download_recordings_from_csv.ps1" -CsvPath $destCsv -OutDir $IncomingDir
}

$queued = 0
$mp3s = Get-ChildItem $IncomingDir -Filter "*.mp3" -File | Sort-Object Name

foreach ($mp3 in $mp3s) {
    if ($queued -ge $BatchLimit) {
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

powershell -ExecutionPolicy Bypass -File "$Root\scripts\cleanup_old_incoming_calls.ps1" -RootDir "$Root\incoming_calls" -DaysToKeep 45

Write-Host "DONE daily pull for $Date queued=$queued batch_limit=$BatchLimit"
