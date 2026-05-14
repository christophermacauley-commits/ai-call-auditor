param(
    [string]$RootDir = "incoming_calls",
    [int]$DaysToKeep = 45,
    [switch]$DryRun
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$goldenFile = Join-Path $repoRoot "golden_cases\golden18.json"
$backupScript = Join-Path $repoRoot "scripts\backup_golden_fixtures.py"

if (Test-Path $backupScript) {
    python $backupScript --label before_incoming_cleanup
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Golden backup failed. Aborting cleanup."
        exit 1
    }
}

$goldenNames = @()

if (Test-Path $goldenFile) {
    $json = Get-Content $goldenFile -Raw | ConvertFrom-Json
    foreach ($case in $json.cases) {
        if ($case.id) { $goldenNames += [string]$case.id }
        if ($case.match) { $goldenNames += [string]$case.match }
    }
}

$goldenNames = $goldenNames | Where-Object { $_ } | Sort-Object -Unique

function Test-ProtectedGoldenFile {
    param([System.IO.FileInfo]$File)

    $lower = $File.Name.ToLowerInvariant()

    foreach ($name in $goldenNames) {
        $n = $name.ToLowerInvariant()
        if ($lower -eq "$n.txt" -or $lower -eq "$n`_report.txt") {
            return $true
        }
        if ($lower.StartsWith($n) -and ($lower.EndsWith(".txt") -or $lower.EndsWith("_report.txt") -or $lower.EndsWith(".mp3") -or $lower.EndsWith(".wav") -or $lower.EndsWith(".m4a"))) {
            return $true
        }
    }

    return $false
}

$cutoff = (Get-Date).AddDays(-$DaysToKeep)

if (-not (Test-Path $RootDir)) {
    Write-Host "No cleanup needed. Folder not found: $RootDir"
    exit 0
}

$files = Get-ChildItem $RootDir -Recurse -File | Where-Object { $_.LastWriteTime -lt $cutoff }

$deleted = 0
$protected = 0

foreach ($file in $files) {
    if (Test-ProtectedGoldenFile $file) {
        Write-Host "SKIP protected golden/test fixture: $($file.FullName)"
        $protected += 1
        continue
    }

    if ($DryRun) {
        Write-Host "WOULD DELETE old file: $($file.FullName)"
    } else {
        Write-Host "DELETE old file: $($file.FullName)"
        Remove-Item $file.FullName -Force
    }
    $deleted += 1
}

$emptyDirs = Get-ChildItem $RootDir -Recurse -Directory | Sort-Object FullName -Descending | Where-Object {
    -not (Get-ChildItem $_.FullName -Force)
}

foreach ($dir in $emptyDirs) {
    if ($DryRun) {
        Write-Host "WOULD DELETE empty folder: $($dir.FullName)"
    } else {
        Write-Host "DELETE empty folder: $($dir.FullName)"
        Remove-Item $dir.FullName -Force
    }
}

Write-Host "DONE cleanup deleted_files=$deleted protected_skipped=$protected dry_run=$DryRun"
