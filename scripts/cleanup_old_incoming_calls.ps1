param(
    [string]$RootDir = "incoming_calls",
    [int]$DaysToKeep = 45
)

$cutoff = (Get-Date).AddDays(-$DaysToKeep)

if (-not (Test-Path $RootDir)) {
    Write-Host "No cleanup needed. Folder not found: $RootDir"
    exit 0
}

$files = Get-ChildItem $RootDir -Recurse -File | Where-Object { $_.LastWriteTime -lt $cutoff }

foreach ($file in $files) {
    Write-Host "DELETE old file: $($file.FullName)"
    Remove-Item $file.FullName -Force
}

$emptyDirs = Get-ChildItem $RootDir -Recurse -Directory | Sort-Object FullName -Descending | Where-Object {
    -not (Get-ChildItem $_.FullName -Force)
}

foreach ($dir in $emptyDirs) {
    Write-Host "DELETE empty folder: $($dir.FullName)"
    Remove-Item $dir.FullName -Force
}

Write-Host "DONE cleanup deleted_files=$($files.Count)"
