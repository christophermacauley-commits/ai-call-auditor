param(
    [Parameter(Mandatory=$true)]
    [string]$CsvPath,

    [Parameter(Mandatory=$true)]
    [string]$OutDir
)

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$lines = Get-Content $CsvPath
$urls = $lines | ForEach-Object {
    if ($_ -match '(http://[^"]+\.mp3)') { $Matches[1] }
}

$downloaded = 0
$skipped = 0

foreach ($url in $urls) {
    $filename = Split-Path $url -Leaf
    $out = Join-Path $OutDir $filename

    if (Test-Path $out) {
        Write-Host "SKIP existing: $filename"
        $skipped += 1
    } else {
        Write-Host "DOWNLOADING: $filename"
        Invoke-WebRequest -Uri $url -OutFile $out
        $downloaded += 1
    }
}

Write-Host "DONE downloaded=$downloaded skipped=$skipped"
