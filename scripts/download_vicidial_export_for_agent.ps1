param(
    [Parameter(Mandatory=$true)]
    [string]$Date,

    [Parameter(Mandatory=$true)]
    [string]$Agent
)

$url = "https://premiersource.vicihost.com/vicidial/user_stats.php?DB=&pause_code_rpt=&park_rpt=&did_id=&did=&begin_date=$Date&end_date=$Date&user=$Agent&submit=&search_archived_data=&NVAuser=&file_download=8"

Start-Process chrome.exe -ArgumentList '--profile-directory="Profile 5"', $url
