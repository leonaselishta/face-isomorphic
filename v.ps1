param(
    [string]$env
)
if ($env -eq "d"){
    .\venv-deepface\Scripts\Activate
}
elseif ($env -eq "m"){
    .\venv\Scripts\Activate
}
else {
    Write-Host "Invalid environment. Please specify 'd' for deepface or 'm' for mediapipe."
}