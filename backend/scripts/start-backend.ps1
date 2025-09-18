<#
Cross-platform start-backend.ps1
Location: vishwas/backend/scripts/start-backend.ps1

This script is written to work both on Windows PowerShell and PowerShell Core (pwsh) on Linux.
It will:
 - ensure .venv exists (create if missing)
 - install requirements (unless -SkipInstall)
 - activate virtualenv in the pwsh session (uses Activate.ps1 path for Windows vs Linux)
 - load .env and set process env vars
 - start uvicorn (blocking) and show logs

Usage (local):
  # Run and install deps if needed
  pwsh .\scripts\start-backend.ps1

  # skip installation
  pwsh .\scripts\start-backend.ps1 -SkipInstall

  # only install and exit
  pwsh .\scripts\start-backend.ps1 -InstallOnly
#>

param(
  [switch]$SkipInstall = $false,
  [switch]$InstallOnly = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Info($m) { Write-Host "[INFO] $m" -ForegroundColor Cyan }
function Ok($m) { Write-Host "[OK] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Err($m) { Write-Host "[ERROR] $m" -ForegroundColor Red }

# Resolve script dir and project backend root
$scriptPath = $MyInvocation.MyCommand.Path
$scriptDir = Split-Path -Parent $scriptPath
$backendRoot = (Resolve-Path (Join-Path $scriptDir "..")) | Select-Object -First 1
Set-Location $backendRoot
Info "Working directory: $($backendRoot.Path)"

# Paths expected
$venvPath = Join-Path $backendRoot ".venv"
$requirementsPath = Join-Path $backendRoot "requirements.txt"
$envPath = Join-Path $backendRoot ".env"
$credentialsHostPath = Join-Path $backendRoot "google-credentials.json"

# 1) Ensure Python exists
try {
  $py = & python --version 2>&1
  Info "Python found: $py"
}
catch {
  Err "Python not found in PATH. Install Python 3.10+ and re-run this script."
  exit 1
}

# 2) Create virtualenv if missing
if (-not (Test-Path $venvPath)) {
  Info "Creating virtual environment at $venvPath ..."
  python -m venv $venvPath
  Ok "Virtual environment created."
}
else {
  Info "Virtual environment already exists at $venvPath"
}

# 3) Determine correct activate script based on platform
$activateScript = $null
$onWindows = $false

try {
  # Works in both Windows PowerShell and PowerShell Core
  if (($PSVersionTable -and $PSVersionTable.PSEdition -eq "Desktop") -or ($env:OS -and $env:OS -match "Windows_NT")) {
    $onWindows = $true
  }
}
catch {
  # fallback: assume Linux if detection fails
  $onWindows = $false
}

if ($onWindows) {
  $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
}
else {
  $activateScript = Join-Path $venvPath "bin/Activate.ps1"
  if (-not (Test-Path $activateScript)) {
    $activateScript = Join-Path $venvPath "bin/activate"
  }
}


if (Test-Path $activateScript) {
  Info "Activating virtual environment using: $activateScript"
  try {
    if ($activateScript -like "*.ps1") {
      # run the PowerShell activation script in this session
      . $activateScript
    }
    else {
      # it's likely a bash activate script; we are in pwsh => prefer using venv python directly.
      Info "Bash-style activate script detected; will use venv python directly."
    }
    Ok "Virtual environment activation attempted."
  }
  catch {
    Warn "Activation failed: $_"
  }
}
else {
  Warn "No activation script found at expected locations. Proceeding without activation (will invoke venv python directly)."
}

# Helper to run pip / python from venv when activation not suitable
function Get-VenvPythonPath {
  if ($onWindows) {
    return Join-Path $venvPath "Scripts\python.exe"
  }
  else {
    return Join-Path $venvPath "bin/python"
  }
}

$venvPython = Get-VenvPythonPath
if (-not (Test-Path $venvPython)) {
  Warn "Venv python not found at $venvPython - falling back to system python."
  $venvPython = "python"
}

# 4) Install requirements (unless skipped)
if (-not $SkipInstall) {
  if (Test-Path $requirementsPath) {
    Info "Installing/updating pip and installing requirements using $venvPython..."
    & $venvPython -m pip install --upgrade pip setuptools wheel
    & $venvPython -m pip install -r $requirementsPath
    Ok "Dependencies installed into venv."
  }
  else {
    Warn "requirements.txt not found at $requirementsPath - skipping pip install."
  }
}
else {
  Info "Skipping dependency installation (--SkipInstall)."
}

# 5) Ensure .env exists; if not, create a basic .env pointing GOOGLE_APPLICATION_CREDENTIALS to google-credentials.json
if (-not (Test-Path $envPath)) {
  Warn ".env not found at backend root. Creating a template .env."
  @"
# Backend .env - fill other variables as needed
GOOGLE_APPLICATION_CREDENTIALS=./google-credentials.json
GCP_PROJECT=
GCP_REGION=us-central1
TEXT_MODEL_ID=gemini-2.5-flash
GOOGLE_SEARCH_API_KEY=
GOOGLE_SEARCH_CX=
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASS=
"@ | Out-File -FilePath $envPath -Encoding utf8
  Ok "Created .env template at $envPath. Edit to add values if needed."
}
else {
  Info ".env found at $envPath"
}

# 6) Ensure google-credentials.json exists where you said it does
if (Test-Path $credentialsHostPath) {
  Info "Found google-credentials.json at $credentialsHostPath"
}
else {
  Warn "google-credentials.json not found at expected path: $credentialsHostPath"
  Warn "If you have the service-account JSON elsewhere, copy it to $credentialsHostPath or update .env accordingly."
}

# 7) Load .env and inject into current process environment safely
try {
  if (Test-Path $envPath) {
    $envText = Get-Content -Raw -Path $envPath -ErrorAction SilentlyContinue
    if ($envText) {
      $lines = $envText -split "`r?`n"
      foreach ($line in $lines) {
        $trimmed = $line.Trim()
        if ($trimmed -eq "" -or $trimmed.StartsWith("#")) { continue }
        if ($trimmed -match '^\s*([^=]+)\s*=(.*)$') {
          $k = $Matches[1].Trim()
          $v = $Matches[2].Trim()
          if ($v -eq "") { continue }
          if ($k -eq "GOOGLE_APPLICATION_CREDENTIALS" -and $v -like './*') {
            $rel = $v -replace '^\./', ''
            $resolved = (Resolve-Path (Join-Path $backendRoot $rel) -ErrorAction SilentlyContinue)
            if ($resolved) {
              $v = $resolved.Path
            }
          }
          try {
            Set-Item -Path ("Env:" + $k) -Value $v -ErrorAction Stop
            Info "Set environment variable $k"
          }
          catch {
            Warn "Failed to set environment variable $k : $_"
            [System.Environment]::SetEnvironmentVariable($k, $v, "Process")
          }
        }
      }
    }
  }
}
catch {
  Warn "Failed to load .env into process environment: $_"
}

# If only installing, exit now
if ($InstallOnly) {
  Ok "Install-only mode: done. Activate the venv and run server manually or re-run without -InstallOnly."
  exit 0
}

# 8) Start uvicorn (use venv python to guarantee correct interpreter)
Info "Starting development server (uvicorn) using $venvPython ..."
Info "Press Ctrl+C to stop the server."

try {
  & $venvPython -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
}
catch {
  Err "Failed to start server: $_"
  exit 1
}
