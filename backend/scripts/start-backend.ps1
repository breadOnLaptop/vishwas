<#
Start-backend.ps1
Location: vishwas/backend/scripts/start-backend.ps1

Purpose:
 - Ensure .venv exists (create if missing)
 - Install requirements (if missing or if -InstallOnly not used)
 - Activate the .venv in this PowerShell session
 - Ensure GOOGLE_APPLICATION_CREDENTIALS in backend .env points to google-credentials.json
 - Start the backend server (uvicorn) from vishwas/backend root

Usage:
  PS> .\start-backend.ps1
  PS> .\start-backend.ps1 -SkipInstall
  PS> .\start-backend.ps1 -InstallOnly
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
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
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
} catch {
    Err "Python not found in PATH. Install Python 3.10+ and re-run this script."
    exit 1
}

# 2) Create virtualenv if missing
if (-not (Test-Path $venvPath)) {
    Info "Creating virtual environment at $venvPath ..."
    python -m venv $venvPath
    Ok "Virtual environment created."
} else {
    Info "Virtual environment already exists at $venvPath"
}

# 3) Activate virtualenv in this session
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    Info "Activating virtual environment..."
    & $activateScript
    Ok "Virtual environment activated in current session."
} else {
    Err "Activation script not found at $activateScript. Activate manually: .\.venv\Scripts\Activate.ps1"
    exit 1
}

# 4) Install requirements (unless skipped)
if (-not $SkipInstall) {
    if (Test-Path $requirementsPath) {
        Info "Installing/updating pip and installing requirements..."
        python -m pip install --upgrade pip setuptools wheel
        pip install -r $requirementsPath
        Ok "Dependencies installed."
    } else {
        Warn "requirements.txt not found at $requirementsPath - skipping pip install."
    }
} else {
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
} else {
    Info ".env found at $envPath"
}

# 6) Ensure google-credentials.json exists where you said it does
if (Test-Path $credentialsHostPath) {
    Info "Found google-credentials.json at $credentialsHostPath"
} else {
    Warn "google-credentials.json not found at expected path: $credentialsHostPath"
    Warn "If you have the service-account JSON elsewhere, copy it to $credentialsHostPath or update .env accordingly."
}

# 7) Load .env and inject into current process environment safely (use Set-Item for dynamic names)
try {
    if (Test-Path $envPath) {
        $envText = Get-Content -Raw -Path $envPath -ErrorAction SilentlyContinue
        if ($envText) {
            # split into lines and process each non-comment, non-empty line
            $lines = $envText -split "`r?`n"
            foreach ($line in $lines) {
                $trimmed = $line.Trim()
                if ($trimmed -eq "" -or $trimmed.StartsWith("#")) { continue }
                # match KEY=VALUE
                if ($trimmed -match '^\s*([^=]+)\s*=(.*)$') {
                    $k = $Matches[1].Trim()
                    $v = $Matches[2].Trim()
                    if ($v -eq "") { continue }
                    # if relative path like ./file -> resolve to absolute path
                    if ($k -eq "GOOGLE_APPLICATION_CREDENTIALS" -and $v -like './*') {
                        $rel = $v -replace '^\./',''
                        $resolved = (Resolve-Path (Join-Path $backendRoot $rel) -ErrorAction SilentlyContinue)
                        if ($resolved) {
                            $v = $resolved.Path
                        } else {
                            # leave as-is (will likely be resolved by SDK if relative to WORKDIR)
                        }
                    }
                    # Set environment variable for current process (so child uvicorn inherits)
                    try {
                        Set-Item -Path ("Env:" + $k) -Value $v -ErrorAction Stop
                        Info "Set environment variable $k"
                    } catch {
                        Warn "Failed to set environment variable $k : $_"
                        # fallback: set via .NET process env
                        [System.Environment]::SetEnvironmentVariable($k, $v, "Process")
                    }
                }
            }
        }
    }
} catch {
    Warn "Failed to load .env into process environment: $_"
}

# If only installing, exit now
if ($InstallOnly) {
    Ok "Install-only mode: done. Activate the venv and run server manually or re-run without -InstallOnly."
    exit 0
}

# 8) Run uvicorn dev server (foreground) with reload
Info "Starting development server (uvicorn) from $backendRoot ..."
Info "Press Ctrl+C to stop the server."

try {
    # Use the app object directly from the package (app:app -> app object defined in app/__init__.py)
    python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
} catch {
    Err "Failed to start server: $_"
    exit 1
}
