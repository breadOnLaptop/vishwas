<#
start-backend.ps1
vishwas/backend/scripts/start-backend.ps1

- Creates/activates .venv (if needed)
- Installs requirements (unless -SkipInstall)
- Optionally installs & authenticates gcloud using service-account JSON (Windows)
- Loads .env into process environment
- Starts uvicorn using the venv python
#>

param(
    [switch]$SkipInstall = $false,
    [switch]$InstallOnly = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Info($m)  { Write-Host "[INFO]  $m" -ForegroundColor Cyan }
function Ok($m)    { Write-Host "[OK]    $m" -ForegroundColor Green }
function Warn($m)  { Write-Host "[WARN]  $m" -ForegroundColor Yellow }
function Err($m)   { Write-Host "[ERROR] $m" -ForegroundColor Red }

################################################################################
# Initialize-GCloudAuth
# Installs gcloud (Windows) if not present and activates a service-account key.
# Returns $true on success (gcloud available and auth attempted), $false otherwise.
################################################################################
function Initialize-GCloudAuth {
    param(
        [Parameter(Mandatory=$true)][string]$KeyFilePath,
        [string]$ProjectId = $null,
        [string[]]$ApisToEnable = @()
    )

    if (-not (Test-Path -Path $KeyFilePath)) {
        Warn "gcloud key file not found at '$KeyFilePath'; skipping gcloud auth."
        return $false
    }

    # Check if gcloud is present
    $gcloudFound = $false
    try {
        $ver = & gcloud --version 2>&1
        $gcloudFound = $true
        Info "gcloud already installed: $ver"
    } catch {
        $gcloudFound = $false
    }

    if (-not $gcloudFound) {
        Info "gcloud not found. Preparing to install Google Cloud SDK (Windows)..."

        # Installation requires admin privileges for system-wide install
        try {
            $isAdmin = ( ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator) )
        } catch {
            $isAdmin = $false
        }

        if (-not $isAdmin) {
            Warn "Installing gcloud requires Administrator privileges. Please re-run this script as Administrator to install gcloud automatically."
            return $false
        }

        $tmpExe = Join-Path $env:TEMP "GoogleCloudSDKInstaller.exe"
        $downloadUrl = "https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe"

        try {
            Info "Downloading Google Cloud SDK installer to $tmpExe ..."
            Invoke-WebRequest -Uri $downloadUrl -OutFile $tmpExe -ErrorAction Stop
            Info "Running installer (silent)..."
            Start-Process -FilePath $tmpExe -ArgumentList "/S" -Wait -NoNewWindow
            Info "Installer finished; attempting to detect gcloud."
        } catch {
            Warn "Failed to download or run the installer: $_"
            return $false
        }

        # quick re-check
        for ($i = 0; $i -lt 5; $i++) {
            Start-Sleep -Seconds 1
            try {
                $ver = & gcloud --version 2>&1
                $gcloudFound = $true
                break
            } catch {
                $gcloudFound = $false
            }
        }

        if (-not $gcloudFound) {
            Warn "gcloud still not found in PATH after install. You may need to restart your shell. Continuing without gcloud."
            return $false
        } else {
            Info "gcloud is now available: $ver"
        }
    }

    # Activate service account using provided key
    try {
        Info "Activating service account using key: $KeyFilePath"
        & gcloud auth activate-service-account --key-file="$KeyFilePath" --quiet
        Info "Service account activated."
    } catch {
        Warn "gcloud auth activate-service-account failed: $_"
        return $false
    }

    # Set project if available
    if (-not $ProjectId) { $ProjectId = $env:GCP_PROJECT }
    if ($ProjectId) {
        try {
            & gcloud config set project $ProjectId --quiet
            Info "gcloud project set to $ProjectId"
        } catch {
            Warn "gcloud config set project failed: $_"
        }
    }

    # Optionally enable APIs (non-interactive; may require permissions/billing)
    if ($ApisToEnable -and $ApisToEnable.Count -gt 0) {
        foreach ($api in $ApisToEnable) {
            try {
                Info "Enabling API: $api ..."
                & gcloud services enable $api --quiet
                Info "Enabled $api"
            } catch {
                Warn "Failed to enable API $api : $_"
            }
        }
    }

    return $true
}

################################################################################
# Main script execution
################################################################################

# Figure out backend root (script located in scripts/)
$scriptPath = $MyInvocation.MyCommand.Path
$scriptDir  = Split-Path -Parent $scriptPath
$backendRoot = (Resolve-Path (Join-Path $scriptDir "..")) | Select-Object -First 1
Set-Location $backendRoot
Info "Working directory: $($backendRoot.Path)"

$venvPath = Join-Path $backendRoot ".venv"
$requirementsPath = Join-Path $backendRoot "requirements.txt"
$envPath = Join-Path $backendRoot ".env"
$credentialsHostPath = Join-Path $backendRoot "google-credentials.json"

# Ensure python exists
try {
    $pyver = & python --version 2>&1
    Info "Python found: $pyver"
} catch {
    Err "Python not found in PATH. Install Python 3.10+ and re-run."
    exit 1
}

# Create venv if missing
if (-not (Test-Path $venvPath)) {
    Info "Creating virtualenv at $venvPath ..."
    python -m venv $venvPath
    Ok "Virtualenv created."
} else {
    Info "Virtualenv already exists at $venvPath"
}

# Determine platform (Windows vs non-Windows)
$onWindows = $false
try {
    if (($PSVersionTable -and $PSVersionTable.PSEdition -eq "Desktop") -or ($env:OS -and $env:OS -match "Windows_NT")) {
        $onWindows = $true
    }
} catch {
    $onWindows = $false
}

# activation script path
if ($onWindows) {
    $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
} else {
    $activateScript = Join-Path $venvPath "bin/Activate.ps1"
    if (-not (Test-Path $activateScript)) {
        $activateScript = Join-Path $venvPath "bin/activate"
    }
}

if (Test-Path $activateScript) {
    Info "Activating venv using $activateScript"
    try {
        if ($activateScript -like "*.ps1") {
            . $activateScript
            Ok "Activated venv via PowerShell activation script."
        } else {
            Info "Detected bash-style activate; will invoke venv python directly for commands."
        }
    } catch {
        Warn "Activation failed: $_"
    }
} else {
    Warn "No activation script found; will use venv python path directly."
}

function Get-VenvPython {
    if ($onWindows) { return Join-Path $venvPath "Scripts\python.exe" } else { return Join-Path $venvPath "bin/python" }
}
$venvPython = Get-VenvPython
if (-not (Test-Path $venvPython)) {
    Warn "Venv python not found at $venvPython; falling back to system python."
    $venvPython = "python"
}

# Optionally initialize gcloud (Windows) - non-fatal if it fails
# Call this if you want host gcloud available for enabling APIs / administrative tasks.
# If you don't want to auto-install gcloud, comment the next block out.
$apiList = @("vision.googleapis.com","aiplatform.googleapis.com","customsearch.googleapis.com")
try {
    $gok = Initialize-GCloudAuth -KeyFilePath $credentialsHostPath -ProjectId $env:GCP_PROJECT -ApisToEnable $apiList
    if (-not $gok) {
        Warn "Initialize-GCloudAuth did not complete successfully (gcloud not available or auth failed). Host-level gcloud tasks may not work."
    } else {
        Ok "gcloud available and authenticated (host-level)."
    }
} catch {
    Warn "Initialize-GCloudAuth raised an error: $_"
}

# Install requirements unless skipped
if (-not $SkipInstall) {
    if (Test-Path $requirementsPath) {
        Info "Installing/updating pip and requirements into venv ($venvPython)..."
        & $venvPython -m pip install --upgrade pip setuptools wheel
        & $venvPython -m pip install -r $requirementsPath
        Ok "Dependencies installed."
    } else {
        Warn "requirements.txt not found at $requirementsPath"
    }
} else {
    Info "Skipping dependency installation (SkipInstall specified)."
}

# Ensure .env exists; if not create a minimal one
if (-not (Test-Path $envPath)) {
    Warn ".env not found; creating template .env"
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
    Ok "Created template .env at $envPath"
} else {
    Info ".env found at $envPath"
}

# Ensure google credentials exist at expected path (warn otherwise)
if (Test-Path $credentialsHostPath) {
    Ok "Found google-credentials.json at $credentialsHostPath"
} else {
    Warn "google-credentials.json not found at $credentialsHostPath; ensure your GOOGLE_APPLICATION_CREDENTIALS in .env is correct"
}

# Load .env into this process environment so child processes inherit variables
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
                        $rel = $v -replace '^\./',''
                        $resolved = (Resolve-Path (Join-Path $backendRoot $rel) -ErrorAction SilentlyContinue)
                        if ($resolved) { $v = $resolved.Path }
                    }
                    try {
                        Set-Item -Path ("Env:" + $k) -Value $v -ErrorAction Stop
                        Info "Set env $k"
                    } catch {
                        Warn "Failed to set env $k : $_ - trying fallback"
                        [System.Environment]::SetEnvironmentVariable($k, $v, "Process")
                    }
                }
            }
        }
    }
} catch {
    Warn "Failed to load .env into environment: $_"
}

# If only installing, exit now
if ($InstallOnly) {
    Ok "Install-only mode: completed. Exiting."
    exit 0
}

# Start uvicorn (using venv python)
Info "Starting uvicorn using $venvPython (app: app)"
try {
    & $venvPython -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
} catch {
    Err "Failed to start uvicorn: $_"
    exit 1
}
