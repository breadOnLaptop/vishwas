<#
start-docker.ps1 (project root)
Usage examples:
  .\start-docker.ps1 -Mode Dev                # dev: build + up detached, tail logs
  .\start-docker.ps1 -Mode Dev -Rebuild       # force rebuild (dev)
  .\start-docker.ps1 -Mode Prod -Rebuild -Tail  # prod build and tail logs

Notes:
- For true production, prefer a dedicated compose file (docker-compose.prod.yml) and a proper secrets backend.
#>

param(
    [ValidateSet("Dev","Prod")]
    [string]$Mode = "Dev",

    [switch]$Rebuild = $false,
    [switch]$Tail = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Info($m) { Write-Host "[INFO] $m" -ForegroundColor Cyan }

if ($Mode -eq "Dev") {
    Info "Selected mode: Dev"
    if ($Rebuild) {
        Info "Rebuilding images (dev) ..."
        docker-compose build --no-cache
    }
    Info "Starting dev compose..."
    docker-compose up -d
    Info "Frontend: http://localhost:3000"
    Info "Backend:  http://localhost:8000"

    if ($Tail) {
        Info "Tailing logs (Ctrl+C to stop)..."
        docker-compose logs -f
    }
} else {
    Info "Selected mode: Prod"
    # build with ENV=prod build-arg so Dockerfile can switch behavior
    if ($Rebuild) {
        Info "Building images for production (with ENV=prod build-arg)..."
        docker-compose build --build-arg ENV=prod
    }
    Info "Starting services in detached mode..."
    docker-compose up -d

    Info "Production containers started."
    if ($Tail) {
        Info "Tailing logs (Ctrl+C to stop)..."
        docker-compose logs -f
    }
}
