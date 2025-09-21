<#
start-dev.ps1
Starts both frontend and backend containers in dev mode with live reload.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Info($m)  { Write-Host "[INFO]  $m" -ForegroundColor Cyan }

# Stop existing containers (ignore errors)
Info "Stopping any running stack..."
docker compose down -v | Out-Null

# Build and start containers
Info "Starting dev stack (build + detached)..."
docker compose up --build -d

# Show frontend/backend URLs
Info "Frontend: http://localhost:3000"
Info "Backend: http://localhost:8000"

# Tail logs
Info "Tailing logs (Ctrl+C to exit display)..."
docker compose logs -f
