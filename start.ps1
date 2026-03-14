Write-Host "--- Vishwas Service Orchestrator ---" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop all services simultaneously." -ForegroundColor Yellow
Write-Host "--------------------------------------------------"

$processes = @()

try {
    # 1. Start Python Brain (Intelligence)
    Write-Host "[1/3] Launching Python Brain..." -ForegroundColor Green
    $brain = Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd intelligence; .\.venv\Scripts\Activate.ps1; python main.py" -PassThru
    $processes += $brain

    # 2. Start Go Bridge (Backend)
    Write-Host "[2/3] Launching Go Bridge..." -ForegroundColor Blue
    $bridge = Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; go run main.go" -PassThru
    $processes += $bridge

    # 3. Start Frontend (React)
    Write-Host "[3/3] Launching React Frontend..." -ForegroundColor Magenta
    $frontend = Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev" -PassThru
    $processes += $frontend

    Write-Host "--------------------------------------------------"
    Write-Host "All services are active. Monitoring for interrupt..." -ForegroundColor Green
    
    # Keep the orchestrator alive
    while($true) { Start-Sleep -Seconds 1 }
} 
finally {
    Write-Host "`n--- Starting Emergency Cleanup ---" -ForegroundColor Red
    
    foreach ($proc in $processes) {
        if ($proc) {
            $id = $proc.Id
            Write-Host "Terminating process tree for PID $id..." -NoNewline
            
            # Check if process exists before attempting to kill
            if (Get-Process -Id $id -ErrorAction SilentlyContinue) {
                # Forcefully terminate the process and all its children (/T)
                & taskkill /F /T /PID $id 2>$null | Out-Null
                Write-Host " [DONE]" -ForegroundColor Green
            } else {
                Write-Host " [ALREADY CLOSED]" -ForegroundColor Gray
            }
        }
    }
    
    Write-Host "--- Cleanup finished. All windows should be closed. ---" -ForegroundColor Cyan
}
