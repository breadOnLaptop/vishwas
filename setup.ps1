Write-Host "--- Starting Vishwas Project Setup ---" -ForegroundColor Cyan

# 1. Environment Checks
if (-not (Test-Path ".env")) {
    Write-Host "--- .env not found. Creating from template ---" -ForegroundColor Yellow
    Copy-Item ".env.template" ".env"
}

if (-not (Test-Path "google-credentials.json")) {
    Write-Host "!!! WARNING: google-credentials.json is missing in root! GCP features will fail." -ForegroundColor Red
}

# 2. Go Setup
Write-Host "--- Setting up Go Bridge ---" -ForegroundColor Blue
$env:GOPATH = "D:\software\programming\go\workspace"
$env:PATH += ";$env:GOPATH\bin\windows_amd64"

Write-Host "Installing Go Protobuf plugins..."
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

cd backend
go mod tidy
cd ..

# 3. Python Setup
Write-Host "--- Setting up Python Brain ---" -ForegroundColor Green
cd intelligence
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install grpcio-tools
cd ..

# 4. Frontend Setup
Write-Host "--- Setting up React Frontend ---" -ForegroundColor Magenta
cd frontend
npm install
cd ..

# 5. Generate Protos
Write-Host "--- Generating API Contracts ---" -ForegroundColor Yellow
./generate_protos.ps1

Write-Host "--- Setup Complete! You can now run ./start.ps1 ---" -ForegroundColor Green
