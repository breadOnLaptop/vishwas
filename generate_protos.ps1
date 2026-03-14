# Generate Go Protos
Write-Host "Generating Go Protos..."
$env:PATH += ";D:\software\programming\go\workspace\bin\windows_amd64"
protoc --proto_path=api-contracts --go_out=backend/internal/gen --go_opt=paths=source_relative v1/intelligence.proto v1/knowledge.proto v1/orchestration.proto
protoc --proto_path=api-contracts --go-grpc_out=backend/internal/gen --go-grpc_opt=paths=source_relative v1/intelligence.proto v1/knowledge.proto v1/orchestration.proto

# Generate Python Protos
Write-Host "Generating Python Protos..."
if (-not (Test-Path "intelligence/gen_proto")) { mkdir "intelligence/gen_proto" }

# Use the venv python
$PYTHON_EXE = "intelligence/.venv/Scripts/python.exe"
# Point explicitly to the file
& $PYTHON_EXE -m grpc_tools.protoc --proto_path=api-contracts --python_out=intelligence/gen_proto --grpc_python_out=intelligence/gen_proto api-contracts/v1/intelligence.proto

# Create __init__.py files
if (-not (Test-Path "intelligence/gen_proto/__init__.py")) { New-Item "intelligence/gen_proto/__init__.py" -ItemType File }
if (-not (Test-Path "intelligence/gen_proto/v1")) { mkdir "intelligence/gen_proto/v1" }
if (-not (Test-Path "intelligence/gen_proto/v1/__init__.py")) { New-Item "intelligence/gen_proto/v1/__init__.py" -ItemType File }

Write-Host "Checking for generated files..."
ls -R intelligence/gen_proto
