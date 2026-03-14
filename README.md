# Vishwas - Hybrid-RAG Misinformation Detection

Vishwas is a project aimed at detecting misinformation using a Hybrid-RAG approach, combining the strengths of Go for orchestration, Python for AI intelligence, and .NET for vector knowledge management.

## 🏗 Project Structure

- **`api-contracts/`**: Shared Protocol Buffer definitions (`.proto`) and documentation for service interaction.
- **`backend/`**: The Go orchestrator service handling the REST API and service coordination.
- **`frontend/`**: The React + Vite web client for user interaction.

## 📚 Documentation

For detailed information on each component, please refer to the `docs/` folder within each directory:

- [API Contracts Documentation](./api-contracts/docs/)
- [Backend Service Documentation](./backend/docs/)
- [Frontend Service Documentation](./frontend/docs/)

## 🚀 Getting Started

### 1. API Contracts
Define your services in `api-contracts/v1/` and generate the code for your preferred language.

### 2. Backend (Go)
```bash
cd backend
go mod download
go run main.go
```

### 3. Frontend (React)
```bash
cd frontend
npm install
npm run dev
```
