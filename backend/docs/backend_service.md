# 🚀 Backend Service (Go Orchestrator)

## Overview
The Backend Service is the central orchestrator written in Go. It handles incoming requests from the frontend, manages file uploads, and communicates with the Intelligence Service for content analysis.

## Key Responsibilities
- **REST API:** Exposes endpoints for the React frontend.
- **Orchestration:** Coordinates analysis by calling internal services (via gRPC/Proto).
- **File Handling:** Processes multi-part uploads (images, PDFs, documents).
- **Validation:** Ensures requests are well-formed before processing.

## API Endpoints (REST)

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/analyze/text` | Analyze raw text content. |
| `POST` | `/api/analyze/image` | Analyze uploaded image content. |
| `POST` | `/api/analyze/document` | Analyze uploaded PDF/Docx files. |
| `POST` | `/api/report` | Report misinformation to authorities. |

## Synchronous Workflow
1. Receives a multipart/form-data request from the Frontend.
2. Extracts fields (text, file, source_url).
3. If it's a file, extracts text/metadata if necessary (or passes it as bytes).
4. Calls `IntelligenceService.AnalyzeContent` using the shared Proto contract.
5. Receives the `AnalyzeResponse`.
6. Sends the final JSON response to the Frontend.

## Dependencies
- **Gin Web Framework:** For handling HTTP requests.
- **Google Generative AI Go SDK:** For direct model interaction (if not using a separate intelligence service).
- **gRPC-Go:** For communication between services.
