# 🧠 Intelligence Service (Python - The Brain)

## Overview
This service is a gRPC server responsible for all heavy AI, LLM, and Vision processing. It acts as the direct interface to **Google Cloud (Vertex/Vision)** or a future **MCP Server**.

## Dynamic Routing
The service is designed to be dynamic:
1.  **GCP Mode**: Default mode; uses Google Vertex AI and Vision.
2.  **MCP Mode**: (Planned) Will route requests through the team's custom MCP server.

## Key Responsibilities
-   **Content Analysis**: Processes text/images/docs to detect misinformation.
-   **LLM Interaction**: Manages prompts and token handling for Gemini models.
-   **Vision AI**: Performs OCR and safe-search filtering on image bytes.
-   **Structured Outputs**: Returns data matching the `intelligence.proto` contract.

## Synchronous Functioning
1.  Receives gRPC call `AnalyzeContent` from the **Go Orchestrator**.
2.  Loads necessary model configurations from `.env`.
3.  Performs analysis using the active provider (GCP/MCP).
4.  Returns a structured `AnalyzeResponse`.

## Requirements
-   Python 3.10+
-   `grpcio`, `grpcio-tools`
-   `google-cloud-aiplatform`, `google-cloud-vision`
-   `python-dotenv`
