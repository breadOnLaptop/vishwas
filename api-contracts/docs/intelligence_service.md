# 🧠 Intelligence Service API Contract

## Overview
The Intelligence Service is responsible for analyzing content (text, images, and documents) to detect misinformation. It leverages Large Language Models (LLMs) and Vision models to provide detailed analysis, scoring, and source verification.

## API Definition (`intelligence.proto`)

### `AnalyzeContent` RPC
This is the primary endpoint for content analysis.

**Request (`AnalyzeRequest`):**
- `text`: Raw text to analyze.
- `image_bytes`: Optional image data.
- `source_url`: URL where the content was found (for context).
- `filename`: Name of the file being uploaded (for document analysis).

**Response (`AnalyzeResponse`):**
- `score`: Truthfulness score (0.0 to 10.0, where 10 is most truthful).
- `color`: Visual representation of the score (e.g., "green", "orange", "red").
- `top_reasons`: Key reasons for the assigned score.
- `user_explanation`: A concise explanation for the end user.
- `explanation`: Detailed technical explanation of the analysis.
- `parsed`: Structured data containing individual claims and their confidence scores.
- `vision`: OCR results and labels if an image was analyzed.
- `top_sources`: List of verified sources or references.

## Synchronous Functioning
1. **Frontend** sends content to the **Backend (Go Orchestrator)**.
2. **Backend** forwards the request to the **Intelligence Service** via gRPC using these contracts.
3. **Intelligence Service** performs the AI analysis and returns the structured `AnalyzeResponse`.
4. **Backend** processes the response and sends it back to the **Frontend**.
