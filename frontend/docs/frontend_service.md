# 🎨 Frontend Service (React + Vite)

## Overview
The Frontend Service is a modern web application built using React and Vite. It provides an intuitive interface for users to upload text, images, or documents for misinformation analysis.

## Key Features
- **Multi-modal Input:** Supports raw text, image uploads, and document uploads.
- **Real-time Feedback:** Shows analysis progress with interactive loaders.
- **Visual Analytics:** Displays truthfulness scores with color-coded results and detailed claim breakdowns.
- **Responsive Design:** Works on desktop and mobile browsers.

## Project Structure
- `src/components/`: Reusable UI components (InputForm, AnalysisResult, etc.).
- `src/api.js`: Centralized API service for communicating with the Go backend.
- `src/index.css`: Tailwind-like utility styles for a modern "glassmorphism" look.

## Configuration
The frontend connects to the backend API via the base URL configured in `src/api.js`.

## Synchronous Functioning
1. User interacts with the UI (enters text or uploads a file).
2. Frontend calls the Go Backend via REST endpoints.
3. Frontend waits for the JSON response.
4. Once received, the UI updates to show the analysis result, including scores, reasons, and sources.

## Development
- Install dependencies: `npm install`
- Start development server: `npm run dev`
