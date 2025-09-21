from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
import os

def create_app():
    app = FastAPI(title="Vishwas - Misinformation Backend")

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routes under /api
    app.include_router(api_router, prefix="/api")

    # Correct path to frontend dist in Docker
    frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frontend/dist"))
    if not os.path.exists(frontend_dist):
        raise RuntimeError(f"Frontend dist folder does not exist at {frontend_dist}")

    # Serve frontend build
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

    return app

app = create_app()
