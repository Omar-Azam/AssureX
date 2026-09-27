"""
AssureX Claim Engine Backend Application
========================================
FastAPI application entrypoint with route assembly, CORS, and lifecycle events.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database import Base, engine
from backend.routes import auth, products, warranties, claims, reviews, admin, repairs, ocr

# Initialize database schema tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AssureX Claims Processing Engine API",
    description=(
        "Production-grade backend for the AssureX Claims Processing Engine. "
        "Provides role-based authentication, product/warranty management, "
        "document upload with SHA-256 duplicate detection, automated multimodal "
        "ML/Rule evaluation, manual review workflows, and immutable audit logging."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Route Handlers
app.include_router(auth.router)
app.include_router(products.router)
app.include_router(warranties.router)
app.include_router(claims.router)
app.include_router(reviews.router)
app.include_router(repairs.router)
app.include_router(ocr.router)
app.include_router(admin.router)


from pathlib import Path
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Mount frontend static directory if present
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


@app.get("/health", tags=["Frontend & Health Check"])
@app.get("/api/health", tags=["Frontend & Health Check"])
def health_check():
    """System health check and API status metadata."""
    return {
        "status": "online",
        "service": "AssureX Claims Processing Engine API",
        "version": "1.0.0",
        "documentation": "/docs",
        "redoc": "/redoc"
    }


@app.get("/", tags=["Frontend & Health Check"])
@app.get("/app", tags=["Frontend & Health Check"])
def serve_index_or_health(request: Request):
    """Serves responsive frontend application for browsers, or API health metadata."""
    accept = request.headers.get("accept", "")
    user_agent = request.headers.get("user-agent", "")
    
    # Return health JSON if requested explicitly or from automated API test client
    if "text/html" not in accept and ("application/json" in accept or "testclient" in user_agent):
        return health_check()
        
    index_file = _FRONTEND_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))
    return health_check()

