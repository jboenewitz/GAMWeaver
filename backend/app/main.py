"""FastAPI application and route registration."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db_service import db_service
from .ml_service import ml_service
from .admin_routes import router as admin_router
from .export_compare_routes import router as export_compare_router
from .user_routes import router as user_router

app = FastAPI(
    title="Bike Rental Prediction API",
    description="API for predicting bike rentals using IGANN (Interpretable Generalized Additive Neural Networks)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check_root():
    """Health check endpoint for platform probes."""
    return {"status": "healthy"}


@app.on_event("startup")
async def ensure_superadmin_user():
    db_service.ensure_superadmin()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Bike Rental Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "load_data": "/api/data/load",
            "train_model": "/api/model/train",
            "predict": "/api/predict",
            "shape_functions": "/api/model/shape-functions",
            "metrics": "/api/model/metrics",
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_trained": ml_service.is_trained}


app.include_router(admin_router)
app.include_router(export_compare_router)
app.include_router(user_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
