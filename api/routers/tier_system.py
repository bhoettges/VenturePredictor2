from fastapi import APIRouter
from api.services.tier_system import get_health_status, get_model_info, get_root_info

router = APIRouter()

@router.get("/")
def root():
    """Root endpoint with API usage info."""
    return get_root_info()

@router.get("/health")
def health():
    """System health check."""
    return get_health_status()

@router.get("/model_info")
def model_info():
    """Get detailed model information."""
    return get_model_info()
