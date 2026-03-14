from fastapi import APIRouter
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from api.services.system import get_health_status, get_model_info, get_root_info

router = APIRouter()

@router.get("/health")
def health_check():
    return get_health_status()

@router.get("/model-info")
def model_info():
    return get_model_info()

@router.get("/")
def root():
    return get_root_info()
