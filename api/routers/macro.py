from fastapi import APIRouter
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from api.services.macro_analysis import get_macro_analysis

router = APIRouter()

@router.get("/makro-analysis")
def makro_analysis():
    """Return the GPRH, VIX, MOVE, and BVP trend analysis for the last year."""
    return get_macro_analysis()
