import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from gpr_analysis import gprh_trend_analysis
from vix_analysis import vix_trend_analysis
from move_analysis import move_trend_analysis
from bvp_analysis import bvp_trend_analysis

def get_macro_analysis():
    """Return the GPRH, VIX, MOVE, and BVP trend analysis for the last year."""
    gprh = gprh_trend_analysis()
    vix = vix_trend_analysis()
    move = move_trend_analysis()
    bvp = bvp_trend_analysis()
    return {"gprh": gprh, "vix": vix, "move": move, "bvp": bvp}
