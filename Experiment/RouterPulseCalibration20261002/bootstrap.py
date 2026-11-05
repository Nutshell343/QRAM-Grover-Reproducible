"""Import the existing lab protocol/analysis without importing pyQCat."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT.parent
for path in (EXPERIMENT / "LabRouterR5R6", EXPERIMENT / "LabPartialCSWAPVZ4"):
    sys.path.insert(0, str(path))

from router_r5_r6 import protocol, analysis, workflow  # noqa: E402
from partial_cswap_vz4.timing_plan import physical_layers  # noqa: E402
