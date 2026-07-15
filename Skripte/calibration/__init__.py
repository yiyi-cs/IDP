"""
=================================================================================
CALIBRATION PACKAGE
=================================================================================
Enthaelt Kalibrierungs-Skripte fuer beide Pipelines.

Skripte:
- offline_calibration.py          : MediaPipe Pupillen-Kalibrierung
- offline_calibration_ptgaze.py   : ptgaze Gaze-Kalibrierung

Version: v3.2
=================================================================================
"""

import sys
from pathlib import Path

# Fuege Projekt-Root zu sys.path hinzu (fuer manuelle Ausfuehrung)
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Package-Info
__version__ = "3.2"
__all__ = [
    "offline_calibration",
    "offline_calibration_ptgaze",
]
