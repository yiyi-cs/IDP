"""
=================================================================================
UTILS PACKAGE
=================================================================================
Enthaelt Hilfsskripte fuer Pipeline-Orchestrierung und Analyse.

Skripte:
- analysis_pipeline.py            : Pipeline-Orchestrierung (debug_0 -> debug_7)
- vp_data_manager.py              : VP-Erkennung und Datei-Verwaltung
- calibration_mode_manager.py     : Kalibrierungs-Varianten-Verwaltung
- adaptive_sync_detection.py      : Adaptive Low-Signal Sync (NEU v2.4)

Version: v2.1
=================================================================================
"""

import sys
from pathlib import Path

# Fuege Projekt-Root zu sys.path hinzu (fuer manuelle Ausfuehrung)
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Package-Info
__version__ = "2.1"
__all__ = [
    "analysis_pipeline",
    "vp_data_manager",
    "calibration_mode_manager",
    "adaptive_sync_detection",  
]
