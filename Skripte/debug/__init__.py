"""
=================================================================================
DEBUG PIPELINE PACKAGE
=================================================================================
Enthaelt alle debug_X Skripte fuer die Eye-Tracking-Analyse-Pipeline.

Skripte:
- debug_0_phase_detection.py      : Automatische Phasen-Detektion (Audio-Marker)
- debug_1_video_analysis.py       : MediaPipe Pupillen-Detektion
- debug_1_ptgaze.py               : ptgaze Gaze-Detektion
- debug_3_eyetracker_load.py      : EyeLink-Daten laden
- debug_4_synchronization.py      : Webcam-EyeLink Synchronisation
- debug_5_partial_calibration.py  : MediaPipe Kalibrierung anwenden
- debug_5_ptgaze.py               : ptgaze Kalibrierung anwenden
- debug_6_extended_comparison.py  : 3-Wege-Vergleich
- debug_7_interactive_timeline.py : Interaktive Visualisierung

Version: v4.0
=================================================================================
"""

import sys
from pathlib import Path

# Fuege Projekt-Root zu sys.path hinzu (fuer manuelle Ausfuehrung)
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Package-Info
__version__ = "4.0"
__all__ = [
    "debug_0_phase_detection",
    "debug_1_video_analysis",
    "debug_1_ptgaze",
    "debug_3_eyetracker_load",
    "debug_4_synchronization",
    "debug_5_partial_calibration",
    "debug_5_ptgaze",
    "debug_6_extended_comparison",
    "debug_7_interactive_timeline",
]
