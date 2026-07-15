"""
=================================================================================
SHARED MODULES PACKAGE
=================================================================================
Enthaelt gemeinsame Module fuer beide Pipelines (MediaPipe + ptgaze).

Module:
- shared_pupil_detection.py       : MediaPipe-basierte Pupillen-Detektion
- shared_gaze_detection_ptgaze.py : ptgaze-basierte Gaze-Detektion
- shared_blink_detection.py       : Einheitliche Blink-Detektion (EAR)
- shared_coordinate_utils.py      : Koordinaten-Transformation (Pixel <-> Grad)

Version: v1.0
=================================================================================
"""

import sys
from pathlib import Path

# Fuege Projekt-Root zu sys.path hinzu (fuer manuelle Ausfuehrung)
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Exports fuer einfachen Import
from .shared_blink_detection import BlinkDetector, create_blink_detector
from .shared_coordinate_utils import (
    CoordinateTransformer, 
    ScreenParameters, 
    CalibrationMetadata,
    create_coordinate_transformer,
    convert_eyelink_to_degrees,
    batch_convert_to_degrees
)

# Package-Info
__version__ = "1.0"
__all__ = [
    "shared_pupil_detection",
    "shared_gaze_detection_ptgaze",
    "shared_blink_detection",
    "shared_coordinate_utils",
    "BlinkDetector",
    "create_blink_detector",
    "CoordinateTransformer",
    "ScreenParameters",
    "CalibrationMetadata",
    "create_coordinate_transformer",
    "convert_eyelink_to_degrees",
    "batch_convert_to_degrees",
]
