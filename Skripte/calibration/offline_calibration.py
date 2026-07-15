"""
=================================================================================
OFFLINE VIDEO-KALIBRIERUNG v3.2.1 - Vereinfacht (nutzt phases_detected.json)
=================================================================================

Aenderungen in v3.2.1:
----------------------
- Package-Struktur (calibration/)
- Smooth-Pursuit ENTFERNT (Legacy)
- input_type in PKL hinzugefuegt

Aenderungen in v3.2:
-------------------
- AudioMarkerDetector ENTFERNT (redundant mit debug_0)
- Legacy-Flow ENTFERNT (nur phases_detected.json)
- Temporal Pooling nutzt phases_detected.json (konsistent!)

Pipeline-Abhaengigkeit:
----------------------
1. debug_0_phase_detection.py >> phases_detected.json
2. offline_calibration.py (DIESER) >> calibration_*.pkl
3. debug_5_partial_calibration.py >> nutzt .pkl

Unterstuetzte Modi:
------------------
- ENABLE_TEMPORAL_POOLING = False: Nutzt phases_detected.json (Standard)
- ENABLE_TEMPORAL_POOLING = True:  Nutzt phases_detected.json fuer Beg/Mid/End

Version: v3.2.1
=================================================================================
"""

# =================================================================================
# PATH SETUP (fuer manuelle Ausfuehrung + Package-Import)
# =================================================================================
import sys
from pathlib import Path

# Fuege Projekt-Root zu sys.path hinzu
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# =================================================================================
# IMPORTS
# =================================================================================
import cv2
import numpy as np
import pandas as pd
import pickle
import time
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[!] MediaPipe nicht installiert!")
    exit(1)

# Shared Detection (v1.3)
from shared.shared_pupil_detection import RobustPupilDetector

# Config
from config import (
    SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX, SCREEN_WIDTH_CM, SCREEN_HEIGHT_CM,
    VIEWING_DISTANCE_CM, MIN_CONFIDENCE, 
    SMOOTHING_WINDOW_CALIBRATION as SMOOTHING_WINDOW,
    USE_QUALITY_WEIGHTED_AVERAGE, USE_MEDIAN_INSTEAD_OF_MEAN,
    FIRST_TRIAL_IS_PRACTICE, FIXATION_CALIBRATION_ENABLED,
    FIXATION_CALIBRATION_START_OFFSET_S, FIXATION_CALIBRATION_WINDOW_DURATION_S,
    OUTPUT_BASE_DIR, CENTER_CALIBRATION_WEIGHT, ENABLE_HEAD_POSE_CORRECTION,
    ENABLE_TEMPORAL_POOLING, TEMPORAL_POOLING_MIN_VIDEOS,
    ENABLE_TEMPORAL_DECAY_WEIGHTING, TEMPORAL_DECAY_FACTOR,
)

# Environment-Variable-Overrides (für master_cli.py)
import os

if 'PIPELINE_OUTPUT_BASE_DIR' in os.environ:
    OUTPUT_BASE_DIR = Path(os.environ['PIPELINE_OUTPUT_BASE_DIR'])

if 'PIPELINE_MAIN_VIDEO_PATH' in os.environ:
    VIDEO_PATH = Path(os.environ['PIPELINE_MAIN_VIDEO_PATH'])

if 'PIPELINE_EYETRACKER_FILE_PATH' in os.environ:
    EYETRACKER_FILE_PATH = Path(os.environ['PIPELINE_EYETRACKER_FILE_PATH'])

if 'PIPELINE_CALIBRATION_PKL_PATH' in os.environ:
    CALIBRATION_PKL_PATH = Path(os.environ['PIPELINE_CALIBRATION_PKL_PATH'])

# ══════════════════════════════════════════════════════════════════════
# KALIBRIERUNGSMODUS (NEU v3.3)
# ══════════════════════════════════════════════════════════════════════

# Lese calib_mode aus Environment (vom Master-Skript)
CALIB_MODE = os.environ.get('PIPELINE_CALIB_MODE', 'FullCalib')

# Konfiguration: Welche Phasen und Punkte pro Modus verwenden?
# Format: {modus: {'phases': [...], 'beg_range': (start, end), 'use_fixations': bool}}
CALIB_MODE_CONFIG = {
    # Alle Punkte
    'FullCalib': {
        'phases': ['beg', 'mid', 'end'],
        'beg_range': (1, 20),           # Alle 20 Punkte
        'use_fixations': True,
        'description': 'Alle Kalibrierungen (beg+mid+end) + Fixationen'
    },
    
    # Nur Anfang und Ende (ohne Mid)
    'BegEnd': {
        'phases': ['beg', 'end'],
        'beg_range': (1, 20),           # Alle 20 Punkte von beg
        'use_fixations': False,
        'description': 'Nur beg + end (ohne mid, ohne Fixationen)'
    },
    
    # Nur Anfang
    'OnlyBeg': {
        'phases': ['beg'],
        'beg_range': (1, 20),           # Alle 20 Punkte
        'use_fixations': False,
        'description': 'Nur beg-Kalibrierung'
    },
    
    # Nur Ende
    'OnlyEnd': {
        'phases': ['end'],
        'beg_range': None,              # Nicht relevant
        'use_fixations': False,
        'description': 'Nur end-Kalibrierung'
    },
    
    # Nur erste 10 Punkte von beg
    'BegFirst10': {
        'phases': ['beg'],
        'beg_range': (1, 10),           # Nur erste 10
        'use_fixations': False,
        'description': 'Nur erste 10 Punkte von beg'
    },
    
    # Nur zweite 10 Punkte von beg
    'BegSecond10': {
        'phases': ['beg'],
        'beg_range': (11, 20),          # Nur zweite 10
        'use_fixations': False,
        'description': 'Nur zweite 10 Punkte von beg'
    },
    
    # Anfang (erste 10) + Ende
    'BegFirst10End': {
        'phases': ['beg', 'end'],
        'beg_range': (1, 10),
        'use_fixations': False,
        'description': 'Erste 10 von beg + end'
    },
    
    # Anfang (erste 10) + Ende + Fix
    'BegFirst10EndFix': {
        'phases': ['beg', 'end'],
        'beg_range': (1, 10),
        'use_fixations': True,
        'description': 'Erste 10 von beg + end + Fixationen'
    },

    # Anfang (zweite 10) + Ende
    'BegSecond10End': {
        'phases': ['beg', 'end'],
        'beg_range': (11, 20),
        'use_fixations': False,
        'description': 'Zweite 10 von beg + end'
    },
    
    # Mit Fixationen (verschiedene Kombinationen)
    'BegEndFix': {
        'phases': ['beg', 'end'],
        'beg_range': (1, 20),
        'use_fixations': True,
        'description': 'beg + end + Fixationen'
    },
    
    'OnlyBegFix': {
        'phases': ['beg'],
        'beg_range': (1, 20),
        'use_fixations': True,
        'description': 'Nur beg + Fixationen'
    },
    
    # Custom - wird dynamisch konfiguriert
    'Custom': {
        'phases': [],                   # Wird zur Laufzeit gesetzt
        'beg_range': (1, 20),
        'use_fixations': True,
        'description': 'Benutzerdefinierte Konfiguration'
    },
    
    # Testmodus
    'CalibrationTest': {
        'phases': ['test'],             # Generisches Label für Testmodus
        'beg_range': (1, 10),           # Alle Punkte
        'use_fixations': False,         # Keine Trials vorhanden
        'description': 'Einzelne Kalibrierung (Testmodus für R²-Vergleich)'
    },
}

# Zusaetzliche Environment-Variablen fuer Custom-Modus
if CALIB_MODE == 'Custom':
    # PIPELINE_CALIB_PHASES: Komma-getrennte Liste (z.B. "beg,end")
    custom_phases = os.environ.get('PIPELINE_CALIB_PHASES', 'beg,end')
    CALIB_MODE_CONFIG['Custom']['phases'] = [p.strip() for p in custom_phases.split(',')]
    
    # PIPELINE_CALIB_BEG_RANGE: "start,end" (z.B. "1,10")
    custom_beg_range = os.environ.get('PIPELINE_CALIB_BEG_RANGE', '1,20')
    range_parts = custom_beg_range.split(',')
    CALIB_MODE_CONFIG['Custom']['beg_range'] = (int(range_parts[0]), int(range_parts[1]))
    
    # PIPELINE_CALIB_USE_FIXATIONS: "true" oder "false"
    custom_fix = os.environ.get('PIPELINE_CALIB_USE_FIXATIONS', 'true').lower() == 'true'
    CALIB_MODE_CONFIG['Custom']['use_fixations'] = custom_fix

# Validiere CALIB_MODE
if CALIB_MODE not in CALIB_MODE_CONFIG:
    print(f"[!] Unbekannter CALIB_MODE: {CALIB_MODE}")
    print(f"    Verfuegbare Modi: {', '.join(CALIB_MODE_CONFIG.keys())}")
    print(f"    Nutze Fallback: FullCalib")
    CALIB_MODE = 'FullCalib'

# ══════════════════════════════════════════════════════════════════════
# KONFIGURATION (direkt im Skript, für schnelle Anpassung)
# ══════════════════════════════════════════════════════════════════════

# Output
OUTPUT_FOLDER = OUTPUT_BASE_DIR  # Nutzt config.py

# QS-Visualisierung
ENABLE_QS_VISUALIZATION = True
ENABLE_PHOTO_SERIES = True

# Pupillen-Extraktion (aus config.py übernommen)
EXTRACTION_WINDOW_START = 0.5  # s nach Marker
EXTRACTION_WINDOW_END = 3.0    # s nach Marker

# Regression
POLYNOMIAL_DEGREE = 2  # 1=Linear, 2=Quadratisch (empfohlen)

# Minimale Punkt-Anzahl
MIN_POINTS_FOR_CALIBRATION = 8

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==================== DATENSTRUKTUREN ====================

@dataclass
class ScreenParameters:
    """Screen-Parameter (Bildschirm + Viewing Distance)"""
    width_px: int
    height_px: int
    width_cm: float
    height_cm: float
    viewing_distance_cm: float
    center_x: int = None
    center_y: int = None
    
    def __post_init__(self):
        if self.center_x is None:
            self.center_x = self.width_px // 2
        if self.center_y is None:
            self.center_y = self.height_px // 2


@dataclass
class CalibrationMetadata:
    """
     v3.2: Vereinfacht - Nur aus phases_detected.json.
    
    Kein mode/n_runs mehr (nicht relevant für phases_detected.json).
    """
    timepoints: List[str] = None  # ['beg'] oder ['beg', 'mid', 'end']
    n_points_total: int = 0
    source: str = 'phases_detected.json'  # v3.2
    
    def __post_init__(self):
        if self.timepoints is None:
            self.timepoints = []

# ==================== PUPILLEN-EXTRAKTOR ====================

class VideoPupilExtractor:
    """
     v3.2: Vereinfacht - Nur Wrapper für shared_pupil_detection.
    
    Kein eigener Code mehr (alles in shared_pupil_detection.py).
    """
    
    def __init__(self):
        self.detector = RobustPupilDetector()
        print(f" VideoPupilExtractor (shared_pupil_detection v1.2)")
    
    def close(self):
        """Schließt MediaPipe Face Mesh"""
        self.detector.close()

# ==================== OFFLINE-KALIBRIERUNG ====================

class OfflineVideoCalibration:
    """
    Offline Video-Kalibrierung v3.2 (vereinfacht).
    
     NEU: Nutzt NUR phases_detected.json (konsistent mit Pipeline!)
     ENTFERNT: Audio-Detektion (redundant mit debug_0)
     ENTFERNT: Legacy-Flow (für alte Daten)
    
    Pipeline:
    ---------
    1. debug_0 >> phases_detected.json (Audio + Video-Zeiten)
    2. offline_calibration >> calibration_*.pkl (dieses Skript)
    3. debug_5 >> nutzt .pkl für Prediction
    
    Version: v3.2
    """
    
    def __init__(self, experimental_video_path: str, screen_params: ScreenParameters,
             calibration_video_path: str = None):
        """
         v3.2: VEREINFACHT - Nur ein Video-Parameter (Experimental).
        
        Kalibrierungsvideo wird aus phases_detected.json geladen!
        
        Args:
            experimental_video_path: Video für Fixations-Kalibrierung (optional)
            screen_params: Bildschirm-Parameter
        """
        
        # ══════════════════════════════════════════════════════════════
        # VIDEOS
        # ══════════════════════════════════════════════════════════════
        
        self.experimental_video_path = experimental_video_path
        self.calibration_video_path = calibration_video_path
        
        #  NEU: Kalibrierungsvideo wird aus phases_detected.json geladen
        # (wird in _run_calibration_from_phases gesetzt)
        
        # ══════════════════════════════════════════════════════════════
        # PARAMETER
        # ══════════════════════════════════════════════════════════════
        
        self.screen = screen_params
        
        # ══════════════════════════════════════════════════════════════
        # DETEKTOREN
        # ══════════════════════════════════════════════════════════════
        
        self.pupil_extractor = VideoPupilExtractor()
        
        # ══════════════════════════════════════════════════════════════
        # TRAINING-DATEN (immer initialisiert!)
        # ══════════════════════════════════════════════════════════════
        
        self.pupil_measurements = []      # [(x, y), ...]
        self.screen_targets = []           # [(x, y), ...]
        self.point_types = []              # ['calibration', 'fixation_center', ...]
        self.head_pose_measurements = []   # [{'yaw': ..., 'pitch': ...}, ...]
        
        #  WICHTIG: Auch wenn ENABLE_HEAD_POSE_CORRECTION=False, 
        #             existiert die Liste (mit None-Werten)
        
        # ══════════════════════════════════════════════════════════════
        # VIDEO-CACHE (Performance-Optimierung!)
        # ══════════════════════════════════════════════════════════════
        
        self._video_cache = {}  # {video_path: cv2.VideoCapture}
        
        # ══════════════════════════════════════════════════════════════
        # INFO-AUSGABE
        # ══════════════════════════════════════════════════════════════
        
        print(f"\n📍 Kalibrierungs-Setup:")
        print(f"   Experimental-Video: {Path(experimental_video_path).name if experimental_video_path else 'Nicht verfügbar'}")
        print(f"   Screen: {screen_params.width_px}x{screen_params.height_px} px "
              f"({screen_params.width_cm}x{screen_params.height_cm} cm)")
        print(f"   Viewing Distance: {screen_params.viewing_distance_cm} cm")
        print(f"   Head-Pose: {' Aktiviert' if ENABLE_HEAD_POSE_CORRECTION else ' Deaktiviert'}")
        print(f"   Temporal Pooling: {' Aktiviert' if ENABLE_TEMPORAL_POOLING else ' Deaktiviert'}")

    # ══════════════════════════════════════════════════════════════════
    # HELPER-METHODEN
    # ══════════════════════════════════════════════════════════════════
    
    def _open_video_cached(self, video_path: str) -> Optional[cv2.VideoCapture]:
        """
        🚀 PERFORMANCE: Öffnet Video EINMAL und cached es.
        
        Statt 32× VideoCapture() bei 32 Punkten >> nur 1×!
        Speedup: ~3-5× schneller bei vielen Punkten.
        
        Args:
            video_path: Pfad zum Video
        
        Returns:
            cv2.VideoCapture oder None bei Fehler
        """
        
        if video_path not in self._video_cache:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"    Kann Video nicht öffnen: {Path(video_path).name}")
                return None
            
            self._video_cache[video_path] = cap
        
        return self._video_cache[video_path]
    
    def _close_video_cache(self):
        """Schließt alle gecachten Videos (am Ende aufrufen!)"""
        for cap in self._video_cache.values():
            cap.release()
        self._video_cache.clear()
    
    def _extract_pupil_for_point(self, video_path: str, timestamp: float,
                                 screen_pos: Tuple[int, int],
                                 timepoint_label: str = "") -> Optional[Dict]:
        """
         ZENTRAL: Extrahiert Pupille + Head-Pose für einen Punkt.
        
        Verhindert Code-Duplikation zwischen run_calibration() und 
        _run_temporal_pooling_calibration().
        
        Args:
            video_path: Video-Pfad
            timestamp: Audio-Marker-Zeit (Sekunden)
            screen_pos: Ziel-Position auf Screen
            timepoint_label: 'beg', 'mid', 'end' oder '' (für point_types)
        
        Returns:
            dict mit:
            - 'pupil_pos': [x, y] oder None
            - 'head_pose': dict oder None (falls ENABLE_HEAD_POSE_CORRECTION)
            - 'point_type': str (für self.point_types)
            oder None bei Fehler
        """
        
        # ══════════════════════════════════════════════════════════════
        # EXTRAKTION (einheitlich - immer extract_pupil_with_head_pose)
        # ══════════════════════════════════════════════════════════════
        
        #  FIX: Nutze IMMER extract_pupil_with_head_pose()
        # Die Methode gibt (pupil, None) zurück wenn Head-Pose deaktiviert
        pupil_pos, head_pose = self.extract_pupil_with_head_pose(
            video_path, timestamp=timestamp
        )
        
        if pupil_pos is None:
            return None
        
        # ══════════════════════════════════════════════════════════════
        # PUNKT-TYP BESTIMMEN
        # ══════════════════════════════════════════════════════════════
        
        is_center = (screen_pos[0] == SCREEN_WIDTH_PX // 2 and 
                    screen_pos[1] == SCREEN_HEIGHT_PX // 2)
        
        if is_center:
            point_type = f'calibration_center_{timepoint_label}' if timepoint_label else 'calibration_center'
        else:
            point_type = f'calibration_{timepoint_label}' if timepoint_label else 'calibration'
        
        return {
            'pupil_pos': pupil_pos,
            'head_pose': head_pose,  # Ist None wenn ENABLE_HEAD_POSE_CORRECTION=False
            'point_type': point_type
        }

    
    def extract_pupil_with_head_pose(self, video_path: str, timestamp: float,
                                     window_start: float = None,
                                     window_end: float = None) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Extrahiert Pupillen + Head-Pose aus Zeitfenster.
        
        Args:
            video_path: Video-Pfad
            timestamp: Zeitstempel (s)
            window_start: Offset Start (s), default: EXTRACTION_WINDOW_START
            window_end: Offset Ende (s), default: EXTRACTION_WINDOW_END
        
        Returns:
            (pupil_position, head_pose_data oder None)
        """
        
        # Defaults aus config.py
        if window_start is None:
            window_start = EXTRACTION_WINDOW_START
        if window_end is None:
            window_end = EXTRACTION_WINDOW_END
        
        cap = self._open_video_cached(video_path) 
        if cap is None:
            return None, None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_time = timestamp + window_start
        end_time = timestamp + window_end
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        pupil_positions = []
        head_yaws = []
        head_pitches = []
        head_rolls = []
        head_confidences = []
        
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            result = self.pupil_extractor.detector.extract_from_frame(frame)
            
            if (result['position'] is not None and 
                result['confidence'] >= MIN_CONFIDENCE):
                pupil_positions.append(result['position'])
                
                if result.get('head_yaw') is not None:
                    head_yaws.append(result['head_yaw'])
                    head_pitches.append(result['head_pitch'])
                    head_rolls.append(result['head_roll'])
                    head_confidences.append(result['head_pose_confidence'])
        
        if len(pupil_positions) < 5:
            return None, None
        
        pupil_final = np.median(pupil_positions, axis=0)
        
        head_pose_data = None
        if len(head_yaws) >= 3:
            head_pose_data = {
                'yaw': float(np.median(head_yaws)),
                'pitch': float(np.median(head_pitches)),
                'roll': float(np.median(head_rolls)),
                'confidence': float(np.mean(head_confidences)),
                'n_frames': len(head_yaws)
            }
        
        return pupil_final, head_pose_data
    
    def _compute_head_pose_reference(self) -> Dict:
        """
        Berechnet Referenz-Head-Pose (Median über alle Punkte).
        
        Wird für debug_5 adaptive Head-Pose-Korrektur benötigt.
        
        Returns:
            dict mit 'yaw', 'pitch', 'roll', 'confidence', 'enabled'
        """
        
        if not ENABLE_HEAD_POSE_CORRECTION:
            return {'enabled': False}
        
        valid_head_poses = [hp for hp in self.head_pose_measurements if hp is not None]
        
        if len(valid_head_poses) < 3:
            print(f"    Zu wenig Head-Pose-Daten ({len(valid_head_poses)}/3)")
            return {'enabled': False}
        
        ref_yaw = np.median([hp['yaw'] for hp in valid_head_poses])
        ref_pitch = np.median([hp['pitch'] for hp in valid_head_poses])
        ref_roll = np.median([hp['roll'] for hp in valid_head_poses])
        ref_confidence = np.mean([hp['confidence'] for hp in valid_head_poses])
        
        print(f"\n🧠 Referenz-Head-Pose (Median über {len(valid_head_poses)} Punkte):")
        print(f"   Yaw:   {ref_yaw:+.2f}°")
        print(f"   Pitch: {ref_pitch:+.2f}°")
        print(f"   Roll:  {ref_roll:+.2f}°")
        
        return {
            'yaw': float(ref_yaw),
            'pitch': float(ref_pitch),
            'roll': float(ref_roll),
            'confidence': float(ref_confidence),
            'n_points_used': len(valid_head_poses),
            'enabled': True
        }
    
    def _compute_point_weights(self, enable_temporal: bool = False) -> np.ndarray:
        """
        Berechnet Gewichte fuer alle Punkt-Typen.
        
        Kombiniert:
        - Punkt-Typ-Gewichtung (calibration, center, fixation)
        - Optional: Temporale Gewichtung (Beg < Mid < End fuer Pooling)
        
        Args:
            enable_temporal: Ob temporale Gewichtung aktiviert (fuer Pooling)
        
        Returns:
            Gewichtungs-Array (normalisiert auf len(point_types))
        """
        
        n_points = len(self.point_types)
        weights = np.ones(n_points)
        
        # Punkt-Typ-Gewichtung (v3.2.1: Smooth-Pursuit entfernt)
        for i, point_type in enumerate(self.point_types):
            if 'center' in point_type:
                if 'calibration' in point_type:
                    weights[i] = 1.0 - CENTER_CALIBRATION_WEIGHT
                elif point_type == 'fixation_center':
                    weights[i] = CENTER_CALIBRATION_WEIGHT
            else:
                weights[i] = 1.0
        
        # Temporale Gewichtung (optional)
        if enable_temporal and ENABLE_TEMPORAL_DECAY_WEIGHTING:
            timepoint_map = {'beg': 0.0, 'mid': 0.5, 'end': 1.0}
            
            for i, point_type in enumerate(self.point_types):
                t = 0.5  # Default
                for key, value in timepoint_map.items():
                    if key in point_type:
                        t = value
                        break
                
                weights[i] *= np.exp(TEMPORAL_DECAY_FACTOR * t)
        
        # Normalisieren
        weights = weights / weights.sum() * len(weights)
        
        return weights
    
    def _pixel_to_degree_x(self, pixels: float) -> float:
        """Konvertiert Pixel zu Grad (X-Achse)"""
        cm_per_px = SCREEN_WIDTH_CM / SCREEN_WIDTH_PX
        cm = pixels * cm_per_px
        deg = np.arctan2(cm, VIEWING_DISTANCE_CM) * 180 / np.pi
        return deg
    
    def _pixel_to_degree_y(self, pixels: float) -> float:
        """Konvertiert Pixel zu Grad (Y-Achse)"""
        cm_per_px = SCREEN_HEIGHT_CM / SCREEN_HEIGHT_PX
        cm = pixels * cm_per_px
        deg = np.arctan2(cm, VIEWING_DISTANCE_CM) * 180 / np.pi
        return deg

    
    # ══════════════════════════════════════════════════════════════════
    # HAUPT-ENTRY-POINT
    # ══════════════════════════════════════════════════════════════════
    
    def run_calibration(self) -> Dict:
        """
         v3.2: Vereinfacht - phases_detected.json ist REQUIRED!
        
        Zwei Modi (gesteuert über config.py):
        1. Standard (ENABLE_TEMPORAL_POOLING = False):
           >> Nutzt phases_detected.json aus OUTPUT_BASE_DIR
        
        2. Temporal Pooling (ENABLE_TEMPORAL_POOLING = True):
           >> Nutzt phases_detected_beg.json, *_mid.json, *_end.json
        
        Returns:
            dict mit:
            - 'success': True/False
            - 'model': Trainiertes Modell (bei success=True)
            - 'filepath': Pfad zur .pkl
            - 'n_points': Anzahl Trainings-Punkte
            - 'r2_scores': (r2_x, r2_y)
        """
        
        print("\n" + "="*70)
        
        if ENABLE_TEMPORAL_POOLING:
            print(" OFFLINE KALIBRIERUNG v3.2 (TEMPORAL POOLING)")
            print("="*70)
            return self._run_temporal_pooling_calibration()
        else:
            print(" OFFLINE KALIBRIERUNG v3.2 (STANDARD)")
            print("="*70)
            return self._run_calibration_from_phases()

    
    def _run_calibration_from_phases(self) -> Dict:
        """
         v3.2: Standard-Kalibrierung aus phases_detected.json.
        
        Workflow:
        ---------
        1. Lade phases_detected.json (aus OUTPUT_BASE_DIR)
        2. Extrahiere Kalibrierpunkte (mit Video-Zeiten!)
        3. Extrahiere Pupillen für jeden Punkt
        4. Optional: Fixationen aus Experimental-Video
        5. Trainiere Modell
        6. Speichere .pkl
        
        Returns:
            model dict oder error dict
        """
        
        # ══════════════════════════════════════════════════════════════
        # SCHRITT 1: Prüfe phases_detected.json
        # ══════════════════════════════════════════════════════════════
        
        phases_json_path = Path(OUTPUT_BASE_DIR) / "phases_detected.json"
        
        if not phases_json_path.exists():
            print(f"\n FEHLER: phases_detected.json nicht gefunden!")
            print(f"   Pfad: {phases_json_path}")
            print(f"   >> Laufe ZUERST: debug_0_phase_detection.py")
            return {
                'success': False,
                'error': 'phases_detected.json fehlt',
                'path_expected': str(phases_json_path)
            }
        
        print(f"\n{'='*70}")
        print("LADE KALIBRIERUNGSDATEN (aus phases_detected.json)")
        print(f"{'='*70}\n")
        print(f"    {phases_json_path.name}")
        
        # ══════════════════════════════════════════════════════════════
        # SCHRITT 2: Lade JSON
        # ══════════════════════════════════════════════════════════════
        
        with open(phases_json_path, 'r') as f:
            phases = json.load(f)
        
        # Extrahiere Metadaten
        metadata = CalibrationMetadata()
        
        if 'sync_info' in phases:
            metadata.mode = phases['sync_info'].get('method', 'unknown')
        
        # ══════════════════════════════════════════════════════════════
        # KALIBRIERUNGSMODUS ANWENDEN (NEU v3.3)
        # ══════════════════════════════════════════════════════════════
        
        mode_config = CALIB_MODE_CONFIG[CALIB_MODE]
        allowed_phases = mode_config['phases']
        beg_range = mode_config['beg_range']
        use_fixations = mode_config['use_fixations']
        
        print(f"\n   [CALIB_MODE] {CALIB_MODE}")
        print(f"   {mode_config['description']}")
        print(f"   Erlaubte Phasen: {', '.join(allowed_phases) if allowed_phases else 'Keine'}")
        if 'beg' in allowed_phases and beg_range:
            print(f"   BEG Punkt-Range: {beg_range[0]}-{beg_range[1]}")
        print(f"   Fixationen: {'Ja' if use_fixations else 'Nein'}")
        
        # Zaehle verfuegbare UND erlaubte Kalibrierungen
        calib_phases_available = []
        
        # WICHTIG: Pruefe ALLE moeglichen Labels (inkl. 'test' fuer CalibrationTest!)
        # Standard-Labels + alle aus allowed_phases
        labels_to_check = list(set(['beg', 'mid', 'end'] + allowed_phases))
        
        for label in labels_to_check:
            calib_key = f'calibration_{label}'
            
            # Pruefe ob Phase in JSON vorhanden
            if calib_key not in phases.get('phases', {}):
                continue
            if 'points' not in phases['phases'][calib_key]:
                continue
            
            n_points_in_json = len(phases['phases'][calib_key]['points'])
            if n_points_in_json == 0:
                continue
            
            # Pruefe ob Phase durch CALIB_MODE erlaubt
            if label not in allowed_phases:
                print(f"   • {label.upper()}: {n_points_in_json} Punkte [UEBERSPRUNGEN - nicht in {CALIB_MODE}]")
                continue
            
            # Bei beg: Punkt-Range anwenden
            if label == 'beg' and beg_range:
                n_points_effective = min(beg_range[1], n_points_in_json) - beg_range[0] + 1
                n_points_effective = max(0, n_points_effective)
                print(f"   • {label.upper()}: {n_points_effective}/{n_points_in_json} Punkte (Range {beg_range[0]}-{beg_range[1]})")
                metadata.n_points_total += n_points_effective
            else:
                print(f"   • {label.upper()}: {n_points_in_json} Punkte")
                metadata.n_points_total += n_points_in_json
            
            calib_phases_available.append(label)
        
        if len(calib_phases_available) == 0:
            print(f"\n    Keine Kalibrierpunkte fuer Modus {CALIB_MODE}!")
            print(f"      Verfuegbare Phasen in JSON: {', '.join([k.replace('calibration_', '') for k in phases.get('phases', {}).keys() if k.startswith('calibration_')])}")
            print(f"      Erlaubte Phasen fuer {CALIB_MODE}: {', '.join(allowed_phases)}")
            return {'success': False, 'error': f'Keine Kalibrierpunkte fuer Modus {CALIB_MODE}'}
        
        metadata.timepoints = calib_phases_available
        # ══════════════════════════════════════════════════════════════
        # VIDEO-PFAD BESTIMMEN (v3.2.2 - Prioritaet korrigiert!)
        # ══════════════════════════════════════════════════════════════
        # Prioritaet:
        # 1. Environment-Variable PIPELINE_MAIN_VIDEO_PATH (vom Master-Skript)
        # 2. video_path aus phases_detected.json
        # 3. VIDEO_PATH aus config.py (Fallback)
        
        video_path = None
        video_source = None
        
        # Prioritaet 1: Environment-Variable (vom Master-Skript)
        if 'PIPELINE_MAIN_VIDEO_PATH' in os.environ:
            env_video = Path(os.environ['PIPELINE_MAIN_VIDEO_PATH'])
            if env_video.exists():
                video_path = str(env_video)
                video_source = 'Environment (PIPELINE_MAIN_VIDEO_PATH)'
                print(f"   [>] Video aus Environment-Variable")
        
        # Prioritaet 2: JSON
        if video_path is None:
            json_video = phases.get('video_path', None)
            if json_video and json_video != '' and Path(json_video).exists():
                video_path = json_video
                video_source = 'phases_detected.json'
                print(f"   [>] Video aus phases_detected.json")
        
        # Prioritaet 3: config.py Fallback
        if video_path is None:
            print(f"   [!] Kein Video in Environment/JSON gefunden")
            
            try:
                from config import VIDEO_PATH as CONFIG_VIDEO_PATH
                if Path(CONFIG_VIDEO_PATH).exists():
                    video_path = str(CONFIG_VIDEO_PATH)
                    video_source = 'config.py (Fallback)'
                    print(f"   [>] Nutze VIDEO_PATH aus config.py als Fallback")
            except ImportError:
                pass
        
        # Validierung
        if video_path is None or not Path(video_path).exists():
            print(f"   [X] Kein gueltiges Video gefunden!")
            print(f"       Environment: {os.environ.get('PIPELINE_MAIN_VIDEO_PATH', 'nicht gesetzt')}")
            print(f"       JSON: {phases.get('video_path', 'nicht vorhanden')}")
            return {'success': False, 'error': 'Video nicht gefunden'}
        
        print(f"   [OK] Video: {Path(video_path).name}")
        print(f"        Quelle: {video_source}")
        
        # ══════════════════════════════════════════════════════════════
        # SCHRITT 3: Extrahiere Pupillen
        # ══════════════════════════════════════════════════════════════
        
        print(f"\n👁️ EXTRAHIERE PUPILLEN ({len(calib_phases_available)} Kalibrierungen)...")
        
        self.pupil_measurements = []
        self.screen_targets = []
        self.point_types = []
        self.head_pose_measurements = []
        
        # Hole beg_range fuer Filterung
        mode_config = CALIB_MODE_CONFIG[CALIB_MODE]
        beg_range = mode_config['beg_range']
        
        for label in calib_phases_available:
            calib_key = f'calibration_{label}'
            calib_phase = phases['phases'][calib_key]
            
            print(f"\n   {'─'*60}")
            print(f"   KALIBRIERUNG: {label.upper()}")
            print(f"   {'─'*60}")
            
            points = calib_phase['points']
            n_total_points = len(points)
            
            # Filtere Punkte bei beg nach Range (NEU v3.3)
            # HINWEIS: Bei 'test' wird keine Filterung angewendet
            if label == 'beg' and beg_range:
                range_start, range_end = beg_range
                points_filtered = [p for p in points if range_start <= p['point_id'] <= range_end]
                print(f"   [FILTER] Punkte {range_start}-{range_end}: {len(points_filtered)}/{len(points)}")
                points = points_filtered
            
            for point in points:
                point_id = point['point_id']
                position = tuple(point['position'])
                video_time = point['video_time_s']
                audio_found = point.get('audio_found_s') is not None
                
                print(f"   [{point_id}/{len(calib_phase['points'])}] @ {video_time:.2f}s >> {position}", end=" ")

                #  Extrahiere (zentrale Methode!)
                result = self._extract_pupil_for_point(
                    video_path, video_time, position,
                    timepoint_label=label
                )
                
                if result is not None:
                    self.pupil_measurements.append(result['pupil_pos'])
                    self.screen_targets.append(position)
                    self.point_types.append(result['point_type'])
                    
                    if ENABLE_HEAD_POSE_CORRECTION:
                        self.head_pose_measurements.append(result['head_pose'])
                    
                    status = ''
                    if not audio_found:
                        status += ' (Audio: ✗)'
                    print(status)
                else:
                    print("")
        
        # ══════════════════════════════════════════════════════════════
        # SCHRITT 4: Optional - Fixationen
        # ══════════════════════════════════════════════════════════════

        # Pruefe ob Fixationen durch CALIB_MODE erlaubt (NEU v3.3)
        mode_config = CALIB_MODE_CONFIG[CALIB_MODE]
        use_fixations_mode = mode_config['use_fixations']
        
        if not use_fixations_mode:
            print(f"\n   [INFO] Fixationen deaktiviert (CALIB_MODE={CALIB_MODE})")
        elif not FIXATION_CALIBRATION_ENABLED:
            print(f"\n   [INFO] Fixationen deaktiviert (FIXATION_CALIBRATION_ENABLED=False in config)")
        elif video_path:
            print(f"\n{'='*70}")
            print("FIXATIONEN ALS ZUSAETZLICHE KALIBRIERPUNKTE")
            print(f"{'='*70}\n")
            
            # Extrahiere aus phases_detected.json
            fixation_timestamps = []
            
            for block_key in ['experiment_block1', 'experiment_block2']:
                if block_key in phases['phases']:
                    for trial in phases['phases'][block_key]['trials']:
                        if trial.get('is_practice', False) and not FIRST_TRIAL_IS_PRACTICE:
                            continue
                        
                        fix_start_s = trial['fixation']['start_video_s']
                        fixation_timestamps.append(fix_start_s)
            
            print(f"    {len(fixation_timestamps)} Fixations gefunden")
            
            n_added = 0
            
            for i, fix_time in enumerate(fixation_timestamps):
                timestamp = fix_time + FIXATION_CALIBRATION_START_OFFSET_S
                screen_pos = (SCREEN_WIDTH_PX // 2, SCREEN_HEIGHT_PX // 2)
                
                print(f"   [{i+1}/{len(fixation_timestamps)}] @ {timestamp:.2f}s", end=" ")
                
                result = self._extract_pupil_for_point(
                    video_path,
                    timestamp, 
                    screen_pos
                )
                
                if result is not None:
                    self.pupil_measurements.append(result['pupil_pos'])
                    self.screen_targets.append(screen_pos)
                    self.point_types.append('fixation_center')
                    
                    if ENABLE_HEAD_POSE_CORRECTION:
                        self.head_pose_measurements.append(result['head_pose'])
                    
                    print("")
                    n_added += 1
                else:
                    print("")
            
            if n_added > 0:
                print(f"\n    {n_added} Fixations hinzugefuegt")
        else:
            print(f"\n   [INFO] Fixationen uebersprungen (kein Video-Pfad)")

        # ══════════════════════════════════════════════════════════════
        # SCHRITT 5: Validierung
        # ══════════════════════════════════════════════════════════════
        
        if len(self.pupil_measurements) < MIN_POINTS_FOR_CALIBRATION:
            print(f"\n Zu wenige Messungen: {len(self.pupil_measurements)}/{MIN_POINTS_FOR_CALIBRATION}")
            return {'success': False, 'error': 'Zu wenige Punkte'}
        
        print(f"\n {len(self.pupil_measurements)} Messungen erfolgreich")
        
        # ══════════════════════════════════════════════════════════════
        # SCHRITT 6: Modell trainieren
        # ══════════════════════════════════════════════════════════════
        
        print(f"\n🎓 TRAINIERE KALIBRIERUNGSMODELL...")
        
        # Head-Pose-Referenz
        reference_head_pose = self._compute_head_pose_reference()
        
        # Modell
        model = self.create_calibration_model()
        model['reference_head_pose'] = reference_head_pose
        model['metadata'] = metadata
        model['source'] = 'phases_detected.json'
        
        # ══════════════════════════════════════════════════════════════
        # SCHRITT 7: Speichern
        # ══════════════════════════════════════════════════════════════
        
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem if video_path else 'unknown'
        filename = f"calibration_{video_name}_{timestamp_str}.pkl"
        filepath = Path(OUTPUT_FOLDER) / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        print("\n" + "="*70)
        print(" KALIBRIERUNG ERFOLGREICH")
        print("="*70)
        print(f"📋 Statistik:")
        print(f"   • Messungen: {len(self.pupil_measurements)}")
        print(f"   • Kalibrierungen: {', '.join(calib_phases_available)}")
        print(f"   • Regression: {'Polynom (Grad ' + str(model['polynomial_degree']) + ')' if model['polynomial_degree'] > 1 else 'Linear'}")
        print(f"   • R²: X={model['r2_scores'][0]:.3f}, Y={model['r2_scores'][1]:.3f}")
        print(f"\n {filepath}")
        print("="*70 + "\n")
        
        self._close_video_cache()
        
        return {
            'success': True,
            'model': model,
            'filepath': str(filepath),
            'n_points': len(self.pupil_measurements),
            'r2_scores': model['r2_scores']
        }

    
    def _run_temporal_pooling_calibration(self) -> Dict:
        """
         v3.2: Temporal Pooling aus phases_detected.json (vereinfacht!).
        
        Lädt phases_detected.json für mehrere Zeitpunkte (beg/mid/end)
        und kombiniert ALLE Punkte für Training.
        
        ENTFERNT: Audio-Detektion (redundant mit debug_0!)
        NEU: Nutzt phases_detected.json für jeden Zeitpunkt
        
        Returns:
            model dict oder error dict
        """
        
        print(f"\n{'='*70}")
        print("TEMPORAL POOLING: LADE ALLE ZEITPUNKTE")
        print(f"{'='*70}")
        
        # ══════════════════════════════════════════════════════════════
        # SCHRITT 1: Prüfe phases_detected.json für jeden Zeitpunkt
        # ══════════════════════════════════════════════════════════════
        
        phases_jsons = {}
        
        for timepoint in ['beg', 'mid', 'end']:
            json_path = Path(OUTPUT_BASE_DIR) / f"phases_detected_{timepoint}.json"
            
            if json_path.exists():
                phases_jsons[timepoint] = json_path
                print(f"    {timepoint.upper()}: {json_path.name}")
            else:
                print(f"    {timepoint.upper()}: Nicht gefunden")
        
        if len(phases_jsons) < TEMPORAL_POOLING_MIN_VIDEOS:
            print(f"\n Zu wenige Zeitpunkte: {len(phases_jsons)}/{TEMPORAL_POOLING_MIN_VIDEOS}")
            print(f"   >> Laufe debug_0 für jeden Zeitpunkt (beg/mid/end)")
            return {
                'success': False,
                'error': f'Zu wenige Zeitpunkte ({len(phases_jsons)}/{TEMPORAL_POOLING_MIN_VIDEOS})'
            }
        
        # Reset Listen
        self.pupil_measurements = []
        self.screen_targets = []
        self.point_types = []
        self.head_pose_measurements = []
        
        # ══════════════════════════════════════════════════════════════
        # SCHRITT 2: Für jeden Zeitpunkt - Pupillen extrahieren
        # ══════════════════════════════════════════════════════════════
        
        for timepoint, json_path in phases_jsons.items():
            with open(json_path, 'r') as f:
                phases = json.load(f)
            
            print(f"\n{'─'*70}")
            print(f"📍 ZEITPUNKT: {timepoint.upper()}")
            print(f"{'─'*70}")
            
            calib_key = f'calibration_{timepoint}'
            
            if calib_key not in phases['phases']:
                print(f"    Keine Kalibrierpunkte in {timepoint.upper()}!")
                continue
            
            calib_phase = phases['phases'][calib_key]
            points = calib_phase.get('points', [])
            video_path = phases.get('video_path', None)
            
            if video_path is None:
                print(f"    Kein Video-Pfad in JSON!")
                continue
            
            print(f"   Video: {Path(video_path).name}")
            print(f"   Punkte: {len(points)}")
            
            n_added = 0
            
            for point in points:
                point_id = point['point_id']
                position = tuple(point['position'])
                video_time = point['video_time_s']
                
                print(f"   [{point_id}/{len(points)}] @ {video_time:.2f}s >> {position}", end=" ")
                
                result = self._extract_pupil_for_point(
                    video_path, video_time, position,
                    timepoint_label=timepoint
                )
                
                if result is not None:
                    self.pupil_measurements.append(result['pupil_pos'])
                    self.screen_targets.append(position)
                    self.point_types.append(result['point_type'])
                    
                    if ENABLE_HEAD_POSE_CORRECTION:
                        self.head_pose_measurements.append(result['head_pose'])
                    
                    print("")
                    n_added += 1
                else:
                    print("")
            
            print(f"\n    {timepoint.upper()}: {n_added} Punkte hinzugefügt")
        
        # ══════════════════════════════════════════════════════════════
        # SCHRITT 3: Optional - Fixationen
        # ══════════════════════════════════════════════════════════════
        
        if FIXATION_CALIBRATION_ENABLED and self.experimental_video_path:
            print(f"\n{'='*70}")
            print("FIXATIONEN ALS ZUSÄTZLICHE KALIBRIERPUNKTE")
            print(f"{'='*70}\n")
            
            # Nutze phases_detected.json (Standard-Version, kein Zeitpunkt)
            phases_main_json = Path(OUTPUT_BASE_DIR) / "phases_detected.json"
            
            if phases_main_json.exists():
                with open(phases_main_json, 'r') as f:
                    phases_main = json.load(f)
                
                fixation_timestamps = []
                
                for block_key in ['experiment_block1', 'experiment_block2']:
                    if block_key in phases_main['phases']:
                        for trial in phases_main['phases'][block_key]['trials']:
                            if trial.get('is_practice', False) and not FIRST_TRIAL_IS_PRACTICE:
                                continue
                            
                            fix_start_s = trial['fixation']['start_video_s']
                            fixation_timestamps.append(fix_start_s)
                
                print(f"    {len(fixation_timestamps)} Fixations gefunden")
                
                n_added = 0
                
                for i, fix_time in enumerate(fixation_timestamps):
                    timestamp = fix_time + FIXATION_CALIBRATION_START_OFFSET_S
                    screen_pos = (SCREEN_WIDTH_PX // 2, SCREEN_HEIGHT_PX // 2)
                    
                    print(f"   [{i+1}/{len(fixation_timestamps)}] @ {timestamp:.2f}s", end=" ")
                    
                    result = self._extract_pupil_for_point(
                        self.experimental_video_path, timestamp, screen_pos
                    )
                    
                    if result is not None:
                        self.pupil_measurements.append(result['pupil_pos'])
                        self.screen_targets.append(screen_pos)
                        self.point_types.append('fixation_center')
                        
                        if ENABLE_HEAD_POSE_CORRECTION:
                            self.head_pose_measurements.append(result['head_pose'])
                        
                        print("")
                        n_added += 1
                    else:
                        print("")
                
                if n_added > 0:
                    print(f"\n    {n_added} Fixations hinzugefügt")
        
        # ══════════════════════════════════════════════════════════════
        # SCHRITT 4: Validierung
        # ══════════════════════════════════════════════════════════════
        
        if len(self.pupil_measurements) < MIN_POINTS_FOR_CALIBRATION:
            print(f"\n Zu wenige Messungen: {len(self.pupil_measurements)}/{MIN_POINTS_FOR_CALIBRATION}")
            return {'success': False, 'error': 'Zu wenige Punkte'}
        
        print(f"\n {len(self.pupil_measurements)} Messungen erfolgreich")
        
        # ══════════════════════════════════════════════════════════════
        # SCHRITT 5: Modell trainieren (mit temporaler Gewichtung!)
        # ══════════════════════════════════════════════════════════════
        
        print(f"\n🎓 TRAINIERE KALIBRIERUNGSMODELL (Temporal Pooling)...")
        
        # Head-Pose-Referenz
        reference_head_pose = self._compute_head_pose_reference()
        
        # Modell
        model = self.create_calibration_model()
        model['reference_head_pose'] = reference_head_pose
        model['temporal_pooling_enabled'] = True
        model['n_timepoints'] = len(phases_jsons)
        model['timepoints'] = list(phases_jsons.keys())
        model['source'] = 'phases_detected_pooling'
        
        # ══════════════════════════════════════════════════════════════
        # SCHRITT 6: Speichern
        # ══════════════════════════════════════════════════════════════
        
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"calibration_temporal_pooling_{timestamp_str}.pkl"
        filepath = Path(OUTPUT_FOLDER) / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        print("\n" + "="*70)
        print(" TEMPORAL POOLING KALIBRIERUNG ERFOLGREICH")
        print("="*70)
        print(f"📋 Statistik:")
        print(f"   • Messungen: {len(self.pupil_measurements)}")
        print(f"   • Zeitpunkte: {', '.join(phases_jsons.keys())}")
        print(f"   • Regression: {'Polynom (Grad ' + str(model['polynomial_degree']) + ')' if model['polynomial_degree'] > 1 else 'Linear'}")
        print(f"   • R²: X={model['r2_scores'][0]:.3f}, Y={model['r2_scores'][1]:.3f}")
        print(f"\n {filepath}")
        print("="*70 + "\n")
        
        self._close_video_cache()
        
        return {
            'success': True,
            'model': model,
            'filepath': str(filepath),
            'n_points': len(self.pupil_measurements),
            'n_timepoints': len(phases_jsons),
            'r2_scores': model['r2_scores']
        }
    
    def _compute_head_pose_reference(self) -> Dict:
        """
        Berechnet Referenz-Head-Pose (Median über alle Punkte).
        
        Wird für debug_5 adaptive Head-Pose-Korrektur benötigt.
        
        Returns:
            dict mit 'yaw', 'pitch', 'roll', 'confidence', 'enabled'
        """
        
        if not ENABLE_HEAD_POSE_CORRECTION:
            return {'enabled': False}
        
        valid_head_poses = [hp for hp in self.head_pose_measurements if hp is not None]
        
        if len(valid_head_poses) < 3:
            print(f"    Zu wenig Head-Pose-Daten ({len(valid_head_poses)}/3)")
            return {'enabled': False}
        
        ref_yaw = np.median([hp['yaw'] for hp in valid_head_poses])
        ref_pitch = np.median([hp['pitch'] for hp in valid_head_poses])
        ref_roll = np.median([hp['roll'] for hp in valid_head_poses])
        ref_confidence = np.mean([hp['confidence'] for hp in valid_head_poses])
        
        print(f"\n🧠 Referenz-Head-Pose (Median über {len(valid_head_poses)} Punkte):")
        print(f"   Yaw:   {ref_yaw:+.2f}°")
        print(f"   Pitch: {ref_pitch:+.2f}°")
        print(f"   Roll:  {ref_roll:+.2f}°")
        
        return {
            'yaw': float(ref_yaw),
            'pitch': float(ref_pitch),
            'roll': float(ref_roll),
            'confidence': float(ref_confidence),
            'n_points_used': len(valid_head_poses),
            'enabled': True
        }
    
    def create_calibration_model(self) -> dict:
        """
         ZENTRAL: Trainiert Kalibrierungsmodell (Linear oder Polynomial).
        
        Nutzt:
        - self.pupil_measurements (Input)
        - self.screen_targets (Output)
        - self.point_types (für Gewichtung)
        
        Gewichtung:
        - Normale Kalibrierpunkte: 1.0
        - Kalibrierungs-Mitten: 1.0 - CENTER_CALIBRATION_WEIGHT
        - Fixations-Mitten: CENTER_CALIBRATION_WEIGHT
        - Smooth-Pursuit: SMOOTH_PURSUIT_WEIGHT
        
        Returns:
            model dict (kompatibel mit debug_5)
        """
        
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Ridge
        
        X = np.array(self.pupil_measurements)
        y_x = np.array([t[0] for t in self.screen_targets])
        y_y = np.array([t[1] for t in self.screen_targets])
        
        # ══════════════════════════════════════════════════════════════
        # GEWICHTUNG (nutzt zentrale Methode!)
        # ══════════════════════════════════════════════════════════════
        
        weights = self._compute_point_weights(enable_temporal=ENABLE_TEMPORAL_POOLING)
        
        # ══════════════════════════════════════════════════════════════
        # MODELL-TRAINING
        # ══════════════════════════════════════════════════════════════
        
        if POLYNOMIAL_DEGREE > 1:
            # Polynomiale Regression
            print(f"   Verwende POLYNOMIALE Regression Grad {POLYNOMIAL_DEGREE}")
            
            poly = PolynomialFeatures(degree=POLYNOMIAL_DEGREE, include_bias=True)
            X_poly = poly.fit_transform(X)
            
            # Ridge für Stabilität (wichtig bei höheren Graden!)
            model_x = Ridge(alpha=1.0).fit(X_poly, y_x, sample_weight=weights)
            model_y = Ridge(alpha=1.0).fit(X_poly, y_y, sample_weight=weights)
            
            r2_x = model_x.score(X_poly, y_x)
            r2_y = model_y.score(X_poly, y_y)
            
            print(f"    R² X={r2_x:.3f}, Y={r2_y:.3f}")
            
            if r2_y < 0.7:
                print(f"    R² Y ist niedrig! Mögliche Ursachen:")
                print(f"      • Kamera nicht auf Augenhöhe")
                print(f"      • Perspektivische Verzerrung zu stark für Polynom")
                print(f"   >> Tipp: Neue Aufnahme mit Kamera EXAKT auf Augenhöhe!")
            
            return {
                'model_x': model_x,
                'model_y': model_y,
                'screen_params': self.screen,
                'r2_scores': (r2_x, r2_y),
                'method': f'offline_calibration_v3.3_poly{POLYNOMIAL_DEGREE}',
                'input_type': 'pupil_pixels',
                'calib_mode': CALIB_MODE,  # NEU v3.3
                'calib_mode_config': CALIB_MODE_CONFIG[CALIB_MODE],  # NEU v3.3
                'polynomial_degree': POLYNOMIAL_DEGREE,
                'poly_transformer': poly,
                'ridge_alpha': 1.0,
                'extraction_window': (EXTRACTION_WINDOW_START, EXTRACTION_WINDOW_END),
                'min_confidence': MIN_CONFIDENCE,
                'smoothing_window': SMOOTHING_WINDOW,
                'use_quality_weighted': USE_QUALITY_WEIGHTED_AVERAGE,
                'calibration_points': self.screen_targets,
                'pupil_points': [list(p) for p in self.pupil_measurements],
                'point_types': self.point_types,
                'n_training_points': len(self.pupil_measurements)
            }
        
        else:
            # Lineare Regression (Grad 1)
            print(f"   Verwende LINEARE Regression (gewichtet)")
            
            model_x = LinearRegression().fit(X, y_x, sample_weight=weights)
            model_y = LinearRegression().fit(X, y_y, sample_weight=weights)
            
            r2_x = model_x.score(X, y_x)
            r2_y = model_y.score(X, y_y)
            
            print(f"    R² X={r2_x:.3f}, Y={r2_y:.3f}")
            
            if r2_y < 0.7:
                print(f"    R² Y ist niedrig! Mögliche Ursachen:")
                print(f"      • Kamera nicht auf Augenhöhe")
                print(f"      • Kopfbewegungen während Kalibrierung")
                print(f"      • Zu wenige Kalibrierpunkte vertikal")
                print(f"   >> Tipp: Neue Aufnahme mit Kamera auf Augenhöhe!")
            
            return {
                'model_x': model_x,
                'model_y': model_y,
                'screen_params': self.screen,
                'r2_scores': (r2_x, r2_y),
                'method': 'offline_calibration_v3.2_linear',
                'input_type': 'pupil_pixels',  # NEU v3.2.1: Konsistent mit ptgaze PKL
                'calib_mode': CALIB_MODE,  # NEU v3.3
                'calib_mode_config': CALIB_MODE_CONFIG[CALIB_MODE],  # NEU v3.3
                'polynomial_degree': 1,
                'poly_transformer': None,
                'extraction_window': (EXTRACTION_WINDOW_START, EXTRACTION_WINDOW_END),
                'min_confidence': MIN_CONFIDENCE,
                'smoothing_window': SMOOTHING_WINDOW,
                'use_quality_weighted': USE_QUALITY_WEIGHTED_AVERAGE,
                'calibration_points': self.screen_targets,
                'pupil_points': [list(p) for p in self.pupil_measurements],
                'point_types': self.point_types,
                'n_training_points': len(self.pupil_measurements)
            }


# ==================== HAUPTPROGRAMM ====================

def main():
    """
     v3.2: Vereinfacht - Nutzt phases_detected.json (REQUIRED!).
    
    Workflow:
    1. Prüfe Config (Temporal Pooling oder Standard)
    2. Validiere phases_detected.json vorhanden
    3. Erstelle OfflineVideoCalibration
    4. run_calibration() >> .pkl
    """
    
    print("\n" + "="*70)
    print(" OFFLINE VIDEO-KALIBRIERUNG v3.2 (VEREINFACHT)")
    print("="*70)
    
    # ══════════════════════════════════════════════════════════════════
    # MODUS-ERKENNUNG
    # ══════════════════════════════════════════════════════════════════
    
    if ENABLE_TEMPORAL_POOLING:
        print(f"\n TEMPORAL POOLING MODUS")
        print(f"   Erwarte: phases_detected_beg.json, *_mid.json, *_end.json")
    else:
        print(f"\n STANDARD MODUS")
        print(f"   Erwarte: phases_detected.json")
    
    # ══════════════════════════════════════════════════════════════════
    # KONFIGURATION
    # ══════════════════════════════════════════════════════════════════
    
    print(f"\n Konfiguration:")
    print(f"   Output: {OUTPUT_FOLDER}")
    print(f"   Experimental-Video: {Path(OUTPUT_BASE_DIR).parent / 'Videos' / 'experiment.mp4' if Path(OUTPUT_BASE_DIR).exists() else 'Auto-Detection'}")
    print(f"   Regression: {'Polynom (Grad ' + str(POLYNOMIAL_DEGREE) + ')' if POLYNOMIAL_DEGREE > 1 else 'Linear'}")
    print(f"   Extraktions-Fenster: {EXTRACTION_WINDOW_START}-{EXTRACTION_WINDOW_END}s nach Marker")
    print(f"   Min. Confidence: {MIN_CONFIDENCE}")
    print(f"   Head-Pose: {' Aktiviert' if ENABLE_HEAD_POSE_CORRECTION else ' Deaktiviert'}")
    print(f"   Fixations-Kalibrierung: {' Aktiviert' if FIXATION_CALIBRATION_ENABLED else ' Deaktiviert'}")
    
    # ══════════════════════════════════════════════════════════════════
    # SCREEN-PARAMETER
    # ══════════════════════════════════════════════════════════════════
    
    screen = ScreenParameters(
        width_px=SCREEN_WIDTH_PX,
        height_px=SCREEN_HEIGHT_PX,
        width_cm=SCREEN_WIDTH_CM,
        height_cm=SCREEN_HEIGHT_CM,
        viewing_distance_cm=VIEWING_DISTANCE_CM
    )
    
    # ══════════════════════════════════════════════════════════════════
    # VALIDIERUNG: phases_detected.json vorhanden?
    # ══════════════════════════════════════════════════════════════════
    
    phases_json_path = Path(OUTPUT_BASE_DIR) / "phases_detected.json"
    
    if not ENABLE_TEMPORAL_POOLING and not phases_json_path.exists():
        print(f"\n FEHLER: phases_detected.json nicht gefunden!")
        print(f"   Pfad: {phases_json_path}")
        print(f"\n📋 Workflow:")
        print(f"   1. Laufe ZUERST: debug_0_phase_detection.py")
        print(f"   2. Dann: offline_calibration.py (dieser)")
        return
    
    if ENABLE_TEMPORAL_POOLING:
        # Prüfe ob mindestens 2 Zeitpunkte vorhanden
        timepoints_found = []
        for tp in ['beg', 'mid', 'end']:
            tp_json = Path(OUTPUT_BASE_DIR) / f"phases_detected_{tp}.json"
            if tp_json.exists():
                timepoints_found.append(tp)
        
        if len(timepoints_found) < TEMPORAL_POOLING_MIN_VIDEOS:
            print(f"\n FEHLER: Zu wenige Zeitpunkte für Temporal Pooling!")
            print(f"   Gefunden: {len(timepoints_found)} ({', '.join(timepoints_found)})")
            print(f"   Benötigt: {TEMPORAL_POOLING_MIN_VIDEOS}")
            print(f"\n📋 Workflow:")
            print(f"   1. Laufe debug_0 für JEDEN Zeitpunkt (beg, mid, end)")
            print(f"   2. Dann: offline_calibration.py mit ENABLE_TEMPORAL_POOLING=True")
            return
        
        print(f"\n    {len(timepoints_found)} Zeitpunkte gefunden: {', '.join(timepoints_found)}")
    else:
        print(f"\n    phases_detected.json gefunden")
    
    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENTAL-VIDEO (für Fixations-Kalibrierung)
    # ══════════════════════════════════════════════════════════════════
    
    experimental_video_path = None
    
    if FIXATION_CALIBRATION_ENABLED:
        # Versuche aus phases_detected.json zu laden
        if phases_json_path.exists():
            with open(phases_json_path, 'r') as f:
                phases = json.load(f)
            experimental_video_path = phases.get('video_path', None)
        
        if experimental_video_path is None:
            print(f"\n    Kein Experimental-Video in phases_detected.json")
            print(f"      >> Fixations-Kalibrierung wird übersprungen")
        else:
            print(f"    Experimental-Video: {Path(experimental_video_path).name}")
    
    # ══════════════════════════════════════════════════════════════════
    # KALIBRIERUNG DURCHFÜHREN
    # ══════════════════════════════════════════════════════════════════
    
    calibrator = None  #  Initialisiere VOR try-Block (für finally!)
    
    try:
        #  KORRIGIERTE REIHENFOLGE (ohne logfile_path!)
        calibrator = OfflineVideoCalibration(
            experimental_video_path=experimental_video_path or "",  #  Erstes Argument
            screen_params=screen,                                   #  Zweites Argument
            calibration_video_path=""                               #  Optional (wird aus JSON geladen)
        )
        
        result = calibrator.run_calibration()
        
        if result['success']:
            print(f"\n🎉 Kalibrierung erfolgreich!")
            print(f"\n Ergebnis:")
            print(f"   • Datei: {Path(result['filepath']).name}")
            print(f"   • Punkte: {result['n_points']}")
            print(f"   • R² X: {result['r2_scores'][0]:.3f}")
            print(f"   • R² Y: {result['r2_scores'][1]:.3f}")
            
            if ENABLE_TEMPORAL_POOLING:
                print(f"   • Zeitpunkte: {result.get('n_timepoints', 'N/A')}")
        else:
            print(f"\n Fehlgeschlagen: {result.get('error')}")
    
    except Exception as e:
        print(f"\n Fehler: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        #  None-Check (falls Fehler vor Zuweisung)
        if calibrator is not None:
            calibrator.pupil_extractor.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Abgebrochen durch Nutzer")

