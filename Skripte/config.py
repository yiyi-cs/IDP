"""
=================================================================================
CONFIG.PY - Zentrale Konfiguration fuer Eye-Tracking-Pipeline v4.0
=================================================================================

STRUKTUR:
---------
1. DATEIPFADE & EXPERIMENT-SETUP
2. ANALYSE-MODUS
3. BILDSCHIRM & HARDWARE
4. PHASE DETECTION (debug_0)
5. PUPILLEN-DETEKTION (debug_1)
6. BLINK DETECTION
7. AUDIO-MARKER-DETEKTION (debug_2)
8. EYELINK-PARAMETER (debug_3)
9. SYNCHRONISATION (debug_4)
10. KALIBRIERUNG (debug_5)
11. FIXATION-DETECTION & TRIAL-ANALYSE (debug_6/7)
12. VISUALISIERUNG & EXPORT

VERSION HISTORY:
----------------
v4.0 (2026-01): Bereinigt - Modus 3 + Smooth-Pursuit entfernt
v3.0 (2025): Vollstaendige Ueberarbeitung
v2.1 (2025): Audio-Marker v2
v1.0 (2025): Initial Version

=================================================================================
"""

import numpy as np
from pathlib import Path
import math

# =================================================================================
# 1. DATEIPFADE & EXPERIMENT-SETUP
# =================================================================================

# Basis-Verzeichnisse
BASE_DIR = Path(__file__).parent
OUTPUT_BASE_DIR = Path(r"Anpassen nicht notwendig")

# ---------------------------------------------------------------------------------
# Video & EyeLink Dateien
# ---------------------------------------------------------------------------------

VIDEO_PATH = Path(r"Anpassen nicht notwendig")
MAIN_VIDEO_PATH = VIDEO_PATH  # Alias fuer Kompatibilitaet

EYETRACKER_FILE_PATH = Path(r"Anpassen nicht notwendig")

CALIBRATION_PKL_PATH = Path(r"Anpassen nicht notwendig")

# ---------------------------------------------------------------------------------
# JSON-Konfigurationsdateien (fuer Phase Detection)
# ---------------------------------------------------------------------------------

EXPERIMENT_SYNC_JSON_PATH = Path(r"Anpassen nicht notwendig")
CALIBRATION_TIMING_JSON_DIR = Path(r"Anpassen nicht notwendig")

# ---------------------------------------------------------------------------------
# Multi-Video (Temporal Pooling)
# ---------------------------------------------------------------------------------

EXPERIMENTAL_VIDEOS_FOLDER = Path(r"Anpassen nicht notwendig")

# ---------------------------------------------------------------------------------
# Experiment-Struktur
# ---------------------------------------------------------------------------------

N_TRIALS = 24                       # Anzahl experimentelle Trials
EXPECTED_TONE_EVENTS = 48           # 24 Fixation + 24 Trial = 48 Toene
FIRST_TRIAL_IS_PRACTICE = True      # Trial 1 ist Uebungslauf (wird Trial 0)

# ---------------------------------------------------------------------------------
# Output-Dateinamen (automatisch generiert)
# ---------------------------------------------------------------------------------

OUTPUT_FILES = {
    # debug_0: Phase Detection
    'debug_0_phases': 'phases_detected.json',
    'debug_0_qc_spectro': 'phase_detection_qc_spectro.png',
    'debug_0_qc_stats': 'phase_detection_qc_stats.png',
    'debug_0_event_table': 'phase_detection_event_table.csv',
    'debug_0_sync_candidates': 'phase_detection_sync_candidates.png',
    
    # debug_1: Pupil Detection
    'debug_1_pupil': 'debug_1_pupil_data.csv',
    'debug_1_ptgaze': 'debug_1_ptgaze_data.csv',
    'debug_1_video': 'debug_1_pupil_video.mp4',
    'debug_1_metadata': 'debug_1_metadata.json',
    
    # debug_2: Audio Markers
    'debug_2_markers': 'debug_2_audio_markers.csv',
    'debug_2_trials': 'debug_2_trial_structure.csv',
    'debug_2_plot': 'debug_2_audio_detection.png',
    
    # debug_3: EyeLink
    'debug_3_eyetracker': 'debug_3_eyetracker_data.csv',
    'debug_3_blocks': 'debug_3_blocks_overview.csv',
    'debug_3_fixations': 'debug_3_fixations.csv',
    'debug_3_metadata': 'debug_3_metadata.json',
    
    # debug_4: Synchronisation
    'debug_4_synced': 'debug_4_pupil_data_synced.csv',
    'debug_4_ptgaze_synced': 'debug_4_ptgaze_gaze_synced.csv',
    'debug_4_offsets': 'debug_4_sync_offsets.csv',
    'debug_4_info': 'debug_4_sync_info.json',
    
    # debug_5: Kalibrierung
    'debug_5_calibrated': 'debug_5_pupil_data_calibrated.csv',
    'debug_5_ptgaze_calibrated': 'debug_5_ptgaze_calibrated.csv',
    'debug_5_info': 'debug_5_calibration_info.json',
    
    # debug_6: Trial-Analyse
    'debug_6_results': 'debug_6_trial_results.csv',
    'debug_6_comparison': 'debug_6_three_way_comparison.csv',
    'debug_6_metrics': 'debug_6_validation_metrics.json',
}


# =================================================================================
# 2. ANALYSE-MODUS
# =================================================================================

ANALYSIS_MODE = 2  # <-- HIER AENDERN

"""
MODUS-UEBERSICHT:
-----------------

MODUS 1: STANDALONE WEBCAM
    - Nur Webcam-Video (keine EyeLink-Daten)
    - Kalibrierung: .pkl (offline_calibration)
    - Pipeline: debug_0 >> debug_1 >> debug_5 >> debug_6
    - Skripte ueberspringen: debug_3, debug_4
    - Genauigkeit: +/-1-2 Grad angestrebt

MODUS 2: COMPARISON (WEBCAM + EYELINK)
    - Webcam + EyeLink parallel aufgenommen
    - Kalibrierung: .pkl (identisch zu Modus 1)
    - Pipeline: debug_0 >> debug_1 >> debug_3 >> debug_4 >> debug_5 >> debug_6
    - Synchronisation: Multi-Anchor (debug_4)
    - Genauigkeit: +/-1-2 Grad angestrebt

HINWEIS: Modus 3 (Legacy) wurde in v4.0 entfernt.
"""


# =================================================================================
# 3. BILDSCHIRM & HARDWARE
# =================================================================================

# ---------------------------------------------------------------------------------
# Bildschirm-Dimensionen
# ---------------------------------------------------------------------------------

SCREEN_WIDTH_PX = 1920              # Pixel (horizontal)
SCREEN_HEIGHT_PX = 1080             # Pixel (vertikal)
SCREEN_WIDTH_CM = 53.2              # Physische Breite (cm)
SCREEN_HEIGHT_CM = 29.9             # Physische Hoehe (cm)
VIEWING_DISTANCE_CM = 70            # Abstand Auge-Bildschirm (cm)

# Abgeleitete Parameter (automatisch berechnet)
SCREEN_CENTER_X = SCREEN_WIDTH_PX // 2
SCREEN_CENTER_Y = SCREEN_HEIGHT_PX // 2

# Maximale plausible Sehwinkel (automatisch berechnet)
MAX_PLAUSIBLE_DEG_X = math.atan2(SCREEN_WIDTH_CM / 2, VIEWING_DISTANCE_CM) * 180 / math.pi
MAX_PLAUSIBLE_DEG_Y = math.atan2(SCREEN_HEIGHT_CM / 2, VIEWING_DISTANCE_CM) * 180 / math.pi

# ---------------------------------------------------------------------------------
# Kamera-Parameter
# ---------------------------------------------------------------------------------

CAMERA_FOCAL_LENGTH = 1000          # Geschaetzte Brennweite (Pixel)


# =================================================================================
# 4. PHASE DETECTION (debug_0)
# =================================================================================

# ---------------------------------------------------------------------------------
# Master-Switch
# ---------------------------------------------------------------------------------

ENABLE_AUTOMATIC_PHASE_DETECTION = True

# ---------------------------------------------------------------------------------
# Sync-Strategie
# ---------------------------------------------------------------------------------

PHASE_DETECTION_SYNC_STRATEGY = 'audio_with_json_fallback'
# Optionen:
# - 'audio_with_json_fallback': Audio-Suche + JSON-Fallback (EMPFOHLEN)
# - 'json_only': Nur JSON (wenn Audio fehlt)
# - 'manual': Manueller Sync-Punkt

# ---------------------------------------------------------------------------------
# Audio-Suche
# ---------------------------------------------------------------------------------

PHASE_DETECTION_AUDIO_SEARCH_WINDOW_S = 5.0     # +/-5s um erwarteten Zeitpunkt
PHASE_DETECTION_AUDIO_MIN_CONFIDENCE = 0.6

# ---------------------------------------------------------------------------------
# Sync-Punkt-Findung (1760 Hz Kalibrierung)
# ---------------------------------------------------------------------------------

CALIBRATION_ISOLATION_MIN_SILENCE_S = 30.0      # Mind. 30s Stille VOR 1. Peak
CALIBRATION_PEAK_INTERVAL_S = 6.5               # Erwarteter Abstand zwischen Peaks
CALIBRATION_VALIDATION_N_PEAKS = 2              # Pruefe naechste 2 Peaks

# ---------------------------------------------------------------------------------
# ADAPTIVE SYNC KONFIGURATION (NEU v2.4)
# ---------------------------------------------------------------------------------

# Aktiviere adaptive Suche bei manueller Eingabe
ENABLE_ADAPTIVE_LOW_SIGNAL_SEARCH = True

# Nur verwenden wenn Audio stark reduziert wurde
ADAPTIVE_SEARCH_AUTO_DETECT = True  # Automatisch erkennen ob nötig

# Manuelle Aktivierung (überschreibt Auto-Detect)
FORCE_ADAPTIVE_SEARCH = False

# ---------------------------------------------------------------------------------
# JSON-Fallback
# ---------------------------------------------------------------------------------

PHASE_DETECTION_JSON_TOLERANCE_MS = 100         # +/-100ms Unsicherheit ohne Audio

# ---------------------------------------------------------------------------------
# Interaktiver Modus
# ---------------------------------------------------------------------------------

PHASE_DETECTION_INTERACTIVE_MODE = 'cli'        # 'cli' oder 'auto'
PHASE_DETECTION_AUTO_SELECT_BEST = False

# ---------------------------------------------------------------------------------
# Validierung
# ---------------------------------------------------------------------------------

PHASE_DETECTION_MIN_DETECTION_RATE = 0.75       # 75% Marker muessen gefunden sein
PHASE_DETECTION_MAX_OFFSET_STD_MS = 200
PHASE_DETECTION_MAX_DEVIATION_MS = 500

# ---------------------------------------------------------------------------------
# QS-Ausgabe
# ---------------------------------------------------------------------------------

SAVE_PHASE_DETECTION_QS = True
PHASE_DETECTION_QS_FILENAME = "phase_detection_qs.png"
PHASE_DETECTION_CREATE_EVENT_TABLE = True

# ---------------------------------------------------------------------------------
# Fallback-Modi
# ---------------------------------------------------------------------------------

ENABLE_AUDIO_OPTIONAL_MODE = True
ENABLE_JSON_OPTIONAL_MODE = True

# JSON-Only Modus
JSON_ONLY_CONFIDENCE = 0.6
JSON_ONLY_TOLERANCE_MS = 200
JSON_ONLY_USE_PREDICTED_TIMES = True
JSON_ONLY_FIXATION_DURATION_S = 2.0
JSON_ONLY_STIMULUS_DURATION_S = 7.0
JSON_ONLY_ITI_S = 0.1

# Template-Modus (Audio ohne JSON)
TEMPLATE_MODE_ENABLED = True
TEMPLATE_MODE_CONFIDENCE = 0.7
TEMPLATE_N_TRIALS = 24
TEMPLATE_N_CALIBRATIONS = 3
TEMPLATE_FIXATION_DURATION_S = 2.0
TEMPLATE_STIMULUS_DURATION_S = 7.0
TEMPLATE_EXPECTED_1760_COUNT = 30
TEMPLATE_EXPECTED_440_COUNT = 24
TEMPLATE_EXPECTED_880_COUNT = 24
TEMPLATE_MIN_MARKER_DETECTION_RATE = 0.7
TEMPLATE_MAX_MARKER_DEVIATION_S = 2.0

# ---------------------------------------------------------------------------------
# Auto-Config aus JSON
# ---------------------------------------------------------------------------------

ENABLE_AUTO_CONFIG_FROM_JSON = True
AUTO_CONFIG_ALLOW_SCREEN_OVERRIDE = True
AUTO_CONFIG_ALLOW_DISTANCE_OVERRIDE = True
AUTO_CONFIG_ALLOW_TIMING_OVERRIDE = True
AUTO_CONFIG_ALLOW_TRIAL_OVERRIDE = True
AUTO_CONFIG_MIN_SCREEN_WIDTH_PX = 800
AUTO_CONFIG_MAX_SCREEN_WIDTH_PX = 3840
AUTO_CONFIG_MIN_VIEWING_DISTANCE_CM = 50
AUTO_CONFIG_MAX_VIEWING_DISTANCE_CM = 80
AUTO_CONFIG_VERBOSE = True

# =================================================================================
# 5. PUPILLEN-DETEKTION (debug_1 / shared_pupil_detection.py)
# =================================================================================

# ---------------------------------------------------------------------------------
# MediaPipe Face Mesh
# ---------------------------------------------------------------------------------

MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5

# Iris-Landmark-Indizes (MediaPipe Face Mesh mit refine_landmarks=True)
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

# ---------------------------------------------------------------------------------
# Plausibilitaets-Checks
# ---------------------------------------------------------------------------------

# Augenabstand (Interpupillary Distance)
EYE_DISTANCE_MIN_PX = 5             # Minimum IPD
EYE_DISTANCE_MAX_PX = 300           # Maximum IPD
EYE_DISTANCE_PENALTY = 0.5          # Confidence-Multiplikator bei Verletzung

AUTO_ADJUST_EYE_DISTANCE_RANGE = False

# ---------------------------------------------------------------------------------
# Qualitaetsgewichtung
# ---------------------------------------------------------------------------------

USE_QUALITY_WEIGHTED_AVERAGE = True
SINGLE_EYE_PENALTY = 0.8            # Multiplikator wenn nur ein Auge erkannt
MIN_LANDMARK_QUALITY = 0.4          # Mind. 2 von 5 Landmarks (40%)
MIN_CONFIDENCE = 0                  # Confidence-Schwelle fuer Frame-Akzeptanz

# ---------------------------------------------------------------------------------
# Outlier-Removal (IQR)
# ---------------------------------------------------------------------------------

ENABLE_OUTLIER_REMOVAL = True
IQR_MULTIPLIER = 1.5                # Tukey's Fences
MIN_SAMPLES_FOR_IQR = 10

# ---------------------------------------------------------------------------------
# Temporale Glaettung
# ---------------------------------------------------------------------------------

SMOOTHING_WINDOW_CALIBRATION = 7    # Frames (ungerade empfohlen)
SMOOTHING_METHOD = 'mean'           # 'mean', 'median', 'gaussian'
SMOOTHING_CENTER = True
SMOOTHING_AUTO_ADJUST_TO_FPS = False

# ---------------------------------------------------------------------------------
# Aggregation (fuer Kalibrierung)
# ---------------------------------------------------------------------------------

USE_MEDIAN_INSTEAD_OF_MEAN = True
AGGREGATION_METHOD = 'median'
SAVE_AGGREGATION_VARIANCE = True

# ---------------------------------------------------------------------------------
# Kalibrierungs-Extraktion
# ---------------------------------------------------------------------------------

EXTRACTION_WINDOW_START = 0.6       # Sekunden NACH Marker (Settling-Time)
EXTRACTION_WINDOW_END = 3.0         # Sekunden NACH Marker

# Abgeleitete Parameter
FIXATION_SETTLING_TIME = EXTRACTION_WINDOW_START
FIXATION_STABLE_DURATION = EXTRACTION_WINDOW_END - EXTRACTION_WINDOW_START


# =================================================================================
# 6. BLINK DETECTION (shared_blink_detection.py)
# =================================================================================

ENABLE_BLINK_DETECTION = True
EAR_BLINK_THRESHOLD = 0.16           # Eye Aspect Ratio Threshold
EAR_CONSEC_FRAMES = 2               # Mindestanzahl Frames fuer Blink

# new added
EAR_SMOOTHING_WINDOW = 3
MAX_EAR_ASYMMETRY = 0.2
TRANSITION_PADDING = 2

EAR_ADAPTIVE_RATIO = 0.65
EAR_BASELINE_MIN_SAMPLES = 10
EAR_BASELINE_WINDOW = 90
EAR_MIN_THRESHOLD = 0.10
# ---------------------------------------------------------------------------------
# Blink Filtering (fuer Auswertung in debug_6/7)
# ---------------------------------------------------------------------------------

ENABLE_BLINK_FILTERING_IN_ANALYSIS = True
POST_BLINK_RECOVERY_MS = 150        # ms nach Blink-Ende ausschliessen
MIN_VALID_SAMPLE_RATIO = 0.5        # Trial nur auswerten wenn >50% valide

# ---------------------------------------------------------------------------------
# Outside Monitor Detection
# ---------------------------------------------------------------------------------

ENABLE_OUTSIDE_MONITOR_FILTERING = True

# =================================================================================
# 7. AUDIO-MARKER-DETEKTION (debug_2)
# =================================================================================

# ---------------------------------------------------------------------------------
# Frequenzen
# ---------------------------------------------------------------------------------

AUDIO_FREQUENCIES = {
    'fixation': 440,                # Hz - Fixationskreuz beginnt
    'trial': 880,                   # Hz - Trial beginnt
    'calibration': 1760,            # Hz - Kalibrierung
    'end': 220                      # Hz - Experiment endet
}

# ---------------------------------------------------------------------------------
# Frequenzspezifische Toleranzen
# ---------------------------------------------------------------------------------

FREQUENCY_TOLERANCES = {
    440: 1,
    880: 50,
    1760: 50,
    220: 25
}
FREQUENCY_TOLERANCE = 50            # Fallback

# ---------------------------------------------------------------------------------
# Peak-Detection
# ---------------------------------------------------------------------------------

AUDIO_MAGNITUDE_PERCENTILE = 20
AUDIO_RELATIVE_THRESHOLD = 0.20
AUDIO_MIN_DISTANCE_S = 2
AUDIO_PROMINENCE = 0.08
AUDIO_CLUSTER_GAP = 0.8

# ---------------------------------------------------------------------------------
# Ton-Detektion
# ---------------------------------------------------------------------------------

TONE_MIN_DURATION_S = 0.02
TONE_MIN_INTERVAL_S = 1
TONE_SIGNAL_THRESHOLD = 0.40
AUDIO_PERCENTILE_THRESHOLD = 90

FREQUENCY_TOLERANCE_HZ = 20
MIN_TONE_DURATION_S = 0.1
MAX_TONE_DURATION_S = 1.0

# ---------------------------------------------------------------------------------
# Timeline-Erwartungen
# ---------------------------------------------------------------------------------

EXPECTED_FIXATION_DURATION_S = 1.9
EXPECTED_TRIAL_DURATION_S = 7.0
FIXATION_PHASE_DURATION_S = 1.9     # Alias
STIMULUS_DURATION_S = 7.0           # Alias

# ---------------------------------------------------------------------------------
# Trial-Struktur-Export
# ---------------------------------------------------------------------------------

SAVE_TRIAL_STRUCTURE = True
DOCUMENT_DETECTION_METHOD = True
TRIAL_ANALYSIS_BUFFER_MS = 200


# =================================================================================
# 8. EYELINK-PARAMETER (debug_3)
# =================================================================================

SAVE_EYETRACKER_SAMPLES = True      # debug_3_eyetracker_data.csv (gross!)
SAVE_EYELINK_FIXATIONS = True       # debug_3_fixations.csv
SAVE_EYELINK_SACCADES = False
SAVE_EYELINK_BLINKS = False

TRACKED_EYE = None                  # None = Auto-detect, 'L' oder 'R'


# =================================================================================
# 9. SYNCHRONISATION (debug_4)
# =================================================================================

USE_LINEAR_DRIFT_CORRECTION = True
MIN_SYNC_ANCHORS = 5

MAX_ACCEPTABLE_SYNC_STD_MS = 50.0
MAX_ACCEPTABLE_DRIFT_MS = 100.0
MIN_OVERLAP_DURATION_S = 180

# =================================================================================
# 10. KALIBRIERUNG (debug_5 / offline_calibration.py)
# =================================================================================

# ---------------------------------------------------------------------------------
# Qualitaets-Schwellen
# ---------------------------------------------------------------------------------

MIN_CALIBRATION_R2 = 0.7

# ---------------------------------------------------------------------------------
# Plausibilitaets-Checks
# ---------------------------------------------------------------------------------

def calculate_max_plausible_deg(viewing_distance_cm: float, screen_dimension_cm: float) -> float:
    """Berechnet maximalen plausiblen Sehwinkel (inkl. 50% Puffer)"""
    half_dimension = screen_dimension_cm / 2
    max_deg = np.arctan2(half_dimension, viewing_distance_cm) * 180 / np.pi
    return max_deg * 1.5

# Werden oben bereits berechnet, hier nochmal fuer Klarheit
# MAX_PLAUSIBLE_DEG_X = calculate_max_plausible_deg(VIEWING_DISTANCE_CM, SCREEN_WIDTH_CM)
# MAX_PLAUSIBLE_DEG_Y = calculate_max_plausible_deg(VIEWING_DISTANCE_CM, SCREEN_HEIGHT_CM)

# ---------------------------------------------------------------------------------
# Pre-Calibration Outlier-Filter
# ---------------------------------------------------------------------------------

ENABLE_PRECALIBRATION_OUTLIER_FILTER = False
PUPIL_OUTLIER_PERCENTILE_LOW = 1
PUPIL_OUTLIER_PERCENTILE_HIGH = 99

# ---------------------------------------------------------------------------------
# Confidence-Kombination
# ---------------------------------------------------------------------------------

COMBINE_SYNC_AND_DETECTION_CONFIDENCE = False

# ---------------------------------------------------------------------------------
# Temporal Pooling (Saxena et al. 2022)
# ---------------------------------------------------------------------------------

ENABLE_TEMPORAL_POOLING = False
TEMPORAL_POOLING_MIN_VIDEOS = 2
ENABLE_TEMPORAL_DECAY_WEIGHTING = True
TEMPORAL_DECAY_FACTOR = 0.2

# ---------------------------------------------------------------------------------
# Double Exponential Filter (Jitter-Reduktion)
# ---------------------------------------------------------------------------------

ENABLE_DOUBLE_EXPONENTIAL_FILTER = False
DOUBLE_EXP_ALPHA = 0.3              # Level-Smoothing
DOUBLE_EXP_BETA = 0.1               # Trend-Smoothing
SAVE_RAW_AND_FILTERED_GAZE = True

# ---------------------------------------------------------------------------------
# Manuelle Gaze-Offset-Korrektur
# ---------------------------------------------------------------------------------

ENABLE_MANUAL_GAZE_OFFSET = False
MANUAL_GAZE_OFFSET_X_DEG = -8.0
MANUAL_GAZE_OFFSET_Y_DEG = 0.0

# ---------------------------------------------------------------------------------
# Manuelle Head-Pose-Korrektur (Pre-Calibration)
# ---------------------------------------------------------------------------------

ENABLE_MANUAL_HEAD_POSE_CORRECTION = False
MANUAL_HEAD_POSE_YAW_DEG = 5.0
MANUAL_HEAD_POSE_PITCH_DEG = 0.0
HEAD_POSE_CORRECTION_PIXELS_PER_DEGREE = 2.5

# ---------------------------------------------------------------------------------
# Adaptive Head-Pose-Korrektur
# ---------------------------------------------------------------------------------

ENABLE_HEAD_POSE_CORRECTION = False

# Head-Pose-Extraktion
ENABLE_HEAD_POSE_EXTRACTION = False
HEAD_POSE_REFERENCE_LANDMARKS = [1, 33, 61, 199, 263, 291]

# Head-Pose-Smoothing
ENABLE_HEAD_POSE_TEMPORAL_SMOOTHING = False
HEAD_POSE_SMOOTHING_WINDOW_FRAMES = 5
HEAD_POSE_SMOOTHING_METHOD = 'median'

# Korrektur-Methode
HEAD_POSE_CORRECTION_METHOD = 'linear_approximation'
HEAD_POSE_CORRECTION_FACTOR_YAW_PX_PER_DEG = 2.8
HEAD_POSE_CORRECTION_FACTOR_PITCH_PX_PER_DEG = 1.5
HEAD_POSE_CORRECTION_FACTOR_ROLL_PX_PER_DEG = 0.0
HEAD_POSE_EYE_CENTER_OFFSET_MM = [0, 0, 50]

# Referenz-Head-Pose
HEAD_POSE_REFERENCE_AGGREGATION = 'median'
HEAD_POSE_MIN_POINTS_FOR_REFERENCE = 8

# Qualitaets-Schwellen
HEAD_POSE_MIN_DEVIATION_FOR_CORRECTION_DEG = 5.0
HEAD_POSE_MAX_PLAUSIBLE_DEVIATION_DEG = 20.0
HEAD_POSE_CORRECTION_CONFIDENCE_THRESHOLD = 0

# Debugging
SAVE_HEAD_POSE_DEBUG_CSV = False
VISUALIZE_HEAD_POSE_IN_QS_VIDEO = False

# ---------------------------------------------------------------------------------
# Fixation-Kalibrierung
# ---------------------------------------------------------------------------------

FIXATION_CALIBRATION_ENABLED = True
FIXATION_CALIBRATION_START_OFFSET_S = 0.5
FIXATION_CALIBRATION_WINDOW_DURATION_S = 1.8

# ---------------------------------------------------------------------------------
# Center-Gewichtung
# ---------------------------------------------------------------------------------

CENTER_CALIBRATION_WEIGHT = 0.5

# =================================================================================
# 11. FIXATION-DETECTION & TRIAL-ANALYSE (debug_6/7)
# =================================================================================

# ---------------------------------------------------------------------------------
# I-VT (Identification by Velocity Threshold)
# ---------------------------------------------------------------------------------

IVT_VELOCITY_THRESHOLD_DEG_S = {
    1: 30.0,                        # Modus 1: Standard
    2: 30.0,                        # Modus 2: Standard
}

IVT_MIN_FIXATION_DURATION_MS = 100
IVT_MAX_FIXATION_DURATION_MS = 8000

SMOOTH_BEFORE_IVT = True
IVT_SMOOTHING_WINDOW = 3

# ---------------------------------------------------------------------------------
# Fixation Temporal Averaging
# ---------------------------------------------------------------------------------

ENABLE_FIXATION_TEMPORAL_AVERAGING = False
FIXATION_BUFFER_SIZE = 5
IVT_MAX_HISTORY_FRAMES = 5
MIN_FIXATIONS_FOR_ANALYSIS = 3

# ---------------------------------------------------------------------------------
# Center of Cancellation (CoC)
# ---------------------------------------------------------------------------------

CALCULATE_COC = True
COC_NORMAL_RANGE = (-0.2, 0.2)

# ---------------------------------------------------------------------------------
# Lateralisierung
# ---------------------------------------------------------------------------------

LEFT_THRESHOLD_DEG = -1.0
RIGHT_THRESHOLD_DEG = 1.0

# ---------------------------------------------------------------------------------
# Trial-Qualitaet
# ---------------------------------------------------------------------------------

MIN_FIXATIONS_PER_TRIAL = 3
MIN_TRIAL_COVERAGE_PERCENT = 50

# ---------------------------------------------------------------------------------
# Head-Pose Stability Filtering
# ---------------------------------------------------------------------------------

ENABLE_HEAD_POSE_FILTERING = True
HEAD_POSE_YAW_THRESHOLD = 10.0
HEAD_POSE_PITCH_THRESHOLD = 10.0
HEAD_POSE_REFERENCE_METHOD = 'median'
SAVE_REJECTED_TRIALS = True


# =================================================================================
# 12. VISUALISIERUNG & EXPORT
# =================================================================================

PLOT_DPI = 150

# ---------------------------------------------------------------------------------
# Detaillierte Exports (Optional)
# ---------------------------------------------------------------------------------

SAVE_DETAILED_FIXATIONS_CSV = False
SAVE_TRIAL_TIMESERIES_CSV = False
CREATE_INDIVIDUAL_TRIAL_PLOTS = False


# =================================================================================
# VALIDIERUNGS-FUNKTION
# =================================================================================

def validate_config():
    """
    Prueft config.py auf Plausibilitaet und Konsistenz.
    """
    
    issues = []
    
    # Check 1: Dateipfade
    if not VIDEO_PATH.exists():
        issues.append(f"[!] VIDEO_PATH nicht gefunden: {VIDEO_PATH}")
    
    if ANALYSIS_MODE == 2 and not EYETRACKER_FILE_PATH.exists():
        issues.append(f"[!] EYETRACKER_FILE_PATH nicht gefunden: {EYETRACKER_FILE_PATH}")
    
    if ANALYSIS_MODE in [1, 2] and not CALIBRATION_PKL_PATH.exists():
        issues.append(f"[!] CALIBRATION_PKL_PATH nicht gefunden: {CALIBRATION_PKL_PATH}")
    
    # Check 2: Experiment-Logik
    if N_TRIALS < 1 or N_TRIALS > 100:
        issues.append(f"[!] N_TRIALS unplausibel: {N_TRIALS} (erwartet: 1-100)")
    
    # Check 3: Bildschirm-Parameter
    if VIEWING_DISTANCE_CM < 30 or VIEWING_DISTANCE_CM > 100:
        issues.append(f"[!] VIEWING_DISTANCE_CM unplausibel: {VIEWING_DISTANCE_CM}cm")
    
    aspect_ratio = SCREEN_WIDTH_PX / SCREEN_HEIGHT_PX
    if aspect_ratio < 1.2 or aspect_ratio > 2.5:
        issues.append(f"[!] Bildschirm-Aspect-Ratio ungewoehnlich: {aspect_ratio:.2f}")
    
    # Check 4: Schwellen-Konsistenz
    if EYE_DISTANCE_MIN_PX >= EYE_DISTANCE_MAX_PX:
        issues.append(f"[!] EYE_DISTANCE: MIN >= MAX")
    
    if LEFT_THRESHOLD_DEG >= RIGHT_THRESHOLD_DEG:
        issues.append(f"[!] LATERALIZATION: LEFT >= RIGHT")
    
    if EXTRACTION_WINDOW_START >= EXTRACTION_WINDOW_END:
        issues.append(f"[!] EXTRACTION_WINDOW: START >= END")

    # Check 5: Phase Detection
    if ENABLE_AUTOMATIC_PHASE_DETECTION:
        if not EXPERIMENT_SYNC_JSON_PATH.exists():
            issues.append(f"[!] EXPERIMENT_SYNC_JSON_PATH nicht gefunden")
    
    # Check 6: Modus-Validierung
    if ANALYSIS_MODE not in [1, 2]:
        issues.append(f"[!] ANALYSIS_MODE ungueltig: {ANALYSIS_MODE} (nur 1 oder 2 erlaubt)")
    
    # Ausgabe
    if issues:
        print(f"\n{'='*70}")
        print("CONFIG-VALIDIERUNG: PROBLEME GEFUNDEN")
        print(f"{'='*70}")
        for issue in issues:
            print(issue)
        print(f"\n[i] Bitte config.py pruefen und korrigieren!\n")
        return False
    else:
        return True


# =================================================================================
# STARTUP
# =================================================================================

print(f"[OK] Config geladen (v4.0)")
print(f"     Modus: {ANALYSIS_MODE}")
print(f"     Output: {OUTPUT_BASE_DIR}")
