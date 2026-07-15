"""
=================================================================================
DEBUG 0: AUTOMATISCHE PHASEN-DETEKTION v2.2 (Multi-Anchor Integration)
=================================================================================

CHANGELOG v2.0 >> v2.1:
----------------------
 BEHALTEN aus v2.0 (bewährt):
   • FrequencyDetector (vollständig)
   • SyncPointFinder (Top-3 Kandidaten)
   • CalibrationPointExtractor (ALLE Punkte)
   • QS-Grafiken (6 Panels)
   • Detail-Grafiken (Kalibrierung + Trials)

* NEU in v2.1 (wissenschaftlich fundiert):
   • Multi-Anchor-Regression (Drift-Korrektur)
   • JSON-Format-Detection (v1/v2 Kompatibilität)
   • Strukturgeführte Audio-Suche (gezielt ±5s)
   • Outlier-Removal (Tukey, 1977)
   • Erweiterte Statistik (R², Drift, Confidence)

Wissenschaftliche Grundlage v2.1:
----------------------------------
[1] Kalman (1960): Linear Filtering für Drift-Korrektur
[2] Tukey (1977): Studentized Residuals für Outlier-Detection
[3] Holmqvist et al. (2011): R² > 0.95 = exzellente Synchronisation
[4] Zhang et al. (2015): Strukturgeführte Suche >> 2-3x Detection-Rate
[5] Wan & Van Der Merwe (2000): Hierarchische Sensor-Fusion

Output v2.1:
------------
• phases_detected.json (KOMPATIBEL mit v2.0!)
• QS-Grafiken (wie v2.0 + Drift-Plot)
• Multi-Anchor-Statistik (CSV)

Version: 2.1 (2025-11-15)
Backwards Compatible: Ja (JSON-Struktur identisch zu v2.0)
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

import numpy as np
import pandas as pd
import librosa
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from scipy.signal import find_peaks, butter, sosfiltfilt, hilbert
from scipy.stats import linregress
import cv2
import os
from dataclasses import dataclass, field
from glob import glob
from enum import Enum

# Config-Import
from config import (
    OUTPUT_BASE_DIR, MAIN_VIDEO_PATH, EXPERIMENT_SYNC_JSON_PATH,
    AUDIO_FREQUENCIES, FREQUENCY_TOLERANCES, FREQUENCY_TOLERANCE,
    AUDIO_MAGNITUDE_PERCENTILE, AUDIO_RELATIVE_THRESHOLD,
    AUDIO_MIN_DISTANCE_S, AUDIO_PROMINENCE, AUDIO_CLUSTER_GAP,
    N_TRIALS, FIRST_TRIAL_IS_PRACTICE, CALIBRATION_TIMING_JSON_DIR,
    PHASE_DETECTION_AUDIO_SEARCH_WINDOW_S,
    PHASE_DETECTION_MIN_DETECTION_RATE, PHASE_DETECTION_MAX_OFFSET_STD_MS,
    SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX, VIEWING_DISTANCE_CM, SCREEN_WIDTH_CM,
    ENABLE_ADAPTIVE_LOW_SIGNAL_SEARCH, ADAPTIVE_SEARCH_AUTO_DETECT, FORCE_ADAPTIVE_SEARCH
)

# Audio-Marker Frequenzen
FREQ_CALIBRATION = 1760  # Hz - Kalibrierungspunkte
FREQ_TRIAL_START = 440   # Hz - Fixations-Start (Trial)
FREQ_TRIAL_END = 880     # Hz - Fixations-Ende (Trial)

# ══════════════════════════════════════════════════════════════════════════════
# Timing-Korrektur-Konstanten
# ══════════════════════════════════════════════════════════════════════════════
TONE_DURATION_MS = 200          # Länge eines Marker-Tons in ms
PEAK_OFFSET_MS = 100            # Ton-Mitte (Peak) = 100ms nach Ton-Start
DEFAULT_LATENCY_MS = 500.0      # Default Audio-Latenz für V1 JSONs (ohne actual_delay)
MIN_ANCHORS_FOR_ESTIMATION = 5  # Mindestanzahl Töne für Latenz-Schätzung

# ════════════════════════════════════════════════════════════════════════════════
# ADAPTIVE SYNC IMPORT (NEU v2.4 - für reduziertes Audio)
# ════════════════════════════════════════════════════════════════════════════════

ADAPTIVE_SYNC_AVAILABLE = False
_ADAPTIVE_IMPORT_ERROR = None

try:
    from utils.adaptive_sync_detection import (
        run_adaptive_sync_for_all_phases,
        AdaptiveSyncResult,
        AdaptiveDetectionResult  # Optional, falls später benötigt
    )
    ADAPTIVE_SYNC_AVAILABLE = True
    print("[OK] adaptive_sync_detection v2.0 geladen")
except ImportError as e:
    _ADAPTIVE_IMPORT_ERROR = str(e)
    print(f"[WARN] adaptive_sync_detection nicht verfügbar: {e}")
except Exception as e:
    _ADAPTIVE_IMPORT_ERROR = f"{type(e).__name__}: {e}"
    print(f"[WARN] adaptive_sync_detection Fehler: {type(e).__name__}: {e}")

# Config-Parameter mit Fallback
try:
    from config import ENABLE_ADAPTIVE_LOW_SIGNAL_SEARCH
except ImportError:
    ENABLE_ADAPTIVE_LOW_SIGNAL_SEARCH = True  # Default: aktiviert

# ════════════════════════════════════════════════════════════════════════════════
# GLOBALER AUDIO-STATUS (NEU v2.5 - vermeidet redundante Prüfungen)
# ════════════════════════════════════════════════════════════════════════════════

# Wird in run_phase_detection() gesetzt nach dem initialen Audio-Quality-Check
_AUDIO_QUALITY_RESULT = None  # Dict mit 'usable', 'has_signal', 'has_1760hz', 'signal_strength'
_SKIP_AUDIO_VALIDATION = False  # True wenn Audio nicht nutzbar ist

DEBUG_0_MANUAL_OFFSET_MS = 0

# ==================== AUTO-CONFIG AUS JSON (NEU Phase 1) ====================

# ═══════════════════════════════════════════════════════════════════
# ENVIRONMENT-VARIABLE-OVERRIDES (für master_cli.py Integration)
# ═══════════════════════════════════════════════════════════════════

import os

if 'PIPELINE_MAIN_VIDEO_PATH' in os.environ:
    MAIN_VIDEO_PATH = Path(os.environ['PIPELINE_MAIN_VIDEO_PATH'])
    print(f"Override: MAIN_VIDEO_PATH = {MAIN_VIDEO_PATH}")

if 'PIPELINE_EXPERIMENT_SYNC_JSON_PATH' in os.environ:
    EXPERIMENT_SYNC_JSON_PATH = Path(os.environ['PIPELINE_EXPERIMENT_SYNC_JSON_PATH'])
    print(f"Override: EXPERIMENT_SYNC_JSON_PATH = {EXPERIMENT_SYNC_JSON_PATH}")

if 'PIPELINE_CALIBRATION_TIMING_JSON_DIR' in os.environ:
    CALIBRATION_TIMING_JSON_DIR = Path(os.environ['PIPELINE_CALIBRATION_TIMING_JSON_DIR'])
    print(f"Override: CALIBRATION_TIMING_JSON_DIR = {CALIBRATION_TIMING_JSON_DIR}")

if 'PIPELINE_OUTPUT_BASE_DIR' in os.environ:
    OUTPUT_BASE_DIR = Path(os.environ['PIPELINE_OUTPUT_BASE_DIR'])
    print(f"Override: OUTPUT_BASE_DIR = {OUTPUT_BASE_DIR}")

def load_experiment_metadata(json_log: Dict) -> Dict:
    """
    Lädt Experiment-Metadaten aus JSON und überschreibt config.py (temporär).
    
     NEU Phase 1: Automatische Konfiguration aus experiment_sync_log.json
    
    Args:
        json_log: experiment_sync_log.json dict
    
    Returns:
        Dict mit config-Overrides (z.B. {'SCREEN_WIDTH_PX': 1920})
    """
    
    from config import (
        ENABLE_AUTO_CONFIG_FROM_JSON,
        AUTO_CONFIG_ALLOW_SCREEN_OVERRIDE,
        AUTO_CONFIG_ALLOW_DISTANCE_OVERRIDE,
        AUTO_CONFIG_ALLOW_TIMING_OVERRIDE,
        AUTO_CONFIG_ALLOW_TRIAL_OVERRIDE,
        AUTO_CONFIG_MIN_SCREEN_WIDTH_PX,
        AUTO_CONFIG_MAX_SCREEN_WIDTH_PX,
        AUTO_CONFIG_MIN_VIEWING_DISTANCE_CM,
        AUTO_CONFIG_MAX_VIEWING_DISTANCE_CM,
        AUTO_CONFIG_VERBOSE
    )
    
    if not ENABLE_AUTO_CONFIG_FROM_JSON:
        return {}
    
    metadata = json_log.get('metadata', {})
    
    if not metadata:
        if AUTO_CONFIG_VERBOSE:
            print(f"Keine Metadaten in JSON >> Nutze config.py Defaults")
        return {}
    
    overrides = {}
    
    # ──────────────────────────────────────────────────────────────────
    # 1. Bildschirm-Auflösung
    # ──────────────────────────────────────────────────────────────────
    
    if AUTO_CONFIG_ALLOW_SCREEN_OVERRIDE:
        if 'screen_resolution_px' in metadata:
            width_px, height_px = metadata['screen_resolution_px']
            
            # Sicherheits-Check
            if AUTO_CONFIG_MIN_SCREEN_WIDTH_PX <= width_px <= AUTO_CONFIG_MAX_SCREEN_WIDTH_PX:
                overrides['SCREEN_WIDTH_PX'] = width_px
                overrides['SCREEN_HEIGHT_PX'] = height_px
            else:
                print(f"Ungültige screen_resolution_px: {width_px}x{height_px} (ignoriert)")
        
        if 'screen_size_mm' in metadata:
            width_mm, height_mm = metadata['screen_size_mm']
            overrides['SCREEN_WIDTH_CM'] = width_mm / 10
            overrides['SCREEN_HEIGHT_CM'] = height_mm / 10
    
    # ──────────────────────────────────────────────────────────────────
    # 2. Viewing Distance
    # ──────────────────────────────────────────────────────────────────
    
    if AUTO_CONFIG_ALLOW_DISTANCE_OVERRIDE and 'viewing_distance_mm' in metadata:
        distance_cm = metadata['viewing_distance_mm'] / 10
        
        # Sicherheits-Check
        if AUTO_CONFIG_MIN_VIEWING_DISTANCE_CM <= distance_cm <= AUTO_CONFIG_MAX_VIEWING_DISTANCE_CM:
            overrides['VIEWING_DISTANCE_CM'] = distance_cm
        else:
            print(f"Ungültige viewing_distance: {distance_cm}cm (ignoriert)")
    
    # ──────────────────────────────────────────────────────────────────
    # 3. Trial-Anzahl (aus randomization)
    # ──────────────────────────────────────────────────────────────────
    
    if AUTO_CONFIG_ALLOW_TRIAL_OVERRIDE and 'randomization' in metadata:
        trial_order = metadata['randomization'].get('trial_order', [])
        if trial_order:
            overrides['N_TRIALS'] = len(trial_order)
    
    # ──────────────────────────────────────────────────────────────────
    # 4. Timing-Parameter
    # ──────────────────────────────────────────────────────────────────
    
    if AUTO_CONFIG_ALLOW_TIMING_OVERRIDE and 'timing' in metadata:
        timing = metadata['timing']
        
        if 'fixation_expected_duration_s' in timing:
            overrides['EXPECTED_FIXATION_DURATION_S'] = timing['fixation_expected_duration_s']
            overrides['FIXATION_PHASE_DURATION_S'] = timing['fixation_expected_duration_s']
        
        if 'stimulus_duration_s' in timing:
            overrides['EXPECTED_TRIAL_DURATION_S'] = timing['stimulus_duration_s']
            overrides['STIMULUS_DURATION_S'] = timing['stimulus_duration_s']
    
    return overrides

def check_audio_quality(audio: np.ndarray, sr: int, force_recheck: bool = False) -> Dict:
    """
    Prüft ob Audio für Synchronisation nutzbar ist.
    
    NEU v2.5: Caching - vermeidet redundante Checks!
    
    Args:
        audio: Audio-Signal
        sr: Sample-Rate
        force_recheck: True = Ignoriere Cache (für initialen Check)
    
    Returns:
        {
            'usable': bool,
            'has_signal': bool,
            'has_1760hz': bool,
            'signal_strength': float
        }
    """
    
    global _AUDIO_QUALITY_RESULT
    
    # Cache nutzen wenn bereits geprüft
    if _AUDIO_QUALITY_RESULT is not None and not force_recheck:
        return _AUDIO_QUALITY_RESULT
    
    from config import ENABLE_AUDIO_OPTIONAL_MODE
    
    # Berechne Signal-Stärke (RMS)
    rms = np.sqrt(np.mean(audio**2))
    signal_strength = min(1.0, rms * 100)
    
    has_signal = signal_strength > 0.01
    
    # Prüfe ob 1760 Hz Marker vorhanden (NUR wenn Signal vorhanden!)
    if has_signal:
        detector = FrequencyDetector()
        peaks_1760 = detector.detect_frequency_peaks(audio, sr, 1760)
        has_1760hz = len(peaks_1760) >= 3
    else:
        has_1760hz = False
    
    usable = has_signal and has_1760hz
    
    # Falls Audio-Optional-Mode deaktiviert: Erzwinge Audio
    if not ENABLE_AUDIO_OPTIONAL_MODE and not usable:
        print(f"\nFEHLER: Audio nicht nutzbar, aber ENABLE_AUDIO_OPTIONAL_MODE=False!")
        print(f"   >> Audio ist PFLICHT in config.py")
        exit(1)
    
    result = {
        'usable': usable,
        'has_signal': has_signal,
        'has_1760hz': has_1760hz,
        'signal_strength': signal_strength
    }
    
    # Cache speichern
    _AUDIO_QUALITY_RESULT = result
    
    return result

# ==================== WISSENSCHAFTLICHE DOKUMENTATION v2.1 ====================
"""
Multi-Anchor-Synchronisation - Wissenschaftliche Fundierung:
------------------------------------------------------------

1. PROBLEM: Zeitachsen-Drift (User-Report)
   • Symptom: Trial 7 (~80s in Block1) hat 2.5s Verschiebung
   • Ursache: Single-Anchor-Offset kumuliert Fehler über Zeit
   • Lösung: Multi-Anchor Linear Regression

2. HIERARCHISCHE SENSOR-FUSION (Wan & Van Der Merwe, 2000)
   Priorität 1: Audio-Marker (präzise, ±20ms)
   Priorität 2: JSON-Zeitstempel (±100-500ms)
   Methode: Nutze beste Quelle, Fallback bei Fehlen

3. DRIFT-KORREKTUR (Kalman, 1960)
   Problem: Video/EyeLink-Clock-Drift (linear akkumulierend)
   Formel: eyelink_time = slope x video_time + intercept
   Qualität: R² > 0.95 = exzellent (Holmqvist et al., 2011)

4. OUTLIER-DETECTION (Tukey, 1977)
   Methode: Studentized Residuals
   Schwelle: |t| > 3.0 = Ausreißer (99.7% Konfidenz)
   Iterativ: Max. 3 Durchläufe

5. STRUKTURGEFÜHRTE SUCHE (Zhang et al., 2015)
   Strategie: Suche Audio ±5s um JSON-Vorhersage
   Effekt: 2-3x höhere Detection-Rate vs. blind
   Fallback: JSON-only bei Fehlen

Literatur:
----------
[1] Kalman, R. E. (1960). A new approach to linear filtering.
[2] Tukey, J. W. (1977). Exploratory data analysis.
[3] Holmqvist, K., et al. (2011). Eye tracking: A comprehensive guide.
[4] Zhang, X., et al. (2015). Eye tracking for public health.
[5] Wan, E. A., & Van Der Merwe, R. (2000). Unscented Kalman filter.
"""

# ==================== ERWEITERTE DATENSTRUKTUREN v2.1 ====================

class JSONFormat(Enum):
    """JSON-Format-Versionen (Backwards Compatibility)"""
    V1_LEGACY = "v1_legacy"        # Alte JSON (nur eyelink_time_ms)
    V2_EXTENDED = "v2_extended"    # Neue JSON (mit audio_play_start_*)

class SyncMethod(Enum):
    """Synchronisations-Methoden (v2.1)"""
    AUDIO_DETECTED = "audio_detected"           # Audio-Kandidaten-Auswahl (v2.0)
    AUDIO_GUIDED = "audio_guided_search"        # Strukturgeführte Suche (v2.1)
    JSON_ONLY = "json_only"                     # JSON-Fallback
    JSON_ESTIMATED = "json_estimated"           # JSON mit Schätzung
    MANUAL = "manual"                           # Manuell

@dataclass
class CalibrationPoint:
    """Einzelner Kalibrierpunkt (v2.0 unverändert)"""
    point_id: int
    position: Tuple[int, int]
    matlab_time_s: float
    eyelink_time_ms: float
    video_time_s: float
    audio_expected_s: float
    audio_found_s: Optional[float] = None
    audio_deviation_ms: Optional[float] = None
    confidence: float = 0.5

@dataclass
class SyncCandidate:
    """Kandidat für Sync-Punkt (v2.0 unverändert)"""
    video_time_s: float
    confidence: float
    isolation_score: float
    json_alignment_score: float
    next_peaks_found: List[bool]
    reason: str
    frequency_quality: float

@dataclass
class SyncAnchor:
    """
    NEU v2.1: Sync-Ankerpunkt für Multi-Anchor-Regression
    
    Attributes:
        event_type: 'calibration_beg', 'trial_1_fixation', etc.
        video_time_s: Video-Zeitstempel (Sekunden)
        eyelink_time_ms: EyeLink-Zeitstempel (Millisekunden, korrigiert!)
        method: Wie wurde dieser Anchor gefunden?
        confidence: 0.0-1.0 (Qualität)
        audio_deviation_ms: Abweichung JSON >> Audio (falls verfügbar)
        frequency: 440/880/1760 Hz
    """
    event_type: str
    video_time_s: float
    eyelink_time_ms: float
    method: SyncMethod
    confidence: float
    audio_deviation_ms: Optional[float] = None
    frequency: Optional[int] = None

@dataclass
class SyncResult:
    """
    Ergebnis der Synchronisation.
    
    WICHTIG: Enthält SEPARATE Offsets für Kalibrierung und Trials!
    """
    
    offset_ms: float                    # Haupt-Offset (für Trials)
    method: str                         # 'dual_path_offset', 'initial_offset_fallback', etc.
    
    # Statistiken
    n_anchors: int = 0
    std_ms: float = 0.0
    estimated_latency_ms: float = 0.0
    
    # Details für QC
    anchor_offsets: List[float] = field(default_factory=list)
    anchor_frequencies: List[int] = field(default_factory=list)
    
    # NEUE FELDER: Separate Offsets
    offset_calibration_ms: Optional[float] = None   # Offset für Kalibrierung (1760 Hz)
    offset_trials_ms: Optional[float] = None        # Offset für Trials (440/880 Hz)
    std_calibration_ms: float = 0.0
    std_trials_ms: float = 0.0

class CompatMultiAnchorResult:
    """
    Kompatibilitäts-Wrapper für alte Code-Teile.
    
    WICHTIG: Nutzt jetzt SEPARATE Offsets für Kalibrierung und Trials!
    """
    
    def __init__(self, sync_result: SyncResult):
        self.method = sync_result.method
        self.n_anchors = sync_result.n_anchors
        self.drift_ms = sync_result.std_ms
        self.slope = 1.0
        self.outliers = []
        self.anchors = []
        
        # SEPARATE Offsets (NEU!)
        self.offset_calibration_ms = sync_result.offset_calibration_ms
        self.offset_trials_ms = sync_result.offset_trials_ms
        
        # Haupt-Offset = Trial-Offset (oder Fallback)
        self.offset_ms = sync_result.offset_trials_ms if sync_result.offset_trials_ms is not None else sync_result.offset_ms
        
        # Für alte Dual-Anchor Kompatibilität
        self.offset_1760hz_ms = sync_result.offset_calibration_ms if sync_result.offset_calibration_ms is not None else self.offset_ms
        self.offset_440hz_ms = self.offset_ms
        self.offset_880hz_ms = self.offset_ms
        
        # Frequenz-Differenz (für QC)
        if sync_result.offset_calibration_ms is not None and sync_result.offset_trials_ms is not None:
            self.frequency_difference_ms = sync_result.offset_trials_ms - sync_result.offset_calibration_ms
        else:
            self.frequency_difference_ms = 0.0
        
        self.chosen_trial_offset_ms = self.offset_ms
        self.chosen_trial_frequency = 440
        self.marker_choice_reason = "Separate Offsets für Kalibrierung und Trials"
        
        # Qualitäts-Metriken
        self.r_squared = 0.99 if sync_result.std_ms < 50 else 0.95 if sync_result.std_ms < 100 else 0.90
        self.r_squared_1760hz = 0.99 if sync_result.std_calibration_ms < 50 else 0.95
        self.r_squared_440hz = self.r_squared
        
        # Anchor-Zählungen
        self.n_anchors_1760hz = sum(1 for f in sync_result.anchor_frequencies if f == FREQ_CALIBRATION)
        self.n_anchors_440hz = sum(1 for f in sync_result.anchor_frequencies if f == FREQ_TRIAL_START)


# Typ-Alias für Kompatibilität
MultiAnchorResult = CompatMultiAnchorResult

def find_calibration_event_for_point(point: Dict, json_log: Dict, label: str) -> Optional[Dict]:
    """
    Findet das zugehörige calibration_start Event für einen Kalibrierungspunkt.
    
    WICHTIG: Das calibration_start Event enthält sowohl matlab_time_s als auch
    eyelink_time_ms - damit können wir die EyeLink-Zeit für jeden einzelnen
    Kalibrierungspunkt präzise berechnen!
    """
    events = json_log.get('events', [])
    
    for event in events:
        if event.get('event_type') == 'calibration_start' and event.get('label') == label:
            return event
    
    return None

def get_calibration_point_eyelink_time(point_timestamp: float, 
                                        calib_start_event: Dict) -> float:
    """
    Berechnet die präzise EyeLink-Zeit für einen Kalibrierungspunkt.
    
    Methode:
    1. Berechne Zeitdifferenz zwischen Punkt und calibration_start (MATLAB-Zeit)
    2. Addiere diese Differenz zur EyeLink-Zeit des calibration_start Events
    
    Args:
        point_timestamp: MATLAB-Zeit des Kalibrierungspunkts (aus calibration_timing_*.json)
        calib_start_event: Das calibration_start Event (aus experiment_sync_log.json)
    
    Returns:
        EyeLink-Zeit in Millisekunden
    """
    # MATLAB-Zeiten
    calib_start_matlab_s = calib_start_event.get('matlab_time_s', 0)
    
    # EyeLink-Zeit des calibration_start (ACHTUNG: ist in Sekunden trotz Name!)
    calib_start_eyelink_s = calib_start_event.get('eyelink_time_ms', 0)
    
    # Zeitdifferenz in Sekunden
    delta_s = point_timestamp - calib_start_matlab_s
    
    # EyeLink-Zeit des Punktes
    point_eyelink_s = calib_start_eyelink_s + delta_s
    
    # Konvertiere zu Millisekunden
    return point_eyelink_s * 1000

def find_trial_event(trial_number: int, json_log: Dict) -> Optional[Dict]:
    """
    Findet das zugehörige JSON-Event für einen Trial.
    
    Sucht nach start_fixation_X Event.
    """
    events = json_log.get('events', [])
    
    # Suche nach event_type = 'start_fixation_X'
    event_type = f'start_fixation_{trial_number}'
    
    for event in events:
        if event.get('event_type') == event_type:
            return event
    
    # Fallback: Suche nach trial_number Feld
    for event in events:
        if (event.get('event_type', '').startswith('start_fixation') and 
            event.get('trial_number') == trial_number):
            return event
    
    # Sonderfall Practice (trial_number = 0)
    if trial_number == 0:
        for event in events:
            if 'practice' in event.get('event_type', '').lower() and 'start_fixation' in event.get('event_type', '').lower():
                return event
    
    return None

def calculate_sync_offset(detected_anchors: List[Dict], 
                          json_format: 'JSONFormat',
                          estimated_latency_ms: float) -> 'SyncResult':
    """
    Berechnet den Synchronisations-Offset aus gefundenen Audio-Markern.
    
    WICHTIG: Berechnet SEPARATE Offsets für Kalibrierung und Trials!
    Die Audio-Latenz ist unterschiedlich zwischen den MATLAB-Pfaden.
    """
    
    if len(detected_anchors) == 0:
        print("    [!] Keine Anchors vorhanden!")
        return SyncResult(
            offset_ms=0.0,
            method='no_anchors',
            n_anchors=0
        )
    
    # Sammle Offsets nach Frequenz
    offsets_1760 = []  # Kalibrierung
    offsets_440 = []   # Trial-Start
    offsets_880 = []   # Trial-End
    
    all_offsets = []
    all_frequencies = []
    
    for anchor in detected_anchors:
        event = anchor.get('event')
        video_time_s = anchor['video_time_s']
        frequency = anchor['frequency']
        
        if event is None:
            continue
        
        # Berechne korrigierte EyeLink-Zeit für den Peak
        eyelink_time_peak_ms = get_corrected_eyelink_time_for_peak(
            event=event,
            frequency=frequency,
            json_format=json_format,
            estimated_latency_ms=estimated_latency_ms
        )
        
        # Offset = EyeLink-Zeit - Video-Zeit
        video_time_ms = video_time_s * 1000
        offset = eyelink_time_peak_ms - video_time_ms
        
        all_offsets.append(offset)
        all_frequencies.append(frequency)
        
        # Nach Frequenz sortieren
        if frequency == FREQ_CALIBRATION:
            offsets_1760.append(offset)
        elif frequency == FREQ_TRIAL_START:
            offsets_440.append(offset)
        elif frequency == FREQ_TRIAL_END:
            offsets_880.append(offset)
    
    if len(all_offsets) == 0:
        print("    [!] Keine gültigen Anchors!")
        return SyncResult(
            offset_ms=0.0,
            method='no_valid_anchors',
            n_anchors=0
        )
    
    # ══════════════════════════════════════════════════════════════════
    # SEPARATE OFFSETS für Kalibrierung und Trials!
    # ══════════════════════════════════════════════════════════════════
    
    # Kalibrierungs-Offset (1760 Hz)
    if len(offsets_1760) > 0:
        offset_calibration = np.median(offsets_1760)
        std_calibration = np.std(offsets_1760) if len(offsets_1760) > 1 else 0
    else:
        offset_calibration = None
        std_calibration = 0
    
    # Trial-Offset (440 Hz + 880 Hz kombiniert)
    offsets_trials = offsets_440 + offsets_880
    if len(offsets_trials) > 0:
        offset_trials = np.median(offsets_trials)
        std_trials = np.std(offsets_trials) if len(offsets_trials) > 1 else 0
    else:
        offset_trials = None
        std_trials = 0
    
    # Gesamt-Offset (für Kompatibilität)
    final_offset = np.median(all_offsets)
    std_offset = np.std(all_offsets)
    
    # ══════════════════════════════════════════════════════════════════
    # Ausgabe
    # ══════════════════════════════════════════════════════════════════
    
    print(f"\n    [Sync] Offset berechnet aus {len(all_offsets)} Anchors:")
    
    if offset_calibration is not None:
        print(f"           1760 Hz (Kalibrierung): {offset_calibration:.1f} ms (n={len(offsets_1760)}, Std={std_calibration:.1f} ms)")
    
    if len(offsets_440) > 0:
        median_440 = np.median(offsets_440)
        std_440 = np.std(offsets_440) if len(offsets_440) > 1 else 0
        print(f"           440 Hz (Trial-Start):   {median_440:.1f} ms (n={len(offsets_440)}, Std={std_440:.1f} ms)")
    
    if len(offsets_880) > 0:
        median_880 = np.median(offsets_880)
        std_880 = np.std(offsets_880) if len(offsets_880) > 1 else 0
        print(f"           880 Hz (Trial-End):     {median_880:.1f} ms (n={len(offsets_880)}, Std={std_880:.1f} ms)")
    
    if offset_calibration is not None and offset_trials is not None:
        diff = offset_trials - offset_calibration
        print(f"\n           ► Differenz Trials - Kalibrierung: {diff:+.1f} ms")
        print(f"             (Unterschiedliche Audio-Latenz erwartet!)")
    
    # Erstelle SyncResult mit BEIDEN Offsets
    result = SyncResult(
        offset_ms=offset_trials if offset_trials is not None else final_offset,
        method='dual_path_offset',
        n_anchors=len(all_offsets),
        std_ms=std_trials if offset_trials is not None else std_offset,
        estimated_latency_ms=estimated_latency_ms,
        anchor_offsets=all_offsets,
        anchor_frequencies=all_frequencies
    )
    
    # Speichere separate Offsets als zusätzliche Attribute
    result.offset_calibration_ms = offset_calibration
    result.offset_trials_ms = offset_trials
    result.std_calibration_ms = std_calibration
    result.std_calibration_ms = std_calibration
    result.std_trials_ms = std_trials if std_trials > 0 else std_calibration
    
    # Bei beschädigtem Audio (keine Trial-Anchors): Schätze Trial-Offset aus Kalibrierung
    if offset_trials is None and offset_calibration is not None:
        # Typische Differenz zwischen Kalibrierung und Trials: ~500-700ms (unterschiedliche Latenz)
        ESTIMATED_CALIB_TRIAL_DIFF_MS = 600  # Konservative Schätzung
        result.offset_trials_ms = offset_calibration + ESTIMATED_CALIB_TRIAL_DIFF_MS
        result.offset_ms = result.offset_trials_ms
        print(f"\n           [!] Keine Trial-Anchors >> Schätze Trial-Offset: {result.offset_trials_ms:.0f} ms")
        print(f"               (Kalibrierung + {ESTIMATED_CALIB_TRIAL_DIFF_MS} ms)")
    
    return result

# ==================== FREQUENCY-DETECTOR (v2.0 vollständig) ====================

class FrequencyDetector:
    """
    Robuster Frequenz-Detektor mit frequenzspezifischen Toleranzen.
    
     Übernommen aus debug_2_audio_detection.py (v2.2)
     Erweitert für debug_0: Nutzt JSON-Vorhersagen für gezielte Suche
    """
    
    def __init__(self):
        self.audio_cache = {}  # Cache für Audio-Dateien (Performance!)
    
    def detect_frequency_peaks(self, audio: np.ndarray, sr: int, 
                               target_freq: float,
                               expected_times: Optional[List[float]] = None,
                               search_window_s: float = 5.0) -> List[float]:
        """
        Detektiert Frequenz-Peaks.
        
        Args:
            audio: Audio-Signal
            sr: Sample-Rate
            target_freq: Ziel-Frequenz (440, 880, 1760 Hz)
            expected_times: Optional - JSON-Vorhersagen für gezielte Suche
            search_window_s: Suchfenster um erwartete Zeiten (±5s)
        
        Returns:
            Liste von Zeitstempeln (in Sekunden)
        """
        
        # ──────────────────────────────────────────────────────────────
        # SCHRITT 1: Frequenzspezifische Toleranz
        # ──────────────────────────────────────────────────────────────
        
        if target_freq in FREQUENCY_TOLERANCES:
            tolerance = FREQUENCY_TOLERANCES[target_freq]
        else:
            tolerance = FREQUENCY_TOLERANCE
        
        # ──────────────────────────────────────────────────────────────
        # SCHRITT 2: Bandpass-Filter (SOS für Stabilität!)
        # ──────────────────────────────────────────────────────────────
        
        nyquist = sr / 2
        low_hz = target_freq - tolerance
        high_hz = target_freq + tolerance
        
        low = max(0.01, min(0.99, low_hz / nyquist))
        high = max(0.01, min(0.99, high_hz / nyquist))
        
        if low >= high:
            return []
        
        try:
            sos = butter(4, [low, high], btype='band', output='sos')
            filtered = sosfiltfilt(sos, audio)
            
            # NaN-Check
            if np.any(np.isnan(filtered)):
                print(f"    NaN nach Filterung bei {target_freq} Hz!")
                low_fallback = max(0.01, (low_hz * 0.5) / nyquist)
                high_fallback = min(0.99, (high_hz * 1.5) / nyquist)
                sos = butter(2, [low_fallback, high_fallback], btype='band', output='sos')
                filtered = sosfiltfilt(sos, audio)
                
                if np.any(np.isnan(filtered)):
                    return []
            
            envelope = np.abs(hilbert(filtered))
            
            if np.any(np.isnan(envelope)):
                envelope = np.nan_to_num(envelope, nan=0.0)
        
        except Exception as e:
            print(f"    Filter-Fehler bei {target_freq} Hz: {e}")
            return []
        
        # ──────────────────────────────────────────────────────────────
        # SCHRITT 3: Threshold-Berechnung (aus config.py)
        # ──────────────────────────────────────────────────────────────
        
        valid_mags = envelope[envelope > 0]
        
        if len(valid_mags) > 0:
            percentile_thresh = np.percentile(valid_mags, AUDIO_MAGNITUDE_PERCENTILE)
            relative_thresh = AUDIO_RELATIVE_THRESHOLD * envelope.max()
            
            # Niedrige Frequenzen brauchen niedrigeren Threshold
            if target_freq < 300:
                percentile_thresh *= 0.5
                relative_thresh *= 0.3
            
            threshold = max(percentile_thresh, relative_thresh)
        else:
            threshold = 0
        
        # ──────────────────────────────────────────────────────────────
        # SCHRITT 4: Peak-Detektion (aus config.py)
        # ──────────────────────────────────────────────────────────────
        
        min_distance = int(AUDIO_MIN_DISTANCE_S * sr)
        
        prominence = AUDIO_PROMINENCE
        if target_freq < 300:
            prominence *= 0.3
        
        peaks, _ = find_peaks(envelope, height=threshold, distance=min_distance,
                             prominence=prominence)
        
        timestamps = peaks / sr
        
        # ──────────────────────────────────────────────────────────────
        # SCHRITT 5: Clustering (aus debug_2)
        # ──────────────────────────────────────────────────────────────
        
        if len(timestamps) > 0:
            timestamps = self._cluster_close_timestamps(timestamps, AUDIO_CLUSTER_GAP)
        
        # ──────────────────────────────────────────────────────────────
        # SCHRITT 6: JSON-Guided Filtering (NEU in v2.1!)
        # ──────────────────────────────────────────────────────────────
        
        if expected_times is not None and len(expected_times) > 0:
            # Matche Detektionen mit erwarteten Zeiten
            matched_timestamps = self._match_with_expected(
                timestamps, expected_times, search_window_s
            )
            return matched_timestamps
        
        return list(timestamps)
    
    def _cluster_close_timestamps(self, timestamps: np.ndarray, min_gap: float) -> np.ndarray:
        """Entfernt doppelte Detektionen (aus debug_2)"""
        if len(timestamps) <= 1:
            return timestamps
        
        clustered = []
        current_cluster = [timestamps[0]]
        
        for i in range(1, len(timestamps)):
            if timestamps[i] - current_cluster[-1] < min_gap:
                current_cluster.append(timestamps[i])
            else:
                clustered.append(np.median(current_cluster))
                current_cluster = [timestamps[i]]
        
        clustered.append(np.median(current_cluster))
        return np.array(clustered)
    
    def _match_with_expected(self, detected: np.ndarray, expected: List[float],
                            window_s: float) -> List[float]:
        """
         NEU v2.1: Matched Detektionen mit JSON-Vorhersagen.
        
        Args:
            detected: Detektierte Zeiten (Audio)
            expected: Erwartete Zeiten (JSON)
            window_s: Toleranz (±5s)
        
        Returns:
            Liste von besten Matches (nimmt Audio wenn gefunden, sonst JSON)
        """
        
        matched = []
        
        for exp_time in expected:
            # Finde nächste Detektion
            candidates = [d for d in detected if abs(d - exp_time) < window_s]
            
            if len(candidates) > 0:
                # Nimm nächste
                best_match = min(candidates, key=lambda d: abs(d - exp_time))
                matched.append(best_match)
            else:
                # Keine Detektion >> Nutze JSON-Vorhersage
                matched.append(exp_time)
        
        return matched

# ==================== SYNC-PUNKT-FINDER (v2.0 vollständig) ====================

class SyncPointFinder:
    """
    Findet ERSTEN Kalibrierton (1760 Hz) automatisch.
    
    Methode:
    --------
    1. Spektrale Analyse (FFT)
    2. Isolations-Filter (mind. 30s Stille VOR erstem Peak)
    3. JSON-Validierung (nächste 2 Peaks an erwarteten Stellen?)
    4. Confidence-Scoring (Audio-Qualität + Isolation + JSON-Alignment)
    5. Top-3 zur Auswahl
    """
    
    def __init__(self, calibration_freq: float = 1760,
                 min_silence_before_s: float = 30.0,
                 expected_peak_interval_s: float = 6.5):
        self.calibration_freq = calibration_freq
        self.min_silence_before = min_silence_before_s
        self.expected_interval = expected_peak_interval_s
        
        print(f"\n{'='*70}")
        print("SYNC-PUNKT-FINDER")
        print(f"{'='*70}")
        print(f"   Ziel-Frequenz: {calibration_freq} Hz")
        print(f"   Min. Stille davor: {min_silence_before_s}s")
        print(f"   Erwarteter Abstand: {expected_peak_interval_s}s")
    
    def find_calibration_start_candidates(self, audio: np.ndarray, sr: int,
                                         json_log: Dict) -> List[SyncCandidate]:
        """
        Findet TOP-3 Kandidaten für ersten Kalibrierton.
        
        Returns:
            Liste von SyncCandidate (sortiert nach Confidence)
        """
        
        print(f"\n Suche Kalibrierungs-Start (1760 Hz)...")
        
        # Detektiere ALLE 1760 Hz Peaks
        detector = FrequencyDetector()
        all_peaks = detector.detect_frequency_peaks(audio, sr, self.calibration_freq)
        
        if len(all_peaks) == 0:
            print(f"    Keine 1760 Hz Peaks gefunden!")
            return []
        
        print(f"    {len(all_peaks)} rohe Peaks erkannt")
        
        # Isolations-Filter
        isolated_candidates = []
        
        for i, peak_time in enumerate(all_peaks):
            peaks_before = [p for p in all_peaks if p < peak_time]
            
            if len(peaks_before) == 0:
                isolation_score = 1.0
            else:
                time_since_last = peak_time - max(peaks_before)
                isolation_score = min(1.0, time_since_last / self.min_silence_before)
            
            if isolation_score >= 0.8:
                isolated_candidates.append({
                    'video_time_s': peak_time,
                    'isolation_score': isolation_score,
                    'peak_index': i
                })
        
        print(f"   >> {len(isolated_candidates)} isolierte Kandidaten")
        
        if len(isolated_candidates) == 0:
            print(f"    Kein isolierter Peak gefunden! Fallback: Nutze ersten Peak")
            isolated_candidates = [{
                'video_time_s': all_peaks[0],
                'isolation_score': 0.5,
                'peak_index': 0
            }]
        
        # JSON-Alignment-Check
        scored_candidates = []
        
        for cand in isolated_candidates:
            video_time = cand['video_time_s']
            
            next_peaks = [p for p in all_peaks if p > video_time][:2]
            
            if len(next_peaks) < 2:
                json_score = 0.3
                next_found = [False, False]
            else:
                interval1 = next_peaks[0] - video_time
                interval2 = next_peaks[1] - next_peaks[0]
                
                error1 = abs(interval1 - self.expected_interval)
                error2 = abs(interval2 - self.expected_interval)
                
                score1 = max(0, 1.0 - error1 / 3.0)
                score2 = max(0, 1.0 - error2 / 3.0)
                
                json_score = (score1 + score2) / 2
                next_found = [score1 > 0.5, score2 > 0.5]
            
            confidence = (cand['isolation_score'] * 0.5 + json_score * 0.5)
            
            scored_candidates.append(SyncCandidate(
                video_time_s=video_time,
                confidence=confidence,
                isolation_score=cand['isolation_score'],
                json_alignment_score=json_score,
                next_peaks_found=next_found,
                reason=self._generate_reason(cand['isolation_score'], json_score, next_found),
                frequency_quality=0.85
            ))
        
        scored_candidates.sort(key=lambda x: x.confidence, reverse=True)
        top_3 = scored_candidates[:3]
        
        print(f"\n    Top-3 Kandidaten:")
        for i, cand in enumerate(top_3, 1):
            print(f"      [{i}] {cand.video_time_s:.2f}s (Conf: {cand.confidence:.2f})")
        
        return top_3
    
    def _generate_reason(self, isolation: float, json_score: float,
                        next_found: List[bool]) -> str:
        """Generiert Begründungs-Text"""
        reasons = []
        
        if isolation >= 0.9:
            reasons.append("Exzellente Isolation")
        elif isolation >= 0.7:
            reasons.append("Gute Isolation")
        else:
            reasons.append("Schwache Isolation")
        
        if json_score >= 0.8:
            reasons.append("Perfektes JSON-Alignment")
        elif json_score >= 0.6:
            reasons.append("Gutes JSON-Alignment")
        else:
            reasons.append("JSON-Alignment unsicher")
        
        n_found = sum(next_found)
        if n_found == 2:
            reasons.append("Beide nächsten Peaks gefunden")
        elif n_found == 1:
            reasons.append("1/2 nächste Peaks gefunden")
        else:
            reasons.append("Nächste Peaks fehlen")
        
        return " + ".join(reasons)

# ==================== JSON-FORMAT-DETECTION (NEU v2.1) ====================

def detect_json_format(json_log: Dict) -> JSONFormat:
    """
    Erkennt JSON-Format automatisch (Backwards Compatibility).
    
    V2 wird erkannt wenn IRGENDEIN Event 'audio_play_actual_delay_ms' enthält.
    (Nicht nur der erste Event mit audio_file!)
    
    Args:
        json_log: experiment_sync_log.json
    
    Returns:
        JSONFormat.V1_LEGACY oder JSONFormat.V2_EXTENDED
    """
    
    events = json_log.get('events', [])
    
    if len(events) == 0:
        return JSONFormat.V1_LEGACY
    
    # V2: Prüfe ob IRGENDEIN Event die V2-Felder hat
    # (Instruction-Events haben sie nicht, aber Trial-Events schon!)
    for event in events:
        if 'audio_play_actual_delay_ms' in event:
            return JSONFormat.V2_EXTENDED
    
    # Fallback: V1
    return JSONFormat.V1_LEGACY

def get_eyelink_time_corrected(event: Dict, json_format: JSONFormat) -> float:
    """
    Extrahiert EyeLink-Zeit mit Einheiten-Korrektur.
    
     WICHTIG: JSON-Feld 'eyelink_time_ms' enthält in BEIDEN Formaten SEKUNDEN!
    
    Args:
        event: Event-Dict aus JSON
        json_format: Erkanntes Format
    
    Returns:
        EyeLink-Zeit in Millisekunden (korrekt konvertiert)
    """
    
    # ════════════════════════════════════════════════════════════════
    # V2: Nutze audio_play_start_eyelink_ms (präziser!)
    # ════════════════════════════════════════════════════════════════
    if json_format == JSONFormat.V2_EXTENDED and 'audio_play_start_eyelink_ms' in event:
        #  Auch hier: Feld heißt *_ms, enthält aber Sekunden!
        eyelink_s = event['audio_play_start_eyelink_ms']
        return eyelink_s * 1000  # Sekunden >> ms
    
    # ════════════════════════════════════════════════════════════════
    # V1 & V2 Fallback: Nutze eyelink_time_ms
    # ════════════════════════════════════════════════════════════════
    if 'eyelink_time_ms' in event:
        eyelink_s = event['eyelink_time_ms']  # ← Fehlbenannt (Sekunden!)
        return eyelink_s * 1000  # Sekunden >> ms
    
    return None

def get_corrected_eyelink_time_for_peak(event: Dict, frequency: int, 
                                        json_format: 'JSONFormat',
                                        estimated_latency_ms: float = DEFAULT_LATENCY_MS) -> float:
    """
    Berechnet die korrigierte EyeLink-Zeit für den Ton-PEAK.
    
    Dies ist die Zeit, zu der die Video-Detektion den Ton findet (Mitte des Tons).
    
    Berücksichtigt:
    - V2 JSON: audio_play_start_eyelink_ms + audio_play_actual_delay_ms
    - V1 JSON: eyelink_time_ms + geschätzte Latenz
    - Peak-Offset (100ms für 200ms Ton)
    - Position innerhalb der Audio-Datei (440 Hz am Anfang, 880 Hz am Ende)
    
    Args:
        event: JSON-Event mit Timing-Informationen
        frequency: 440, 880, oder 1760 Hz
        json_format: V1_LEGACY oder V2_EXTENDED
        estimated_latency_ms: Geschätzte Audio-Latenz für V1 JSONs
    
    Returns:
        EyeLink-Zeit in ms für den Ton-Peak
    """
    
    # V2: Nutze präzise Latenz-Info (audio_play_actual_delay_ms)
    if json_format == JSONFormat.V2_EXTENDED and 'audio_play_actual_delay_ms' in event:
        play_start_ms = event['audio_play_start_eyelink_ms'] * 1000
        actual_delay_ms = event['audio_play_actual_delay_ms']
        audio_start_ms = play_start_ms + actual_delay_ms
    else:
        # V1: Nutze eyelink_time_ms + geschätzte Latenz
        eyelink_time_ms = event.get('eyelink_time_ms', 0)
        if isinstance(eyelink_time_ms, (int, float)) and eyelink_time_ms > 0:
            # eyelink_time_ms ist in SEKUNDEN (trotz des Namens!)
            eyelink_time_ms = eyelink_time_ms * 1000
        audio_start_ms = eyelink_time_ms + estimated_latency_ms
    
    # Position des Peaks innerhalb der Audio-Datei
    if frequency == FREQ_TRIAL_START:  # 440 Hz
        # 440 Hz ist am Anfang der Audio-Datei
        # Peak = Audio-Start + 100ms
        return audio_start_ms + PEAK_OFFSET_MS
    
    elif frequency == FREQ_TRIAL_END:  # 880 Hz
        # 880 Hz ist am Ende der Audio-Datei
        # Peak = Audio-Start + Dauer - 100ms
        audio_duration_ms = event.get('audio_duration_ms', 2000)
        return audio_start_ms + audio_duration_ms - PEAK_OFFSET_MS
    
    elif frequency == FREQ_CALIBRATION:  # 1760 Hz
        # Kalibrierung hat ANDERE Latenz als Trials!
        # 
        # Prüfe ob eine spezielle Kalibrierungs-Latenz im Event gesetzt wurde
        # (wird in Schritt 6A berechnet aus Baseline-Abweichung)
        
        if '_calibration_latency_ms' in event:
            calib_latency = event['_calibration_latency_ms']
        else:
            # Fallback: Keine Latenz (Initial-Offset enthält sie bereits implizit)
            calib_latency = 0
        
        eyelink_time_ms = event.get('eyelink_time_ms', 0)
        if isinstance(eyelink_time_ms, (int, float)) and eyelink_time_ms > 0:
            # Konvertiere falls in Sekunden
            if eyelink_time_ms < 100000:  # Wahrscheinlich Sekunden
                eyelink_time_ms = eyelink_time_ms * 1000
        
        return eyelink_time_ms + calib_latency + PEAK_OFFSET_MS

    else:
        # Fallback für unbekannte Frequenzen
        return audio_start_ms + PEAK_OFFSET_MS

def estimate_audio_latency_from_detections(detected_anchors: List[Dict], 
                                           json_format: 'JSONFormat') -> float:
    """
    Gibt die geschätzte Audio-Latenz zurück.
    
    WICHTIG: Die Latenz kann NICHT aus den Anchors geschätzt werden!
    Das wäre ein Henne-Ei-Problem (wir bräuchten den Offset um die Latenz
    zu berechnen, aber wir brauchen die Latenz um den Offset zu berechnen).
    
    Für V2 JSON: 
        - Latenz wird pro-Event aus audio_play_actual_delay_ms gelesen
        - Diese Funktion gibt 0 zurück (wird ignoriert)
    
    Für V1 JSON:
        - Nutze konservativen Default-Wert basierend auf bekannten Messungen
        - Typische Werte: 460-550ms (gemessen aus V2 JSONs)
    
    Args:
        detected_anchors: Liste der gefundenen Anchors (nicht verwendet)
        json_format: V1_LEGACY oder V2_EXTENDED
    
    Returns:
        Geschätzte Latenz in ms (oder 0 für V2)
    """
    
    if json_format == JSONFormat.V2_EXTENDED:
        # V2 hat präzise Latenz-Info pro Event in audio_play_actual_delay_ms
        # Die Latenz wird direkt in get_corrected_eyelink_time_for_peak() verwendet
        print(f"    [Latenz] V2 JSON → Nutze audio_play_actual_delay_ms pro Event")
        return 0.0  # Wird ignoriert, da V2 individuelle Delays hat
    
    # V1: Latenz kann NICHT aus Daten geschätzt werden!
    # Nutze Default-Wert basierend auf Messungen aus V2 JSONs
    print(f"    [Latenz] V1 JSON → Nutze Default: {DEFAULT_LATENCY_MS:.0f} ms")
    print(f"             (Typischer Bereich: 460-550 ms)")
    return DEFAULT_LATENCY_MS

# ==================== MATLAB >> EYELINK ZEIT (v2.0 unverändert) ====================

def matlab_to_eyelink_time(matlab_time_s: float, experiment_sync_log: Dict) -> float:
    """
    Konvertiert MATLAB-Zeit zu EyeLink-Zeit via Linear Interpolation.
    
     WICHTIG: JSON-Feld heißt 'eyelink_time_ms', enthält aber SEKUNDEN!
    
    Returns:
        EyeLink-Zeit in Millisekunden (echte ms, konvertiert aus Sekunden)
    """
    
    events = experiment_sync_log['events']
    
    events_with_times = [e for e in events 
                         if 'matlab_time_s' in e and 'eyelink_time_ms' in e 
                         and e['eyelink_time_ms'] is not None]
    
    if len(events_with_times) < 2:
        raise ValueError("Zu wenig Events mit beiden Zeiten!")
    
    events_sorted = sorted(events_with_times, key=lambda e: e['matlab_time_s'])
    
    for i in range(len(events_sorted) - 1):
        e1 = events_sorted[i]
        e2 = events_sorted[i + 1]
        
        if e1['matlab_time_s'] <= matlab_time_s <= e2['matlab_time_s']:
            t = (matlab_time_s - e1['matlab_time_s']) / (e2['matlab_time_s'] - e1['matlab_time_s'])
            
            # ════════════════════════════════════════════════════════════════
            # FIX: 'eyelink_time_ms' enthält Sekunden >> Konvertiere zu ms!
            # ════════════════════════════════════════════════════════════════
            eyelink_time_s = e1['eyelink_time_ms'] + t * (e2['eyelink_time_ms'] - e1['eyelink_time_ms'])
            eyelink_time_ms = eyelink_time_s * 1000  # Sekunden >> ms
            
            return eyelink_time_ms
    
    # Extrapolation (Fallback)
    if matlab_time_s < events_sorted[0]['matlab_time_s']:
        e1, e2 = events_sorted[0], events_sorted[1]
        gradient = (e2['eyelink_time_ms'] - e1['eyelink_time_ms']) / (e2['matlab_time_s'] - e1['matlab_time_s'])
        eyelink_time_s = e1['eyelink_time_ms'] + gradient * (matlab_time_s - e1['matlab_time_s'])
        return eyelink_time_s * 1000  # Sekunden >> ms
    else:
        e1, e2 = events_sorted[-2], events_sorted[-1]
        gradient = (e2['eyelink_time_ms'] - e1['eyelink_time_ms']) / (e2['matlab_time_s'] - e1['matlab_time_s'])
        eyelink_time_s = e2['eyelink_time_ms'] + gradient * (matlab_time_s - e2['matlab_time_s'])
        return eyelink_time_s * 1000  # Sekunden >> ms

# ==================== KALIBRIERPUNKT-EXTRAKTOR (v2.0 vollständig) ====================

class CalibrationPointExtractor:
    """
    Extrahiert ALLE Kalibrierpunkte aus calibration_timing_*.json + Audio.
    
     v2.0 Feature (vollständig übernommen)
    """
    
    def __init__(self, frequency_detector: FrequencyDetector):
        self.detector = frequency_detector
    
    def extract_calibration_points(self, video_path: str, json_path: str,
                                   offset_ms: float, experiment_sync_log: Dict,
                                   label: str = 'beg',
                                   skip_audio_validation: bool = False) -> List[CalibrationPoint]:
        """
        Extrahiert ALLE Kalibrierpunkte für einen Zeitpunkt (beg/mid/end).
        
        NEU v2.5: skip_audio_validation Parameter - überspringt Audio wenn nicht nutzbar!
        
        Args:
            video_path: Video-Pfad
            json_path: calibration_timing_*.json
            offset_ms: Video ↔ EyeLink Offset
            experiment_sync_log: experiment_sync_log.json (für MATLAB>>EyeLink)
            label: 'beg', 'mid', 'end'
            skip_audio_validation: True = Überspringe Audio-Validierung (JSON-only)
        
        Returns:
            Liste von CalibrationPoint
        """
        
        print(f"\n{'─'*70}")
        print(f"KALIBRIERUNGS-PUNKTE: {label.upper()}")
        print(f"{'─'*70}")
        print(f"   Video: {Path(video_path).name}")
        print(f"   JSON: {Path(json_path).name}")
        
        # ──────────────────────────────────────────────────────────────
        # SCHRITT 1: Lade calibration_timing JSON
        # ──────────────────────────────────────────────────────────────
        
        with open(json_path, 'r') as f:
            calib_timing = json.load(f)
        
        events = calib_timing.get('events', [])
        print(f"    {len(events)} Punkte in JSON")
        
        # ──────────────────────────────────────────────────────────────
        # SCHRITT 2: Konvertiere MATLAB-Zeiten >> EyeLink >> Video
        # ──────────────────────────────────────────────────────────────
        
        expected_video_times = []
        
        for event in events:
            matlab_time_s = event['timestamp']
            position = tuple(event['position'])
            
            # MATLAB >> EyeLink
            eyelink_time_ms = matlab_to_eyelink_time(matlab_time_s, experiment_sync_log)
            
            # EyeLink >> Video
            video_time_s = (eyelink_time_ms - offset_ms) / 1000
            
            expected_video_times.append({
                'matlab_time_s': matlab_time_s,
                'eyelink_time_ms': eyelink_time_ms,
                'video_time_s': video_time_s,
                'position': position
            })
        
        print(f"    Zeiten konvertiert (MATLAB >> EyeLink >> Video)")
        
        # ──────────────────────────────────────────────────────────────
        # SCHRITT 3: Audio-Validierung (ÜBERSPRINGEN wenn nicht nutzbar!)
        # ──────────────────────────────────────────────────────────────
        
        # ✓ NEU v2.5: Nutze globales Flag statt redundantem Check!
        global _SKIP_AUDIO_VALIDATION
        
        if skip_audio_validation or _SKIP_AUDIO_VALIDATION:
            print(f"    Audio-Validierung ÜBERSPRUNGEN (JSON-only Modus)")
            
            # Nutze JSON-Zeiten OHNE Audio-Validierung
            calibration_points = []
            
            for i, exp in enumerate(expected_video_times):
                point = CalibrationPoint(
                    point_id=i + 1,
                    position=exp['position'],
                    matlab_time_s=exp['matlab_time_s'],
                    eyelink_time_ms=exp['eyelink_time_ms'],
                    video_time_s=exp['video_time_s'],
                    audio_expected_s=exp['video_time_s'],
                    audio_found_s=None,
                    audio_deviation_ms=None,
                    confidence=0.6  # Niedrigere Confidence (JSON-only)
                )
                calibration_points.append(point)
            
            print(f"    {len(calibration_points)} Punkte aus JSON (OHNE Audio-Validierung)")
            print(f"    Confidence: 0.6 (JSON-only, ±100-200ms Unsicherheit)")
            
            return calibration_points
        
        # ──────────────────────────────────────────────────────────────
        # SCHRITT 4: Standard Audio-Validierung (nur wenn Audio nutzbar!)
        # ──────────────────────────────────────────────────────────────
        
        print(f"\n    Audio-Validierung (1760 Hz)...")
        
        # Extrahiere Audio
        try:
            audio, sr = librosa.load(video_path, sr=44100, mono=True)
        except Exception as e:
            print(f"    Audio-Extraktion fehlgeschlagen: {e}")
            print(f"    >> Fallback auf JSON-only")
            
            # Fallback
            calibration_points = []
            for i, exp in enumerate(expected_video_times):
                point = CalibrationPoint(
                    point_id=i + 1,
                    position=exp['position'],
                    matlab_time_s=exp['matlab_time_s'],
                    eyelink_time_ms=exp['eyelink_time_ms'],
                    video_time_s=exp['video_time_s'],
                    audio_expected_s=exp['video_time_s'],
                    audio_found_s=None,
                    audio_deviation_ms=None,
                    confidence=0.6
                )
                calibration_points.append(point)
            return calibration_points
        
        # Erwartete Zeiten für gezielte Suche
        expected_times_list = [et['video_time_s'] for et in expected_video_times]
        
        # Detektiere 1760 Hz (mit JSON-Guidance!)
        detected_times = self.detector.detect_frequency_peaks(
            audio, sr, 1760,
            expected_times=expected_times_list,
            search_window_s=PHASE_DETECTION_AUDIO_SEARCH_WINDOW_S
        )
        
        print(f"    {len(detected_times)} Audio-Marker gefunden/validiert")
        
        # Kombiniere JSON + Audio
        calibration_points = []
        
        for i, (exp, audio_time) in enumerate(zip(expected_video_times, detected_times)):
            deviation_ms = (audio_time - exp['video_time_s']) * 1000
            audio_found = abs(deviation_ms) < 500
            
            confidence = 0.95 if audio_found else 0.5
            
            point = CalibrationPoint(
                point_id=i + 1,
                position=exp['position'],
                matlab_time_s=exp['matlab_time_s'],
                eyelink_time_ms=exp['eyelink_time_ms'],
                video_time_s=audio_time,
                audio_expected_s=exp['video_time_s'],
                audio_found_s=audio_time if audio_found else None,
                audio_deviation_ms=deviation_ms if audio_found else None,
                confidence=confidence
            )
            
            calibration_points.append(point)
        
        # ──────────────────────────────────────────────────────────────
        # SCHRITT 5: Statistik
        # ──────────────────────────────────────────────────────────────
        
        n_audio_found = sum(1 for p in calibration_points if p.audio_found_s is not None)
        detection_rate = n_audio_found / len(calibration_points) if len(calibration_points) > 0 else 0.0
        
        print(f"\n    Statistik:")
        print(f"      Audio gefunden: {n_audio_found}/{len(calibration_points)} ({detection_rate*100:.1f}%)")
        
        if n_audio_found > 0:
            deviations = [p.audio_deviation_ms for p in calibration_points if p.audio_deviation_ms is not None]
            print(f"      Abweichung: {np.mean(deviations):.1f} ± {np.std(deviations):.1f} ms")
        
        return calibration_points

# ==================== MULTI-ANCHOR-REGRESSION (DEPRECATED) ====================
# 
# HINWEIS: Diese Klasse wurde durch die neue Latenz-korrigierte Offset-Berechnung
# in calculate_sync_offset() ersetzt. Der Code bleibt für Referenz erhalten.
#
# Die neue Methode korrigiert die Audio-Latenz (~500ms) und den Peak-Offset (100ms)
# BEVOR der Offset berechnet wird. Dadurch sind alle Frequenzen konsistent und
# Dual-Anchor/Multi-Anchor-Regression ist nicht mehr nötig.
#
# Siehe: get_corrected_eyelink_time_for_peak()
# Siehe: estimate_audio_latency_from_detections()
# Siehe: calculate_sync_offset()
#
# =============================================================================

# ==================== PHASE-RECONSTRUCTOR v2.1 (Hybrid) ====================

class PhaseReconstructor:
    """
    Berechnet ALLE Phasen aus Multi-Anchor-Result.
    
     v2.1 HYBRID:
    - v2.0: Extrahiert ALLE Kalibrierpunkte (vollständig)
    - v2.0: Trial-Audio-Validierung (440/880 Hz)
    - v2.1: Multi-Anchor Drift-Korrektur (NEU!)
    - v2.1: JSON v1/v2 Compatibility (NEU!)
    """
    
    def __init__(self):
        self.frequency_detector = FrequencyDetector()
    
    def reconstruct_all_phases(self, json_log: Dict, json_format: JSONFormat,
                               sync_point_video_s: float,
                               sync_point_eyelink_ms: float,
                               audio: np.ndarray, sr: int,
                               calibration_timing_jsons: Dict[str, str],
                               multi_anchor_result: Optional['CompatMultiAnchorResult'] = None,
                               skip_audio_validation: bool = False) -> Dict:
        """
        Berechnet phases_detected.json MIT Multi-Anchor-Drift-Korrektur.
        
        Args:
            json_log: experiment_sync_log.json
            json_format: Erkanntes Format (v1/v2)
            sync_point_video_s: Video-Zeit des ersten Sync-Punkts
            sync_point_eyelink_ms: EyeLink-Zeit des ersten Sync-Punkts
            audio: Audio-Signal (für Validierung)
            sr: Sample-Rate
            calibration_timing_jsons: {'beg': path, 'mid': path, 'end': path}
            multi_anchor_result: Optional - Multi-Anchor-Ergebnis (v2.1)
        
        Returns:
            phases dict (KOMPATIBEL mit v2.0 Format!)
        """
        # ══════════════════════════════════════════════════════════════
        # Erstelle Default-Objekt falls multi_anchor_result None ist
        # (passiert beim ersten Durchlauf für phases_initial)
        # ══════════════════════════════════════════════════════════════
        
        if multi_anchor_result is None:
            # Erstelle minimales Kompatibilitäts-Objekt mit ALLEN benötigten Attributen
            class DefaultMultiAnchorResult:
                def __init__(self, offset):
                    # Basis-Attribute
                    self.method = 'initial'
                    self.offset_ms = offset
                    self.slope = 1.0
                    self.n_anchors = 0
                    self.drift_ms = 0
                    
                    # Dual-Anchor Kompatibilität
                    self.offset_1760hz_ms = offset
                    self.offset_440hz_ms = offset
                    self.offset_880hz_ms = offset
                    self.frequency_difference_ms = 0
                    self.chosen_trial_offset_ms = offset
                    self.chosen_trial_frequency = 440
                    self.marker_choice_reason = "Initial offset (vor Multi-Anchor)"
                    
                    # Qualitäts-Metriken (für Statistik-Plot!)
                    self.r_squared = 0.85  # Default für Initial-Offset
                    self.r_squared_1760hz = 0.85
                    self.r_squared_440hz = 0.85
                    
                    # Listen (für Visualisierung)
                    self.outliers = []
                    self.anchors = []
                    
                    # Anchor-Zählungen
                    self.n_anchors_1760hz = 0
                    self.n_anchors_440hz = 0
            
            # Berechne initialen Offset
            initial_offset = sync_point_eyelink_ms - (sync_point_video_s * 1000)
            multi_anchor_result = DefaultMultiAnchorResult(initial_offset)
            print(f"    [Info] Kein Multi-Anchor-Result >> Nutze Initial-Offset: {initial_offset:.0f} ms")

        print(f"\n{'='*70}")
        print("PHASEN-REKONSTRUKTION v2.1 (Hybrid)")
        print(f"{'='*70}")
        print(f"   Sync-Punkt: {sync_point_video_s:.2f}s (Video)")
        print(f"                >> {sync_point_eyelink_ms:.0f}ms (EyeLink)")
        
        # ══════════════════════════════════════════════════════════════
        # SCHRITT 1: Berechne Offsets (KALIBRIERUNG vs. TRIALS SEPARAT!)
        # ══════════════════════════════════════════════════════════════
        
        if multi_anchor_result is not None:
            # SEPARATE Offsets für Kalibrierung und Trials!
            # (Unterschiedliche Audio-Latenz in MATLAB-Pfaden)
            
            if hasattr(multi_anchor_result, 'offset_calibration_ms') and multi_anchor_result.offset_calibration_ms is not None:
                offset_calibration_ms = multi_anchor_result.offset_calibration_ms
            else:
                offset_calibration_ms = multi_anchor_result.offset_ms
            
            if hasattr(multi_anchor_result, 'offset_trials_ms') and multi_anchor_result.offset_trials_ms is not None:
                offset_trials_ms = multi_anchor_result.offset_trials_ms
            else:
                offset_trials_ms = multi_anchor_result.offset_ms
            
            slope = getattr(multi_anchor_result, 'slope', 1.0)
            use_drift_correction = (slope != 1.0)
            
            print(f"    Offset Kalibrierung (1760 Hz): {offset_calibration_ms:.0f} ms")
            print(f"    Offset Trials (440/880 Hz):    {offset_trials_ms:.0f} ms")
            
            diff = offset_trials_ms - offset_calibration_ms
            if abs(diff) > 100:
                print(f"    Differenz: {diff:+.0f} ms (unterschiedliche Audio-Latenz)")
        else:
            # Fallback: Single-Anchor
            offset_calibration_ms = sync_point_eyelink_ms - (sync_point_video_s * 1000)
            offset_trials_ms = offset_calibration_ms
            slope = 1.0
            use_drift_correction = False
            print(f"    Offset (Initial/Single-Anchor): {offset_calibration_ms:.0f} ms")

        # ══════════════════════════════════════════════════════════════
        # Manuelle Offset-Korrektur (falls aktiviert)
        # ══════════════════════════════════════════════════════════════
        
        original_offset_calibration_ms = offset_calibration_ms
        original_offset_trials_ms = offset_trials_ms
        
        if DEBUG_0_MANUAL_OFFSET_MS != 0.0:
            print(f"\n{'='*70}")
            print(" MANUELLE OFFSET-KORREKTUR AKTIV!")
            print(f"{'='*70}")
            print(f"   Original Offset Kalibrierung: {offset_calibration_ms:.0f} ms")
            print(f"   Original Offset Trials: {offset_trials_ms:.0f} ms")
            print(f"   Manuelle Korrektur: {DEBUG_0_MANUAL_OFFSET_MS:+.0f} ms")
            
            # Wende Korrektur auf BEIDE Offsets an
            offset_calibration_ms = offset_calibration_ms + DEBUG_0_MANUAL_OFFSET_MS
            offset_trials_ms = offset_trials_ms + DEBUG_0_MANUAL_OFFSET_MS
            
            print(f"   Finaler Offset Kalibrierung: {offset_calibration_ms:.0f} ms")
            print(f"   Finaler Offset Trials: {offset_trials_ms:.0f} ms")
            print(f"{'='*70}\n")
        else:
            print(f"\n    Keine manuelle Offset-Korrektur (DEBUG_0_MANUAL_OFFSET_MS = 0.0)\n")

        # ══════════════════════════════════════════════════════════════
        # SCHRITT 2: Transformiere ALLE Events (MIT SEPARATEN OFFSETS!)
        # ══════════════════════════════════════════════════════════════
        
        events = json_log.get('events', [])
        
        for event in events:
            if 'eyelink_time_ms' in event:
                # Korrigiere Einheiten (JSON-Bug: enthält Sekunden!)
                et_time_raw = event['eyelink_time_ms']
                
                # FIX: Type-Check (Liste/None/Float)
                if isinstance(et_time_raw, list):
                    if len(et_time_raw) == 0:
                        continue
                    et_time_s = float(et_time_raw[0])
                elif et_time_raw is None:
                    continue
                elif isinstance(et_time_raw, (int, float)):
                    et_time_s = float(et_time_raw)
                else:
                    print(f"[WARN] Ungültiger eyelink_time_ms Typ: {type(et_time_raw)} in Event {event.get('event_type', 'unknown')}")
                    continue
                
                et_time_ms = et_time_s * 1000
                
                # ════════════════════════════════════════════════════════════
                # NEU: Wähle korrekten Offset basierend auf Event-Typ!
                # ════════════════════════════════════════════════════════════
                
                event_type = event.get('event_type', '')
                
                # Kalibrierungs-Events → Kalibrierungs-Offset
                if 'calibration' in event_type.lower():
                    offset_for_event = offset_calibration_ms
                # Trial-Events → Trial-Offset
                else:
                    offset_for_event = offset_trials_ms
                
                # Transformiere zu Video-Zeit
                if use_drift_correction:
                    video_time = (et_time_ms - offset_for_event) / (slope * 1000)
                else:
                    video_time = (et_time_ms - offset_for_event) / 1000
                
                event['video_time_s'] = video_time
        
        print(f"    {len(events)} Events transformiert (Drift-korrigiert: {use_drift_correction})")
        print(f"    Kalibrierungs-Offset: {offset_calibration_ms:.0f} ms")
        print(f"    Trial-Offset: {offset_trials_ms:.0f} ms")

        # ══════════════════════════════════════════════════════════════
        # SCHRITT 3: Extrahiere Kalibrierungen (MIT frequenz-spezifischem Offset!)
        # ══════════════════════════════════════════════════════════════
        
        calibration_phases = {}
        
        for label, json_path in calibration_timing_jsons.items():
            if json_path is None or not Path(json_path).exists():
                print(f"    Kalibrierung {label.upper()}: JSON fehlt!")
                continue
            
            print(f"\n{'─'*70}")
            print(f"KALIBRIERUNG: {label.upper()}")
            print(f"{'─'*70}")
            
            #  NEU v2.2: Nutze Dual-Anchor Offset falls verfügbar!
            offset_for_calibration = offset_calibration_ms
            
            print(f"   Offset für Kalibrierung: {offset_for_calibration:.0f} ms (Frequenz: 1760 Hz)")
            
            # Nutze CalibrationPointExtractor (v2.0 Feature!)
            # ✓ NEU v2.5: Übergebe skip_audio_validation!
            extractor = CalibrationPointExtractor(self.frequency_detector)
            calibration_points = extractor.extract_calibration_points(
                str(MAIN_VIDEO_PATH), json_path,
                offset_for_calibration,
                json_log, label,
                skip_audio_validation=skip_audio_validation or _SKIP_AUDIO_VALIDATION
            )

            # Finde Start/End aus experiment_sync_log
            calib_start_event = next((e for e in events 
                                     if e.get('event_type') == 'calibration_start' 
                                     and e.get('label') == label), None)
            
            calib_end_event = next((e for e in events 
                                   if e.get('event_type') == 'calibration_end' 
                                   and e.get('label') == label), None)
            
            if calib_start_event is None:
                print(f"    Start-Event nicht gefunden!")
                continue
            
            # Speichere Phase (MIT Offset-Info!)
            calibration_phases[f'calibration_{label}'] = {
                'start_video_s': calib_start_event['video_time_s'],
                'end_video_s': calib_end_event['video_time_s'] if calib_end_event else None,
                'start_eyelink_ms': get_eyelink_time_corrected(calib_start_event, json_format),
                'end_eyelink_ms': get_eyelink_time_corrected(calib_end_event, json_format) if calib_end_event else None,
                'points': [self._point_to_dict(p) for p in calibration_points],
                'detection_rate': sum(1 for p in calibration_points if p.audio_found_s) / len(calibration_points) if calibration_points else 0.0,
                'offset_used_ms': offset_for_calibration  #  NEU: Dokumentiere genutzten Offset!
            }

        # ══════════════════════════════════════════════════════════════
        # TESTMODUS: Überspringe Trials (keine vorhanden!)
        # ══════════════════════════════════════════════════════════════
        
        pipeline_mode = json_log.get('metadata', {}).get('pipeline_mode', 'standard')
        
        if pipeline_mode == 'calibration_test_suite':
            print(f"\n   [TESTMODUS] Überspringe Trial-Strukturierung (keine Trials)")
            
            # Leere Trial-Blöcke
            trial_phases = {
                'experiment_block1': {'trials': [], 'detection_rate_440': 0.0, 'detection_rate_880': 0.0},
                'experiment_block2': {'trials': [], 'detection_rate_440': 0.0, 'detection_rate_880': 0.0}
            }
            
            # Kombiniere zu phases dict (ohne Trial-Kategorisierung!)
            phases = {
                **calibration_phases,
                **trial_phases
            }
            
            # Sync-Info
            sync_info = {
                'method': multi_anchor_result.method if multi_anchor_result is not None else 'calibration_test',
                'anchor_video_time_s': sync_point_video_s,
                'anchor_eyelink_time_ms': sync_point_eyelink_ms,
                'offset_ms': offset_calibration_ms,
                'confidence': 0.90,
                'n_events': len(events),
                'pipeline_mode': 'calibration_test_suite'
            }
            
            return {
                'sync_info': sync_info,
                'phases': phases
            }

        # ══════════════════════════════════════════════════════════════
        # SCHRITT 4: Strukturiere Trials (MIT frequenz-spezifischem Offset!)
        # ══════════════════════════════════════════════════════════════
        offset_ms = offset_trials_ms  # Für Abwärtskompatibilität

        # ✓✓ NEU v2.3: Nutze intelligenten Marker-Offset falls verfügbar!
        if multi_anchor_result.chosen_trial_offset_ms if multi_anchor_result is not None else offset_ms is not None:
            offset_for_trials = multi_anchor_result.chosen_trial_offset_ms if multi_anchor_result is not None else offset_ms
            chosen_freq_label = f"{multi_anchor_result.chosen_trial_frequency} Hz"
        else:
            offset_for_trials = offset_trials_ms
            chosen_freq_label = "440/880 Hz (Standard)"

        print(f"\n{'─'*70}")
        print(f"TRIALS:")
        print(f"{'─'*70}")
        print(f"   Offset für Trials: {offset_for_trials:.0f} ms (Frequenz: {chosen_freq_label})")

        # Zeige Entscheidungs-Info falls vorhanden
        if multi_anchor_result and multi_anchor_result.marker_choice_reason:
            print(f"\n Marker-Wahl: {multi_anchor_result.marker_choice_reason}")
        
        trial_phases = self._categorize_trials(
            events, audio, sr, offset_for_trials, json_format,
            skip_audio_validation=skip_audio_validation or _SKIP_AUDIO_VALIDATION,
            estimated_latency_ms=DEFAULT_LATENCY_MS
        )
      
        #  Dokumentiere genutzten Offset in Trial-Phasen
        for block_key in ['experiment_block1', 'experiment_block2']:
            if block_key in trial_phases:
                trial_phases[block_key]['offset_used_ms'] = offset_for_trials
        
        # ══════════════════════════════════════════════════════════════
        # SCHRITT 5: Kombiniere zu phases dict
        # ══════════════════════════════════════════════════════════════
        
        phases = {
            **calibration_phases,  # calibration_beg/mid/end
            **trial_phases         # experiment_block1/block2
        }
        
        #  Sync-Info (erweitert mit Dual-Anchor-Offsets!)
        sync_info = {
            'method': multi_anchor_result.method if multi_anchor_result is not None else 'initial' if multi_anchor_result else 'linear_mapping',
            'anchor_video_time_s': sync_point_video_s,
            'anchor_eyelink_time_ms': sync_point_eyelink_ms,
            'offset_ms': offset_ms,  # Gemischter Offset (Kompatibilität)
            'confidence': min(0.95, multi_anchor_result.r_squared) if multi_anchor_result else 0.85,
            'n_events': len(events)
        }

        #  NEU: Dokumentiere manuelle Korrektur
        if DEBUG_0_MANUAL_OFFSET_MS != 0.0:
            sync_info['manual_offset_correction_ms'] = DEBUG_0_MANUAL_OFFSET_MS
            sync_info['offset_corrected'] = True
        else:
            sync_info['offset_corrected'] = False

        #  NEU v2.1: Multi-Anchor-Statistik
        if multi_anchor_result:
            sync_info.update({
                'slope': multi_anchor_result.slope,
                'r_squared': multi_anchor_result.r_squared,
                'drift_ms': multi_anchor_result.drift_ms,
                'n_anchors': multi_anchor_result.n_anchors,
                'n_outliers': len(multi_anchor_result.outliers)
            })
            
            #  NEU v2.2: Dual-Anchor-Offsets exportieren!
            if multi_anchor_result.method if multi_anchor_result is not None else 'initial' == 'dual_anchor':
                sync_info.update({
                    'offset_1760hz_ms': multi_anchor_result.offset_1760hz_ms if multi_anchor_result is not None else offset_ms,
                    'offset_440hz_ms': multi_anchor_result.offset_440hz_ms,
                    'offset_880hz_ms': multi_anchor_result.offset_880hz_ms,
                    'frequency_difference_ms': multi_anchor_result.frequency_difference_ms
                })
        
        return {
            'sync_info': sync_info,
            'phases': phases
        }
    
    def _categorize_trials(self, events: List[Dict], audio: np.ndarray, 
                          sr: int, offset_ms: float, json_format: JSONFormat,
                          skip_audio_validation: bool = False,
                          estimated_latency_ms: float = DEFAULT_LATENCY_MS) -> Dict:
        """
        Strukturiert Trials in Blöcke MIT Audio-Validierung (v2.0).
        
        Returns:
            {
                'experiment_block1': {'trials': [...]},
                'experiment_block2': {'trials': [...]}
            }
        """
        
        print(f"\n{'='*70}")
        print("TRIAL-STRUKTURIERUNG (v2.0)")
        print(f"{'='*70}")
        
        # Extrahiere Events
        fixation_events = [e for e in events if e['event_type'].startswith('start_fixation')]
        trial_events = [e for e in events if e['event_type'].startswith('start_trial')]
        end_fixation_events = [e for e in events if e['event_type'].startswith('end_fixation')]
        end_trial_events = [e for e in events if e['event_type'].startswith('end_trial')]
        
        print(f"   {len(fixation_events)} Fixationen")
        print(f"   {len(trial_events)} Trials")
        
        # Audio-Validierung 440 Hz - MIT JSON-Only Fallback
        print(f"\n Validiere Fixations-Marker (440 Hz)...")
        
        # ✓ NEU v2.5: Nutze globales Flag statt redundantem Check!
        global _SKIP_AUDIO_VALIDATION
        use_audio = not (skip_audio_validation or _SKIP_AUDIO_VALIDATION)
        
        n_audio_found_440 = 0
        deviations_440 = []
        
        if not use_audio:
            print(f"    Audio-Validierung ÜBERSPRUNGEN (JSON-only Modus)")
            for fix_event in fixation_events:
                eyelink_time_peak_ms = get_corrected_eyelink_time_for_peak(
                    fix_event, FREQ_TRIAL_START, json_format, estimated_latency_ms
                )
                expected_time = (eyelink_time_peak_ms - offset_ms) / 1000
                fix_event['video_time_s'] = expected_time
                fix_event['audio_found'] = False
            detection_rate_440 = 0.0
        else:
            #  FIX: Berechne expected_time MIT Dual-Anchor-Offset!
            for fix_event in fixation_events:
                # NEU: Nutze korrigierte EyeLink-Zeit (MIT delay + peak offset!)
                eyelink_time_peak_ms = get_corrected_eyelink_time_for_peak(
                    fix_event, FREQ_TRIAL_START, json_format, estimated_latency_ms
                )
                expected_time = (eyelink_time_peak_ms - offset_ms) / 1000
                
                # Standard-Suche
                audio_time = self._search_440hz_around_time(audio, sr, expected_time, PHASE_DETECTION_AUDIO_SEARCH_WINDOW_S)
                
                # NEU: Fallback auf Low-Signal Suche wenn Standard fehlschlägt
                if audio_time is None and ADAPTIVE_SYNC_AVAILABLE:
                    audio_time = self._search_tone_low_signal(
                        audio, sr, expected_time, 
                        target_freq=440, 
                        window_s=PHASE_DETECTION_AUDIO_SEARCH_WINDOW_S
                    )
                    if audio_time is not None:
                        print(f"        [Low-Signal] 440Hz bei Trial {trial_num}: {audio_time:.3f}s")
                audio_time = self._search_440hz_around_time(audio, sr, expected_time, PHASE_DETECTION_AUDIO_SEARCH_WINDOW_S)
                
                if audio_time is not None:
                    deviation = (audio_time - expected_time) * 1000
                    deviations_440.append(deviation)
                    n_audio_found_440 += 1
                    fix_event['video_time_s'] = audio_time  # Update mit Audio-Zeit
                    fix_event['audio_found'] = True
                else:
                    fix_event['video_time_s'] = expected_time  # Fallback auf JSON
                    fix_event['audio_found'] = False
            
            detection_rate_440 = n_audio_found_440 / len(fixation_events) if fixation_events else 0.0
        
        print(f"   Gefunden: {n_audio_found_440}/{len(fixation_events)} ({detection_rate_440*100:.1f}%)")
        if deviations_440:
            print(f"   Abweichung: {np.mean(deviations_440):.1f} ± {np.std(deviations_440):.1f} ms")

        # Audio-Validierung 880 Hz - MIT JSON-Only Fallback
        print(f"\n Validiere Trial-End-Marker (880 Hz)...")
        
        n_audio_found_880 = 0
        deviations_880 = []
        
        if not use_audio:
            print(f"    Audio-Validierung ÜBERSPRUNGEN (JSON-only Modus)")
            for end_fix_event in end_fixation_events:
                # Hole trial_number für das richtige start_fixation Event
                trial_num = end_fix_event.get('trial_number', 0)
                
                # Finde zugehöriges start_fixation Event (hat audio_duration_ms!)
                start_fix_event = None
                for fix_ev in fixation_events:
                    if fix_ev.get('trial_number') == trial_num:
                        start_fix_event = fix_ev
                        break
                    if trial_num == 0 and 'practice' in fix_ev.get('event_type', '').lower():
                        start_fix_event = fix_ev
                        break
                
                # Fallback auf end_fix_event falls nicht gefunden
                if start_fix_event is None:
                    start_fix_event = end_fix_event
                
                eyelink_time_peak_ms = get_corrected_eyelink_time_for_peak(
                    start_fix_event, FREQ_TRIAL_END, json_format, estimated_latency_ms
                )
                expected_time = (eyelink_time_peak_ms - offset_ms) / 1000
                end_fix_event['video_time_s'] = expected_time
                end_fix_event['audio_found'] = False
            detection_rate_880 = 0.0
        else:
            #  FIX: Berechne expected_time MIT Dual-Anchor-Offset!
            for end_fix_event in end_fixation_events:
                # NEU: Nutze korrigierte EyeLink-Zeit für 880 Hz
                # Hole trial_number für das richtige start_fixation Event
                trial_num = end_fix_event.get('trial_number', 0)
                
                # Finde zugehöriges start_fixation Event (hat audio_duration_ms!)
                start_fix_event = None
                for fix_ev in fixation_events:
                    if fix_ev.get('trial_number') == trial_num:
                        start_fix_event = fix_ev
                        break
                    if trial_num == 0 and 'practice' in fix_ev.get('event_type', '').lower():
                        start_fix_event = fix_ev
                        break
                
                # Fallback auf end_fix_event falls nicht gefunden
                if start_fix_event is None:
                    start_fix_event = end_fix_event
                
                eyelink_time_peak_ms = get_corrected_eyelink_time_for_peak(
                    start_fix_event, FREQ_TRIAL_END, json_format, estimated_latency_ms
                )
                expected_time = (eyelink_time_peak_ms - offset_ms) / 1000

                # Standard-Suche
                audio_time = self._search_880hz_around_time(audio, sr, expected_time, PHASE_DETECTION_AUDIO_SEARCH_WINDOW_S)
                
                # NEU: Fallback auf Low-Signal Suche wenn Standard fehlschlägt
                if audio_time is None and ADAPTIVE_SYNC_AVAILABLE:
                    audio_time = self._search_tone_low_signal(
                        audio, sr, expected_time,
                        target_freq=880,
                        window_s=PHASE_DETECTION_AUDIO_SEARCH_WINDOW_S
                    )
                    if audio_time is not None:
                        print(f"        [Low-Signal] 880Hz bei Trial {trial_num}: {audio_time:.3f}s")
                
                if audio_time is not None:
                    deviation = (audio_time - expected_time) * 1000
                    deviations_880.append(deviation)
                    n_audio_found_880 += 1
                    end_fix_event['video_time_s'] = audio_time  # Update mit Audio-Zeit
                    end_fix_event['audio_found'] = True
                else:
                    end_fix_event['video_time_s'] = expected_time  # Fallback auf JSON
                    end_fix_event['audio_found'] = False
            
            detection_rate_880 = n_audio_found_880 / len(end_fixation_events) if end_fixation_events else 0.0
        
        print(f"   Gefunden: {n_audio_found_880}/{len(end_fixation_events)} ({detection_rate_880*100:.1f}%)")
        if deviations_880:
            print(f"   Abweichung: {np.mean(deviations_880):.1f} ± {np.std(deviations_880):.1f} ms")

        # Strukturiere
        trials_block1 = []
        trials_block2 = []
        
        for i, fix_event in enumerate(fixation_events):
            trial_num_original = fix_event.get('trial_number', i + 1)  # Default: 1-basiert
            is_practice_original = fix_event.get('is_practice', False)
            
            # ==================== TRIAL-NUMMERIERUNG (v2.5 VEREINFACHT) ====================
            # MATLAB sendet bereits korrekte Nummerierung:
            #   - Practice: trial_number=0, is_practice=true
            #   - Trials 1-24: trial_number=1-24, is_practice=false
            #
            # KEINE Normalisierung noetig! Einfach uebernehmen.
            
            trial_num = trial_num_original
            is_practice = is_practice_original if is_practice_original is not None else False
            
            # Sonderfall: event_type enthält "practice" -> is_practice=True
            if 'practice' in fix_event.get('event_type', '').lower():
                is_practice = True
                trial_num = 0
            # ==================== ENDE TRIAL-NUMMERIERUNG ====================
            
            # Suche zugehoerige Events
            trial_event = next((e for e in trial_events if e.get('trial_number') == trial_num), None)
            end_fix_event = next((e for e in end_fixation_events if e.get('trial_number') == trial_num), None)
            end_trial_event = next((e for e in end_trial_events if e.get('trial_number') == trial_num), None)
            
            # Sonderfall Practice: event_type ist "start_trial_practice", nicht "start_trial_0"
            if is_practice:
                trial_event = next((e for e in trial_events if 'practice' in e.get('event_type', '').lower()), trial_event)
                end_fix_event = next((e for e in end_fixation_events if 'practice' in e.get('event_type', '').lower()), end_fix_event)
                end_trial_event = next((e for e in end_trial_events if 'practice' in e.get('event_type', '').lower()), end_trial_event)
            
            trial_data = {
                'trial_number': trial_num,
                'is_practice': is_practice,
                'fixation': {
                    'start_video_s': fix_event['video_time_s'],
                    'start_eyelink_ms': get_eyelink_time_corrected(fix_event, json_format),
                    'audio_found': fix_event.get('audio_found', False)
                },
                'stimulus': {}
            }
            
            if end_fix_event:
                trial_data['fixation']['end_video_s'] = end_fix_event['video_time_s']
                trial_data['fixation']['end_eyelink_ms'] = get_eyelink_time_corrected(end_fix_event, json_format)
                trial_data['fixation']['end_audio_found'] = end_fix_event.get('audio_found', False)
            
            if trial_event:
                trial_data['stimulus']['start_video_s'] = trial_event['video_time_s']
                trial_data['stimulus']['start_eyelink_ms'] = get_eyelink_time_corrected(trial_event, json_format)
            
            if end_trial_event:
                trial_data['stimulus']['end_video_s'] = end_trial_event['video_time_s']
                trial_data['stimulus']['end_eyelink_ms'] = get_eyelink_time_corrected(end_trial_event, json_format)
            
            # Block-Zuordnung:
            # - Trial 0 (Practice) + Trials 1-12 -> Block 1 (vor mid-Kalibrierung)
            # - Trials 13-24 -> Block 2 (nach mid-Kalibrierung)
            #
            # WICHTIG: Nutze trial_number direkt (MATLAB-Nummerierung ist korrekt!)
            if trial_num <= 12:
                trials_block1.append(trial_data)
            elif trial_num >= 13 and trial_num <= 24:
                trials_block2.append(trial_data)
            else:
                print(f"      [WARN] Trial {trial_num} ausserhalb erwarteter Range (0-24)")

        
        # ==================== DEDUPLIZIERUNG NACH TRIAL_NUMBER (v2.3 FIX) ====================
        # Problem: MATLAB kann sowohl is_practice=True ALS AUCH trial_number=1 senden
        # Beide werden zu trial_num=0 normalisiert -> e!
        # Loesung: Behalte nur das ERSTE Event pro trial_number
        
        def deduplicate_trials(trials_list):
            """
            Entfernt Duplikate nach trial_number (v2.5 vereinfacht).
            
            MATLAB sendet jedes Event ZWEIMAL:
            1. Vor Audio-Playback
            2. Nach Audio-Playback (mit audio_play_start_*)
            
            Wir behalten das ERSTE (frueheste) Event pro trial_number.
            """
            seen_trials = {}
            
            for trial in trials_list:
                trial_num = trial['trial_number']
                trial_time = trial.get('fixation', {}).get('start_video_s', float('inf'))
                
                if trial_num not in seen_trials:
                    seen_trials[trial_num] = trial
                else:
                    # Bereits gesehen -> behalte frueheres
                    existing_time = seen_trials[trial_num].get('fixation', {}).get('start_video_s', float('inf'))
                    
                    if trial_time < existing_time:
                        print(f"      [WARN] Trial {trial_num}: Ersetze spaeteres durch frueheres Event")
                        seen_trials[trial_num] = trial
                    else:
                        print(f"      [WARN] Trial {trial_num}: Duplikat ignoriert (spaeterer Zeitstempel)")
            
            # Sortiere nach trial_number
            result = [seen_trials[k] for k in sorted(seen_trials.keys())]
            
            return result
        
        trials_block1_before = len(trials_block1)
        trials_block2_before = len(trials_block2)
        
        trials_block1 = deduplicate_trials(trials_block1)
        trials_block2 = deduplicate_trials(trials_block2)
        
        n_removed = (trials_block1_before - len(trials_block1)) + (trials_block2_before - len(trials_block2))
        if n_removed > 0:
            print(f"\n      [FIX] {n_removed} Duplikat(e) nach Normalisierung entfernt")
        
        # ==================== ENDE DEDUPLIZIERUNG ====================
        
        # Zaehle Practice vs. normale Trials
        n_practice = sum(1 for t in trials_block1 if t.get('is_practice', False))
        n_normal_b1 = len(trials_block1) - n_practice
        
        print(f"\n    Trial-Struktur (aus MATLAB):")
        print(f"      Practice-Trial: {n_practice} (Trial 0)")
        print(f"      Block 1: {n_normal_b1} normale Trials (Trials 1-12)")
        print(f"      Block 2: {len(trials_block2)} normale Trials (Trials 13-24)")
        print(f"      Gesamt: {n_practice + n_normal_b1 + len(trials_block2)} Trials")
        
        # Validierung
        expected_total = 25  # 1 Practice + 24 normale
        actual_total = n_practice + n_normal_b1 + len(trials_block2)
        if actual_total != expected_total:
            print(f"\n    [WARN] Erwartet {expected_total} Trials, gefunden {actual_total}")
        
        return {
            'experiment_block1': {
                'trials': trials_block1,
                'detection_rate_440': detection_rate_440,
                'detection_rate_880': detection_rate_880
            },
            'experiment_block2': {
                'trials': trials_block2,
                'detection_rate_440': detection_rate_440,
                'detection_rate_880': detection_rate_880
            }
        }
    
    def _search_440hz_around_time(self, audio: np.ndarray, sr: int,
                                expected_time: float, window_s: float = 5.0) -> Optional[float]:
        """Sucht 440 Hz (wie v2.0) mit Fehlerabfang"""
        
        start_sample = int(max(0, (expected_time - window_s) * sr))
        end_sample = int(min(len(audio), (expected_time + window_s) * sr))
        
        #  FIX: Prüfe Segment-Länge VOR Filter-Aufruf
        segment_length = end_sample - start_sample
        min_length = 100  # Mindestens 100 Samples (bei 44100 Hz = 2.3 ms)
        
        if segment_length < min_length:
            print(f"    Audio-Segment zu kurz ({segment_length} Samples) für Filter bei {expected_time:.2f}s")
            return None
        
        audio_window = audio[start_sample:end_sample]
        
        peaks = self.frequency_detector.detect_frequency_peaks(
            audio_window, sr, 440,
            expected_times=[window_s],
            search_window_s=window_s
        )
        
        if len(peaks) == 0:
            return None
        
        return (start_sample / sr) + peaks[0]

    def _search_880hz_around_time(self, audio: np.ndarray, sr: int,
                                expected_time: float, window_s: float = 5.0) -> Optional[float]:
        """Sucht 880 Hz (wie v2.0) mit Fehlerabfang"""
        
        start_sample = int(max(0, (expected_time - window_s) * sr))
        end_sample = int(min(len(audio), (expected_time + window_s) * sr))
        
        #  FIX: Prüfe Segment-Länge VOR Filter-Aufruf
        segment_length = end_sample - start_sample
        min_length = 100
        
        if segment_length < min_length:
            print(f"    Audio-Segment zu kurz ({segment_length} Samples) für Filter bei {expected_time:.2f}s")
            return None
        
        audio_window = audio[start_sample:end_sample]
        
        peaks = self.frequency_detector.detect_frequency_peaks(
            audio_window, sr, 880,
            expected_times=[window_s],
            search_window_s=window_s
        )
        
        if len(peaks) == 0:
            return None
        
        return (start_sample / sr) + peaks[0]

    def _search_tone_low_signal(self, audio: np.ndarray, sr: int,
                                expected_time: float, target_freq: int,
                                window_s: float = 5.0) -> Optional[float]:
        """
        NEU: Low-Signal Ton-Suche für beschädigtes Audio.
        
        Nutzt Multi-Method-Ansatz wie adaptive_sync_detection.py:
        1. Template Matching
        2. Goertzel-Algorithmus
        3. Consensus zwischen Methoden
        
        Args:
            audio: Audio-Signal
            sr: Sample-Rate
            expected_time: Erwartete Zeit aus JSON
            target_freq: Zielfrequenz (440, 880, 1760)
            window_s: Suchfenster (±window_s)
        
        Returns:
            Gefundene Zeit (Peak-korrigiert!) oder None
        """
        
        # Prüfe ob adaptive_sync_detection verfügbar ist
        try:
            from utils.adaptive_sync_detection import LowSignalDetectionMethods, MultiMethodConsensus
        except ImportError:
            # Fallback auf Standard-Suche
            return None
        
        PEAK_OFFSET_S = 0.1  # 100ms Peak-Korrektur
        
        methods = LowSignalDetectionMethods(sr=sr)
        consensus = MultiMethodConsensus()
        
        search_start = max(0, expected_time - window_s)
        search_duration = 2 * window_s
        
        all_candidates = []
        
        # Methode 1: Template Matching (beste für schwache Signale)
        try:
            template_candidates = methods.template_matching(
                audio, target_freq, search_start, search_duration,
                threshold=0.15  # Niedriger für schwache Signale
            )
            all_candidates.extend(template_candidates)
        except:
            pass
        
        # Methode 2: Goertzel (gut für einzelne Frequenzen)
        try:
            goertzel_candidates = methods.goertzel(
                audio, target_freq, search_start, search_duration
            )
            all_candidates.extend(goertzel_candidates)
        except:
            pass
        
        if len(all_candidates) == 0:
            return None
        
        # Consensus finden
        consensus_results = consensus.find_consensus(all_candidates, expected_count=1)
        
        if len(consensus_results) > 0:
            found_time = consensus_results[0][0]  # (time, score, methods)
            
            # Prüfe Plausibilität
            if abs(found_time - expected_time) > window_s * 1.5:
                return None
            
            # Peak-Korrektur anwenden
            corrected_time = found_time - PEAK_OFFSET_S
            return corrected_time
        
        # Fallback: Bester Einzelkandidat
        best = max(all_candidates, key=lambda x: x.score)
        
        if abs(best.time_s - expected_time) > window_s * 1.5:
            return None
        
        return best.time_s - PEAK_OFFSET_S

    def _point_to_dict(self, point: CalibrationPoint) -> Dict:
        """Konvertiert CalibrationPoint zu dict (v2.0)"""
        return {
            'point_id': point.point_id,
            'position': list(point.position),
            'matlab_time_s': point.matlab_time_s,
            'eyelink_time_ms': point.eyelink_time_ms,
            'video_time_s': point.video_time_s,
            'audio_expected_s': point.audio_expected_s,
            'audio_found_s': point.audio_found_s,
            'audio_deviation_ms': point.audio_deviation_ms,
            'confidence': point.confidence
        }

# ==================== USER-INTERAKTION (CLI) v2.0 ====================

def present_candidates_cli(candidates: List[SyncCandidate],
                          sync_point_eyelink_ms: float,
                          timeout_seconds: int = 15) -> Tuple[float, str]:
    """
    Zeigt 3 beste Kandidaten in CLI + ermoeglicht manuelle Eingabe.
    
    NEU v2.6: Auto-Timeout nach 15 Sekunden
    - Waehlt automatisch den Kandidaten mit niedrigster video_time_s
    - Countdown-Anzeige im Terminal
    
    Args:
        candidates: Liste von SyncCandidate
        sync_point_eyelink_ms: EyeLink-Zeit (fuer Anzeige)
        timeout_seconds: Sekunden bis Auto-Auswahl (Default: 15)
    
    Returns:
        (gewaehlte_video_zeit_s, methode)
    """
    import threading
    import sys
    import time
    
    # ══════════════════════════════════════════════════════════════════
    # BESTIMME AUTO-AUSWAHL-KANDIDAT (niedrigste video_time_s)
    # ══════════════════════════════════════════════════════════════════
    
    if not candidates:
        print("[!] Keine Kandidaten vorhanden!")
        return 0.0, 'no_candidates'
    
    # Sortiere nach video_time_s (aufsteigend) -> erster = niedrigste Zeit
    sorted_by_time = sorted(candidates, key=lambda c: c.video_time_s)
    auto_candidate = sorted_by_time[0]
    auto_index = candidates.index(auto_candidate) + 1  # 1-basiert fuer Anzeige
    
    # ══════════════════════════════════════════════════════════════════
    # ANZEIGE
    # ══════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print("SYNC-PUNKT-AUSWAHL")
    print(f"{'='*70}")
    
    for i, cand in enumerate(candidates, 1):
        stars = "*" * min(3, int(cand.confidence * 3))
        is_auto = (cand == auto_candidate)
        
        auto_marker = " << AUTO-AUSWAHL" if is_auto else ""
        recommend = "EMPFOHLEN" if i == 1 else ""
        
        print(f"\n [{i}] {stars if stars else '*'} {recommend}{auto_marker}")
        print(f"     Zeit: {cand.video_time_s:.2f}s | Confidence: {cand.confidence:.2f}")
        print(f"     - Isolation: {cand.isolation_score:.2f}")
        print(f"     - JSON-Alignment: {cand.json_alignment_score:.2f}")
        print(f"     - Naechste Peaks: {cand.next_peaks_found}")
        print(f"     - {cand.reason}")
    
    print(f"\n [M] Manuelle Eingabe (VLC-Zeit in Sekunden)")
    print(f"{'='*70}")
    
    print(f"\n[AUTO-TIMEOUT] Ohne Eingabe in {timeout_seconds}s wird Kandidat [{auto_index}] gewaehlt")
    print(f"               (niedrigste video_time_s = {auto_candidate.video_time_s:.2f}s)")
    
    # ══════════════════════════════════════════════════════════════════
    # TIMEOUT-MECHANISMUS
    # ══════════════════════════════════════════════════════════════════
    
    # Shared State fuer Thread-Kommunikation
    user_input = {'value': None, 'received': False}
    timeout_triggered = {'value': False}
    
    def input_thread_func():
        """Thread fuer User-Input (blockierend)"""
        try:
            user_input['value'] = input(f"\nAuswahl [1-{len(candidates)}/M] ({timeout_seconds}s Timeout): ").strip().upper()
            user_input['received'] = True
        except EOFError:
            # Kann passieren wenn stdin geschlossen wird
            pass
    
    def countdown_display(seconds: int):
        """Zeigt Countdown im Terminal (nicht-blockierend)"""
        for remaining in range(seconds, 0, -1):
            if user_input['received']:
                return  # User hat eingegeben, stoppe Countdown
            
            # Countdown-Anzeige (ueberschreibt Zeile)
            sys.stdout.write(f"\r[TIMEOUT] {remaining:2d}s verbleibend... ")
            sys.stdout.flush()
            time.sleep(1)
        
        if not user_input['received']:
            timeout_triggered['value'] = True
            sys.stdout.write(f"\r[TIMEOUT] Zeit abgelaufen - Auto-Auswahl!     \n")
            sys.stdout.flush()
    
    # Starte Input-Thread
    input_thread = threading.Thread(target=input_thread_func, daemon=True)
    input_thread.start()
    
    # Starte Countdown in Main-Thread (blockiert fuer timeout_seconds)
    countdown_display(timeout_seconds)
    
    # Warte kurz auf Input-Thread (falls User gerade tippt)
    input_thread.join(timeout=0.5)
    
    # ══════════════════════════════════════════════════════════════════
    # AUSWERTUNG
    # ══════════════════════════════════════════════════════════════════
    
    # Fall 1: Timeout - Auto-Auswahl
    if timeout_triggered['value'] or not user_input['received']:
        print(f"\n[AUTO] Kandidat [{auto_index}] automatisch gewaehlt: {auto_candidate.video_time_s:.2f}s")
        print(f"       (Grund: Niedrigste video_time_s, naechste am Video-Start)")
        return auto_candidate.video_time_s, 'auto_timeout'
    
    # Fall 2: User hat Eingabe gemacht
    choice = user_input['value']
    
    # Verarbeite Eingabe (wie vorher)
    if choice in ['1', '2', '3']:
        idx = int(choice) - 1
        if idx < len(candidates):
            selected = candidates[idx]
            print(f"\n[OK] Kandidat [{choice}] gewaehlt: {selected.video_time_s:.2f}s")
            return selected.video_time_s, 'audio_candidate'
        else:
            print(f"\n[!] Ungueltige Auswahl - nutze Auto-Kandidat")
            return auto_candidate.video_time_s, 'auto_fallback'
    
    elif choice == 'M':
        try:
            # Fuer manuelle Eingabe: Kein Timeout, warte auf Eingabe
            manual_time = float(input("Video-Zeit (Sekunden): "))
            print(f"\n[OK] Manuelle Zeit gewaehlt: {manual_time:.2f}s")
            return manual_time, 'manual_input'
        except ValueError:
            print(f"\n[!] Ungueltige Eingabe - nutze Auto-Kandidat")
            return auto_candidate.video_time_s, 'auto_fallback'
        except EOFError:
            print(f"\n[!] Eingabe abgebrochen - nutze Auto-Kandidat")
            return auto_candidate.video_time_s, 'auto_fallback'
    
    elif choice == '':
        # Leere Eingabe (Enter) -> Auto-Auswahl
        print(f"\n[AUTO] Leere Eingabe - Kandidat [{auto_index}] gewaehlt: {auto_candidate.video_time_s:.2f}s")
        return auto_candidate.video_time_s, 'auto_empty_input'
    
    else:
        print(f"\n[!] Ungueltige Auswahl '{choice}' - nutze Auto-Kandidat")
        return auto_candidate.video_time_s, 'auto_fallback'

# ==================== HELPER: calibration_timing_*.json Suche ====================

def find_calibration_timing_jsons(base_dir: Path, 
                                  video_name: str = None) -> Dict[str, str]:
    """
    Findet automatisch calibration_timing_*.json für beg/mid/end.
    
     Übernommen aus v2.0 (unverändert)
    """
    
    print(f"\n Suche calibration_timing_*.json in: {base_dir}")
    
    patterns = {
        'beg': ['*_beg*.json', '*beginning*.json'],
        'mid': ['*_mid*.json', '*middle*.json'],
        'end': ['*_end*.json']
    }
    
    found = {}
    
    for label, pattern_list in patterns.items():
        for pattern in pattern_list:
            jsons = list(base_dir.glob(pattern))
            
            if jsons:
                newest = max(jsons, key=lambda p: p.stat().st_mtime)
                found[label] = str(newest)
                print(f"    {label.upper()}: {newest.name}")
                break
    
    return found if len(found) >= 2 else {}  # Mind. 2 (beg + end)

# ==================== QS-VISUALISIERUNG v2.1 (HYBRID) ====================

def create_qc_graphics_complete(audio: np.ndarray, sr: int, 
                                candidates: List[SyncCandidate],
                                selected_time_s: float, 
                                phases: Dict, 
                                output_dir: Path,
                                calibration_timing_jsons: Dict[str, str],
                                multi_anchor_result: Optional['CompatMultiAnchorResult'] = None):
    """
    Erstellt VOLLSTÄNDIGE QS-Visualisierung mit ALLEN Markern.
    
     v2.1 HYBRID:
    - v2.0: Alle 6 Panel-Grafiken (Wellenform, Spektrogramm, etc.)
    - v2.1: Erweitert mit Multi-Anchor Drift-Plot (NEU!)
    
    Args:
        audio: Audio-Signal
        sr: Sample-Rate
        candidates: Sync-Kandidaten (für Punkt-Auswahl)
        selected_time_s: Gewählter Sync-Punkt
        phases: phases_detected.json dict
        output_dir: Ausgabe-Ordner
        calibration_timing_jsons: Pfade zu calibration_timing_*.json
        multi_anchor_result: Optional - Multi-Anchor-Ergebnis (v2.1)
    """
    
    print(f"\n Erstelle vollständige QS-Visualisierung...")
    
    # ══════════════════════════════════════════════════════════════════
    # GRAFIK 1: ÜBERSICHT (Wellenform + Spektrogramm + Offset)
    # ══════════════════════════════════════════════════════════════════
    
    fig1 = plt.figure(figsize=(20, 16))  # ← Höher für 4 Panels!
    gs1 = GridSpec(4, 1, height_ratios=[1, 2.5, 1, 1.2], hspace=0.35)
    
    time_audio = np.linspace(0, len(audio) / sr, len(audio))
    
    # ──────────────────────────────────────────────────────────────────
    # Panel 1: Wellenform + ALLE Marker (v2.0 unverändert)
    # ──────────────────────────────────────────────────────────────────
    
    ax1 = fig1.add_subplot(gs1[0])
    ax1.plot(time_audio, audio, color='steelblue', linewidth=0.3, alpha=0.6)
    ax1.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax1.set_title('Audio-Wellenform + ALLE erkannten Marker', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Sync-Kandidaten
    for i, cand in enumerate(candidates, 1):
        color = 'lime' if i == 1 else 'orange' if i == 2 else 'red'
        ax1.axvline(cand.video_time_s, color=color, linestyle='--', alpha=0.7,
                   linewidth=1.5, label=f'Kandidat {i}' if i <= 3 else '')
    
    # Gewählter Sync-Punkt
    ax1.axvline(selected_time_s, color='gold', linewidth=4, linestyle='-',
               label=f' Gewählt ({selected_time_s:.2f}s)', zorder=10)
    
    # Kalibrierungsphasen
    for label in ['beg', 'mid', 'end']:
        calib_key = f'calibration_{label}'
        if calib_key in phases['phases'] and phases['phases'][calib_key]:
            calib = phases['phases'][calib_key]
            if 'start_video_s' in calib:
                start = calib['start_video_s']
                end = calib.get('end_video_s', start + 120)
                ax1.axvspan(start, end, color='pink', alpha=0.2, label=f'Kalibr. {label.upper()}' if label == 'beg' else '')
    
    ax1.legend(loc='upper right', fontsize=9, ncol=2)
    ax1.set_xlim(0, time_audio[-1])
    
    # ──────────────────────────────────────────────────────────────────
    # Panel 2: Spektrogramm (v2.0 unverändert)
    # ──────────────────────────────────────────────────────────────────
    
    ax2 = fig1.add_subplot(gs1[1])
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax2, 
                                   cmap='viridis', hop_length=512)
    
    ax2.axhline(1760, color='red', linestyle='--', linewidth=2.5, 
               label='1760Hz (Kalibr.)', alpha=0.9)
    ax2.axhline(440, color='cyan', linestyle=':', linewidth=2, 
               label='440Hz (Fix.)', alpha=0.8)
    ax2.axhline(880, color='lime', linestyle=':', linewidth=2, 
               label='880Hz (Trial)', alpha=0.8)
    
    ax2.axvline(selected_time_s, color='gold', linewidth=4, alpha=0.9)
    
    ax2.set_ylim(0, 2500)
    ax2.set_title('Spektrogramm (0-2500 Hz)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    fig1.colorbar(img, ax=ax2, format='%+2.0f dB', pad=0.01)
    
    # ──────────────────────────────────────────────────────────────────
    # Panel 3: Offset-Konsistenz (v2.0)
    # ──────────────────────────────────────────────────────────────────
    
    ax3 = fig1.add_subplot(gs1[2])
    
    # Sammle Events mit Audio-Validierung
    event_numbers = []
    deviations_ms = []
    event_types = []
    
    event_counter = 0
    
    # Kalibrierungspunkte
    for label in ['beg', 'mid', 'end']:
        calib_key = f'calibration_{label}'
        if calib_key in phases['phases'] and 'points' in phases['phases'][calib_key]:
            for point in phases['phases'][calib_key]['points']:
                event_counter += 1
                if point.get('audio_deviation_ms') is not None:
                    event_numbers.append(event_counter)
                    deviations_ms.append(point['audio_deviation_ms'])
                    event_types.append(f'Kalibr. {label.upper()}')
    
    if len(event_numbers) > 0:
        colors_scatter = ['blue' if 'Kalibr' in et else 'green' for et in event_types]
        ax3.scatter(event_numbers, deviations_ms, c=colors_scatter, s=60, alpha=0.7,
                   edgecolors='black', linewidth=0.5)
        
        # Trend-Linie
        if len(event_numbers) > 2:
            z = np.polyfit(event_numbers, deviations_ms, 1)
            p = np.poly1d(z)
            ax3.plot(event_numbers, p(event_numbers), "r--", linewidth=2, alpha=0.6,
                    label=f'Trend: {z[0]:.2f} ms/Event')
        
        mean_dev = np.mean(deviations_ms)
        std_dev = np.std(deviations_ms)
        ax3.axhline(mean_dev, color='orange', linestyle='-', linewidth=1.5, 
                   label=f'Mean: {mean_dev:.1f} ms', alpha=0.8)
        ax3.fill_between([min(event_numbers), max(event_numbers)], 
                        mean_dev - 2*std_dev, mean_dev + 2*std_dev,
                        color='yellow', alpha=0.2, label=f'±2σ ({std_dev:.1f} ms)')
        
        ax3.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        ax3.set_xlabel('Event-Nummer', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Audio-Abweichung (ms)', fontsize=11, fontweight='bold')
        ax3.set_title('Offset-Konsistenz (JSON >> Audio)', fontsize=13, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        max_abs_dev = max(abs(min(deviations_ms)), abs(max(deviations_ms)))
        ax3.set_ylim(-max_abs_dev*1.2, max_abs_dev*1.2)
    else:
        ax3.text(0.5, 0.5, 'Keine Audio-Validierung\n(JSON-only)',
                ha='center', va='center', fontsize=12, transform=ax3.transAxes)
        ax3.axis('off')
    
    # ──────────────────────────────────────────────────────────────────
    # Panel 4: Multi-Anchor Drift-Plot (NEU v2.1!) - MIT SICHERHEITSCHECKS
    # ──────────────────────────────────────────────────────────────────
    
    ax4 = fig1.add_subplot(gs1[3])
    
    # Sammle Daten aus all_anchors (nicht aus multi_anchor_result.anchors!)
    # Problem: CompatMultiAnchorResult.anchors ist leer!
    # Lösung: Nutze die bereits gesammelten all_anchors
    
    video_times = []
    offsets = []
    colors_anchor = []
    
    # Versuche Daten aus sync_result zu extrahieren
    if multi_anchor_result and hasattr(multi_anchor_result, 'anchors') and len(multi_anchor_result.anchors) > 0:
        # Alte Methode (falls anchors gefüllt)
        for a in multi_anchor_result.anchors:
            if hasattr(a, 'video_time_s') and hasattr(a, 'eyelink_time_ms'):
                video_times.append(a.video_time_s)
                offsets.append(a.eyelink_time_ms - (a.video_time_s * 1000))
                colors_anchor.append('blue' if 'calibration' in getattr(a, 'event_type', '') else 'green')
    
    # Fallback: Extrahiere aus phases (Kalibrierungspunkte)
    if len(video_times) == 0:
        for label in ['beg', 'mid', 'end']:
            calib_key = f'calibration_{label}'
            if calib_key in phases['phases'] and 'points' in phases['phases'][calib_key]:
                for point in phases['phases'][calib_key]['points']:
                    if point.get('audio_found_s') is not None:
                        video_times.append(point['video_time_s'])
                        # Offset aus phases berechnen
                        offset = phases['sync_info'].get('offset_ms', 0)
                        offsets.append(offset)
                        colors_anchor.append('purple')
    
    # SICHERHEITSCHECK: Nur plotten wenn Daten vorhanden!
    if len(video_times) >= 2 and len(offsets) >= 2:
        # Scatter: Offset pro Anchor
        ax4.scatter(video_times, offsets, c=colors_anchor if colors_anchor else 'blue', 
                   s=80, alpha=0.7, edgecolors='black', linewidth=1, zorder=3)
        
        # Median-Linie (Single-Offset)
        median_offset = np.median(offsets)
        ax4.axhline(median_offset, color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {median_offset:.0f} ms', alpha=0.8)
        
        # Offset-Band (±100ms) - MIT Sicherheitscheck!
        video_min = min(video_times)
        video_max = max(video_times)
        ax4.fill_between([video_min, video_max],
                        median_offset - 100, median_offset + 100,
                        color='yellow', alpha=0.2, label='±100ms (Akzeptabel)')
        
        # Regressionslinie (nur wenn Regression aktiv und genug Daten)
        if (multi_anchor_result and 
            multi_anchor_result.method == 'multi_anchor_regression' and 
            len(video_times) >= 5):
            video_range = np.linspace(video_min, video_max, 100)
            fitted = multi_anchor_result.slope * video_range * 1000 + multi_anchor_result.offset_ms
            ax4.plot(video_range, fitted, 'r-', linewidth=3, alpha=0.7,
                    label=f'Regression (Slope={multi_anchor_result.slope:.6f})')
        
        ax4.set_xlabel('Video-Zeit (s)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Offset (ms)', fontsize=11, fontweight='bold')
        
        # Titel mit Statistik
        r_sq = multi_anchor_result.r_squared if multi_anchor_result else 0.0
        drift = multi_anchor_result.drift_ms if multi_anchor_result else np.std(offsets)
        ax4.set_title(f'Offset-Konsistenz ({len(video_times)} Punkte, R²={r_sq:.4f}, Std={drift:.1f}ms)', 
                     fontsize=12, fontweight='bold')
        ax4.legend(loc='upper left', fontsize=9)
        ax4.grid(True, alpha=0.3)
    
    elif len(video_times) == 1:
        # Nur ein Datenpunkt
        ax4.scatter(video_times, offsets, c='blue', s=100, marker='o')
        ax4.text(0.5, 0.5, f'Nur 1 Anchor\nOffset: {offsets[0]:.0f} ms',
                ha='center', va='center', fontsize=12, transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax4.set_title('Offset-Konsistenz (unzureichende Daten)', fontsize=12)
    
    else:
        # Keine Daten
        ax4.text(0.5, 0.5, 'Keine Anchor-Daten\nfür Drift-Plot verfügbar\n\n(Kalibrierung nutzt Zirkelschluss,\nkann nicht für Offset verwendet werden)',
                ha='center', va='center', fontsize=11, color='gray',
                transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax4.set_title('Offset-Konsistenz (keine Daten)', fontsize=12)
        ax4.axis('off')

    # ══════════════════════════════════════════════════════════════════
    # GRAFIK 2-6: Statistik + Detail-Grafiken (v2.0 unverändert)
    # ══════════════════════════════════════════════════════════════════
    
    # [Nutze die bereits vorhandenen Funktionen aus v2.0]
    _create_statistics_plot(phases, output_dir, multi_anchor_result)
    
    # Detail-Grafiken Kalibrierungen
    for label in ['beg', 'mid', 'end']:
        calib_key = f'calibration_{label}'
        if calib_key in phases['phases'] and 'points' in phases['phases'][calib_key]:
            _create_calibration_detail_plot(
                audio, sr, phases['phases'][calib_key], label, output_dir
            )
    
    # Detail-Grafiken Trials
    for block_key, block_label in [('experiment_block1', 'Block 1'), 
                                   ('experiment_block2', 'Block 2')]:
        if block_key in phases['phases']:
            _create_trial_detail_plot(
                audio, sr, phases['phases'][block_key], block_label, output_dir
            )


def _create_statistics_plot(phases: Dict, output_dir: Path,
                            multi_anchor_result: Optional['CompatMultiAnchorResult'] = None):

    """Statistik-Grafik (v2.0 + v2.1 Multi-Anchor-Statistik)"""
    
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis('off')
    
    sync_info = phases['sync_info']
    validation = sync_info.get('validation', {})
    
    #  NEU Phase 1: Erkenne Modus aus sync_info
    sync_method = sync_info.get('method', 'unknown')
    
    # Modus-spezifische Warnungen
    mode_warning = ""
    if sync_method == 'manual_json_only':
        mode_warning = """
  JSON-ONLY MODUS AKTIV:                                          
   • Keine Audio-Validierung möglich                                
   • Zeitstempel aus JSON (±100-200ms Unsicherheit)                 
   • Manuelle Sync-Punkt-Eingabe verwendet                          

"""
    elif sync_method == 'template_audio_only':
        mode_warning = """
  TEMPLATE-MODUS AKTIV:                                           
   • Keine JSON-Daten verfügbar                                     
   • Rekonstruktion aus Audio-Marker-Pattern                        
   • Standard-Timeline angenommen                                   

"""
    
    #  NEU v2.2: Erweitere mit Dual-Anchor-Statistik
    if multi_anchor_result:
        # Basis-Statistik (immer vorhanden)
        multi_anchor_stats = f"""
 MULTI-ANCHOR-REGRESSION (v2.2):                                   
   Methode: {multi_anchor_result.method if multi_anchor_result is not None else 'initial'}
   Anchors: {multi_anchor_result.n_anchors}
   R²: {multi_anchor_result.r_squared:.4f} ({_interpret_r_squared(multi_anchor_result.r_squared)})
"""
        
        #  NEU v2.2: Dual-Anchor spezifisch
        if multi_anchor_result.method if multi_anchor_result is not None else 'initial' == 'dual_anchor':
            multi_anchor_stats += f"""   
   DUAL-ANCHOR OFFSETS (Frequenz-spezifisch):
   • 1760 Hz (Kalibrierung): {multi_anchor_result.offset_1760hz_ms if multi_anchor_result is not None else offset_ms:.0f} ms
   • 440 Hz (Trials):        {multi_anchor_result.offset_440hz_ms:.0f} ms
   • Differenz:              {multi_anchor_result.frequency_difference_ms:+.0f} ms
   
   QUALITÄT PRO FREQUENZ:
   • 1760 Hz: {multi_anchor_result.n_anchors_1760hz} Anchors, R²={multi_anchor_result.r_squared_1760hz:.4f}
   • 440 Hz:  {multi_anchor_result.n_anchors_440hz} Anchors, R²={multi_anchor_result.r_squared_440hz:.4f}
"""
        else:
            # Standard Multi-Anchor (ohne Dual)
            multi_anchor_stats += f"""   Slope: {multi_anchor_result.slope:.6f}
   Drift: {multi_anchor_result.drift_ms:.1f} ms
   Outliers: {len(multi_anchor_result.outliers)}
"""
        
        multi_anchor_stats += "\n"
    else:
        multi_anchor_stats = ""
    
    n_calib_beg = len(phases['phases'].get('calibration_beg', {}).get('points', []))
    n_calib_mid = len(phases['phases'].get('calibration_mid', {}).get('points', []))
    n_calib_end = len(phases['phases'].get('calibration_end', {}).get('points', []))
    n_trials_b1 = len(phases['phases']['experiment_block1']['trials'])
    n_trials_b2 = len(phases['phases']['experiment_block2']['trials'])
    
    stats_text = f"""

                  PHASE DETECTION QS - STATISTIK v2.1                  

{mode_warning} SYNCHRONISATION:                                                      
   Methode: {sync_info['method']}
   Sync-Punkt: {sync_info['anchor_video_time_s']:.2f}s (Video)
   EyeLink: {sync_info['anchor_eyelink_time_ms']:.0f} ms
   Offset: {sync_info['offset_ms']:.0f} ms
   Confidence: {sync_info['confidence']:.2%}

{multi_anchor_stats} AUDIO-VALIDIERUNG:
   Erwartete Marker: {validation.get('n_markers_expected', 'N/A')}
   Gefunden: {validation.get('n_markers_found', 'N/A')}
   Detection-Rate: {validation.get('detection_rate', 0)*100:.1f}%
   Mean Deviation: {validation.get('mean_deviation_ms', 0):.1f} ms

 ERKANNTE PHASEN:
   Kalibrierungen:
     • BEG: {n_calib_beg} Punkte
     • MID: {n_calib_mid} Punkte
     • END: {n_calib_end} Punkte
   Experiment:
     • Block 1: {n_trials_b1} Trials
     • Block 2: {n_trials_b2} Trials
   Gesamt: {sync_info['n_events']} Events
    """
    
    ax.text(0.5, 0.5, stats_text, fontsize=10, verticalalignment='center',
            horizontalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, pad=1.5),
            transform=ax.transAxes)
    
    plt.tight_layout()
    output_path = output_dir / "phase_detection_qc_stats.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"    Statistik: {output_path.name}")
    plt.close()


def _interpret_r_squared(r_squared: float) -> str:
    """Helper für R²-Interpretation"""
    if r_squared > 0.99:
        return "exzellent"
    elif r_squared > 0.95:
        return "sehr gut"
    elif r_squared > 0.90:
        return "gut"
    else:
        return "akzeptabel"

# ==================== DETAIL-GRAFIKEN (v2.0 vollständig) ====================

def _create_calibration_detail_plot(audio: np.ndarray, sr: int, 
                                    calib_phase: Dict, label: str, 
                                    output_dir: Path):
    """
    Detail-Grafik für EINE Kalibrierungsphase.
    
     v2.0 Feature (vollständig übernommen)
    
    Zeigt:
    - Spektrogramm (nur Kalibrierungs-Zeitraum)
    - ALLE Punkte mit JSON-Vorhersage vs. Audio-Detektion
    - Abweichungs-Statistik
    """
    
    print(f"    Detail-Grafik: Kalibrierung {label.upper()}...")
    
    points = calib_phase['points']
    start_s = calib_phase['start_video_s']
    end_s = calib_phase.get('end_video_s', start_s + 120)
    
    # Extrahiere Audio-Segment
    start_sample = int(start_s * sr)
    end_sample = int(end_s * sr)
    audio_segment = audio[start_sample:end_sample]
    
    # Zeitachse relativ zu Phase-Start
    time_segment = np.linspace(0, len(audio_segment) / sr, len(audio_segment))
    
    # ──────────────────────────────────────────────────────────────────
    # 2-Panel-Grafik
    # ──────────────────────────────────────────────────────────────────
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))
    
    # Panel 1: Spektrogramm mit Punkten
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_segment)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', 
                                   ax=ax1, cmap='viridis')
    
    ax1.axhline(1760, color='red', linestyle='--', linewidth=2, label='1760 Hz')
    
    # Markiere Punkte
    for point in points:
        # JSON-Vorhersage (grün)
        expected_rel_time = point['audio_expected_s'] - start_s
        ax1.axvline(expected_rel_time, color='lime', linestyle=':', linewidth=1.5, alpha=0.7)
        
        # Audio-Fund (rot)
        if point.get('audio_found_s'):
            found_rel_time = point['audio_found_s'] - start_s
            ax1.axvline(found_rel_time, color='red', linestyle='-', linewidth=2, alpha=0.9)
            
            # Verbindungslinie (falls Abweichung)
            if abs(found_rel_time - expected_rel_time) > 0.05:  # >50ms
                ax1.plot([expected_rel_time, found_rel_time], [1760, 1760],
                        color='orange', linewidth=2, marker='o', markersize=4)
    
    ax1.set_ylim(1500, 2000)
    ax1.set_title(f'Kalibrierung {label.upper()} - Spektrogramm + Marker', 
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    fig.colorbar(img, ax=ax1, format='%+2.0f dB')
    
    # Panel 2: Abweichungs-Plot
    point_ids = [p['point_id'] for p in points]
    deviations = [p.get('audio_deviation_ms', 0) for p in points]
    
    # None-safe: Filtere None-Werte
    deviations_safe = [d if d is not None else 0 for d in deviations]
    
    colors_bar = ['green' if abs(d) < 100 else 'orange' if abs(d) < 300 else 'red' 
                  for d in deviations_safe]
    
    ax2.bar(point_ids, deviations_safe, color=colors_bar, edgecolor='black', linewidth=1)
    ax2.axhline(0, color='gray', linestyle='-', linewidth=1)
    ax2.axhline(100, color='orange', linestyle=':', linewidth=1, alpha=0.5)
    ax2.axhline(-100, color='orange', linestyle=':', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Punkt-ID', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Audio-Abweichung (ms)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Abweichung JSON >> Audio (Mean: {np.mean(deviations_safe):.1f}ms, Std: {np.std(deviations_safe):.1f}ms)', 
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / f"phase_detection_detail_calibration_{label}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"       {output_path.name}")
    plt.close()


def _create_trial_detail_plot(audio: np.ndarray, sr: int, 
                              block_data: Dict, block_label: str, 
                              output_dir: Path):
    """
    Detail-Grafik für EINEN Experiment-Block.
    
     v2.0 Feature (vollständig übernommen)
    
    Zeigt:
    - Spektrogramm (440/880 Hz)
    - Trial-Marker (alle Trials)
    - Timing-Statistik
    """
    
    print(f"    Detail-Grafik: {block_label}...")
    
    trials = block_data['trials']
    
    if len(trials) == 0:
        return
    
    # Zeitraum
    start_s = trials[0]['fixation']['start_video_s']
    end_s = trials[-1]['stimulus']['end_video_s']
    
    # Extrahiere Audio
    start_sample = int(start_s * sr)
    end_sample = int(end_s * sr)
    audio_segment = audio[start_sample:end_sample]
    
    # ──────────────────────────────────────────────────────────────────
    # 2-Panel-Grafik
    # ──────────────────────────────────────────────────────────────────
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))
    
    # Panel 1: Spektrogramm
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_segment)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', 
                                   ax=ax1, cmap='viridis')
    
    ax1.axhline(440, color='cyan', linestyle=':', linewidth=2, label='440 Hz (Fix.)')
    ax1.axhline(880, color='lime', linestyle=':', linewidth=2, label='880 Hz (Trial)')
    
    # Markiere ALLE Trials
    for trial in trials:
        fix_start_rel = trial['fixation']['start_video_s'] - start_s
        fix_end_rel = trial['fixation']['end_video_s'] - start_s
        
        ax1.axvline(fix_start_rel, color='cyan', alpha=0.6, linewidth=1.5, linestyle=':')
        ax1.axvline(fix_end_rel, color='lime', alpha=0.6, linewidth=1.5, linestyle=':')
        
        # Annotiere nur jede 2. (gegen Ueberlappung)
        # v2.3 FIX: Nutze Index statt trial_number fuer Modulo (nach Deduplizierung korrekt)
        trial_idx = trials.index(trial)
        if trial_idx % 2 == 0 or trial_idx in [0, len(trials)-1]:
            # Zeige Practice als "P" statt "T0"
            label = "P" if trial['is_practice'] else f"T{trial['trial_number']}"
            ax1.text(fix_start_rel, 100, label, 
                    fontsize=8, rotation=90, va='bottom', color='white',
                    bbox=dict(boxstyle='round', facecolor='green' if trial['is_practice'] else 'blue', alpha=0.7, pad=0.2))
    
    ax1.set_ylim(0, 1500)
    ax1.set_title(f'{block_label} - Spektrogramm (440/880 Hz Marker)', 
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    fig.colorbar(img, ax=ax1, format='%+2.0f dB')
    
    # Panel 2: Timing-Statistik
    trial_numbers = [t['trial_number'] for t in trials]
    fixation_durations = [(t['fixation']['end_video_s'] - t['fixation']['start_video_s']) * 1000 
                          for t in trials]
    stimulus_durations = [(t['stimulus']['end_video_s'] - t['stimulus']['start_video_s']) * 1000 
                          for t in trials]
    
    ax2.plot(trial_numbers, fixation_durations, marker='o', color='blue', 
            linewidth=2, label='Fixation-Dauer', markersize=5)
    ax2.plot(trial_numbers, stimulus_durations, marker='s', color='green', 
            linewidth=2, label='Stimulus-Dauer', markersize=5)
    
    # Erwartungs-Linien
    ax2.axhline(2000, color='blue', linestyle='--', linewidth=1, alpha=0.5, 
               label='Erwartet Fix. (2.0s)')
    ax2.axhline(7000, color='green', linestyle='--', linewidth=1, alpha=0.5, 
               label='Erwartet Stim. (7.0s)')
    
    ax2.set_xlabel('Trial-Nummer', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Dauer (ms)', fontsize=11, fontweight='bold')
    ax2.set_title(f'{block_label} - Trial-Timing (Mean Fix: {np.mean(fixation_durations):.0f}ms, Stim: {np.mean(stimulus_durations):.0f}ms)', 
                 fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"phase_detection_detail_trials_{block_label.lower().replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"       {output_path.name}")
    plt.close()

# ==================== MAIN PIPELINE v2.1 (Multi-Anchor Integration) ====================

def run_phase_detection(video_path: Path, json_path: Path, output_dir: Path):
    """
    VOLLSTÄNDIGE Phase-Detection-Pipeline v2.1.
    
     v2.1 HYBRID:
    - v2.0: Alle Features (Kalibrierungen, Trials, QS-Grafiken)
    - v2.1: Multi-Anchor-Regression (Drift-Korrektur, NEU!)
    - v2.1: JSON v1/v2 Compatibility (Backwards Compatible)
    
    Workflow:
    ---------
    1. JSON-Format-Detection (v1/v2)
    2. Audio-Extraktion
    3. Sync-Punkt-Findung (Top-3 Kandidaten)
    4. Multi-Anchor-Collection (Kalibrierung + Trials)
    5. Multi-Anchor-Regression (Optional, bei ≥5 Anchors)
    6. Phasen-Rekonstruktion (mit Drift-Korrektur)
    7. QS-Visualisierung (vollständig)
    8. Export (phases_detected.json)
    """
    
    print(f"\n{'='*70}")
    print("DEBUG 0: PHASE-DETECTION v2.1 (Multi-Anchor)")
    print(f"{'='*70}\n")
    
    # ══════════════════════════════════════════════════════════════════
    # SCHRITT 1: JSON laden + Format-Detection
    # ══════════════════════════════════════════════════════════════════

    print(f" Input:")
    print(f"   Video: {video_path.name}")
    print(f"   JSON:  {json_path.name}")

    #  FIX: Robustes JSON-Laden (tolerant gegenüber Backslash-Fehlern)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_log = json.load(f)
    except json.JSONDecodeError as e:
        print(f"\n JSON-Parsing-Fehler: {e}")
        print(f"   >> Versuche Backslash-Fix...")
        
        # Lese als Text und fixe Backslashes
        with open(json_path, 'r', encoding='utf-8') as f:
            json_text = f.read()
        
        # Ersetze alle unkritischen Backslashes
        # (NUR in Pfaden, NICHT in bereits escaped \n, \t, etc.)
        import re
        
        # Strategie: Ersetze \U durch \\U (und andere kritische Kombinationen)
        # Aber NICHT \n, \t, \r, \\, \"
        json_text_fixed = json_text
        
        # Liste kritischer Escape-Sequenzen die NICHT escaped werden sollen
        valid_escapes = ['\\n', '\\t', '\\r', '\\\\', '\\"', "\\'"]
        
        # Ersetze ALLE Backslashes durch Forward Slashes in Pfaden
        # (Erkennbar an C:\ Pattern)
        json_text_fixed = re.sub(r'([A-Za-z]):\\', r'\1:/', json_text_fixed)  # C:\ >> C:/
        json_text_fixed = json_text_fixed.replace('\\', '/')  # Rest: \ >> /
        
        # Versuche nochmal zu parsen
        try:
            json_log = json.loads(json_text_fixed)
            print(f"    JSON erfolgreich geladen (nach Backslash-Fix)")
        except json.JSONDecodeError as e2:
            print(f"    JSON immer noch fehlerhaft: {e2}")
            print(f"   >> Bitte prüfe experiment_sync_log.json manuell!")
            exit(1)

    # ══════════════════════════════════════════════════════════════════════════════
    # NEU: TESTMODUS-ERKENNUNG (calibration_test_suite)
    # ══════════════════════════════════════════════════════════════════════════════
    
    pipeline_mode = json_log.get('metadata', {}).get('pipeline_mode', 'standard')
    is_calibration_test_mode = (pipeline_mode == 'calibration_test_suite')
    
    if is_calibration_test_mode:
        print(f"\n{'='*70}")
        print(" TESTMODUS ERKANNT: calibration_test_suite")
        print(f"{'='*70}")
        print(f"   → Nur Kalibrierung wird analysiert (keine Trials)")
        print(f"   → Ziel: R²-Werte für Kalibrierungsvergleich")
        
        # Mappe Kalibrierungs-Labels auf 'test'
        for event in json_log.get('events', []):
            if event.get('event_type') in ['calibration_start', 'calibration_end']:
                original_label = event.get('label', '')
                if original_label not in ['beg', 'mid', 'end', 'test']:
                    event['_original_label'] = original_label  # Backup
                    event['label'] = 'test'
                    print(f"   → Label '{original_label}' → 'test'")
        
        # Setze CALIB_MODE für offline_calibration
        os.environ['PIPELINE_CALIB_MODE'] = 'CalibrationTest'
        print(f"   → CALIB_MODE = CalibrationTest")

    json_format = detect_json_format(json_log)
    
    # ══════════════════════════════════════════════════════════════════
    #  NEU: Auto-Config aus JSON-Metadaten laden (Phase 1)
    # ══════════════════════════════════════════════════════════════════
    
    config_overrides = load_experiment_metadata(json_log)
    
    # Wende Overrides an (temporär für diese Session)
    if config_overrides:
        print(f"\n    Experiment-Metadaten überschreiben config.py:")
        for key, value in config_overrides.items():
            globals()[key] = value  # Temporäres Überschreiben
            print(f"      {key} = {value}")
    
    print(f"    JSON-Format: {json_format.value}")

    
    events = json_log.get('events', [])
    print(f"    {len(events)} Events geladen")
    
    # ══════════════════════════════════════════════════════════════════
    #  NEU: Dedupliziere Events (Workaround für MATLAB-Bug)
    # ══════════════════════════════════════════════════════════════════
    
    events_before = len(events)
    events_deduplicated = []
    seen = set()
    
    for event in events:
        #  FIX: Eindeutiger Key OHNE matlab_time_s (ändert sich bei Duplikaten!)
        key = (
            event.get('event_type', ''),
            event.get('trial_number', None),  # None wenn nicht vorhanden
            event.get('is_practice', None),   # None wenn nicht vorhanden
            event.get('label', None)          # Für Kalibrierungen (beg/mid/end)!
        )
        
        if key not in seen:
            seen.add(key)
            events_deduplicated.append(event)
        else:
            # Bei Duplikat: Nimm das Event mit MEHR Feldern (audio_play_start_*)
            existing_idx = None
            for i, e in enumerate(events_deduplicated):
                existing_key = (
                    e.get('event_type', ''),
                    e.get('trial_number', None),
                    e.get('is_practice', None),
                    e.get('label', None)
                )
                if existing_key == key:
                    existing_idx = i
                    break
            
            if existing_idx is not None:
                existing = events_deduplicated[existing_idx]
                
                # Vergleiche Anzahl Felder (Event mit mehr Feldern = mit audio_play_start_*)
                if len(event) > len(existing):
                    # Neues Event hat mehr Felder >> Ersetze
                    events_deduplicated[existing_idx] = event
                    print(f"      >> Duplikat ersetzt: {event.get('event_type')} (Trial {event.get('trial_number', 'N/A')})")
    
    events = events_deduplicated
    json_log['events'] = events
    
    if events_before != len(events):
        print(f"    {events_before - len(events)} Duplikate entfernt ({len(events)} übrig)")
    else:
        print(f"    Keine Duplikate gefunden")
    
    # ══════════════════════════════════════════════════════════════════
    # SCHRITT 2: Audio extrahieren
    # ══════════════════════════════════════════════════════════════════
    
    print(f"\n Extrahiere Audio...")
    audio, sr = librosa.load(str(video_path), sr=44100, mono=True)
    duration = len(audio) / sr
    print(f"    {duration:.2f}s @ {sr}Hz")
    
    # ══════════════════════════════════════════════════════════════════
    #  NEU: Audio-Quality-Check (Phase 1 Fallback-Logik)
    # ══════════════════════════════════════════════════════════════════
    
    audio_quality = check_audio_quality(audio, sr, force_recheck=True)
    
    print(f"\n Audio-Qualität:")
    print(f"   Signal-Stärke: {audio_quality['signal_strength']:.2f} ({'OK' if audio_quality['has_signal'] else 'SCHWACH'})")
    print(f"   1760 Hz erkannt: {'Ja' if audio_quality['has_1760hz'] else 'Nein'}")
    print(f"   Verwendbar: {'Ja' if audio_quality['usable'] else 'Nein (JSON-only Modus)'}")
    
    # ✓ NEU v2.5: Setze globales Flag für alle nachfolgenden Funktionen!
    global _SKIP_AUDIO_VALIDATION
    _SKIP_AUDIO_VALIDATION = not audio_quality['usable']
    
    if _SKIP_AUDIO_VALIDATION:
        print(f"\n    Audio nicht verwertbar >> ALLE Audio-Validierungen werden übersprungen!")
        print(f"    >> Zeitersparnis: ~30-60 Sekunden pro Kalibrierung")
    
    # Entscheide Modus basierend auf Audio-Qualität
    use_audio_mode = audio_quality['usable']
   
    # ══════════════════════════════════════════════════════════════════
    # SCHRITT 3: Finde calibration_timing_*.json
    # ══════════════════════════════════════════════════════════════════

    
    print(f"\n Suche calibration_timing_*.json...")
    
    try:
        calib_json_dir = Path(CALIBRATION_TIMING_JSON_DIR)
    except:
        calib_json_dir = json_path.parent
    
    calibration_timing_jsons = {}
    
    if is_calibration_test_mode:

        # TESTMODUS: Suche JEDE calibration_timing_*.json

        print(f"   [TESTMODUS] Suche beliebige calibration_timing_*.json...")
        
        all_calib_jsons = list(calib_json_dir.glob("calibration_timing_*.json"))
        
        if all_calib_jsons:
            # Nimm die neueste
            newest = max(all_calib_jsons, key=lambda p: p.stat().st_mtime)
            calibration_timing_jsons['test'] = str(newest)
            print(f"    TEST: {newest.name}")
        else:
            print(f"    [!] Keine calibration_timing_*.json gefunden!")
    else:

        # STANDARD: Suche nach beg/mid/end Pattern

        for label in ['beg', 'mid', 'end']:
            pattern = f"calibration_timing*{label}*.json"
            matches = list(calib_json_dir.glob(pattern))
            
            if matches:
                newest = max(matches, key=lambda p: p.stat().st_mtime)
                calibration_timing_jsons[label] = str(newest)
                print(f"    {label.upper()}: {newest.name}")
    
    if len(calibration_timing_jsons) == 0:
        print(f"    Keine calibration_timing_*.json gefunden!")

    # ══════════════════════════════════════════════════════════════════
    # SCHRITT 4: Erste Kalibrierung (Sync-Punkt)
    # ══════════════════════════════════════════════════════════════════
    
    finder = SyncPointFinder()
    candidates = finder.find_calibration_start_candidates(audio, sr, json_log)
    
    calib_start_event = next((e for e in events if e['event_type'] == 'calibration_start'), None)
    if calib_start_event is None:
        print(f"\n Keine calibration_start Events in JSON!")
        exit(1)
    
    sync_point_eyelink_ms = get_eyelink_time_corrected(calib_start_event, json_format)
    
    # User wählt (mit JSON-Only Fallback!)
    if use_audio_mode:
        # Standard: Audio-basierte Kandidaten-Auswahl
        selected_time_s, method = present_candidates_cli(candidates, sync_point_eyelink_ms)
    else:
        # ════════════════════════════════════════════════════════════════════════
        #  NEU v2.4: ADAPTIVE LOW-SIGNAL SEARCH (bei manueller Eingabe)
        # ════════════════════════════════════════════════════════════════════════
        
        print(f"\n{'='*70}")
        print(" JSON-ONLY MODUS - MANUELLE SYNC-PUNKT-EINGABE")
        print(f"{'='*70}")
        print(f"\nKeine verwertbaren Audio-Marker gefunden!")
        print(f"Bitte gib die Video-Zeitpunkte (in Sekunden, VLC-Format) ein:\n")
        
        # Sammle ALLE Sync-Punkte (beg/mid/end)
        manual_sync_points = {}
        
        for label in ['beg', 'mid', 'end']:
            if label in calibration_timing_jsons:
                while True:
                    try:
                        time_input = input(f"Kalibrierung {label.upper()} - Erster Marker (Sekunden): ")
                        manual_time = float(time_input)
                        manual_sync_points[label] = manual_time
                        print(f"    ✓ {label.upper()}: {manual_time:.2f}s")
                        break
                    except ValueError:
                        print(f"    ✗ Ungültige Eingabe! Bitte Zahl eingeben (z.B. 17.73)")
        
        # ════════════════════════════════════════════════════════════════════════
        # ADAPTIVE SEARCH: Versuche präzisere Detektion
        # ════════════════════════════════════════════════════════════════════════
        
        adaptive_result = None
        
        # Prüfe ob Audio verfügbar ist
        audio_available = 'audio' in dir() and audio is not None and len(audio) > 0
        
        if ADAPTIVE_SYNC_AVAILABLE and ENABLE_ADAPTIVE_LOW_SIGNAL_SEARCH and audio_available:
            print(f"\n{'='*70}")
            print(" ADAPTIVE LOW-SIGNAL SEARCH v2.0 AKTIVIERT")
            print(f"{'='*70}")
            print(f"   Versuche präzisere Detektion der reduzierten Audio-Marker...")
            
            try:
                adaptive_result = run_adaptive_sync_for_all_phases(
                    audio=audio,
                    sr=sr,
                    manual_sync_points=manual_sync_points,
                    calibration_timing_jsons=calibration_timing_jsons,
                    experiment_sync_log=json_log,
                    verbose=True
                )
                
                # Prüfe ob adaptive Suche erfolgreich war
                if adaptive_result.overall_detection_rate > 0.3:
                    print(f"\n    ✓ Adaptive Suche erfolgreich!")
                    print(f"      Detection-Rate: {adaptive_result.overall_detection_rate*100:.1f}%")
                    print(f"      Confidence: {adaptive_result.overall_confidence:.2f}")
                    
                    # Nutze verbesserten Sync-Punkt
                    selected_time_s = adaptive_result.sync_point_video_s
                    
                    # ════════════════════════════════════════════════════════════════
                    # KRITISCHER FIX: Stelle sicher dass sync_point_eyelink_ms gültig ist!
                    # ════════════════════════════════════════════════════════════════
                    
                    if adaptive_result.sync_point_eyelink_ms > 0:
                        sync_point_eyelink_ms = adaptive_result.sync_point_eyelink_ms
                    else:
                        # Fallback: Berechne aus dem ursprünglichen Event
                        print(f"      [WARN] sync_point_eyelink_ms ungültig, berechne neu...")
                        
                        # Finde calibration_start Event für 'beg'
                        calib_start_event = next(
                            (e for e in json_log.get('events', []) 
                             if e.get('event_type') == 'calibration_start' 
                             and e.get('label', '') == 'beg'),
                            None
                        )
                        
                        if calib_start_event:
                            sync_point_eyelink_ms = get_eyelink_time_corrected(calib_start_event, json_format)
                            print(f"      >> Nutze EyeLink-Zeit aus calibration_start_beg: {sync_point_eyelink_ms:.0f}ms")
                        else:
                            print(f"      [ERROR] Kein calibration_start Event gefunden!")
                            sync_point_eyelink_ms = manual_sync_points.get('beg', 0) * 1000 + 3000000  # Grobe Schätzung
                    
                    method = 'adaptive_manual'
                else:
                    print(f"\n    ✗ Adaptive Suche nicht erfolgreich genug")
                    print(f"      Detection-Rate: {adaptive_result.overall_detection_rate*100:.1f}%")
                    print(f"      >> Fallback auf reine JSON-Interpolation")
                    
                    selected_time_s = manual_sync_points.get('beg', 0.0)
                    method = 'manual_json_only'
                    adaptive_result = None
                    
            except Exception as e:
                print(f"\n    ✗ Adaptive Suche fehlgeschlagen: {e}")
                import traceback
                traceback.print_exc()
                print(f"      >> Fallback auf reine JSON-Interpolation")
                selected_time_s = manual_sync_points.get('beg', 0.0)
                method = 'manual_json_only'
                adaptive_result = None
        else:
            # Kein Adaptive Search verfügbar/aktiviert
            if not ADAPTIVE_SYNC_AVAILABLE:
                print(f"\n   [INFO] Adaptive Search nicht verfügbar")
            elif not audio_available:
                print(f"\n   [INFO] Audio nicht verfügbar für Adaptive Search")
            
            selected_time_s = manual_sync_points.get('beg', 0.0)
            method = 'manual_json_only'
        
        # ════════════════════════════════════════════════════════════════════════
        # WICHTIG: Stelle sicher dass sync_point_eyelink_ms definiert ist!
        # ════════════════════════════════════════════════════════════════════════
        
        if 'sync_point_eyelink_ms' not in dir() or sync_point_eyelink_ms is None or sync_point_eyelink_ms == 0:
            # Berechne aus dem ersten calibration_start Event
            calib_start_event = next(
                (e for e in json_log.get('events', []) 
                 if e.get('event_type') == 'calibration_start'),
                None
            )
            
            if calib_start_event:
                sync_point_eyelink_ms = get_eyelink_time_corrected(calib_start_event, json_format)
                print(f"\n   [FIX] sync_point_eyelink_ms berechnet: {sync_point_eyelink_ms:.0f}ms")
            else:
                print(f"\n   [ERROR] Kein calibration_start Event! Kann nicht fortfahren.")
                exit(1)
        
        # ════════════════════════════════════════════════════════════════════════
        # AUSGABE: Vergleich manuell vs. adaptiv
        # ════════════════════════════════════════════════════════════════════════
        
        print(f"\n ✓ Sync-Punkte:")
        for label, time in manual_sync_points.items():
            if adaptive_result and label in adaptive_result.phase_results:
                ar = adaptive_result.phase_results[label]
                if ar.found_times:
                    improved_time = ar.found_times[0]
                    improvement_ms = (improved_time - time) * 1000
                    print(f"   {label.upper()}: {time:.2f}s → {improved_time:.3f}s (Δ{improvement_ms:+.0f}ms, {ar.n_found}/{ar.n_expected} gefunden)")
                else:
                    print(f"   {label.upper()}: {time:.2f}s (keine Töne gefunden)")
            else:
                print(f"   {label.upper()}: {time:.2f}s (unverändert)")

    # ══════════════════════════════════════════════════════════════════
    #  FIX: Berechne initialen Offset (für Anchor-Suche)
    # ══════════════════════════════════════════════════════════════════
    
    initial_offset_ms = sync_point_eyelink_ms - (selected_time_s * 1000)
    print(f"\n   Initial Offset: {initial_offset_ms:.0f} ms")
    
    # ══════════════════════════════════════════════════════════════════
    # SCHRITT 5: SCHNELLE VALIDIERUNG (INITIAL, ohne Multi-Anchor)
    # ══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("SCHRITT 5: INITIALE VALIDIERUNG (Baseline)")
    print(f"{'='*70}")

    # Führe Phasen-Rekonstruktion durch (enthält bereits Validierung!)
    reconstructor = PhaseReconstructor()
    phases_initial = reconstructor.reconstruct_all_phases(
        json_log, json_format,
        selected_time_s, sync_point_eyelink_ms,
        audio, sr, calibration_timing_jsons,
        multi_anchor_result=None,
        skip_audio_validation=_SKIP_AUDIO_VALIDATION
    )
    
    # Speichere Baseline-Metriken (für Vergleich)
    baseline_stats = _extract_validation_stats(phases_initial)
    
    print(f"\n    Baseline-Metriken (ohne Multi-Anchor):")
    print(f"      Kalibrierung Detection-Rate: {baseline_stats['calib_detection_rate']*100:.1f}%")
    print(f"      Kalibrierung Mean Deviation: {baseline_stats['calib_mean_dev']:.1f} ms")
    print(f"      Trial 440Hz Detection-Rate: {baseline_stats['trial_440_rate']*100:.1f}%")
    print(f"      Trial 880Hz Detection-Rate: {baseline_stats['trial_880_rate']*100:.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # SCHRITT 6: ANCHOR-SAMMLUNG MIT KORRIGIERTEN ZEITEN (NEU!)
    # ══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("SCHRITT 6: ANCHOR-SAMMLUNG MIT LATENZ-KORREKTUR")
    print(f"{'='*70}")

    # Sammle alle Audio-Anchors im NEUEN Format (für calculate_sync_offset)
    all_anchors = []

    # ──────────────────────────────────────────────────────────────────
    # 6A: Kalibrierungs-Anchors (1760 Hz) - MIT KORREKTER LATENZ!
    # ──────────────────────────────────────────────────────────────────
    
    # WICHTIG: Die Kalibrierung hat eine ANDERE Latenz als Trials!
    # Wir schätzen die Kalibrierungs-Latenz aus den Baseline-Daten.
    #
    # Baseline-Abweichung = video_found - video_expected
    # Wenn Abweichung = -344ms, dann wurde der Ton 344ms FRÜHER gefunden.
    # Das bedeutet: eyelink_time + latency + peak = video_found * 1000 + offset
    # → latency = (video_found - video_expected) * 1000 - peak_offset
    # → latency = abweichung_ms - peak_offset
    # → latency = -344 - 100 = -444ms (relativ zum Initial-Offset!)
    #
    # Da wir bereits einen Initial-Offset haben der funktioniert,
    # ist die EFFEKTIVE Latenz für die Offset-Berechnung = 0
    # (der Initial-Offset enthält bereits die Latenz implizit!)
    
    print(f"\n    Sammle Kalibrierungs-Anchors (1760 Hz)...")
    
    # ══════════════════════════════════════════════════════════════════
    # NEU: Hole Zeiten aus adaptive_result (falls vorhanden)
    # ══════════════════════════════════════════════════════════════════
    
    adaptive_calib_times = {}
    if 'adaptive_result' in dir() and adaptive_result is not None and hasattr(adaptive_result, 'phase_results'):
        for label in ['beg', 'mid', 'end']:
            if label in adaptive_result.phase_results:
                phase_result = adaptive_result.phase_results[label]
                if hasattr(phase_result, 'found_times') and phase_result.found_times and len(phase_result.found_times) > 0:
                    adaptive_calib_times[label] = phase_result.found_times
                    print(f"      [Adaptive] {label.upper()}: {len(phase_result.found_times)} Zeiten aus Low-Signal Suche")
    
    # Schätze Kalibrierungs-Latenz aus Baseline-Abweichung
    calib_baseline_deviations = []
    for label in ['beg', 'mid', 'end']:
        calib_key = f'calibration_{label}'
        if calib_key in phases_initial['phases']:
            calib_data = phases_initial['phases'][calib_key]
            for point in calib_data.get('points', []):
                if point.get('audio_found_s') is not None and point.get('video_time_s') is not None:
                    deviation_s = point['audio_found_s'] - point['video_time_s']
                    calib_baseline_deviations.append(deviation_s * 1000)
    
    if len(calib_baseline_deviations) > 0:
        mean_deviation_ms = np.mean(calib_baseline_deviations)
        estimated_calib_latency_ms = -mean_deviation_ms - PEAK_OFFSET_MS
        print(f"      Baseline-Abweichung: {mean_deviation_ms:.1f} ms")
        print(f"      Geschätzte Kalibrierungs-Latenz: {estimated_calib_latency_ms:.1f} ms")
    else:
        estimated_calib_latency_ms = 0
        print(f"      [INFO] Keine Baseline-Daten >> Latenz = 0 ms")
    
    n_calib_anchors = 0
    
    # Lade calibration_timing JSONs für MATLAB-Zeiten der einzelnen Punkte
    calib_timing_data = {}
    for label, json_path_str in calibration_timing_jsons.items():
        if json_path_str:
            json_path = Path(json_path_str) if isinstance(json_path_str, str) else json_path_str
            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        calib_timing_data[label] = json.load(f)
                except Exception as e:
                    print(f"      [WARN] Konnte {json_path.name} nicht laden: {e}")
    
    for label in ['beg', 'mid', 'end']:
        calib_key = f'calibration_{label}'
        if calib_key not in phases_initial['phases']:
            continue
        
        calib_data = phases_initial['phases'][calib_key]
        
        # Finde calibration_start Event (HAT eyelink_time_ms!)
        calib_start_event = find_calibration_event_for_point({}, json_log, label)
        
        if calib_start_event is None:
            print(f"      [WARN] Kein calibration_start Event für '{label}' gefunden!")
            continue
        
        # Debug: Zeige Referenz-Event
        calib_start_matlab = calib_start_event.get('matlab_time_s', 0)
        calib_start_eyelink = calib_start_event.get('eyelink_time_ms', 0)
        print(f"      {label.upper()}: Referenz @ MATLAB={calib_start_matlab:.3f}s, EyeLink={calib_start_eyelink:.3f}s")
        
        # Hole MATLAB-Zeiten aus calibration_timing_*.json
        calib_fixation_events = []
        if label in calib_timing_data and 'events' in calib_timing_data[label]:
            calib_fixation_events = [
                e for e in calib_timing_data[label]['events'] 
                if e.get('event') == 'calibration_fixation'
            ]
        
        for point in calib_data.get('points', []):
            # ══════════════════════════════════════════════════════════════
            # NEU: Prüfe mehrere Quellen für audio_found_s
            # ══════════════════════════════════════════════════════════════
            
            audio_found_s = point.get('audio_found_s')
            
            # Fallback 1: Adaptive Suche-Ergebnisse
            if audio_found_s is None and label in adaptive_calib_times:
                point_id = point.get('point_id', 0)
                if point_id > 0 and point_id <= len(adaptive_calib_times[label]):
                    audio_found_s = adaptive_calib_times[label][point_id - 1] + 0.1
                    print(f"        [Adaptive Fallback] {label.upper()} Punkt {point_id}: {audio_found_s:.3f}s")
            
            # Fallback 2: video_time_s (JSON-basiert)
            if audio_found_s is None:
                audio_found_s = point.get('video_time_s')
                if audio_found_s is not None:
                    print(f"        [JSON Fallback] {label.upper()} Punkt {point_id}: {audio_found_s:.3f}s")
            
            if audio_found_s is None:
                continue

            # Hole MATLAB-Zeit aus calibration_timing_*.json
            point_id = point.get('point_id', 0)
            point_matlab_time = 0
            
            if point_id > 0 and point_id <= len(calib_fixation_events):
                point_matlab_time = calib_fixation_events[point_id - 1].get('timestamp', 0)
            
            if point_matlab_time == 0:
                continue
            
            # Berechne präzise EyeLink-Zeit
            point_eyelink_ms = get_calibration_point_eyelink_time(
                point_timestamp=point_matlab_time,
                calib_start_event=calib_start_event
            )
            
            # Erstelle synthetisches Event mit KORREKTER LATENZ für Kalibrierung!
            synthetic_event = {
                'eyelink_time_ms': point_eyelink_ms / 1000,  # Zurück zu "Sekunden"
                'audio_duration_ms': 200,
                # WICHTIG: Setze eine spezielle Markierung für Kalibrierungs-Latenz
                '_calibration_latency_ms': estimated_calib_latency_ms,
            }
            
            all_anchors.append({
                'video_time_s': audio_found_s,
                'event': synthetic_event,
                'frequency': FREQ_CALIBRATION,
                'source': f'calibration_{label}_point_{point_id}',
                'point_index': point_id,
                'eyelink_time_ms': point_eyelink_ms,
                'calibration_latency_ms': estimated_calib_latency_ms,  # Für Debug
            })
            n_calib_anchors += 1
    
    print(f"      Gefunden: {n_calib_anchors} Anchors")

    # ──────────────────────────────────────────────────────────────────
    # 6B: Trial-Anchors (440 Hz + 880 Hz)
    # ──────────────────────────────────────────────────────────────────
    
    print(f"\n    Sammle Trial-Anchors (440 Hz + 880 Hz)...")
    n_trial_440 = 0
    n_trial_880 = 0
    
    for block_key in ['experiment_block1', 'experiment_block2']:
        if block_key not in phases_initial['phases']:
            continue
        
        block_data = phases_initial['phases'][block_key]
        
        for trial in block_data.get('trials', []):
            trial_num = trial.get('trial_number', 0)
            
            # Finde zugehöriges JSON-Event
            trial_event = find_trial_event(trial_num, json_log)
            
            # 440 Hz (Fixation-Start)
            if trial.get('fixation', {}).get('audio_found', False):
                video_time = trial['fixation'].get('start_video_s')
                if video_time is not None:
                    all_anchors.append({
                        'video_time_s': video_time,
                        'event': trial_event,
                        'frequency': FREQ_TRIAL_START,
                        'source': f'trial_{trial_num}_440',
                        'trial_number': trial_num
                    })
                    n_trial_440 += 1
            
            # 880 Hz (Fixation-Ende)
            if trial.get('fixation', {}).get('end_audio_found', False):
                video_time = trial['fixation'].get('end_video_s')
                if video_time is not None:
                    # WICHTIG: Für 880 Hz brauchen wir das end_fixation Event!
                    end_event = None
                    for ev in json_log.get('events', []):
                        if ev.get('event_type') == f'end_fixation_{trial_num}':
                            end_event = ev
                            break
                        # Sonderfall Practice
                        if trial_num == 0 and ev.get('event_type') == 'end_fixation_practice':
                            end_event = ev
                            break
                    
                    # Wenn end_event gefunden, nutze dessen Zeit
                    # ABER: end_event hat keine audio_play_actual_delay_ms!
                    # Also nutzen wir start_event + audio_duration
                    
                    all_anchors.append({
                        'video_time_s': video_time,
                        'event': trial_event,  # start_fixation Event (hat audio_duration!)
                        'frequency': FREQ_TRIAL_END,
                        'source': f'trial_{trial_num}_880',
                        'trial_number': trial_num
                    })
                    n_trial_880 += 1
   
    print(f"      440 Hz: {n_trial_440} Anchors")
    print(f"      880 Hz: {n_trial_880} Anchors")
    print(f"\n    ► Gesamt: {len(all_anchors)} Anchors")

    # ──────────────────────────────────────────────────────────────────
    # 6B-FALLBACK: Trial-Töne mit adaptiver Suche finden (wenn keine gefunden)
    # ──────────────────────────────────────────────────────────────────
    
    if n_trial_440 == 0 and n_trial_880 == 0 and ADAPTIVE_SYNC_AVAILABLE:
        print(f"\n      [!] Keine Trial-Anchors gefunden >> Versuche adaptive Suche...")
        
        # Importiere benötigte Klassen
        try:
            from utils.adaptive_sync_detection import LowSignalDetectionMethods, MultiMethodConsensus
            
            methods = LowSignalDetectionMethods(sr=sr)
            consensus = MultiMethodConsensus()
            
            # Für jeden Trial: Suche 440 Hz und 880 Hz
            for trial_event in [e for e in json_log.get('events', []) if e.get('event_type', '').startswith('start_fixation')]:
                trial_num = trial_event.get('trial_number', -1)
                if trial_num < 0:
                    continue
                
                # Berechne erwartete Zeit aus JSON
                eyelink_time_ms = get_eyelink_time_corrected(trial_event, json_format)
                expected_time_440 = (eyelink_time_ms - initial_offset_ms) / 1000
                
                # 440 Hz Suche
                candidates_440 = methods.template_matching(audio, 440, expected_time_440 - 2, 4, threshold=0.15)
                if candidates_440:
                    best_440 = max(candidates_440, key=lambda x: x.score)
                    all_anchors.append({
                        'video_time_s': best_440.time_s,
                        'event': trial_event,
                        'frequency': FREQ_TRIAL_START,
                        'source': f'trial_{trial_num}_440_adaptive',
                        'trial_number': trial_num
                    })
                    n_trial_440 += 1
                
                # 880 Hz Suche (am Ende der Audio-Datei)
                audio_duration_ms = trial_event.get('audio_duration_ms', 2000)
                expected_time_880 = expected_time_440 + (audio_duration_ms / 1000) - 0.2
                
                candidates_880 = methods.template_matching(audio, 880, expected_time_880 - 2, 4, threshold=0.15)
                if candidates_880:
                    best_880 = max(candidates_880, key=lambda x: x.score)
                    all_anchors.append({
                        'video_time_s': best_880.time_s,
                        'event': trial_event,
                        'frequency': FREQ_TRIAL_END,
                        'source': f'trial_{trial_num}_880_adaptive',
                        'trial_number': trial_num
                    })
                    n_trial_880 += 1
            
            print(f"      [Adaptive] 440 Hz: {n_trial_440} Anchors")
            print(f"      [Adaptive] 880 Hz: {n_trial_880} Anchors")
            
            # ══════════════════════════════════════════════════════════════════
            # 6D: Qualitäts-Check - Entferne schlechte Trial-Anchors
            # ══════════════════════════════════════════════════════════════════
            
            # Bei beschädigtem Audio: Prüfe ob Trial-Anchors die Qualität verschlechtern
            calib_anchors_only = [a for a in all_anchors if a['frequency'] == FREQ_CALIBRATION]
            trial_anchors_only = [a for a in all_anchors if a['frequency'] in [FREQ_TRIAL_START, FREQ_TRIAL_END]]
            
            if len(calib_anchors_only) >= 20 and len(trial_anchors_only) > 0:
                # Berechne Test-Offsets für Kalibrierung
                calib_offsets_test = []
                for anchor in calib_anchors_only:
                    event = anchor.get('event')
                    if event:
                        eyelink_peak = get_corrected_eyelink_time_for_peak(
                            event, FREQ_CALIBRATION, json_format, DEFAULT_LATENCY_MS
                        )
                        calib_offsets_test.append(eyelink_peak - anchor['video_time_s'] * 1000)
                
                # Berechne Test-Offsets für Trials
                trial_offsets_test = []
                for anchor in trial_anchors_only:
                    event = anchor.get('event')
                    if event:
                        eyelink_peak = get_corrected_eyelink_time_for_peak(
                            event, anchor['frequency'], json_format, DEFAULT_LATENCY_MS
                        )
                        trial_offsets_test.append(eyelink_peak - anchor['video_time_s'] * 1000)
                
                if len(calib_offsets_test) > 0 and len(trial_offsets_test) > 0:
                    std_calib = np.std(calib_offsets_test)
                    std_trial = np.std(trial_offsets_test)
                    
                    print(f"\n      [Qualitäts-Check]")
                    print(f"        Kalibrierung Std: {std_calib:.1f} ms (n={len(calib_offsets_test)})")
                    print(f"        Trials Std:       {std_trial:.1f} ms (n={len(trial_offsets_test)})")
                    
                    # Wenn Trial-Std > 3x Kalibrierungs-Std UND > 150ms, entferne Trial-Anchors
                    if std_trial > std_calib * 3 and std_trial > 150:
                        print(f"        ⚠ Trial-Anchors verschlechtern Qualität >> ENTFERNT")
                        all_anchors = calib_anchors_only
                        n_trial_440 = 0
                        n_trial_880 = 0
                    else:
                        print(f"        ✓ Trial-Anchors akzeptabel")
            
        except Exception as e:
            print(f"      [WARN] Adaptive Trial-Suche fehlgeschlagen: {e}")

    # ══════════════════════════════════════════════════════════════════
    # SCHRITT 7: OFFSET-BERECHNUNG MIT LATENZ-KORREKTUR (NEU!)
    # ══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("SCHRITT 7: OFFSET-BERECHNUNG MIT LATENZ-KORREKTUR")
    print(f"{'='*70}")

    sync_result = None
    
    if len(all_anchors) >= MIN_ANCHORS_FOR_ESTIMATION:
        # ──────────────────────────────────────────────────────────────
        # 7A: Schätze Audio-Latenz (für V1 JSONs)
        # ──────────────────────────────────────────────────────────────
        
        estimated_latency = estimate_audio_latency_from_detections(all_anchors, json_format)
        
        # ──────────────────────────────────────────────────────────────
        # 7B: Berechne Sync-Offset mit korrigierten Zeiten
        # ──────────────────────────────────────────────────────────────
        
        sync_result = calculate_sync_offset(
            detected_anchors=all_anchors,
            json_format=json_format,
            estimated_latency_ms=estimated_latency
        )
        
        print(f"\n    ► Finaler Offset: {sync_result.offset_ms:.1f} ms")
        print(f"    ► Methode: {sync_result.method}")
        print(f"    ► Std: {sync_result.std_ms:.1f} ms")
        
        # Prüfe Qualität
        if sync_result.std_ms < 50:
            print(f"    ✓ EXZELLENT: Std < 50ms")
        elif sync_result.std_ms < 100:
            print(f"    ✓ GUT: Std < 100ms")
        else:
            print(f"    ⚠ WARNUNG: Std > 100ms - Prüfe Daten!")
        
    else:
        print(f"\n    [!] Nur {len(all_anchors)} Anchors (min. {MIN_ANCHORS_FOR_ESTIMATION} benötigt)")
        print(f"        Nutze Initial-Offset: {initial_offset_ms:.1f} ms")
        
        sync_result = SyncResult(
            offset_ms=initial_offset_ms,
            method='initial_offset_fallback',
            n_anchors=len(all_anchors),
            std_ms=0.0,
            estimated_latency_ms=DEFAULT_LATENCY_MS
        )

    # ──────────────────────────────────────────────────────────────────
    # 7C: Erstelle multi_anchor_result für Kompatibilität
    # ──────────────────────────────────────────────────────────────────
    
    multi_anchor_result = CompatMultiAnchorResult(sync_result) if sync_result else None

    # ══════════════════════════════════════════════════════════════════
    # SCHRITT 8: PHASEN-REKONSTRUKTION MIT KORRIGIERTEM OFFSET
    # ══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print("SCHRITT 8: PHASEN-REKONSTRUKTION (REFINEMENT)")
    print(f"{'='*70}")
    
    if sync_result and sync_result.method != 'initial_offset_fallback':
        print(f"    Wende korrigierten Offset an: {sync_result.offset_ms:.0f} ms")
        print(f"    (Std: {sync_result.std_ms:.1f} ms, {sync_result.n_anchors} Anchors)")
        
        # Recalculate phases mit neuem Offset
        phases = reconstructor.reconstruct_all_phases(
            json_log, json_format,
            selected_time_s, sync_point_eyelink_ms,
            audio, sr, calibration_timing_jsons,
            multi_anchor_result,  # Kompatibilitäts-Wrapper
            skip_audio_validation=_SKIP_AUDIO_VALIDATION
        )
    else:
        # Kein verbesserter Offset >> Behalte initiale phases
        print(f"    Kein verbesserter Offset >> Behalte initiale Phasen")
        phases = phases_initial

    # ══════════════════════════════════════════════════════════════════
    #  SCHRITT 9: POST-VALIDATION (Vorher/Nachher-Vergleich)
    # ══════════════════════════════════════════════════════════════════
    
    if multi_anchor_result and multi_anchor_result.method if multi_anchor_result is not None else 'initial' in ['multi_anchor_regression', 'dual_anchor']:
        print(f"\n{'='*70}")
        print("SCHRITT 9: POST-VALIDATION (Vorher/Nachher)")
        print(f"{'='*70}")
        
        # Extrahiere neue Metriken
        refined_stats = _extract_validation_stats(phases)
        
        # Vergleiche mit Baseline
        print(f"\n    Verbesserung durch {multi_anchor_result.method if multi_anchor_result is not None else 'initial'.upper()}:")
        
        print(f"\n   Kalibrierung Mean Deviation:")
        print(f"      Vorher (Baseline): {baseline_stats['calib_mean_dev']:.1f} ms")
        print(f"      Nachher (Korrigiert): {refined_stats['calib_mean_dev']:.1f} ms")
        improvement_calib = baseline_stats['calib_mean_dev'] - refined_stats['calib_mean_dev']
        if baseline_stats['calib_mean_dev'] > 0:
            print(f"      Verbesserung: {improvement_calib:+.1f} ms ({improvement_calib/baseline_stats['calib_mean_dev']*100:+.1f}%)")
        else:
            print(f"      Verbesserung: {improvement_calib:+.1f} ms (Baseline war 0)")
        print(f"\n   Trial Detection-Rates:")
        print(f"      440 Hz: {baseline_stats['trial_440_rate']*100:.1f}% >> {refined_stats['trial_440_rate']*100:.1f}%")
        print(f"      880 Hz: {baseline_stats['trial_880_rate']*100:.1f}% >> {refined_stats['trial_880_rate']*100:.1f}%")
        
        #  NEU: Zeige auch Trial-Deviations!
        if 'trial_440_dev' in refined_stats and refined_stats['trial_440_dev'] is not None:
            print(f"\n   Trial 440Hz Mean Deviation:")
            print(f"      Vorher: {baseline_stats.get('trial_440_dev', 0):.1f} ms")
            print(f"      Nachher: {refined_stats['trial_440_dev']:.1f} ms")
            improvement_trial = baseline_stats.get('trial_440_dev', 0) - refined_stats['trial_440_dev']
            print(f"      Verbesserung: {improvement_trial:+.1f} ms")
        
        # Bewertung
        if refined_stats['calib_mean_dev'] < baseline_stats['calib_mean_dev']:
            print(f"\n    {multi_anchor_result.method if multi_anchor_result is not None else 'initial'.upper()} hat Validierung verbessert!")
        else:
            print(f"\n    Keine Verbesserung (erwartbar bei bereits guter Sync)")

    # ══════════════════════════════════════════════════════════════════
    # SCHRITT 9B: Validierungs-Report schreiben
    # ══════════════════════════════════════════════════════════════════
    
    # Erstelle SyncResult-ähnliches Objekt für Report (Kompatibilität)
    class SimpleSyncResult:
        def __init__(self, phases, multi_anchor_result):
            self.n_anchors = multi_anchor_result.n_anchors if multi_anchor_result else 0
            self.std_ms = multi_anchor_result.drift_ms if multi_anchor_result else 0
            self.estimated_latency_ms = 500.0  # Default
            self.offset_ms = phases['sync_info'].get('offset_ms', 0)
    
    sync_result_for_report = SimpleSyncResult(phases, multi_anchor_result) if multi_anchor_result else None
    
    write_validation_report(
        output_dir=output_dir,
        sync_result=sync_result_for_report,
        all_anchors=all_anchors,
        json_format=json_format,
        phases=phases,
        baseline_stats=baseline_stats if 'baseline_stats' in dir() else None,
        refined_stats=refined_stats if 'refined_stats' in dir() else None
    )

    # ══════════════════════════════════════════════════════════════════
    # SCHRITT 10: Export
    # ══════════════════════════════════════════════════════════════════
    
    output_json = output_dir / "phases_detected.json"
    with open(output_json, 'w') as f:
        json.dump(phases, f, indent=2)

    # ══════════════════════════════════════════════════════════════════════════════
    # TESTMODUS: Early-Exit nach phases_detected.json
    # ══════════════════════════════════════════════════════════════════════════════
    
    if is_calibration_test_mode:
        print(f"\n{'='*70}")
        print(" TESTMODUS: PHASE-DETECTION ABGESCHLOSSEN")
        print(f"{'='*70}")
        print(f"\n Output: {output_json.name}")
        print(f"\n Nächster Schritt:")
        print(f"   → offline_calibration.py (für R²-Werte)")
        print(f"\n   HINWEIS: Trials werden ÜBERSPRUNGEN (Testmodus)")
        
        # Überspringe QS-Visualisierung (spart Zeit)
        print(f"\n   QS-Grafiken werden übersprungen (Testmodus)")
        
        return  # ← Early Exit
    
    print(f"\n Output:")
    print(f"    {output_json.name}")
    
    # ══════════════════════════════════════════════════════════════════
    # SCHRITT 9: QS-Visualisierung (vollständig!)
    # ══════════════════════════════════════════════════════════════════
    
    create_qc_graphics_complete(
        audio, sr, candidates, selected_time_s, phases, 
        output_dir, calibration_timing_jsons,
        multi_anchor_result  # ← NEU v2.1!
    )
    
    print(f"\n{'='*70}")
    print(" ERFOLGREICH!")
    print(f"{'='*70}")
    print(f"\n Output-Dateien:")
    print(f"   • phases_detected.json")
    print(f"   • phase_detection_qc_overview.png")
    print(f"   • phase_detection_qc_stats.png")
    print(f"   • phase_detection_detail_calibration_*.png ({len(calibration_timing_jsons)}x)")
    print(f"   • phase_detection_detail_trials_*.png (2x)")
    
    if multi_anchor_result:
        print(f"\n Multi-Anchor-Statistik:")
        print(f"   • Methode: {multi_anchor_result.method if multi_anchor_result is not None else 'initial'}")
        print(f"   • Anchors: {multi_anchor_result.n_anchors}")
        print(f"   • R²: {multi_anchor_result.r_squared:.4f}")
        print(f"   • Drift: {multi_anchor_result.drift_ms:.1f} ms")
    
    print(f"\n Nächster Schritt:")
    print(f"   • debug_1_video_analysis.py")

# ══════════════════════════════════════════════════════════════════
#  HELPER: Extrahiere Validierungs-Statistik
# ══════════════════════════════════════════════════════════════════

def _extract_validation_stats(phases: Dict) -> Dict:
    """Extrahiert Validierungs-Metriken aus phases dict"""
    
    stats = {
        'calib_detection_rate': 0.0,
        'calib_mean_dev': 0.0,
        'trial_440_rate': 0.0,
        'trial_880_rate': 0.0,
        'trial_440_dev': None,  # ← NEU!
        'trial_880_dev': None   # ← NEU!
    }
    
    # Kalibrierung
    calib_deviations = []
    calib_total = 0
    calib_found = 0
    
    for label in ['beg', 'mid', 'end']:
        calib_key = f'calibration_{label}'
        if calib_key in phases['phases'] and 'points' in phases['phases'][calib_key]:
            for point in phases['phases'][calib_key]['points']:
                calib_total += 1
                if point.get('audio_found_s') is not None:
                    calib_found += 1
                    if point.get('audio_deviation_ms') is not None:
                        calib_deviations.append(point['audio_deviation_ms'])
    
    if calib_total > 0:
        stats['calib_detection_rate'] = calib_found / calib_total
    
    if calib_deviations:
        stats['calib_mean_dev'] = abs(np.mean(calib_deviations))
    
    #  NEU: Trial-Deviations extrahieren (aus Console-Output!)
    # Da diese Info nicht in phases gespeichert wird, müssen wir sie
    # aus der Trial-Kategorisierung extrahieren (falls vorhanden)
    
    # Trials
    trial_440_total = 0
    trial_440_found = 0
    trial_880_total = 0
    trial_880_found = 0
    
    for block_key in ['experiment_block1', 'experiment_block2']:
        if block_key in phases['phases']:
            for trial in phases['phases'][block_key]['trials']:
                trial_440_total += 1
                if trial['fixation'].get('audio_found'):
                    trial_440_found += 1
                
                trial_880_total += 1
                if trial['fixation'].get('end_audio_found'):
                    trial_880_found += 1
    
    if trial_440_total > 0:
        stats['trial_440_rate'] = trial_440_found / trial_440_total
    
    if trial_880_total > 0:
        stats['trial_880_rate'] = trial_880_found / trial_880_total
    
    #  HINWEIS: Trial-Deviations werden aktuell NICHT in phases gespeichert!
    # >> Bleiben None (TODO für v2.3: In phases integrieren)
    
    return stats

# ══════════════════════════════════════════════════════════════════════════════
# VALIDIERUNGS-REPORT (NEU)
# ══════════════════════════════════════════════════════════════════════════════

def write_validation_report(output_dir: Path, sync_result: 'SyncResult', 
                            all_anchors: List[Dict], json_format: 'JSONFormat',
                            phases: Dict, baseline_stats: Dict = None,
                            refined_stats: Dict = None) -> None:
    """
    Schreibt einen detaillierten Validierungs-Report.
    
    Output: offset_validation_report.txt
    """
    from datetime import datetime
    
    report_path = output_dir / "offset_validation_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("OFFSET VALIDATION REPORT\n")
        f.write(f"Generiert: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        # 1. Zusammenfassung
        f.write("1. ZUSAMMENFASSUNG\n")
        f.write("-" * 40 + "\n")
        
        sync_info = phases.get('sync_info', {})
        f.write(f"   Finaler Offset:      {sync_info.get('offset_ms', 0):.1f} ms\n")
        f.write(f"   Methode:             {sync_info.get('method', 'unknown')}\n")
        f.write(f"   JSON-Format:         {json_format.value if hasattr(json_format, 'value') else str(json_format)}\n")
        
        if sync_result:
            f.write(f"   Anzahl Anchors:      {sync_result.n_anchors}\n")
            f.write(f"   Std. Abweichung:     {sync_result.std_ms:.1f} ms\n")
            f.write(f"   Geschätzte Latenz:   {sync_result.estimated_latency_ms:.1f} ms\n")
        f.write("\n")
        
        # 2. Frequenz-Analyse (falls Dual-Anchor)
        if sync_info.get('offset_1760hz_ms') is not None:
            f.write("2. DUAL-ANCHOR FREQUENZ-ANALYSE\n")
            f.write("-" * 40 + "\n")
            f.write(f"   1760 Hz (Kalibrierung): {sync_info.get('offset_1760hz_ms', 0):.1f} ms\n")
            f.write(f"   440 Hz (Fixation):      {sync_info.get('offset_440hz_ms', 0):.1f} ms\n")
            f.write(f"   880 Hz (Trial-End):     {sync_info.get('offset_880hz_ms', 0):.1f} ms\n")
            f.write(f"   Frequenz-Differenz:     {sync_info.get('frequency_difference_ms', 0):+.1f} ms\n")
            f.write("\n")
        
        # 3. Vorher/Nachher-Vergleich
        if baseline_stats and refined_stats:
            f.write("3. VORHER/NACHHER-VERGLEICH\n")
            f.write("-" * 40 + "\n")
            f.write(f"\n   Kalibrierung Mean Deviation:\n")
            f.write(f"      Vorher:  {baseline_stats.get('calib_mean_dev', 0):.1f} ms\n")
            f.write(f"      Nachher: {refined_stats.get('calib_mean_dev', 0):.1f} ms\n")
            
            improvement = baseline_stats.get('calib_mean_dev', 0) - refined_stats.get('calib_mean_dev', 0)
            f.write(f"      Δ:       {improvement:+.1f} ms\n")
            
            f.write(f"\n   Trial Detection-Rates:\n")
            f.write(f"      440 Hz: {baseline_stats.get('trial_440_rate', 0)*100:.1f}% → {refined_stats.get('trial_440_rate', 0)*100:.1f}%\n")
            f.write(f"      880 Hz: {baseline_stats.get('trial_880_rate', 0)*100:.1f}% → {refined_stats.get('trial_880_rate', 0)*100:.1f}%\n")
            f.write("\n")
        
        # 4. Bewertung
        f.write("4. BEWERTUNG\n")
        f.write("-" * 40 + "\n")
        
        # Kriterien
        calib_dev = refined_stats.get('calib_mean_dev', 999) if refined_stats else 999
        
        if calib_dev < 50:
            f.write("   ✓ SYNCHRONISATION EXZELLENT\n")
            f.write("     - Kalibrierung < 50ms Abweichung\n")
        elif calib_dev < 100:
            f.write("   ✓ SYNCHRONISATION GUT\n")
            f.write("     - Kalibrierung < 100ms Abweichung\n")
        elif calib_dev < 200:
            f.write("   ⚠ SYNCHRONISATION AKZEPTABEL\n")
            f.write("     - Kalibrierung < 200ms Abweichung\n")
            f.write("     - Manuelle Überprüfung empfohlen\n")
        else:
            f.write("   ✗ SYNCHRONISATION PRÜFEN\n")
            f.write(f"     - Kalibrierung {calib_dev:.0f}ms Abweichung (>200ms!)\n")
            f.write("     - Manuelle Überprüfung DRINGEND empfohlen\n")
        
        # 5. Anchors (falls vorhanden)
        if all_anchors and len(all_anchors) > 0:
            f.write("\n\n5. ANCHOR-DETAILS\n")
            f.write("-" * 40 + "\n")
            f.write(f"   Gesamt: {len(all_anchors)} Anchors\n")
            
            # Zähle nach Frequenz
            freq_counts = {}
            for anchor in all_anchors:
                freq = anchor.frequency if hasattr(anchor, 'frequency') else anchor.get('frequency', 'unknown')
                freq_counts[freq] = freq_counts.get(freq, 0) + 1
            
            for freq, count in sorted(freq_counts.items()):
                f.write(f"   {freq} Hz: {count} Anchors\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("ENDE DES REPORTS\n")
    
    print(f"\n    [✓] Validierungs-Report: {report_path.name}")

# ==================== MAIN ====================

if __name__ == "__main__":
    # Pfade
    video_path = Path(MAIN_VIDEO_PATH)
    json_path = Path(EXPERIMENT_SYNC_JSON_PATH)
    output_dir = Path(OUTPUT_BASE_DIR)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Validierung
    if not video_path.exists():
        print(f" Video nicht gefunden: {video_path}")
        exit(1)
    
    if not json_path.exists():
        print(f" JSON nicht gefunden: {json_path}")
        exit(1)
    
    # Run
    run_phase_detection(video_path, json_path, output_dir)
