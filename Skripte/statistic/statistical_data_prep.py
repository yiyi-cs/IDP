"""
=================================================================================
STATISTICAL_DATA_PREP.PY v1.0 - Erstellt einheitlichen Analysedatensatz
=================================================================================

Kernfunktionen:
---------------
1. Laden der drei Datenquellen (MediaPipe, ptgaze, EyeLink)
2. EyeLink Pixel → Grad konvertieren
3. Downsampling EyeLink (2000 Hz → Video-Framerate)
4. Lag-Korrektur via Cross-Correlation
5. Frame-Level Merge (timestamp-basiert)
6. Optional: Einheitliche Jitter-Reduktion
7. Ausschluss-Flags setzen (NICHT anwenden!)
8. VP-übergreifende Aggregation

Wissenschaftliche Prinzipien:
-----------------------------
- Minimal vorverarbeitete Daten für maximale Transparenz
- Ausschlüsse erst in R (dokumentierbar, reproduzierbar)
- Lag-Korrektur via Cross-Correlation (Standard in Eye-Tracking)

Output:
-------
- analysis_frame_level.csv (Long-Format für LMM)
- lag_correction_report.json (Dokumentation der Lag-Korrektur)

CSV-Struktur:
# Identifikatoren
vp_id                   # z.B. "abc1"
trial_id                # 1-24 (Practice ausgeschlossen)
frame                   # Frame-Nummer im Trial
timestamp_ms            # Zeitstempel (synchronisiert)

# Stimulus-Info
stimulus_id             # 1-12 (welches Bild)
flip_status             # 1=normal, 2=gespiegelt

# Unabhängige Variablen
method                  # "mediapipe" oder "ptgaze"
framerate               # "25hz" oder "60hz"
calib_config            # "FullCalib", "OnlyBegFix", etc.

# Abhängige Variablen (Gaze in Grad)
cv_deg_x                # CV-Methode X
cv_deg_y                # CV-Methode Y
eyelink_deg_x           # EyeLink X (Referenz)
eyelink_deg_y           # EyeLink Y (Referenz)

# Qualitäts-Flags (für Ausschluss in R)
is_blink                # TRUE/FALSE
eyes_closed             # TRUE/FALSE
outside_monitor         # TRUE/FALSE
exclude_blink           # TRUE/FALSE (is_blink | eyes_closed)
exclude_outside         # TRUE/FALSE
exclude_any             # TRUE/FALSE (kombiniert, OHNE Confidence)

# Lag-Korrektur
lag_applied_ms          # Angewendeter Lag pro VP

Version: 1.0
Datum: 2025-01
=================================================================================
"""

# =================================================================================
# PATH SETUP
# =================================================================================
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# =================================================================================
# IMPORTS
# =================================================================================
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
from datetime import datetime
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import loadmat

from config import (
    SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX,
    SCREEN_WIDTH_CM, SCREEN_HEIGHT_CM,
    VIEWING_DISTANCE_CM,
    MIN_VALID_SAMPLE_RATIO,
    N_TRIALS, FIRST_TRIAL_IS_PRACTICE
)

from shared.shared_coordinate_utils import (
    CoordinateTransformer,
    create_coordinate_transformer,
    batch_convert_to_degrees
)

# Type-Hint Import (nur für IDE/Type-Checker, nicht zur Laufzeit)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from statistic.statistical_pipeline import EyeLinkCache

# =================================================================================
# KONFIGURATION
# =================================================================================

# Lag-Korrektur
LAG_SEARCH_RANGE_MS = (-500, 500)  # Suchbereich für Cross-Correlation
EXPECTED_LAG_RANGE_MS = (100, 300)  # Erwarteter Bereich (CV verzögert)
LAG_CORRELATION_THRESHOLD = 0.3    # Mindest-Korrelation für validen Lag

# Downsampling
EYELINK_SAMPLE_RATE_HZ = 2000
DOWNSAMPLE_METHOD = 'nearest'  # 'nearest', 'linear', 'mean_window'

# Jitter-Filter (optional, für beide Methoden gleich)
ENABLE_UNIFIED_JITTER_FILTER = False
JITTER_FILTER_ALPHA = 0.3  # Double Exponential Alpha
JITTER_FILTER_BETA = 0.1   # Double Exponential Beta

# Ausschluss-Schwellenwerte (werden als Flags gesetzt, nicht angewendet!)
CONFIDENCE_THRESHOLD = 0.7
BLINK_RECOVERY_MS = 150  # Post-Blink Recovery Zeit

# Trial-Konfiguration
PRACTICE_TRIAL_NUMBER = 0
EXPERIMENTAL_TRIALS = list(range(1, N_TRIALS + 1))  # 1-24


# =================================================================================
# DATENSTRUKTUREN
# =================================================================================

@dataclass
class LagCorrectionResult:
    """Ergebnis der Lag-Korrektur für eine VP/Config."""
    vp_id: str
    config_key: str
    method: str  # 'mediapipe' oder 'ptgaze'
    
    optimal_lag_ms: float
    correlation_at_lag: float
    lag_valid: bool
    
    # Diagnose
    search_range_ms: Tuple[float, float] = field(default_factory=lambda: LAG_SEARCH_RANGE_MS)
    correlation_curve: Optional[List[float]] = None
    lag_values_ms: Optional[List[float]] = None
    
    def to_dict(self) -> Dict:
        return {
            'vp_id': self.vp_id,
            'config_key': self.config_key,
            'method': self.method,
            'optimal_lag_ms': self.optimal_lag_ms,
            'correlation_at_lag': self.correlation_at_lag,
            'lag_valid': self.lag_valid,
            'search_range_ms': list(self.search_range_ms)
        }


@dataclass
class DataPrepResult:
    """Ergebnis der Datenaufbereitung für eine VP/Config."""
    vp_id: str
    video_fps: str
    calib_mode: str
    
    # Daten
    frame_level_df: pd.DataFrame
    
    # Lag-Korrektur
    lag_mediapipe: Optional[LagCorrectionResult] = None
    lag_ptgaze: Optional[LagCorrectionResult] = None
    
    # Statistiken
    n_frames_total: int = 0
    n_frames_valid: int = 0
    n_trials: int = 0
    
    # Qualität
    success: bool = False
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

# =================================================================================
# STIMULUS-INFO EXTRAKTION
# =================================================================================

def extract_stimulus_info(vp_folder: Path, vp_id: str) -> Optional[Dict]:
    """
    Extrahiert Stimulus-IDs und Flip-Status aus JSON oder MAT-Datei.
    
    Priorität:
    1. experiment_sync_log.json (neuere VPs, enthält 'randomization')
    2. Settings_fv_*.mat (alle VPs, enthält 'rand_A' und 'rand_A_flip')
    
    Args:
        vp_folder: Pfad zum VP-Hauptordner (z.B. Ergebnisse/abc1/)
        vp_id: VP-Code (z.B. 'abc1')
    
    Returns:
        Dict mit 'trial_order' (1-12), 'flip_status' (1/2), 'source' (json/mat)
        oder None falls nicht gefunden
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # Strategie 1: JSON (falls vorhanden, neuere VPs)
    # ─────────────────────────────────────────────────────────────────────────
    
    json_path = vp_folder / 'experiment_sync_log.json'
    if json_path.exists():
        try:
            # Robustes Laden: Repariere ungültige Backslash-Escapes
            with open(json_path, 'r', encoding='utf-8') as f:
                json_content = f.read()
            
            # Fix: Ersetze ungültige Escape-Sequenzen in Windows-Pfaden
            # \U, \E, \F, \D, \K, etc. sind keine gültigen JSON-Escapes
            # Wir ersetzen einzelne Backslashes durch doppelte (außer bei gültigen Escapes)
            import re
            
            def fix_backslashes(content: str) -> str:
                """
                Repariert ungültige Backslash-Escapes in JSON.
                
                Gültige JSON-Escapes: backslash + " \\ / b f n r t uXXXX
                Alles andere (z.B. backslash+U, backslash+D) muss escaped werden.
                """
                # Ersetze Backslashes, die NICHT von gültigen Escape-Zeichen gefolgt werden
                # Lookbehind für nicht-escaped Backslash, Lookahead für ungültige Escape-Zeichen
                fixed = re.sub(
                    r'\\(?!["\\/bfnrtu])',  # Backslash NICHT gefolgt von gültigen Escapes
                    r'\\\\',                 # Ersetze durch doppelten Backslash
                    content
                )
                return fixed
            
            json_content_fixed = fix_backslashes(json_content)
            data = json.loads(json_content_fixed)
            
            # Prüfe ob randomization in metadata vorhanden
            if 'metadata' in data and 'randomization' in data['metadata']:
                rand_info = data['metadata']['randomization']
                
                if 'trial_order' in rand_info and 'flip_status' in rand_info:
                    print(f"  [OK] Stimulus-Info aus JSON: {json_path.name}")
                    return {
                        'trial_order': rand_info['trial_order'],
                        'flip_status': rand_info['flip_status'],
                        'source': 'json'
                    }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [!] JSON-Fehler: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Strategie 2: MAT-Datei (Fallback, alle VPs)
    # ─────────────────────────────────────────────────────────────────────────
    
    # Suche nach Settings_fv_*.mat im VP-Ordner
    mat_pattern = f"Settings_fv_{vp_id}*.mat"
    mat_files = list(vp_folder.glob(mat_pattern))
    
    # Falls nicht gefunden, versuche allgemeineres Pattern
    if not mat_files:
        mat_files = list(vp_folder.glob("Settings_fv_*.mat"))
    
    if mat_files:
        # Nehme die neueste Datei (falls mehrere)
        mat_path = sorted(mat_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        
        try:
            mat_data = loadmat(str(mat_path), squeeze_me=False)
            
            # Extrahiere aus Settings-Struktur
            # Settings ist ein numpy structured array
            settings = mat_data['Settings']
            
            # Zugriff auf nested struct: Settings.rand_A
            rand_A = settings['rand_A'][0, 0].flatten().astype(int).tolist()
            rand_A_flip = settings['rand_A_flip'][0, 0].flatten().astype(int).tolist()
            
            print(f"  [OK] Stimulus-Info aus MAT: {mat_path.name}")
            return {
                'trial_order': rand_A,
                'flip_status': rand_A_flip,
                'source': 'mat'
            }
        
        except Exception as e:
            print(f"  [!] MAT-Fehler ({mat_path.name}): {e}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Keine Stimulus-Info gefunden
    # ─────────────────────────────────────────────────────────────────────────
    
    print(f"  [!] Keine Stimulus-Info gefunden für VP {vp_id}")
    return None


def map_trial_to_stimulus(trial_assignment: int, stimulus_info: Dict) -> Tuple[Optional[int], Optional[int]]:
    """
    Mappt eine Trial-Nummer auf Stimulus-ID und Flip-Status.
    
    Args:
        trial_assignment: Trial-Nummer (1-24, 0 = Practice)
        stimulus_info: Dict aus extract_stimulus_info()
    
    Returns:
        (stimulus_id, flip_status) oder (None, None) für Practice/ungültig
    """
    # Practice-Trial (0) hat keinen Stimulus
    if trial_assignment == 0:
        return None, None
    
    # Trial-Index (0-basiert)
    trial_idx = trial_assignment - 1
    
    if stimulus_info is None:
        return None, None
    
    trial_order = stimulus_info.get('trial_order', [])
    flip_status = stimulus_info.get('flip_status', [])
    
    if trial_idx < 0 or trial_idx >= len(trial_order):
        return None, None
    
    return trial_order[trial_idx], flip_status[trial_idx]

# =================================================================================
# DOUBLE EXPONENTIAL FILTER (einheitlich für beide Methoden)
# =================================================================================

class DoubleExponentialFilter:
    """
    Double Exponential Smoothing (Holt's Linear Trend).
    
    Identisch für MediaPipe und ptgaze, wenn aktiviert.
    """
    
    def __init__(self, alpha: float = JITTER_FILTER_ALPHA, 
                 beta: float = JITTER_FILTER_BETA):
        self.alpha = alpha
        self.beta = beta
        self.level = None
        self.trend = 0.0
    
    def __call__(self, x: float) -> float:
        if np.isnan(x):
            return np.nan
        
        if self.level is None:
            self.level = x
            return x
        
        prev_level = self.level
        self.level = self.alpha * x + (1 - self.alpha) * (prev_level + self.trend)
        self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend
        
        return self.level
    
    def reset(self):
        self.level = None
        self.trend = 0.0
    
    def filter_series(self, series: pd.Series) -> pd.Series:
        """Filtert eine ganze Series."""
        self.reset()
        return series.apply(self.__call__)


# =================================================================================
# STATISTICAL DATA PREP
# =================================================================================

class StatisticalDataPrep:
    """
    Bereitet Daten für statistische Analyse auf.
    
    Workflow:
    ---------
    1. load_vp_data() - Lädt alle drei Quellen
    2. convert_eyelink_to_degrees() - Pixel → Grad
    3. downsample_eyelink() - 2000 Hz → Video-Framerate
    4. compute_lag_correction() - Cross-Correlation
    5. merge_frame_level() - Timestamp-basierter Merge
    6. apply_jitter_filter() - Optional, einheitlich
    7. set_exclusion_flags() - Flags setzen (nicht anwenden!)
    8. export_analysis_dataset() - Long-Format CSV
    """
    
    def __init__(self):
        self.transformer = create_coordinate_transformer()
        self.jitter_filter_x = DoubleExponentialFilter()
        self.jitter_filter_y = DoubleExponentialFilter()
    
    # =========================================================================
    # SCHRITT 1: DATEN LADEN
    # =========================================================================
    
    def load_vp_data(self, run_folder: Path) -> Dict[str, pd.DataFrame]:
        """
        Lädt alle drei Datenquellen aus einem Run-Ordner.
        
        Args:
            run_folder: Pfad zum Run_* Ordner
        
        Returns:
            Dict mit 'mediapipe', 'ptgaze', 'eyelink' DataFrames
        """
        data = {}
        
        # MediaPipe
        mp_path = run_folder / 'debug_5_pupil_data_calibrated.csv'
        if mp_path.exists():
            data['mediapipe'] = pd.read_csv(mp_path)
            print(f"  [OK] MediaPipe: {len(data['mediapipe'])} Frames")
        else:
            print(f"  [!!] MediaPipe nicht gefunden: {mp_path.name}")
            data['mediapipe'] = None
        
        # ptgaze
        pt_path = run_folder / 'debug_5_ptgaze_calibrated.csv'
        if pt_path.exists():
            data['ptgaze'] = pd.read_csv(pt_path)
            print(f"  [OK] ptgaze: {len(data['ptgaze'])} Frames")
        else:
            print(f"  [!!] ptgaze nicht gefunden: {pt_path.name}")
            data['ptgaze'] = None
        
        # EyeLink
        el_path = run_folder / 'debug_3_eyetracker_data.csv'
        if el_path.exists():
            data['eyelink'] = pd.read_csv(el_path)
            print(f"  [OK] EyeLink: {len(data['eyelink'])} Samples")
        else:
            print(f"  [!!] EyeLink nicht gefunden: {el_path.name}")
            data['eyelink'] = None
        
        return data
    
    # =========================================================================
    # SCHRITT 2: EYELINK PIXEL → GRAD
    # =========================================================================
    
    def convert_eyelink_to_degrees(self, eyelink_df: pd.DataFrame) -> pd.DataFrame:
        """
        Konvertiert EyeLink Gaze-Koordinaten von Pixel zu Grad.
        
        Args:
            eyelink_df: DataFrame mit x_pos, y_pos (Pixel)
        
        Returns:
            DataFrame mit zusätzlichen eyelink_deg_x, eyelink_deg_y Spalten
        """
        if eyelink_df is None or eyelink_df.empty:
            return eyelink_df
        
        print(f"\n  [>] EyeLink Pixel → Grad konvertieren...")
        
        # Batch-Konvertierung
        deg_x, deg_y = batch_convert_to_degrees(
            eyelink_df['x_pos'].values,
            eyelink_df['y_pos'].values,
            self.transformer
        )
        
        eyelink_df = eyelink_df.copy()
        eyelink_df['eyelink_deg_x'] = deg_x
        eyelink_df['eyelink_deg_y'] = deg_y
        
        # Statistik
        valid = eyelink_df['eyelink_deg_x'].notna()
        print(f"      Konvertiert: {valid.sum()}/{len(eyelink_df)} "
              f"({valid.sum()/len(eyelink_df)*100:.1f}%)")
        
        if valid.sum() > 0:
            print(f"      X: {eyelink_df.loc[valid, 'eyelink_deg_x'].min():.2f} "
                  f"bis {eyelink_df.loc[valid, 'eyelink_deg_x'].max():.2f} deg")
            print(f"      Y: {eyelink_df.loc[valid, 'eyelink_deg_y'].min():.2f} "
                  f"bis {eyelink_df.loc[valid, 'eyelink_deg_y'].max():.2f} deg")
        
        return eyelink_df
    
    # =========================================================================
    # SCHRITT 3: EYELINK DOWNSAMPLING
    # =========================================================================
    
    def downsample_eyelink(self, eyelink_df: pd.DataFrame, 
                          target_timestamps_ms: np.ndarray,
                          method: str = DOWNSAMPLE_METHOD) -> pd.DataFrame:
        """
        Downsampelt EyeLink-Daten auf Video-Framerate.
        
        Args:
            eyelink_df: EyeLink DataFrame (2000 Hz)
            target_timestamps_ms: Ziel-Zeitstempel (Video-Frames)
            method: 'nearest', 'linear', 'mean_window'
        
        Returns:
            Downgesampletes DataFrame
        """
        if eyelink_df is None or eyelink_df.empty:
            return None
        
        print(f"\n  [>] EyeLink Downsampling ({method})...")
        print(f"      Von {len(eyelink_df)} auf {len(target_timestamps_ms)} Samples")
        
        # Sortiere nach Zeit
        eyelink_sorted = eyelink_df.sort_values('timestamp_ms').reset_index(drop=True)
        
        # Erstelle Ergebnis-DataFrame
        result_data = {
            'timestamp_ms': target_timestamps_ms,
            'eyelink_deg_x': np.full(len(target_timestamps_ms), np.nan),
            'eyelink_deg_y': np.full(len(target_timestamps_ms), np.nan),
            'eyelink_pupil_size': np.full(len(target_timestamps_ms), np.nan),
            'eyelink_valid': np.zeros(len(target_timestamps_ms), dtype=bool)
        }
        
        if method == 'nearest':
            # Nearest-Neighbor Interpolation
            for i, target_ts in enumerate(target_timestamps_ms):
                # Finde nächsten EyeLink-Sample
                idx = np.abs(eyelink_sorted['timestamp_ms'] - target_ts).idxmin()
                
                # Nur wenn innerhalb 2ms (bei 2000 Hz = 0.5ms Abstand)
                if abs(eyelink_sorted.loc[idx, 'timestamp_ms'] - target_ts) <= 2:
                    result_data['eyelink_deg_x'][i] = eyelink_sorted.loc[idx, 'eyelink_deg_x']
                    result_data['eyelink_deg_y'][i] = eyelink_sorted.loc[idx, 'eyelink_deg_y']
                    result_data['eyelink_pupil_size'][i] = eyelink_sorted.loc[idx, 'pupil_size']
                    result_data['eyelink_valid'][i] = True
        
        elif method == 'linear':
            # Lineare Interpolation
            valid_mask = eyelink_sorted['eyelink_deg_x'].notna()
            
            if valid_mask.sum() > 1:
                interp_x = interp1d(
                    eyelink_sorted.loc[valid_mask, 'timestamp_ms'],
                    eyelink_sorted.loc[valid_mask, 'eyelink_deg_x'],
                    kind='linear', bounds_error=False, fill_value=np.nan
                )
                interp_y = interp1d(
                    eyelink_sorted.loc[valid_mask, 'timestamp_ms'],
                    eyelink_sorted.loc[valid_mask, 'eyelink_deg_y'],
                    kind='linear', bounds_error=False, fill_value=np.nan
                )
                
                result_data['eyelink_deg_x'] = interp_x(target_timestamps_ms)
                result_data['eyelink_deg_y'] = interp_y(target_timestamps_ms)
                result_data['eyelink_valid'] = ~np.isnan(result_data['eyelink_deg_x'])
        
        elif method == 'mean_window':
            # Mittelwert über Zeitfenster
            window_half_ms = 1000 / (len(target_timestamps_ms) / 
                                     (target_timestamps_ms[-1] - target_timestamps_ms[0]) * 1000) / 2
            
            for i, target_ts in enumerate(target_timestamps_ms):
                mask = (
                    (eyelink_sorted['timestamp_ms'] >= target_ts - window_half_ms) &
                    (eyelink_sorted['timestamp_ms'] < target_ts + window_half_ms)
                )
                
                if mask.sum() > 0:
                    result_data['eyelink_deg_x'][i] = eyelink_sorted.loc[mask, 'eyelink_deg_x'].mean()
                    result_data['eyelink_deg_y'][i] = eyelink_sorted.loc[mask, 'eyelink_deg_y'].mean()
                    result_data['eyelink_pupil_size'][i] = eyelink_sorted.loc[mask, 'pupil_size'].mean()
                    result_data['eyelink_valid'][i] = True
        
        result_df = pd.DataFrame(result_data)
        
        n_valid = result_df['eyelink_valid'].sum()
        print(f"      Valide: {n_valid}/{len(result_df)} ({n_valid/len(result_df)*100:.1f}%)")
        
        return result_df
    
    # =========================================================================
    # SCHRITT 4: LAG-KORREKTUR
    # =========================================================================
    
    def compute_lag_correction(self, cv_df: pd.DataFrame, 
                               eyelink_df: pd.DataFrame,
                               method: str,
                               vp_id: str,
                               config_key: str) -> LagCorrectionResult:
        """
        Berechnet optimalen Lag via Cross-Correlation.
        
        CV-Daten sind typischerweise 100-300ms verzögert gegenüber EyeLink.
        
        Args:
            cv_df: CV DataFrame (MediaPipe oder ptgaze)
            eyelink_df: Downgesampletes EyeLink DataFrame
            method: 'mediapipe' oder 'ptgaze'
            vp_id: VP-ID für Dokumentation
            config_key: Config für Dokumentation
        
        Returns:
            LagCorrectionResult
        """
        print(f"\n  [>] Lag-Korrektur ({method})...")
        
        # Extrahiere X-Koordinaten (höhere Varianz als Y)
        cv_col = 'gaze_deg_x_calib'
        el_col = 'eyelink_deg_x'
        
        # Merge auf gemeinsame Zeitachse
        merged = pd.merge(
            cv_df[['timestamp_ms_synced', cv_col]],
            eyelink_df[['timestamp_ms', el_col]],
            left_on='timestamp_ms_synced',
            right_on='timestamp_ms',
            how='inner'
        )
        
        if len(merged) < 100:
            print(f"      [!!] Zu wenig überlappende Daten: {len(merged)}")
            return LagCorrectionResult(
                vp_id=vp_id, config_key=config_key, method=method,
                optimal_lag_ms=0, correlation_at_lag=0, lag_valid=False
            )
        
        # Entferne NaN
        valid_mask = merged[cv_col].notna() & merged[el_col].notna()
        cv_signal = merged.loc[valid_mask, cv_col].values
        el_signal = merged.loc[valid_mask, el_col].values
        
        if len(cv_signal) < 100:
            print(f"      [!!] Zu wenig valide Daten: {len(cv_signal)}")
            return LagCorrectionResult(
                vp_id=vp_id, config_key=config_key, method=method,
                optimal_lag_ms=0, correlation_at_lag=0, lag_valid=False
            )
        
        # Normalisiere Signale
        cv_signal = (cv_signal - np.mean(cv_signal)) / np.std(cv_signal)
        el_signal = (el_signal - np.mean(el_signal)) / np.std(el_signal)
        
        # Cross-Correlation
        correlation = signal.correlate(cv_signal, el_signal, mode='full')
        lags = signal.correlation_lags(len(cv_signal), len(el_signal), mode='full')
        
        # Berechne Lag in ms (basierend auf Sample-Rate)
        sample_rate_hz = len(merged) / ((merged['timestamp_ms'].max() - 
                                         merged['timestamp_ms'].min()) / 1000)
        lag_ms = lags / sample_rate_hz * 1000
        
        # Suche im erwarteten Bereich
        search_mask = (lag_ms >= LAG_SEARCH_RANGE_MS[0]) & (lag_ms <= LAG_SEARCH_RANGE_MS[1])
        
        if not search_mask.any():
            print(f"      [!!] Kein Lag im Suchbereich")
            return LagCorrectionResult(
                vp_id=vp_id, config_key=config_key, method=method,
                optimal_lag_ms=0, correlation_at_lag=0, lag_valid=False
            )
        
        # Finde Maximum
        correlation_normalized = correlation / len(cv_signal)
        search_corr = correlation_normalized[search_mask]
        search_lags = lag_ms[search_mask]
        
        max_idx = np.argmax(search_corr)
        optimal_lag = search_lags[max_idx]
        max_corr = search_corr[max_idx]
        
        # Keine Validierung mehr - Lag wird immer dokumentiert
        # Entscheidung über Anwendung erfolgt in R
        
        print(f"      Optimaler Lag: {optimal_lag:.1f} ms")
        print(f"      Korrelation: {max_corr:.3f}")
        
        # Info-Ausgabe wenn außerhalb typischem Bereich (nur zur Info, keine Filterung)
        if optimal_lag < EXPECTED_LAG_RANGE_MS[0] or optimal_lag > EXPECTED_LAG_RANGE_MS[1]:
            print(f"      [Info] Lag außerhalb typischem Bereich "
                  f"({EXPECTED_LAG_RANGE_MS[0]}-{EXPECTED_LAG_RANGE_MS[1]} ms)")
        
        return LagCorrectionResult(
            vp_id=vp_id,
            config_key=config_key,
            method=method,
            optimal_lag_ms=float(optimal_lag),
            correlation_at_lag=float(max_corr),
            lag_valid=True,  # Immer True - keine Filterung mehr
            correlation_curve=correlation_normalized[search_mask].tolist(),
            lag_values_ms=search_lags.tolist()
        )
    
    # =========================================================================
    # SCHRITT 5: FRAME-LEVEL MERGE
    # =========================================================================
    
    def merge_frame_level(self, mediapipe_df: pd.DataFrame,
                         ptgaze_df: pd.DataFrame,
                         eyelink_df: pd.DataFrame,
                         lag_mp: LagCorrectionResult,
                         lag_pt: LagCorrectionResult) -> pd.DataFrame:
        """
        Merged alle drei Datenquellen auf Frame-Level.
        
        Args:
            mediapipe_df: MediaPipe DataFrame
            ptgaze_df: ptgaze DataFrame
            eyelink_df: Downgesampletes EyeLink DataFrame
            lag_mp: Lag-Korrektur für MediaPipe
            lag_pt: Lag-Korrektur für ptgaze
        
        Returns:
            Gemergtes DataFrame im Long-Format
        
        Hinweis:
            Falls vorhanden, wird die `phase_type` Spalte aus debug_4 v3.3+
            übernommen ('fixation', 'stimulus', 'unassigned').
        """
        print(f"\n  [>] Frame-Level Merge...")
        
        results = []
        
        # Basis: Timestamps aus MediaPipe (oder ptgaze falls MP fehlt)
        if mediapipe_df is not None:
            base_timestamps = mediapipe_df['timestamp_ms_synced'].values
        elif ptgaze_df is not None:
            base_timestamps = ptgaze_df['timestamp_ms_synced'].values
        else:
            print(f"      [!!] Keine CV-Daten vorhanden!")
            return pd.DataFrame()
        
        # ─────────────────────────────────────────────────────────────────
        # MediaPipe Daten
        # ─────────────────────────────────────────────────────────────────
        
        if mediapipe_df is not None:
            mp_data = mediapipe_df.copy()
            
            # Lag-korrigierter Timestamp berechnen (für R-Analyse)
            # Wird NICHT für den Merge verwendet, nur dokumentiert
            mp_data['timestamp_ms_lag_corrected'] = (
                mp_data['timestamp_ms_synced'] - lag_mp.optimal_lag_ms
            )
            
            mp_data['method'] = 'mediapipe'
            mp_data['lag_applied_ms'] = lag_mp.optimal_lag_ms  # Immer dokumentieren
            
            # Relevante Spalten (inkl. phase_type falls vorhanden)
            base_cols = [
                'frame', 'timestamp_ms_synced', 'timestamp_ms_lag_corrected',
                'trial_assignment', 'method', 'lag_applied_ms',
                'gaze_deg_x_calib', 'gaze_deg_y_calib',
                'confidence', 'is_blink', 'eyes_closed', 'outside_monitor',
                'plausibility_check'
            ]
            
            # phase_type hinzufügen falls vorhanden (NEU ab debug_4 v3.3)
            if 'phase_type' in mp_data.columns:
                base_cols.append('phase_type')
            
            mp_export = mp_data[base_cols].copy()
            
            mp_export.rename(columns={
                'gaze_deg_x_calib': 'cv_deg_x',
                'gaze_deg_y_calib': 'cv_deg_y',
                'confidence': 'cv_confidence'
            }, inplace=True)
            
            results.append(mp_export)
        
        # ─────────────────────────────────────────────────────────────────
        # ptgaze Daten
        # ─────────────────────────────────────────────────────────────────
        
        if ptgaze_df is not None:
            pt_data = ptgaze_df.copy()
            
            # Lag-korrigierter Timestamp berechnen (für R-Analyse)
            # Wird NICHT für den Merge verwendet, nur dokumentiert
            pt_data['timestamp_ms_lag_corrected'] = (
                pt_data['timestamp_ms_synced'] - lag_pt.optimal_lag_ms
            )
            
            pt_data['method'] = 'ptgaze'
            pt_data['lag_applied_ms'] = lag_pt.optimal_lag_ms  # Immer dokumentieren
            
            # Relevante Spalten (inkl. phase_type falls vorhanden)
            base_cols = [
                'frame', 'timestamp_ms_synced', 'timestamp_ms_lag_corrected',
                'trial_assignment', 'method', 'lag_applied_ms',
                'gaze_deg_x_calib', 'gaze_deg_y_calib',
                'confidence', 'is_blink', 'eyes_closed', 'outside_monitor',
                'plausibility_check'
            ]
            
            # phase_type hinzufügen falls vorhanden (NEU ab debug_4 v3.3)
            if 'phase_type' in pt_data.columns:
                base_cols.append('phase_type')
            
            pt_export = pt_data[base_cols].copy()
            
            pt_export.rename(columns={
                'gaze_deg_x_calib': 'cv_deg_x',
                'gaze_deg_y_calib': 'cv_deg_y',
                'confidence': 'cv_confidence'
            }, inplace=True)
            
            results.append(pt_export)
        
        # Kombiniere
        if not results:
            return pd.DataFrame()
        
        combined = pd.concat(results, ignore_index=True)
        
        # ─────────────────────────────────────────────────────────────────
        # EyeLink Daten mergen
        # ─────────────────────────────────────────────────────────────────
        
        if eyelink_df is not None and not eyelink_df.empty:
            # FIX: Merge auf timestamp_ms_synced (nicht lag_corrected!)
            # EyeLink wurde auf diese Timestamps downgesampelt
            combined = pd.merge(
                combined,
                eyelink_df[['timestamp_ms', 'eyelink_deg_x', 'eyelink_deg_y', 'eyelink_valid']],
                left_on='timestamp_ms_synced',
                right_on='timestamp_ms',
                how='left',
                suffixes=('', '_el')
            )
            
            # Bereinige doppelte timestamp_ms Spalte
            if 'timestamp_ms' in combined.columns and 'timestamp_ms_synced' in combined.columns:
                combined.drop(columns=['timestamp_ms'], inplace=True, errors='ignore')
        else:
            combined['eyelink_deg_x'] = np.nan
            combined['eyelink_deg_y'] = np.nan
            combined['eyelink_valid'] = False
        
        # Statistik
        n_with_eyelink = combined['eyelink_deg_x'].notna().sum()
        print(f"      Ergebnis: {len(combined)} Zeilen "
              f"({combined['method'].nunique()} Methoden)")
        print(f"      Mit EyeLink-Daten: {n_with_eyelink}/{len(combined)} "
              f"({n_with_eyelink/len(combined)*100:.1f}%)")
        
        # Phase-Statistik (falls vorhanden)
        if 'phase_type' in combined.columns:
            n_fixation = (combined['phase_type'] == 'fixation').sum()
            n_stimulus = (combined['phase_type'] == 'stimulus').sum()
            n_unassigned = (combined['phase_type'] == 'unassigned').sum()
            print(f"      Phasen: Fixation={n_fixation}, Stimulus={n_stimulus}, Unassigned={n_unassigned}")
        
        return combined
    
    # =========================================================================
    # SCHRITT 6: JITTER-FILTER (optional, einheitlich)
    # =========================================================================
    
    def apply_jitter_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Wendet einheitlichen Jitter-Filter auf beide CV-Methoden an.
        
        WICHTIG: Nur wenn ENABLE_UNIFIED_JITTER_FILTER = True
        """
        if not ENABLE_UNIFIED_JITTER_FILTER:
            print(f"\n  [>] Jitter-Filter: DEAKTIVIERT")
            df['cv_deg_x_filtered'] = df['cv_deg_x']
            df['cv_deg_y_filtered'] = df['cv_deg_y']
            return df
        
        print(f"\n  [>] Jitter-Filter anwenden (α={JITTER_FILTER_ALPHA}, β={JITTER_FILTER_BETA})...")
        
        df = df.copy()
        
        # Pro Methode und Trial separat filtern
        for method in df['method'].unique():
            for trial in df['trial_assignment'].unique():
                mask = (df['method'] == method) & (df['trial_assignment'] == trial)
                
                if mask.sum() == 0:
                    continue
                
                # Filter für X
                filter_x = DoubleExponentialFilter(JITTER_FILTER_ALPHA, JITTER_FILTER_BETA)
                df.loc[mask, 'cv_deg_x_filtered'] = filter_x.filter_series(df.loc[mask, 'cv_deg_x'])
                
                # Filter für Y
                filter_y = DoubleExponentialFilter(JITTER_FILTER_ALPHA, JITTER_FILTER_BETA)
                df.loc[mask, 'cv_deg_y_filtered'] = filter_y.filter_series(df.loc[mask, 'cv_deg_y'])
        
        # Statistik
        for method in df['method'].unique():
            mask = df['method'] == method
            std_before = df.loc[mask, 'cv_deg_x'].std()
            std_after = df.loc[mask, 'cv_deg_x_filtered'].std()
            reduction = (1 - std_after / std_before) * 100 if std_before > 0 else 0
            print(f"      {method}: Jitter-Reduktion X = {reduction:.1f}%")
        
        return df
    
    # =========================================================================
    # SCHRITT 7: AUSSCHLUSS-FLAGS
    # =========================================================================
    
    def set_exclusion_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Setzt Ausschluss-Flags (wendet sie NICHT an!).
        
        Flags:
        - exclude_blink: Blink oder Augen geschlossen
        - exclude_outside: Außerhalb Monitor
        - exclude_any: Mindestens ein Ausschlussgrund
        
        HINWEIS: Confidence-basierter Ausschluss ist DEAKTIVIERT
                 (MediaPipe-Confidence Bug wird später behoben)
        """
        print(f"\n  [>] Ausschluss-Flags setzen...")
        
        df = df.copy()
        
        # Blink-Flag (Blink oder Augen geschlossen)
        df['exclude_blink'] = df['is_blink'] | df['eyes_closed']
        
        # Outside Monitor
        df['exclude_outside'] = df['outside_monitor']
        
        # Confidence-Ausschluss DEAKTIVIERT
        # df['exclude_low_confidence'] = df['cv_confidence'] < CONFIDENCE_THRESHOLD
        df['exclude_low_confidence'] = False  # Immer False
        
        # Implausible (Plausibilitäts-Check)
        if 'plausibility_check' in df.columns:
            df['exclude_implausible'] = ~df['plausibility_check']
        else:
            df['exclude_implausible'] = False
        
        # Kombiniert (OHNE Confidence!)
        df['exclude_any'] = (
            df['exclude_blink'] |
            df['exclude_outside'] |
            df['exclude_implausible']
        )
        
        # Statistik
        n_total = len(df)
        n_blink = df['exclude_blink'].sum()
        n_outside = df['exclude_outside'].sum()
        n_implausible = df['exclude_implausible'].sum()
        n_any = df['exclude_any'].sum()
        
        print(f"      Blink/Eyes Closed: {n_blink} ({n_blink/n_total*100:.1f}%)")
        print(f"      Outside Monitor: {n_outside} ({n_outside/n_total*100:.1f}%)")
        print(f"      Implausible: {n_implausible} ({n_implausible/n_total*100:.1f}%)")
        print(f"      Confidence: DEAKTIVIERT (Bug-Fix pending)")
        print(f"      Gesamt ausschließbar: {n_any} ({n_any/n_total*100:.1f}%)")
        
        return df
    
    # =========================================================================
    # HAUPTMETHODE: PROCESS VP
    # =========================================================================
    
    def process_vp(self, run_folder: Path, vp_id: str, 
                  video_fps: str, calib_mode: str,
                  eyelink_cache: 'EyeLinkCache' = None) -> DataPrepResult:
        """
        Verarbeitet eine VP/Config komplett.
        
        Args:
            run_folder: Pfad zum Run_* Ordner
            vp_id: VP-ID (z.B. 'abc1')
            video_fps: '25hz' oder '60hz'
            calib_mode: z.B. 'FullCalib'
            eyelink_cache: Optionaler zentraler EyeLink-Cache
        
        Returns:
            DataPrepResult
        """
        config_key = f"{video_fps}_{calib_mode}"
        
        print(f"\n{'='*70}")
        print(f"VERARBEITE: {vp_id} / {config_key}")
        print(f"{'='*70}")
        
        result = DataPrepResult(
            vp_id=vp_id,
            video_fps=video_fps,
            calib_mode=calib_mode,
            frame_level_df=pd.DataFrame()
        )
        
        try:
            # 1. CV-Daten laden (MediaPipe, ptgaze)
            data = self._load_cv_data(run_folder)
            
            if data['mediapipe'] is None and data['ptgaze'] is None:
                result.error_message = "Keine CV-Daten gefunden"
                return result
            
            # 2. EyeLink-Daten (aus Cache oder direkt laden)
            if eyelink_cache is not None:
                # Zentraler Cache: Lädt und konvertiert nur einmal pro VP
                target_ts = (data['mediapipe'] if data['mediapipe'] is not None 
                            else data['ptgaze'])['timestamp_ms_synced'].values
                
                data['eyelink_downsampled'] = eyelink_cache.get_downsampled(
                    vp_id, target_ts, video_fps, run_folder
                )
            else:
                # Fallback: Direkt laden (alte Methode)
                eyelink_raw = self._load_eyelink(run_folder)
                
                if eyelink_raw is not None:
                    eyelink_raw = self.convert_eyelink_to_degrees(eyelink_raw)
                    
                    target_ts = (data['mediapipe'] if data['mediapipe'] is not None 
                                else data['ptgaze'])['timestamp_ms_synced'].values
                    data['eyelink_downsampled'] = self.downsample_eyelink(
                        eyelink_raw, target_ts
                    )
                else:
                    data['eyelink_downsampled'] = None
            
            # 3. Lag-Korrektur
            if data['mediapipe'] is not None and data['eyelink_downsampled'] is not None:
                result.lag_mediapipe = self.compute_lag_correction(
                    data['mediapipe'], data['eyelink_downsampled'],
                    'mediapipe', vp_id, config_key
                )
            else:
                result.lag_mediapipe = LagCorrectionResult(
                    vp_id=vp_id, config_key=config_key, method='mediapipe',
                    optimal_lag_ms=0, correlation_at_lag=0, lag_valid=False
                )
            
            if data['ptgaze'] is not None and data['eyelink_downsampled'] is not None:
                result.lag_ptgaze = self.compute_lag_correction(
                    data['ptgaze'], data['eyelink_downsampled'],
                    'ptgaze', vp_id, config_key
                )
            else:
                result.lag_ptgaze = LagCorrectionResult(
                    vp_id=vp_id, config_key=config_key, method='ptgaze',
                    optimal_lag_ms=0, correlation_at_lag=0, lag_valid=False
                )
            
            # 4. Frame-Level Merge
            merged = self.merge_frame_level(
                data['mediapipe'], data['ptgaze'], data['eyelink_downsampled'],
                result.lag_mediapipe, result.lag_ptgaze
            )
            
            if merged.empty:
                result.error_message = "Merge fehlgeschlagen"
                return result
            
            # 5. Jitter-Filter (optional)
            merged = self.apply_jitter_filter(merged)
            
            # 6. Ausschluss-Flags
            merged = self.set_exclusion_flags(merged)
            
            # 7. VP-Info hinzufügen
            merged['vp_id'] = vp_id
            merged['video_fps'] = video_fps
            merged['calib_config'] = calib_mode
            
            # Practice-Trial markieren
            merged['is_practice'] = merged['trial_assignment'] == PRACTICE_TRIAL_NUMBER
            
            # 8. Stimulus-Info hinzufügen
            vp_main_folder = run_folder.parent.parent
            stimulus_info = extract_stimulus_info(vp_main_folder, vp_id)
            
            if stimulus_info is not None:
                stimulus_mapping = merged['trial_assignment'].apply(
                    lambda t: map_trial_to_stimulus(t, stimulus_info)
                )
                merged['stimulus_id'] = stimulus_mapping.apply(lambda x: x[0])
                merged['flip_status'] = stimulus_mapping.apply(lambda x: x[1])
                
                print(f"  [OK] Stimulus-Info hinzugefügt (Quelle: {stimulus_info['source']})")
                print(f"       Stimuli: {merged['stimulus_id'].dropna().nunique()} verschiedene")
            else:
                merged['stimulus_id'] = np.nan
                merged['flip_status'] = np.nan
                result.warnings.append(f"Keine Stimulus-Info für VP {vp_id}")
            
            # Ergebnis
            result.frame_level_df = merged
            result.n_frames_total = len(merged)
            result.n_frames_valid = (~merged['exclude_any']).sum()
            result.n_trials = merged['trial_assignment'].nunique()
            result.success = True
            
            print(f"\n  [OK] Verarbeitung erfolgreich!")
            print(f"      Frames: {result.n_frames_total}")
            print(f"      Valide: {result.n_frames_valid} ({result.n_frames_valid/result.n_frames_total*100:.1f}%)")
            print(f"      Trials: {result.n_trials}")
            
        except Exception as e:
            result.error_message = str(e)
            print(f"\n  [!!] FEHLER: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def _load_cv_data(self, run_folder: Path) -> Dict[str, pd.DataFrame]:
        """Lädt nur CV-Daten (MediaPipe, ptgaze)."""
        data = {}
        
        mp_path = run_folder / 'debug_5_pupil_data_calibrated.csv'
        if mp_path.exists():
            data['mediapipe'] = pd.read_csv(mp_path)
            print(f"  [OK] MediaPipe: {len(data['mediapipe'])} Frames")
        else:
            print(f"  [!!] MediaPipe nicht gefunden")
            data['mediapipe'] = None
        
        pt_path = run_folder / 'debug_5_ptgaze_calibrated.csv'
        if pt_path.exists():
            data['ptgaze'] = pd.read_csv(pt_path)
            print(f"  [OK] ptgaze: {len(data['ptgaze'])} Frames")
        else:
            print(f"  [!!] ptgaze nicht gefunden")
            data['ptgaze'] = None
        
        return data
    
    def _load_eyelink(self, run_folder: Path) -> Optional[pd.DataFrame]:
        """Lädt EyeLink-Daten (Fallback ohne Cache)."""
        el_path = run_folder / 'debug_3_eyetracker_data.csv'
        if el_path.exists():
            df = pd.read_csv(el_path)
            print(f"  [OK] EyeLink: {len(df)} Samples")
            return df
        else:
            print(f"  [!!] EyeLink nicht gefunden")
            return None

# =================================================================================
# MAIN (TEST)
# =================================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STATISTICAL DATA PREP v1.0 - TEST")
    print("="*70)
    
    # Test mit Beispiel-Pfad
    test_run = Path(r"C:\Users\...\Ergebnisse\abc1\Analyse\Run_25hz_FullCalib_mode2_001")
    
    if test_run.exists():
        prep = StatisticalDataPrep()
        result = prep.process_vp(test_run, 'abc1', '25hz', 'FullCalib')
        
        if result.success:
            print(f"\n[OK] Test erfolgreich!")
            print(f"    Spalten: {list(result.frame_level_df.columns)}")
        else:
            print(f"\n[!!] Test fehlgeschlagen: {result.error_message}")
    else:
        print(f"\n[!] Test-Ordner nicht gefunden")
        print(f"    Bitte Pfad anpassen oder Modul importieren")
