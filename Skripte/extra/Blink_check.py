#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
ADAPTIVE BLINK-ANNOTATION TOOL v2.0 - OPTIMIERT
================================================================================
Mit zusätzlichen Validierungsmechanismen gegen Falsch-Positive:
- Minimale UND maximale Blink-Dauer
- EAR-Percentil als zusätzliches Kriterium
- Refractory Period zwischen Blinks
- Verbesserte HMM-Initialisierung
- Optionale Visualisierung zur Diagnose
================================================================================
"""

import os
import re
import glob
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import binary_dilation, binary_erosion

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("⚠️  hmmlearn nicht installiert!")

# Optional: Visualisierung
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

warnings.filterwarnings('ignore')

# ================================================================================
# KONFIGURATION - HIER PARAMETER ANPASSEN!
# ================================================================================

# Basispfad
BASE_PATH = r"C:\Users\imanu\Documents\Imanuel\Studium\Psy\Master\Pupillendetektion\Ergebnisse"

# === EXKLUSIONSZEITEN ===
EXCLUSION_MS_BEFORE = 80  # ms VOR Blink
EXCLUSION_MS_AFTER = 80   # ms NACH Blink

# === BLINK-DAUER FILTER (WICHTIG!) ===
MIN_BLINK_DURATION_MS = 40    # Minimum
MAX_BLINK_DURATION_MS = 2000  # Maximum

# === REFRACTORY PERIOD ===
# Minimaler Abstand zwischen zwei Blinks (verhindert "Blink-Cluster")
MIN_INTER_BLINK_INTERVAL_MS = 200

# === EAR-PERCENTIL VALIDIERUNG (SEHR WICHTIG!) ===
# Ein Frame gilt nur als Blink, wenn der EAR-Wert unter diesem Percentil liegt
# z.B. 15 bedeutet: EAR muss zu den niedrigsten 15% gehören
EAR_PERCENTILE_THRESHOLD = 10  # Erhöhen = weniger Blinks, Senken = mehr Blinks

# === ALTERNATIVE: Fester EAR-Multiplikator ===
# Blink nur wenn EAR < (Median * Multiplikator)
# z.B. 0.7 = EAR muss unter 70% des Median-Werts liegen
USE_MEDIAN_MULTIPLIER = True  # True = Median-Methode, False = Percentil-Methode

# === EAR-MULTIPLIKATOR PRO FRAMERATE (manuelle Anpassung pro VP)===
# ogt7 25HZ = 0.74 60HZ = 0.785
# oem4 25HZ = 0.74 60HZ = 0.74
# mhe9 25HZ = 0.78 60HZ = 0.78
# ldj9 25HZ = 0.81 60HZ = 0.81
# kdn8 25HZ = 0.83 60HZ = 0.81
# jkl7 25HZ = 0.72 60HZ = 0.72
# fgt6 25HZ = 0.78 60HZ = 0.76
# fbn6 25HZ = 0.88 60HZ = 0.74
# egf5 25HZ = 0.66 60HZ = 0.48
# bjs4 25HZ = 0.71 60HZ = 0.74
# beo7 25HZ = 0.805 60HZ = 0.74
# kro3 25HZ = 0.805 60HZ = 0.74

EAR_MEDIAN_MULTIPLIER_25HZ = 0.74 #Anpassen
EAR_MEDIAN_MULTIPLIER_60HZ = 0.785 #Anpassen

# === SAVITZKY-GOLAY FILTER ===
SAVGOL_WINDOW = 11
SAVGOL_POLY = 3

# === HMM PARAMETER ===
HMM_N_STATES = 2
HMM_N_ITER = 100
HMM_COVARIANCE_TYPE = "full"  # "full" ist robuster als "diag"

# === HYBRIDMODUS ===
# Kombiniert HMM mit Peak-Detection für bessere Genauigkeit
USE_HYBRID_DETECTION = True

# === VISUALISIERUNG ===
# Erstellt Diagnose-Plots für jede Datei
CREATE_DIAGNOSTIC_PLOTS = True  # Auf False setzen wenn nicht benötigt
PLOT_OUTPUT_DIR = os.path.join(BASE_PATH, "_diagnostic_plots")

# === DATEIEN ===
TARGET_CSV_FILES = [
    "debug_5_ptgaze_calibrated.csv",
    "debug_5_pupil_data_calibrated.csv"
]

CREATE_BACKUP = True

# ================================================================================
# LOGGING
# ================================================================================

def setup_logging():
    log_dir = os.path.join(BASE_PATH, "_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"blink_annotation_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ================================================================================
# HILFSFUNKTIONEN
# ================================================================================

def get_vp_ids(base_path: str) -> List[str]:
    pattern = re.compile(r'^[a-zA-Z]{3}\d$')
    vp_ids = []
    if not os.path.exists(base_path):
        return vp_ids
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and pattern.match(item):
            vp_ids.append(item)
    return sorted(vp_ids)


def find_csv_files(base_path: str, vp_id: str) -> List[str]:
    csv_files = []
    vp_analyse_path = os.path.join(base_path, vp_id, "Analyse")
    if not os.path.exists(vp_analyse_path):
        return csv_files
    
    run_patterns = [
        "Run_25hz_BegFirst10EndFix_mode2_*",
        "Run_25hz_FullCalib_mode2_*",
        "Run_60hz_BegFirst10EndFix_mode2_*",
        "Run_60hz_FullCalib_mode2_*"
    ]
    
    for pattern in run_patterns:
        for run_dir in glob.glob(os.path.join(vp_analyse_path, pattern)):
            for target_file in TARGET_CSV_FILES:
                csv_path = os.path.join(run_dir, target_file)
                if os.path.exists(csv_path):
                    csv_files.append(csv_path)
    return csv_files


def is_already_processed(csv_path: str) -> bool:
    try:
        df = pd.read_csv(csv_path, nrows=10)
        if 'blink_source' in df.columns:
            return df['blink_source'].str.contains('hmm_adaptive_v2', na=False).any()
    except Exception:
        pass
    return False


def calculate_framerate(timestamps_ms: np.ndarray) -> float:
    if len(timestamps_ms) < 2:
        return 30.0
    diffs = np.diff(timestamps_ms)
    valid_diffs = diffs[(diffs > 0) & (diffs < 200)]
    if len(valid_diffs) == 0:
        return 30.0
    return 1000.0 / np.median(valid_diffs)


# ================================================================================
# SIGNALVERARBEITUNG
# ================================================================================

def smooth_ear_signal(ear_values: np.ndarray, window: int = SAVGOL_WINDOW, 
                      poly: int = SAVGOL_POLY) -> np.ndarray:
    """Glättet das EAR-Signal mit Savitzky-Golay-Filter"""
    smoothed = ear_values.copy().astype(float)
    valid_mask = ~np.isnan(smoothed) & np.isfinite(smoothed)
    
    if np.sum(valid_mask) <= window:
        return smoothed
    
    if not np.all(valid_mask):
        indices = np.arange(len(smoothed))
        smoothed = np.interp(indices, indices[valid_mask], smoothed[valid_mask])
    
    try:
        smoothed = savgol_filter(smoothed, window, poly)
    except Exception as e:
        logger.warning(f"Savitzky-Golay-Filter fehlgeschlagen: {e}")
    
    return smoothed


def calculate_adaptive_threshold(ear_values: np.ndarray, framerate: float) -> Tuple[float, float, float]:
    """
    Berechnet adaptive Schwellenwerte basierend auf der EAR-Verteilung.
    Verwendet unterschiedliche Multiplikatoren für 25Hz und 60Hz.
    
    Returns:
        percentile_threshold, median_threshold, multiplier_used
    """
    valid_ear = ear_values[~np.isnan(ear_values) & np.isfinite(ear_values)]
    
    if len(valid_ear) == 0:
        return 0.2, 0.2, 0.0
    
    # Wähle Multiplier basierend auf Framerate
    if framerate < 35:  # 25Hz Kamera
        multiplier = EAR_MEDIAN_MULTIPLIER_25HZ
    else:  # 60Hz Kamera
        multiplier = EAR_MEDIAN_MULTIPLIER_60HZ
    
    percentile_threshold = np.percentile(valid_ear, EAR_PERCENTILE_THRESHOLD)
    median_threshold = np.median(valid_ear) * multiplier
    
    return percentile_threshold, median_threshold, multiplier


def fit_hmm_blink_detection(ear_smoothed: np.ndarray) -> np.ndarray:
    """Fittet HMM mit verbesserter Initialisierung"""
    if not HMM_AVAILABLE:
        return np.zeros(len(ear_smoothed), dtype=int)
    
    n = len(ear_smoothed)
    valid_mask = ~np.isnan(ear_smoothed) & np.isfinite(ear_smoothed)
    valid_indices = np.where(valid_mask)[0]
    ear_valid = ear_smoothed[valid_mask].reshape(-1, 1)
    
    if len(ear_valid) < 50:
        return np.zeros(n, dtype=int)
    
    # Bessere Initialisierung basierend auf Percentilen
    ear_median = np.median(ear_valid)
    ear_std = np.std(ear_valid)
    
    # Initiale Mittelwerte: oben (75. Percentil) und unten (25. Percentil)
    init_means = np.array([
        [np.percentile(ear_valid, 75)],  # Zustand 0: Augen offen
        [np.percentile(ear_valid, 25)]   # Zustand 1: Augen geschlossen
    ])
    
    model = hmm.GaussianHMM(
        n_components=HMM_N_STATES,
        covariance_type=HMM_COVARIANCE_TYPE,
        n_iter=HMM_N_ITER,
        random_state=42,
        init_params="c",  # Nicht die Mittelwerte initialisieren
        params="stmc"
    )
    
    # Setze initiale Mittelwerte
    model.means_ = init_means
    
    # Setze Übergangswahrscheinlichkeiten (Blinks sind selten!)
    # Hohe Wahrscheinlichkeit im gleichen Zustand zu bleiben
    model.transmat_ = np.array([
        [0.99, 0.01],  # Von offen: 99.5% bleiben offen, 0.5% schließen
        [0.03, 0.97]     # Von geschlossen: 5% öffnen, 95% bleiben geschlossen
    ])
    
    model.startprob_ = np.array([0.99, 0.01])  # Fast immer mit offenen Augen starten
    
    try:
        model.fit(ear_valid)
        states_valid = model.predict(ear_valid)
        
        means = model.means_.flatten()
        closed_state = np.argmin(means)
        
        logger.info(f"  HMM-Mittelwerte: offen={means[1-closed_state]:.4f}, geschlossen={means[closed_state]:.4f}")
        
        states_valid = (states_valid == closed_state).astype(int)
        
        states = np.zeros(n, dtype=int)
        states[valid_indices] = states_valid
        
        return states
        
    except Exception as e:
        logger.error(f"HMM-Fitting fehlgeschlagen: {e}")
        return np.zeros(n, dtype=int)


def detect_blinks_hybrid(ear_smoothed: np.ndarray, 
                         timestamps_ms: np.ndarray,
                         ear_threshold: float) -> np.ndarray:
    """
    Hybride Blink-Detektion: Kombiniert HMM mit Peak-Detection und EAR-Validierung.
    
    Diese Methode ist robuster gegen Falsch-Positive.
    """
    n = len(ear_smoothed)
    
    # 1. HMM-basierte Detektion
    hmm_states = fit_hmm_blink_detection(ear_smoothed)
    
    # 2. Peak-Detection auf invertiertem Signal (Blinks = Dips im EAR)
    inverted_ear = -ear_smoothed
    inverted_ear = np.nan_to_num(inverted_ear, nan=np.nanmin(inverted_ear))
    
    # Finde Peaks (= Dips im Original)
    peaks, properties = find_peaks(
        inverted_ear,
        height=np.percentile(inverted_ear, 85),  # Nur signifikante Dips
        distance=int(MIN_INTER_BLINK_INTERVAL_MS / np.median(np.diff(timestamps_ms))),
        prominence=0.02  # Mindest-Prominenz
    )
    
    # 3. Kombiniere beide Methoden
    # Ein Blink-Frame muss BEIDE Kriterien erfüllen:
    # - HMM sagt "geschlossen" ODER in der Nähe eines Peaks
    # - EAR unter dem adaptiven Threshold
    
    combined_states = np.zeros(n, dtype=int)
    
    # Markiere Bereiche um Peaks
    peak_window_frames = 5  # Frames um jeden Peak
    for peak in peaks:
        start = max(0, peak - peak_window_frames)
        end = min(n, peak + peak_window_frames + 1)
        combined_states[start:end] = 1
    
    # Kombiniere mit HMM
    combined_states = np.maximum(combined_states, hmm_states)
    
    # 4. EAR-Validierung: Nur Frames mit niedrigem EAR behalten
    ear_valid_mask = ear_smoothed < ear_threshold
    combined_states = combined_states & ear_valid_mask.astype(int)
    
    return combined_states


def extract_and_validate_blinks(states: np.ndarray, 
                                 timestamps_ms: np.ndarray,
                                 ear_values: np.ndarray,
                                 ear_threshold: float) -> Dict:
    """
    Extrahiert Blink-Events mit umfassender Validierung.
    
    Validierungskriterien:
    1. Minimale Blink-Dauer
    2. Maximale Blink-Dauer
    3. Minimaler Abstand zwischen Blinks
    4. EAR muss unter Threshold liegen
    """
    n = len(states)
    is_blink = np.zeros(n, dtype=bool)
    eyes_closed = np.zeros(n, dtype=bool)
    blink_count = np.zeros(n, dtype=int)
    
    # Finde zusammenhängende Blink-Regionen
    state_diff = np.diff(states, prepend=0, append=0)
    blink_starts = np.where(state_diff == 1)[0]
    blink_ends = np.where(state_diff == -1)[0]
    
    if len(blink_starts) == 0:
        return {
            'is_blink': is_blink, 
            'eyes_closed': eyes_closed, 
            'blink_count': blink_count,
            'n_blinks': 0,
            'blink_events': [],
            'rejected_blinks': {'too_short': 0, 'too_long': 0, 'too_close': 0, 'ear_too_high': 0}
        }
    
    # Stelle sicher, dass Start/Ende-Paare korrekt sind
    if len(blink_ends) == 0 or (len(blink_starts) > 0 and blink_ends[0] < blink_starts[0]):
        if len(blink_ends) > 0:
            blink_ends = blink_ends[1:]
    
    if len(blink_starts) > len(blink_ends):
        blink_ends = np.append(blink_ends, n)
    
    min_pairs = min(len(blink_starts), len(blink_ends))
    blink_starts = blink_starts[:min_pairs]
    blink_ends = blink_ends[:min_pairs]
    
    # Validierung
    valid_blinks = []
    rejected = {'too_short': 0, 'too_long': 0, 'too_close': 0, 'ear_too_high': 0}
    last_blink_end_time = -np.inf
    
    for start_idx, end_idx in zip(blink_starts, blink_ends):
        end_idx = min(end_idx, n - 1)
        
        # Berechne Dauer
        blink_duration_ms = timestamps_ms[end_idx] - timestamps_ms[start_idx]
        
        # 1. Minimale Dauer
        if blink_duration_ms < MIN_BLINK_DURATION_MS:
            rejected['too_short'] += 1
            continue
        
        # 2. Maximale Dauer
        if blink_duration_ms > MAX_BLINK_DURATION_MS:
            rejected['too_long'] += 1
            continue
        
        # 3. Abstand zum vorherigen Blink
        if timestamps_ms[start_idx] - last_blink_end_time < MIN_INTER_BLINK_INTERVAL_MS:
            rejected['too_close'] += 1
            continue
        
        # 4. EAR-Validierung: Der mittlere EAR während des Blinks muss unter Threshold liegen
        blink_ear_values = ear_values[start_idx:end_idx+1]
        mean_blink_ear = np.nanmean(blink_ear_values)
        
        if mean_blink_ear > ear_threshold:
            rejected['ear_too_high'] += 1
            continue
        
        # Blink ist valide!
        valid_blinks.append((start_idx, end_idx))
        last_blink_end_time = timestamps_ms[end_idx]
        
        # Markiere Blink
        is_blink[start_idx:end_idx+1] = True
    
    # Exklusionszonen berechnen (150ms vor UND nach)
    for start_idx, end_idx in valid_blinks:
        start_time = timestamps_ms[start_idx]
        end_time = timestamps_ms[min(end_idx, n-1)]
        
        exclusion_start_time = start_time - EXCLUSION_MS_BEFORE
        exclusion_end_time = end_time + EXCLUSION_MS_AFTER
        
        exclusion_mask = (timestamps_ms >= exclusion_start_time) & (timestamps_ms <= exclusion_end_time)
        eyes_closed[exclusion_mask] = True
    
    # Blink-Count setzen
    current_count = 0
    for start_idx, end_idx in valid_blinks:
        current_count += 1
        if end_idx + 1 < n:
            blink_count[end_idx + 1:] = current_count
    
    logger.info(f"  Blink-Validierung: {len(valid_blinks)} akzeptiert, "
                f"{rejected['too_short']} zu kurz, {rejected['too_long']} zu lang, "
                f"{rejected['too_close']} zu nah, {rejected['ear_too_high']} EAR zu hoch")
    
    return {
        'is_blink': is_blink,
        'eyes_closed': eyes_closed,
        'blink_count': blink_count,
        'n_blinks': len(valid_blinks),
        'blink_events': valid_blinks,
        'rejected_blinks': rejected
    }


# ================================================================================
# VISUALISIERUNG (DIAGNOSE)
# ================================================================================

def create_diagnostic_plot(csv_path: str, 
                           timestamps_ms: np.ndarray,
                           ear_raw: np.ndarray,
                           ear_smoothed: np.ndarray,
                           blink_results: Dict,
                           ear_threshold: float,
                           output_dir: str):
    """Erstellt einen Diagnose-Plot für visuelle Inspektion"""
    
    if not MATPLOTLIB_AVAILABLE:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    
    # Zeitachse in Sekunden
    time_sec = (timestamps_ms - timestamps_ms[0]) / 1000
    
    # Plot 1: EAR-Signal mit Threshold
    ax1 = axes[0]
    ax1.plot(time_sec, ear_raw, alpha=0.5, label='EAR (roh)', color='lightblue')
    ax1.plot(time_sec, ear_smoothed, label='EAR (geglättet)', color='blue')
    ax1.axhline(y=ear_threshold, color='red', linestyle='--', label=f'Threshold ({ear_threshold:.3f})')
    ax1.set_ylabel('EAR')
    ax1.legend(loc='upper right')
    ax1.set_title('Eye Aspect Ratio (EAR) mit adaptivem Threshold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Blink-Markierungen
    ax2 = axes[1]
    ax2.plot(time_sec, ear_smoothed, color='blue', alpha=0.7)
    
    # Markiere Blinks
    for start_idx, end_idx in blink_results['blink_events']:
        ax2.axvspan(time_sec[start_idx], time_sec[min(end_idx, len(time_sec)-1)], 
                    color='red', alpha=0.3, label='Blink' if start_idx == blink_results['blink_events'][0][0] else '')
    
    ax2.set_ylabel('EAR')
    ax2.set_title(f"Erkannte Blinks: {blink_results['n_blinks']}")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Exklusionszonen
    ax3 = axes[2]
    ax3.fill_between(time_sec, 0, blink_results['eyes_closed'].astype(int), 
                     color='orange', alpha=0.5, label='Ausgeschlossen (eyes_closed)')
    ax3.fill_between(time_sec, 0, blink_results['is_blink'].astype(int) * 0.5, 
                     color='red', alpha=0.7, label='is_blink')
    ax3.set_ylabel('Status')
    ax3.set_xlabel('Zeit (Sekunden)')
    ax3.set_title('Exklusionszonen (±150ms um jeden Blink)')
    ax3.legend(loc='upper right')
    ax3.set_ylim(-0.1, 1.1)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Speichern
    vp_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(csv_path))))
    run_name = os.path.basename(os.path.dirname(csv_path))
    file_name = os.path.basename(csv_path).replace('.csv', '')
    
    plot_filename = f"{vp_id}_{run_name}_{file_name}_diagnostic.png"
    plot_path = os.path.join(output_dir, plot_filename)
    
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Diagnose-Plot: {plot_filename}")


# ================================================================================
# CSV VERARBEITUNG
# ================================================================================

def process_csv_file(csv_path: str, create_backup: bool = CREATE_BACKUP) -> bool:
    """Verarbeitet eine CSV-Datei mit allen Optimierungen"""
    
    logger.info(f"Verarbeite: {os.path.basename(csv_path)}")
    
    try:
        df = pd.read_csv(csv_path)
        n_rows = len(df)
        
        required_cols = ['timestamp_ms', 'avg_ear']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"  Fehlende Spalten: {missing_cols}")
            return False
        
        # Backup
        if create_backup:
            backup_dir = os.path.join(os.path.dirname(csv_path), "_backups")
            os.makedirs(backup_dir, exist_ok=True)
            backup_filename = os.path.basename(csv_path).replace(
                '.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            shutil.copy2(csv_path, os.path.join(backup_dir, backup_filename))
        
        # Daten extrahieren
        timestamps_ms = df['timestamp_ms'].values.astype(float)
        ear_raw = df['avg_ear'].values.astype(float)
        
        framerate = calculate_framerate(timestamps_ms)
        logger.info(f"  Framerate: {framerate:.1f} Hz")
        
        # EAR glätten
        ear_smoothed = smooth_ear_signal(ear_raw)
        
        # Adaptive Schwellenwerte berechnen (mit framerate-spezifischem Multiplier)
        percentile_threshold, median_threshold, multiplier_used = calculate_adaptive_threshold(ear_smoothed, framerate)
        
        if USE_MEDIAN_MULTIPLIER:
            ear_threshold = median_threshold
            logger.info(f"  EAR-Threshold (Median×{multiplier_used} @ {framerate:.0f}Hz): {ear_threshold:.4f}")
        else:
            ear_threshold = percentile_threshold
            logger.info(f"  EAR-Threshold ({EAR_PERCENTILE_THRESHOLD}. Perzentil): {ear_threshold:.4f}")
        
        # Blink-Detektion
        if USE_HYBRID_DETECTION:
            states = detect_blinks_hybrid(ear_smoothed, timestamps_ms, ear_threshold)
        else:
            states = fit_hmm_blink_detection(ear_smoothed)
        
        # Blinks extrahieren und validieren
        blink_results = extract_and_validate_blinks(
            states, timestamps_ms, ear_smoothed, ear_threshold
        )
        
        # Spalten aktualisieren
        df['is_blink'] = blink_results['is_blink']
        df['eyes_closed'] = blink_results['eyes_closed']
        df['blink_count'] = blink_results['blink_count']
        df['blink_source'] = 'hmm_adaptive_v2'
        
        # Speichern
        df.to_csv(csv_path, index=False)
        
        # Statistik
        n_blinks = blink_results['n_blinks']
        n_excluded = np.sum(blink_results['eyes_closed'])
        exclusion_pct = (n_excluded / n_rows * 100) if n_rows > 0 else 0
        
        logger.info(f"  ✓ Erkannte Blinks: {n_blinks}")
        logger.info(f"  ✓ Ausgeschlossene Frames: {n_excluded} ({exclusion_pct:.1f}%)")
        
        # Diagnose-Plot erstellen
        if CREATE_DIAGNOSTIC_PLOTS:
            create_diagnostic_plot(
                csv_path, timestamps_ms, ear_raw, ear_smoothed,
                blink_results, ear_threshold, PLOT_OUTPUT_DIR
            )
        
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Fehler: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


# ================================================================================
# BENUTZERINTERAKTION
# ================================================================================

def print_header():
    print("\n" + "="*70)
    print("   ADAPTIVE BLINK-ANNOTATION TOOL v2.0 - OPTIMIERT")
    print("="*70)
    print(f"   Exklusion: {EXCLUSION_MS_BEFORE}ms vor | {EXCLUSION_MS_AFTER}ms nach Blink")
    print(f"   Blink-Dauer: {MIN_BLINK_DURATION_MS}-{MAX_BLINK_DURATION_MS}ms")
    print(f"   EAR-Threshold: Median × {EAR_MEDIAN_MULTIPLIER_25HZ} (25Hz) / {EAR_MEDIAN_MULTIPLIER_60HZ} (60Hz)" if USE_MEDIAN_MULTIPLIER 
          else f"   EAR-Threshold: {EAR_PERCENTILE_THRESHOLD}. Perzentil")
    print(f"   Hybrid-Modus: {'Aktiviert' if USE_HYBRID_DETECTION else 'Deaktiviert'}")
    print(f"   Diagnose-Plots: {'Aktiviert' if CREATE_DIAGNOSTIC_PLOTS else 'Deaktiviert'}")
    print("="*70)


def select_processing_mode() -> Tuple[str, Optional[List[str]]]:
    print("\nVerarbeitungsmodus:")
    print("  [1] Alle VPs")
    print("  [2] Nur nicht verarbeitete")
    print("  [3] Einzelne VPs")
    print("  [q] Beenden")
    
    while True:
        choice = input("\n→ Auswahl: ").strip().lower()
        
        if choice == '1':
            return 'all', None
        elif choice == '2':
            return 'unprocessed', None
        elif choice == '3':
            vp_ids = get_vp_ids(BASE_PATH)
            print(f"\n  Verfügbare VPs: {', '.join(vp_ids)}")
            vp_input = input("  VP-IDs (kommagetrennt): ").strip()
            if vp_input:
                selected = [vp.strip().lower() for vp in vp_input.split(',')]
                valid = [vp for vp in selected if vp in vp_ids]
                if valid:
                    return 'single', valid
            print("  Keine gültigen VPs!")
        elif choice == 'q':
            return 'quit', None


def main():
    print_header()
    
    if not os.path.exists(BASE_PATH):
        print(f"\n✗ Basispfad nicht gefunden: {BASE_PATH}")
        return
    
    if not HMM_AVAILABLE:
        print("\n✗ hmmlearn nicht installiert!")
        return
    
    all_vps = get_vp_ids(BASE_PATH)
    print(f"\n  Gefundene VPs: {len(all_vps)}")
    
    mode, vp_list = select_processing_mode()
    
    if mode == 'quit':
        return
    
    vps_to_process = vp_list if mode == 'single' else all_vps
    
    stats = {'vps': 0, 'total': 0, 'success': 0, 'skip': 0, 'error': 0}
    
    for vp_id in vps_to_process:
        csv_files = find_csv_files(BASE_PATH, vp_id)
        if not csv_files:
            continue
        
        stats['vps'] += 1
        logger.info(f"\n{'─'*50}\nVP: {vp_id}\n{'─'*50}")
        
        for csv_path in csv_files:
            stats['total'] += 1
            
            if mode == 'unprocessed' and is_already_processed(csv_path):
                stats['skip'] += 1
                continue
            
            if process_csv_file(csv_path):
                stats['success'] += 1
            else:
                stats['error'] += 1
    
    print(f"\n{'='*70}")
    print(f"  FERTIG: {stats['success']}/{stats['total']} erfolgreich")
    if CREATE_DIAGNOSTIC_PLOTS:
        print(f"  Diagnose-Plots: {PLOT_OUTPUT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
