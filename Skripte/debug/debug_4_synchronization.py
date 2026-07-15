"""
=================================================================================
DEBUG 4: SYNCHRONISATION v3.2 (VEREINFACHT MIT phases_detected.json)
=================================================================================
NEU in v3.2:
------------
- BLINK-MERGE: MediaPipe Blink-Daten werden auf ptgaze uebertragen
- Grund: MediaPipe EAR-basierte Blink-Detection ist stabiler (besonders bei 25hz)
- Methode: Nearest-Neighbor (binary) / Linear Interpolation (continuous)

NEU in v3.1:
------------
- Package-Struktur (debug/)
- ptgaze-Synchronisation integriert (gleicher Offset!)

NEU in v3.0:
------------
- Nutzt Offset aus phases_detected.json (debug_0)
- KEINE Audio-Marker-Detektion mehr (bereits gemacht!)
- KEINE Block-Erkennung mehr (bereits gemacht!)
- KEINE Offset-Berechnung mehr (bereits gemacht!)
- ~70% weniger Code

Methode:
--------
synced_time = video_time + offset  (aus phases_detected.json)

Dann: Trial-Zuordnung aus phases_detected.json

WICHTIG:
--------
- Modus 1 (Standalone): debug_4 wird UEBERSPRUNGEN
- Modus 2/3 (Comparison): debug_4 wendet Offset an + ordnet Samples zu

Timeline:
---------
debug_0 >> debug_1 >> [debug_2] >> debug_3 >> debug_4 >> debug_5 >> debug_6
          |                                |
    phases_detected.json       Nutzt Offset aus phases_detected.json

Version: 3.1 (2025-01)
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
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import *

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

# ==================== HELPER: JSON SERIALIZATION ====================

def make_json_serializable(obj):
    """Konvertiert numpy-Typen zu Python-Standard (für json.dump)"""
    if isinstance(obj, (np.bool_, np.generic)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

# ==================== MAIN ====================

print(f"\n{'='*70}")
print("DEBUG 4: SYNCHRONISATION v3.0 (VEREINFACHT)")
print(f"{'='*70}\n")

if ANALYSIS_MODE == 1:
    print(" MODUS 1: Synchronisation nicht benötigt")
    print("\nNächster Schritt: debug_5_calibration.py")
    exit(0)

print(f"Modus: {ANALYSIS_MODE} (Webcam + EyeLink)")

# ==================== SCHRITT 1: PRÜFE phases_detected.json ====================

phases_json_path = Path(OUTPUT_BASE_DIR) / "phases_detected.json"

if not phases_json_path.exists():
    print(f"\n phases_detected.json nicht gefunden!")
    print(f"   >> Fallback: Legacy Workflow (debug_4 v2.5)")
    print(f"   >> EMPFEHLUNG: Laufe debug_0_phase_detection.py zuerst!")
    print(f"\n   Für optimalen Workflow:")
    print(f"   1. python debug_0_phase_detection.py")
    print(f"   2. python debug_1_video_analysis.py")
    print(f"   3. python debug_3_eyetracker_load.py")
    print(f"   4. python debug_4_synchronization.py (← du bist hier)")
    
    # Hier könnte Legacy-Code stehen (aus v2.5), aber wir forcieren Phase 1
    print(f"\n Abbruch! Laufe debug_0 zuerst.")
    exit(1)

print(f" phases_detected.json gefunden!")
print(f"   Nutze Phase 1 Workflow (KEINE Audio-Detektion)")

# ==================== SCHRITT 2: LADE phases_detected.json ====================

with open(phases_json_path, 'r') as f:
    phases = json.load(f)

sync_info = phases['sync_info']
offset_ms = sync_info['offset_ms']
sync_method = sync_info['method']
confidence = sync_info.get('confidence', 0.95)

print(f"\n Sync-Info aus debug_0:")
print(f"   Methode: {sync_method}")
print(f"   Offset: {offset_ms:.0f} ms")
print(f"   Confidence: {confidence:.2%}")

# Validierungs-Info (falls Audio gemacht wurde)
validation = sync_info.get('validation', {})
if validation:
    print(f"\n   Audio-Validierung:")
    print(f"   • Detection-Rate: {validation.get('detection_rate', 0)*100:.1f}%")
    print(f"   • Mean Deviation: {validation.get('mean_deviation_ms', 0):.1f} ms")

# ==================== SCHRITT 3: LADE PUPILLEN-DATEN ====================

print(f"\n{'='*70}")
print("LADE PUPILLEN-DATEN")
print(f"{'='*70}\n")

pupil_csv = os.path.join(OUTPUT_BASE_DIR, "debug_1_pupil_data.csv")
if not os.path.exists(pupil_csv):
    print(f" FEHLER: debug_1_video_analysis.py fehlt!")
    exit(1)

pupil_data = pd.read_csv(pupil_csv)
print(f"Geladen: {len(pupil_data)} Pupillen-Samples")
print(f"  Video-Zeit: {pupil_data['timestamp_ms'].min():.0f} - {pupil_data['timestamp_ms'].max():.0f} ms")

# Spalten-Erkennung (Rückwärtskompatibel!)
if 'avg_pupil_x_px_raw' in pupil_data.columns:
    x_col, y_col = 'avg_pupil_x_px_raw', 'avg_pupil_y_px_raw'
    print(f"   CSV-Format: v2.2")
elif 'avg_pupil_x_px_final' in pupil_data.columns:
    x_col, y_col = 'avg_pupil_x_px_final', 'avg_pupil_y_px'
    print(f"   CSV-Format: v2.1 (legacy)")
else:
    print(f"\n FEHLER: Keine Pupillenspalten!")
    exit(1)

# ==================== SCHRITT 3B: LADE PTGAZE-DATEN (NEU v3.1) ====================

print(f"\n{'='*70}")
print("LADE PTGAZE-DATEN")
print(f"{'='*70}\n")

ptgaze_csv = os.path.join(OUTPUT_BASE_DIR, "debug_1_ptgaze_data.csv")
ptgaze_data = None
ptgaze_available = False

if os.path.exists(ptgaze_csv):
    ptgaze_data = pd.read_csv(ptgaze_csv)
    ptgaze_available = True
    print(f"Geladen: {len(ptgaze_data)} ptgaze-Samples")
    print(f"  Video-Zeit: {ptgaze_data['timestamp_ms'].min():.0f} - {ptgaze_data['timestamp_ms'].max():.0f} ms")
    
    # Spalten-Check
    required_cols = ['timestamp_ms', 'gaze_pitch_deg', 'gaze_yaw_deg']
    missing_cols = [col for col in required_cols if col not in ptgaze_data.columns]
    if missing_cols:
        print(f"  [WARNUNG] Fehlende Spalten: {missing_cols}")
        print(f"  >> ptgaze-Sync wird uebersprungen")
        ptgaze_available = False
    else:
        print(f"  Spalten OK: gaze_pitch_deg, gaze_yaw_deg vorhanden")
else:
    print(f"[INFO] debug_1_ptgaze_data.csv nicht gefunden")
    print(f"  >> ptgaze-Sync wird uebersprungen (nur MediaPipe)")
    print(f"  >> Falls ptgaze gewuenscht: Laufe debug_1_ptgaze.py zuerst")

# ==================== SCHRITT 4: WENDE OFFSET AN (EINE ZEILE!) ====================

pupil_data_synced = pupil_data.copy()

#  NEU v2.2: Prüfe ob Dual-Anchor-Offsets vorhanden
if sync_method == 'dual_anchor' and 'offset_880hz_ms' in sync_info:
    # Dual-Anchor: Nutze Trial-spezifischen Offset
    offset_trials = sync_info['offset_880hz_ms']
    
    print(f" Dual-Anchor erkannt!")
    print(f"   Nutze Trial-Offset: {offset_trials:.0f} ms (880 Hz)")
    print(f"   (Kalibrierungs-Offset: {sync_info.get('offset_1760hz_ms', 0):.0f} ms)")
    
    # Wende Trial-Offset an (besser für Trial-Analyse!)
    pupil_data_synced['timestamp_ms_synced'] = pupil_data_synced['timestamp_ms'] + offset_trials
    offset_used = offset_trials
else:
    # Standard: Gemischter Offset
    pupil_data_synced['timestamp_ms_synced'] = pupil_data_synced['timestamp_ms'] + offset_ms
    offset_used = offset_ms
    print(f" Single-Offset: {offset_ms:.0f} ms")

pupil_data_synced['sync_method'] = sync_method
pupil_data_synced['sync_confidence'] = confidence

print(f"  {len(pupil_data_synced)} Samples synchronisiert")
print(f"  Neue Zeitachse: {pupil_data_synced['timestamp_ms_synced'].min():.0f} - {pupil_data_synced['timestamp_ms_synced'].max():.0f} ms")

# ==================== SCHRITT 4B: WENDE OFFSET AUF PTGAZE AN (NEU v3.1) ====================

ptgaze_data_synced = None

if ptgaze_available and ptgaze_data is not None:
    print(f"\n{'='*70}")
    print("SYNCHRONISIERE PTGAZE-DATEN")
    print(f"{'='*70}\n")
    
    ptgaze_data_synced = ptgaze_data.copy()
    
    # Wende GLEICHEN Offset an wie fuer MediaPipe!
    ptgaze_data_synced['timestamp_ms_synced'] = ptgaze_data_synced['timestamp_ms'] + offset_used
    ptgaze_data_synced['sync_method'] = sync_method
    ptgaze_data_synced['sync_confidence'] = confidence
    
    print(f"  Offset angewendet: {offset_used:.0f} ms (identisch zu MediaPipe)")
    print(f"  {len(ptgaze_data_synced)} Samples synchronisiert")
    print(f"  Neue Zeitachse: {ptgaze_data_synced['timestamp_ms_synced'].min():.0f} - {ptgaze_data_synced['timestamp_ms_synced'].max():.0f} ms")

# ==================== SCHRITT 5: TRIAL-ZUORDNUNG AUS phases_detected.json ====================

print(f"\n{'='*70}")
print("TRIAL-ZUORDNUNG")
print(f"{'='*70}\n")

# Extrahiere ALLE Trials aus phases_detected.json
all_trials = []

for block_key in ['experiment_block1', 'experiment_block2']:
    if block_key in phases['phases']:
        block_trials = phases['phases'][block_key]['trials']
        all_trials.extend(block_trials)

print(f"Trials gefunden: {len(all_trials)}")

# Ordne jedem Pupillen-Sample ein Trial und eine Phase zu
pupil_data_synced['trial_assignment'] = 0
pupil_data_synced['phase_type'] = 'unassigned'  # NEU: 'fixation', 'stimulus', oder 'unassigned'

for trial in all_trials:
    trial_num = trial['trial_number']
    
    # Zeitgrenzen aus phases_detected.json
    fix_start = trial['fixation']['start_eyelink_ms']
    fix_end = trial['fixation']['end_eyelink_ms']
    stim_start = trial['stimulus']['start_eyelink_ms']
    stim_end = trial['stimulus']['end_eyelink_ms']
    
    # Maske für Fixationsphase (~1-3 Sekunden)
    mask_fixation = (
        (pupil_data_synced['timestamp_ms_synced'] >= fix_start) & 
        (pupil_data_synced['timestamp_ms_synced'] < fix_end)
    )
    pupil_data_synced.loc[mask_fixation, 'trial_assignment'] = trial_num
    pupil_data_synced.loc[mask_fixation, 'phase_type'] = 'fixation'
    
    # Maske für Stimulus/Free Exploration (~7 Sekunden)
    mask_stimulus = (
        (pupil_data_synced['timestamp_ms_synced'] >= stim_start) & 
        (pupil_data_synced['timestamp_ms_synced'] < stim_end)
    )
    pupil_data_synced.loc[mask_stimulus, 'trial_assignment'] = trial_num
    pupil_data_synced.loc[mask_stimulus, 'phase_type'] = 'stimulus'

# Statistik
n_assigned = (pupil_data_synced['trial_assignment'] > 0).sum()
n_fixation = (pupil_data_synced['phase_type'] == 'fixation').sum()
n_stimulus = (pupil_data_synced['phase_type'] == 'stimulus').sum()
n_unassigned = (pupil_data_synced['phase_type'] == 'unassigned').sum()

print(f"\nTrial-Zuordnung (MediaPipe):")
print(f"  Zugeordnet: {n_assigned}/{len(pupil_data_synced)} Samples ({n_assigned/len(pupil_data_synced)*100:.1f}%)")
print(f"\n  Phasen-Statistik:")
print(f"    Fixation:   {n_fixation:>6} Samples ({n_fixation/len(pupil_data_synced)*100:>5.1f}%)")
print(f"    Stimulus:   {n_stimulus:>6} Samples ({n_stimulus/len(pupil_data_synced)*100:>5.1f}%)")
print(f"    Unassigned: {n_unassigned:>6} Samples ({n_unassigned/len(pupil_data_synced)*100:>5.1f}%)")

# TRIAL-ZUORDNUNG FUER PTGAZE (NEU v3.1)

n_assigned_ptgaze = 0

if ptgaze_available and ptgaze_data_synced is not None:
    print(f"\nTrial-Zuordnung (ptgaze):")
    
    # Initialisiere trial_assignment und phase_type
    ptgaze_data_synced['trial_assignment'] = 0
    ptgaze_data_synced['phase_type'] = 'unassigned'  
    
    for trial in all_trials:
        trial_num = trial['trial_number']
        
        # Zeitgrenzen aus phases_detected.json
        fix_start = trial['fixation']['start_eyelink_ms']
        fix_end = trial['fixation']['end_eyelink_ms']
        stim_start = trial['stimulus']['start_eyelink_ms']
        stim_end = trial['stimulus']['end_eyelink_ms']
        
        # Maske für Fixationsphase
        mask_fixation = (
            (ptgaze_data_synced['timestamp_ms_synced'] >= fix_start) & 
            (ptgaze_data_synced['timestamp_ms_synced'] < fix_end)
        )
        ptgaze_data_synced.loc[mask_fixation, 'trial_assignment'] = trial_num
        ptgaze_data_synced.loc[mask_fixation, 'phase_type'] = 'fixation'
        
        # Maske für Stimulus/Free Exploration
        mask_stimulus = (
            (ptgaze_data_synced['timestamp_ms_synced'] >= stim_start) & 
            (ptgaze_data_synced['timestamp_ms_synced'] < stim_end)
        )
        ptgaze_data_synced.loc[mask_stimulus, 'trial_assignment'] = trial_num
        ptgaze_data_synced.loc[mask_stimulus, 'phase_type'] = 'stimulus'
    
    n_assigned_ptgaze = (ptgaze_data_synced['trial_assignment'] > 0).sum()
    n_fixation_ptgaze = (ptgaze_data_synced['phase_type'] == 'fixation').sum()
    n_stimulus_ptgaze = (ptgaze_data_synced['phase_type'] == 'stimulus').sum()
    n_unassigned_ptgaze = (ptgaze_data_synced['phase_type'] == 'unassigned').sum()
    
    print(f"  Zugeordnet: {n_assigned_ptgaze}/{len(ptgaze_data_synced)} Samples ({n_assigned_ptgaze/len(ptgaze_data_synced)*100:.1f}%)")
    print(f"\n  Phasen-Statistik (ptgaze):")
    print(f"    Fixation:   {n_fixation_ptgaze:>6} Samples ({n_fixation_ptgaze/len(ptgaze_data_synced)*100:>5.1f}%)")
    print(f"    Stimulus:   {n_stimulus_ptgaze:>6} Samples ({n_stimulus_ptgaze/len(ptgaze_data_synced)*100:>5.1f}%)")
    print(f"    Unassigned: {n_unassigned_ptgaze:>6} Samples ({n_unassigned_ptgaze/len(ptgaze_data_synced)*100:>5.1f}%)")

    # ══════════════════════════════════════════════════════════════════════
    # BLINK-MERGE: MediaPipe -> ptgaze (NEU v3.2)
    # ══════════════════════════════════════════════════════════════════════
    # Problem: ptgaze Blink-Detection ist bei 25hz Videos instabil
    # Loesung: Uebernehme Blink-Daten von MediaPipe (stabilere EAR-Werte)
    
    print(f"\n  [BLINK-MERGE] Uebertrage MediaPipe Blink-Daten auf ptgaze...")
    
    # Definiere Blink-Spalten
    blink_cols_binary = ['is_blink', 'eyes_closed']  # Boolean -> Nearest-Neighbor
    blink_cols_continuous = ['left_ear', 'right_ear', 'avg_ear']  # Float -> Linear Interpolation
    blink_cols_counter = ['blink_count']  # Int -> Forward-Fill
    
    all_blink_cols = blink_cols_binary + blink_cols_continuous + blink_cols_counter
    
    # Pruefe welche Spalten in MediaPipe vorhanden sind
    available_blink_cols = [col for col in all_blink_cols if col in pupil_data_synced.columns]
    missing_blink_cols = [col for col in all_blink_cols if col not in pupil_data_synced.columns]
    
    if not available_blink_cols:
        print(f"    [!] Keine Blink-Spalten in MediaPipe-Daten gefunden")
        print(f"        Blink-Merge uebersprungen")
    else:
        print(f"    Verfuegbare Spalten: {', '.join(available_blink_cols)}")
        if missing_blink_cols:
            print(f"    Fehlende Spalten: {', '.join(missing_blink_cols)}")
        
        # Sortiere beide DataFrames nach Zeit (wichtig fuer Interpolation!)
        pupil_sorted = pupil_data_synced.sort_values('timestamp_ms_synced').reset_index(drop=True)
        ptgaze_sorted = ptgaze_data_synced.sort_values('timestamp_ms_synced').reset_index(drop=True)
        
        # MediaPipe Timestamps und ptgaze Timestamps
        mp_times = pupil_sorted['timestamp_ms_synced'].values
        pt_times = ptgaze_sorted['timestamp_ms_synced'].values
        
        n_merged = 0
        
        # Binary Spalten (is_blink, eyes_closed) -> Nearest-Neighbor
        for col in blink_cols_binary:
            if col in pupil_sorted.columns:
                mp_values = pupil_sorted[col].values
                
                # Nearest-Neighbor Interpolation
                # Finde fuer jeden ptgaze-Timestamp den naechsten MediaPipe-Timestamp
                indices = np.searchsorted(mp_times, pt_times)
                indices = np.clip(indices, 0, len(mp_times) - 1)
                
                # Pruefe ob linker oder rechter Nachbar naeher ist
                left_indices = np.clip(indices - 1, 0, len(mp_times) - 1)
                right_indices = indices
                
                left_dist = np.abs(pt_times - mp_times[left_indices])
                right_dist = np.abs(pt_times - mp_times[right_indices])
                
                nearest_indices = np.where(left_dist <= right_dist, left_indices, right_indices)
                
                # Uebertrage Werte
                ptgaze_sorted[col] = mp_values[nearest_indices]
                n_merged += 1
        
        # Continuous Spalten (EAR-Werte) -> Lineare Interpolation
        for col in blink_cols_continuous:
            if col in pupil_sorted.columns:
                mp_values = pupil_sorted[col].values
                
                # Behandle NaN-Werte in MediaPipe-Daten
                valid_mask = ~np.isnan(mp_values)
                
                if valid_mask.sum() > 1:
                    # Lineare Interpolation nur mit validen Werten
                    ptgaze_sorted[col] = np.interp(
                        pt_times,
                        mp_times[valid_mask],
                        mp_values[valid_mask]
                    )
                    n_merged += 1
                else:
                    ptgaze_sorted[col] = np.nan
        
        # Counter Spalten (blink_count) -> Forward-Fill von naechstem Zeitpunkt
        for col in blink_cols_counter:
            if col in pupil_sorted.columns:
                mp_values = pupil_sorted[col].values
                
                # Finde naechsten (nicht zukuenftigen) MediaPipe-Timestamp
                indices = np.searchsorted(mp_times, pt_times, side='right') - 1
                indices = np.clip(indices, 0, len(mp_times) - 1)
                
                ptgaze_sorted[col] = mp_values[indices]
                n_merged += 1
        
        # Markiere Quelle der Blink-Daten
        ptgaze_sorted['blink_source'] = 'mediapipe'
        
        # Berechne maximale Zeitdifferenz (fuer QS)
        indices = np.searchsorted(mp_times, pt_times)
        indices = np.clip(indices, 0, len(mp_times) - 1)
        left_indices = np.clip(indices - 1, 0, len(mp_times) - 1)
        
        time_diffs = np.minimum(
            np.abs(pt_times - mp_times[indices]),
            np.abs(pt_times - mp_times[left_indices])
        )
        max_time_diff = np.max(time_diffs)
        mean_time_diff = np.mean(time_diffs)
        
        # Aktualisiere ptgaze_data_synced
        ptgaze_data_synced = ptgaze_sorted
        
        print(f"    [OK] {n_merged} Blink-Spalten uebertragen")
        print(f"    Zeitliche Genauigkeit:")
        print(f"      - Mittlere Abweichung: {mean_time_diff:.1f} ms")
        print(f"      - Maximale Abweichung: {max_time_diff:.1f} ms")
        
        # Warnung bei grosser Zeitdifferenz
        if max_time_diff > 100:  # > 100ms
            print(f"    [!] WARNUNG: Grosse Zeitdifferenz bei Blink-Merge!")
            print(f"        Moeglicherweise unterschiedliche Frameraten")

# Verteilung
for block_key in ['experiment_block1', 'experiment_block2']:
    if block_key in phases['phases']:
        trials_in_block = [t['trial_number'] for t in phases['phases'][block_key]['trials']]
        n_samples = int((pupil_data_synced['trial_assignment'].isin(trials_in_block)).sum())  # ← FIX!
        print(f"  {block_key}: {n_samples} Samples (Trials {min(trials_in_block)}-{max(trials_in_block)})")

# ==================== SCHRITT 6: VALIDIERUNG (OPTIONAL) ====================

print(f"\n{'='*70}")
print("VALIDIERUNG (gegen EyeLink-Blöcke)")
print(f"{'='*70}\n")

# Lade EyeLink-Blöcke (falls vorhanden)
blocks_csv = os.path.join(OUTPUT_BASE_DIR, "debug_3_blocks_overview.csv")

if not os.path.exists(blocks_csv):
    print(f" debug_3_blocks_overview.csv nicht gefunden")
    print(f"   Validierung übersprungen (nicht kritisch)")
    blocks_df = None
else:
    blocks_df = pd.read_csv(blocks_csv)
    print(f"Geladen: {len(blocks_df)} EyeLink-Blöcke")
    
    # Spalten-Check
    start_col = 'start_time_ms' if 'start_time_ms' in blocks_df.columns else 'start_time'
    end_col = 'end_time_ms' if 'end_time_ms' in blocks_df.columns else 'end_time'
    
    # Coverage-Check
    n_in_fixation = 0
    for _, block_row in blocks_df[blocks_df['block_type'] == 'fixation'].iterrows():
        block_start = block_row[start_col]
        block_end = block_row[end_col]
        
        pupil_in_block = pupil_data_synced[
            (pupil_data_synced['timestamp_ms_synced'] >= block_start) &
            (pupil_data_synced['timestamp_ms_synced'] <= block_end)
        ]
        n_in_fixation += len(pupil_in_block)
    
    video_fps = len(pupil_data) / ((pupil_data['timestamp_ms'].max() - pupil_data['timestamp_ms'].min()) / 1000)
    if blocks_df is not None and len(blocks_df) > 0:
        # Berechne durchschnittliche Fixations-Dauer aus debug_3
        fixation_blocks = blocks_df[blocks_df['block_type'] == 'fixation']
        avg_fixation_duration_s = fixation_blocks['duration_ms'].mean() / 1000
        
        print(f"    Durchschnittliche Fixations-Dauer (aus debug_3): {avg_fixation_duration_s:.2f}s")
        
        expected_per_fixation = avg_fixation_duration_s * video_fps
    else:
        # Fallback
        expected_per_fixation = EXPECTED_FIXATION_DURATION_S * video_fps
    n_fixation_phases = len(blocks_df[blocks_df['block_type'] == 'fixation'])
    avg_per_fixation = n_in_fixation / n_fixation_phases if n_fixation_phases > 0 else 0
    coverage = (avg_per_fixation / expected_per_fixation * 100) if expected_per_fixation > 0 else 0
    
    print(f"\nFixationsphasen: {n_fixation_phases}")
    print(f"Pupillen-Samples darin: {n_in_fixation}")
    print(f"  Ø {avg_per_fixation:.1f} Samples/Fixation")
    print(f"  Erwartet: {expected_per_fixation:.0f}")
    print(f"\nCoverage: {coverage:.1f}%")
    
    if coverage >= 80:
        print(f" Gute Coverage")
    elif coverage >= 50:
        print(f" Geringe Coverage")
    else:
        print(f" Sehr geringe Coverage")

# ==================== SCHRITT 7: VISUALISIERUNG ====================

print(f"\n{'='*70}")
print("VISUALISIERUNG")
print(f"{'='*70}\n")

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

# ─────────────────────────────────────────────────────────────────
# Panel 1: Offset-Übersicht (einfach, da konstant!)
# ─────────────────────────────────────────────────────────────────

ax1 = fig.add_subplot(gs[0, :])

# Zeige nur Offset-Linie (da konstant aus debug_0)
ax1.axhline(offset_ms, color='green', linewidth=3, label=f'Offset: {offset_ms:.0f} ms')
ax1.axhline(0, color='gray', linestyle=':', linewidth=1)

# Falls Validierungs-Daten vorhanden (aus debug_0)
if validation and validation.get('mean_deviation_ms'):
    std_dev = validation.get('std_deviation_ms', 0)
    ax1.fill_between([0, len(all_trials)], 
                     offset_ms - std_dev, offset_ms + std_dev,
                     alpha=0.2, color='green', 
                     label=f'±Std ({std_dev:.1f} ms)')

ax1.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
ax1.set_ylabel('Offset (ms)', fontsize=12, fontweight='bold')
ax1.set_title('Synchronisations-Offset (aus debug_0)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, len(all_trials))

# Info-Text
info_text = (f"Methode: {sync_method}\n"
             f"Confidence: {confidence:.2%}\n"
             f"Audio-Validated: {'Ja' if validation else 'Nein'}")
ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ─────────────────────────────────────────────────────────────────
# Panel 2: Trial-Zuordnungs-Statistik
# ─────────────────────────────────────────────────────────────────

ax2 = fig.add_subplot(gs[1, 0])

# Zähle Samples pro Trial
trial_sample_counts = []
trial_numbers = []

for trial in all_trials:
    trial_num = trial['trial_number']
    n_samples = (pupil_data_synced['trial_assignment'] == trial_num).sum()
    trial_sample_counts.append(n_samples)
    trial_numbers.append(trial_num)

ax2.bar(trial_numbers, trial_sample_counts, color='steelblue', edgecolor='black', linewidth=0.5)
ax2.axhline(np.mean(trial_sample_counts), color='red', linestyle='--', 
           linewidth=2, label=f'Mean: {np.mean(trial_sample_counts):.0f}')
ax2.set_xlabel('Trial Number', fontsize=11, fontweight='bold')
ax2.set_ylabel('Anzahl Samples', fontsize=11, fontweight='bold')
ax2.set_title('Pupillen-Samples pro Trial', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# ─────────────────────────────────────────────────────────────────
# Panel 3: Coverage-Check (falls EyeLink vorhanden)
# ─────────────────────────────────────────────────────────────────

ax3 = fig.add_subplot(gs[1, 1])

if blocks_df is not None:
    # Berechne Coverage pro Fixationsphase
    coverage_per_phase = []
    
    for _, block_row in blocks_df[blocks_df['block_type'] == 'fixation'].iterrows():
        block_start = block_row[start_col]
        block_end = block_row[end_col]
        
        pupil_in_block = pupil_data_synced[
            (pupil_data_synced['timestamp_ms_synced'] >= block_start) &
            (pupil_data_synced['timestamp_ms_synced'] <= block_end)
        ]
        
        phase_coverage = (len(pupil_in_block) / expected_per_fixation * 100) if expected_per_fixation > 0 else 0
        coverage_per_phase.append(phase_coverage)
    
    trial_nums_for_coverage = range(1, len(coverage_per_phase) + 1)
    
    colors = ['green' if c >= 80 else 'orange' if c >= 50 else 'red' 
             for c in coverage_per_phase]
    
    ax3.bar(trial_nums_for_coverage, coverage_per_phase, color=colors, 
           edgecolor='black', linewidth=0.5)
    ax3.axhline(80, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Gut (≥80%)')
    ax3.axhline(50, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='OK (≥50%)')
    ax3.set_xlabel('Trial Number', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Coverage (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Fixationsphasen-Coverage', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 100)
else:
    ax3.text(0.5, 0.5, 'Keine EyeLink-Blöcke\n(debug_3 fehlt)', 
            ha='center', va='center', fontsize=12, transform=ax3.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax3.axis('off')

plt.suptitle('DEBUG 4: Synchronisation v3.0 (Vereinfacht mit debug_0)', 
             fontsize=16, fontweight='bold')

plot_path = os.path.join(OUTPUT_BASE_DIR, "debug_4_sync_quality.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f" Plot: {plot_path}")
plt.close()

# ==================== SCHRITT 8: EXPORT ====================

print(f"\n{'='*70}")
print("EXPORT")
print(f"{'='*70}\n")

output_path = os.path.join(OUTPUT_BASE_DIR, "debug_4_pupil_data_synced.csv")
pupil_data_synced.to_csv(output_path, index=False)
print(f" Pupil-Data: {output_path}")

# EXPORT PTGAZE (NEU v3.1)

if ptgaze_available and ptgaze_data_synced is not None:
    ptgaze_output_path = os.path.join(OUTPUT_BASE_DIR, "debug_4_ptgaze_gaze_synced.csv")
    ptgaze_data_synced.to_csv(ptgaze_output_path, index=False)
    print(f" ptgaze-Data: {ptgaze_output_path}")

# Sync-Info (erweitert) - v3.1 mit ptgaze-Support
sync_info_export = make_json_serializable({
    'sync_method': sync_method,
    'offset_ms': float(offset_ms),
    'offset_used_ms': float(offset_used),  # NEU: Tatsaechlich genutzter Offset
    'confidence': float(confidence),
    'source': 'phases_detected.json',
    'debug_0_version': phases['sync_info'].get('method', 'unknown'),
    
    # MediaPipe-Statistiken
    'mediapipe': {
        'n_samples_synced': int(len(pupil_data_synced)),
        'n_samples_assigned_to_trials': int(n_assigned),
        'assignment_rate': float(n_assigned / len(pupil_data_synced)),
        'n_fixation_samples': int(n_fixation),
        'n_stimulus_samples': int(n_stimulus),
        'n_unassigned_samples': int(n_unassigned),
    },
    
    # ptgaze-Statistiken (NEU v3.2 mit Blink-Merge)
    'ptgaze': {
        'available': ptgaze_available,
        'n_samples_synced': int(len(ptgaze_data_synced)) if ptgaze_available else 0,
        'n_samples_assigned_to_trials': int(n_assigned_ptgaze) if ptgaze_available else 0,
        'assignment_rate': float(n_assigned_ptgaze / len(ptgaze_data_synced)) if ptgaze_available and len(ptgaze_data_synced) > 0 else 0.0,
        'n_fixation_samples': int(n_fixation_ptgaze) if ptgaze_available else 0,
        'n_stimulus_samples': int(n_stimulus_ptgaze) if ptgaze_available else 0,
        'n_unassigned_samples': int(n_unassigned_ptgaze) if ptgaze_available else 0,
        'same_offset_as_mediapipe': True,
        'blink_source': 'mediapipe',
        'blink_merge_applied': ptgaze_available,
    },

    'n_trials': int(len(all_trials)),
    'csv_version': '3.3',  
    'validation': {
        'coverage_percent': float(coverage) if blocks_df is not None else None,
        'n_fixation_blocks': int(n_fixation_phases) if blocks_df is not None else None,
        'audio_validated': bool(validation)
    }
})

sync_json = os.path.join(OUTPUT_BASE_DIR, "debug_4_sync_info.json")
with open(sync_json, 'w') as f:
    json.dump(sync_info_export, f, indent=2)
print(f" Sync-Info: {sync_json}")

print(f"\n{'='*70}")
print(" ERFOLGREICH!")
print(f"{'='*70}")
print(f"\n Statistik:")
print(f"   • Samples: {len(pupil_data_synced)}")
print(f"   • Trial-Zuordnung: {n_assigned} ({n_assigned/len(pupil_data_synced)*100:.1f}%)")
print(f"   • Offset: {offset_used:.0f} ms")  # ← FIX: Nutze offset_used statt offset_ms!
print(f"   • Methode: {sync_method} (aus debug_0)")

if blocks_df is not None:
    print(f"   • Coverage: {coverage:.1f}%")

# ptgaze-Statistik (NEU v3.2 mit Blink-Merge)
if ptgaze_available:
    print(f"\n   [ptgaze Sync]")
    print(f"   - Samples: {len(ptgaze_data_synced)}")
    print(f"   - Trial-Zuordnung: {n_assigned_ptgaze} ({n_assigned_ptgaze/len(ptgaze_data_synced)*100:.1f}%)")
    print(f"   - Offset: {offset_used:.0f} ms (identisch zu MediaPipe)")
    print(f"   - Blink-Daten: von MediaPipe (stabiler)")
    print(f"   - Output: debug_4_ptgaze_gaze_synced.csv")
else:
    print(f"\n   [ptgaze Sync]")
    print(f"   - Nicht verfuegbar (debug_1_ptgaze_data.csv fehlt)")

#  NEU: Zeige Dual-Anchor-Info falls aktiv
if sync_method == 'dual_anchor' and 'offset_880hz_ms' in sync_info:
    print(f"\n   💡 Dual-Anchor Details:")
    print(f"      Kalibrierung (1760 Hz): {sync_info['offset_1760hz_ms']:.0f} ms")
    print(f"      Trials (880 Hz):        {sync_info['offset_880hz_ms']:.0f} ms")
    print(f"      Differenz:              {sync_info.get('frequency_difference_ms', 0):+.0f} ms")
    print(f"      >> Daten nutzen Trial-Offset für optimale Korrelation!")
