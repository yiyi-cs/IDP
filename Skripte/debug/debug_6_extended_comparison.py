"""
=================================================================================
DEBUG_6_EXTENDED_COMPARISON.PY - 3-Wege-Vergleich v1.1
=================================================================================
Erweitert debug_6_trial_comparison.py fuer ptgaze-Integration:

3-Wege-Vergleich:
- MediaPipe vs. ptgaze vs. EyeLink (Modus 2)
- MediaPipe vs. ptgaze (Modus 1)

Neue Funktionalitaet:
- Dual-Input: Laedt BEIDE Pipelines (mediapipe + ptgaze)
- Cross-Comparison: Vergleicht alle Paare
- Erweiterte Metriken: Bland-Altman, ICC, Lateralization Agreement
- Neue Visualisierungen: 3-Methoden-Scatter, Dual-Zeitreihen

NEU in v1.1:
------------
- Package-Struktur (debug/)
- Path-Setup fuer manuelle Ausfuehrung

Version: v1.1 (basierend auf debug_6_trial_comparison v3.0)
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
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from config import *

# Environment-Variable-Overrides
if 'PIPELINE_OUTPUT_BASE_DIR' in os.environ:
    OUTPUT_BASE_DIR = Path(os.environ['PIPELINE_OUTPUT_BASE_DIR'])

# ==================== HELPER-FUNKTIONEN ====================

def load_gaze_data(method: str) -> pd.DataFrame:
    """
    Laedt kalibrierte Gaze-Daten.
    
    Args:
        method: 'mediapipe' oder 'ptgaze'
    
    Returns:
        DataFrame mit kalibrierten Daten
    """
    
    if method == 'mediapipe':
        csv_path = os.path.join(OUTPUT_BASE_DIR, "debug_5_pupil_data_calibrated.csv")
    elif method == 'ptgaze':
        csv_path = os.path.join(OUTPUT_BASE_DIR, "debug_5_ptgaze_calibrated.csv")
    else:
        raise ValueError(f"Unbekannte Methode: {method}")
    
    if not os.path.exists(csv_path):
        print(f" FEHLER: {csv_path} nicht gefunden!")
        print(f"   >> Laufe zuerst: {'debug_5_partial_calibration.py' if method == 'mediapipe' else 'debug_5_ptgaze.py'}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f" {method.upper()} Daten geladen: {len(df)} Samples")
    
    return df

def calculate_trial_mean(df: pd.DataFrame, trial_num: int, method: str, 
                        phases: dict = None,
                        filter_blinks: bool = True,
                        filter_outside_monitor: bool = True) -> dict:
    """
    Berechnet Trial-Mean für eine Methode MIT Blink/Monitor-Filterung.
    
    Args:
        df: Gaze-Daten (kalibriert) ODER komplette EyeLink-Daten
        trial_num: Trial-Nummer
        method: 'mediapipe', 'ptgaze', oder 'eyelink'
        phases: Nur für EyeLink nötig (Trial-Grenzen aus phases_detected.json)
        filter_blinks: Frames mit eyes_closed ausschließen
        filter_outside_monitor: Frames außerhalb Monitor ausschließen
    
    Returns:
        dict mit mean_x, mean_y, n_samples, n_filtered_blink, n_filtered_outside, valid_ratio
    """
    
    # ═══════════════════════════════════════════════════════════════
    # METHODEN-SPEZIFISCHE TRIAL-EXTRAKTION
    # ═══════════════════════════════════════════════════════════════
    
    if method in ['mediapipe', 'ptgaze']:
        # Diese CSVs haben 'trial_number' Spalte
        if 'trial_number' not in df.columns:
            print(f"     FEHLER: 'trial_number' fehlt in {method} Daten!")
            return {'mean_x': np.nan, 'mean_y': np.nan, 'n_samples': 0, 
                    'n_filtered_blink': 0, 'n_filtered_outside': 0, 'valid_ratio': 0.0}
        
        trial_data = df[df['trial_number'] == trial_num].copy()
        
        # ═══════════════════════════════════════════════════════════════
        # NEU: NUR STIMULUS-PHASE AUSWERTEN
        # ═══════════════════════════════════════════════════════════════
        if 'phase_type' in trial_data.columns:
            trial_data = trial_data[trial_data['phase_type'] == 'stimulus']
    
    elif method == 'eyelink':
        # EyeLink-Daten haben KEINE trial_number >> nutze phases
        if phases is None:
            print(f"     FEHLER: 'phases' nötig für EyeLink!")
            return {'mean_x': np.nan, 'mean_y': np.nan, 'n_samples': 0,
                    'n_filtered_blink': 0, 'n_filtered_outside': 0, 'valid_ratio': 0.0}
        
        # Finde Trial in phases_detected.json
        trial_info = None
        for block_key in ['experiment_block1', 'experiment_block2']:
            if block_key not in phases.get('phases', {}):
                continue
            
            for t in phases['phases'][block_key]['trials']:
                if t['trial_number'] == trial_num and not t.get('is_practice', False):
                    trial_info = t
                    break
            
            if trial_info:
                break
        
        if trial_info is None:
            return {'mean_x': np.nan, 'mean_y': np.nan, 'n_samples': 0,
                    'n_filtered_blink': 0, 'n_filtered_outside': 0, 'valid_ratio': 0.0}
        
        # Extrahiere Zeitgrenzen
        start_ms = trial_info['fixation']['start_eyelink_ms']
        end_ms = trial_info['stimulus']['end_eyelink_ms']
        
        time_col = 'timestamp_ms' if 'timestamp_ms' in df.columns else 'timestamp'
        
        trial_data = df[
            (df[time_col] >= start_ms) &
            (df[time_col] <= end_ms)
        ].copy()
    
    else:
        raise ValueError(f"Unbekannte Methode: {method}")
    
    n_total = len(trial_data)
    
    if n_total == 0:
        return {'mean_x': np.nan, 'mean_y': np.nan, 'n_samples': 0,
                'n_filtered_blink': 0, 'n_filtered_outside': 0, 'valid_ratio': 0.0}
    
    # ═══════════════════════════════════════════════════════════════
    # FILTERUNG (NEU!)
    # ═══════════════════════════════════════════════════════════════
    
    n_filtered_blink = 0
    n_filtered_outside = 0
    
    # 1. Blink-Filterung
    if filter_blinks and method in ['mediapipe', 'ptgaze']:
        if 'eyes_closed' in trial_data.columns:
            blink_mask = trial_data['eyes_closed'] == True
            n_filtered_blink = blink_mask.sum()
            
            # Post-Blink Recovery (100-200ms nach Blink)
            if n_filtered_blink > 0 and 'timestamp_ms' in trial_data.columns:
                try:
                    from config import POST_BLINK_RECOVERY_MS
                except ImportError:
                    POST_BLINK_RECOVERY_MS = 150
                
                # Finde Blink-Ende-Zeitpunkte
                blink_indices = trial_data[blink_mask].index
                
                for idx in blink_indices:
                    blink_end_time = trial_data.loc[idx, 'timestamp_ms']
                    recovery_end_time = blink_end_time + POST_BLINK_RECOVERY_MS
                    
                    # Markiere Recovery-Frames
                    recovery_mask = (
                        (trial_data['timestamp_ms'] > blink_end_time) &
                        (trial_data['timestamp_ms'] <= recovery_end_time)
                    )
                    blink_mask = blink_mask | recovery_mask
                
                n_filtered_blink = blink_mask.sum()
            
            trial_data = trial_data[~blink_mask]
    
    # 2. Outside-Monitor-Filterung
    if filter_outside_monitor and method in ['mediapipe', 'ptgaze']:
        if 'outside_monitor' in trial_data.columns:
            outside_mask = trial_data['outside_monitor'] == True
            n_filtered_outside = outside_mask.sum()
            trial_data = trial_data[~outside_mask]
    
    # ═══════════════════════════════════════════════════════════════
    # BERECHNE MEAN
    # ═══════════════════════════════════════════════════════════════
    
    n_after_filter = len(trial_data)
    valid_ratio = n_after_filter / n_total if n_total > 0 else 0.0
    
    # Prüfe Minimum Valid Sample Ratio
    try:
        from config import MIN_VALID_SAMPLE_RATIO
    except ImportError:
        MIN_VALID_SAMPLE_RATIO = 0.5
    
    if valid_ratio < MIN_VALID_SAMPLE_RATIO:
        return {
            'mean_x': np.nan, 
            'mean_y': np.nan, 
            'n_samples': n_after_filter,
            'n_filtered_blink': n_filtered_blink,
            'n_filtered_outside': n_filtered_outside,
            'valid_ratio': valid_ratio,
            'rejected_reason': f'valid_ratio {valid_ratio:.1%} < {MIN_VALID_SAMPLE_RATIO:.0%}'
        }
    
    if method in ['mediapipe', 'ptgaze']:
        valid = trial_data.dropna(subset=['gaze_deg_x_calib', 'gaze_deg_y_calib'])
        
        if len(valid) == 0:
            return {'mean_x': np.nan, 'mean_y': np.nan, 'n_samples': 0,
                    'n_filtered_blink': n_filtered_blink, 'n_filtered_outside': n_filtered_outside,
                    'valid_ratio': valid_ratio}
        
        mean_x = valid['gaze_deg_x_calib'].mean()
        mean_y = valid['gaze_deg_y_calib'].mean()
    
    elif method == 'eyelink':
        valid = trial_data.dropna(subset=['x_pos', 'y_pos'])
        
        if len(valid) == 0:
            return {'mean_x': np.nan, 'mean_y': np.nan, 'n_samples': 0,
                    'n_filtered_blink': n_filtered_blink, 'n_filtered_outside': n_filtered_outside,
                    'valid_ratio': valid_ratio}
        
        # Konversion Pixel zu Grad
        deg_x_list = []
        for _, row in valid.iterrows():
            offset_cm = (row['x_pos'] - SCREEN_WIDTH_PX//2) / (SCREEN_WIDTH_PX / SCREEN_WIDTH_CM)
            deg = np.arctan2(offset_cm, VIEWING_DISTANCE_CM) * 180 / np.pi
            deg_x_list.append(deg)
        
        mean_x = np.mean(deg_x_list)
        mean_y = np.nan
    
    return {
        'mean_x': mean_x,
        'mean_y': mean_y,
        'n_samples': len(valid),
        'n_filtered_blink': n_filtered_blink,
        'n_filtered_outside': n_filtered_outside,
        'valid_ratio': valid_ratio
    }

# ==================== 3-WEGE-VERGLEICH ====================

class ThreeWayComparison:
    """
    Vergleicht 3 Methoden: MediaPipe, ptgaze, EyeLink.
    
    Funktioniert auch mit nur 2 Methoden (wenn EyeLink fehlt).
    """
    
    def __init__(self, mode: int):
        self.mode = mode
    
    def compare(self, mediapipe_data: pd.DataFrame, 
                ptgaze_data: pd.DataFrame,
                eyelink_data: pd.DataFrame = None,
                phases: dict = None) -> pd.DataFrame:
        """
        Erstellt Trial-Vergleich MIT Blink/Monitor-Filterung.
        
        NEU: Detaillierte Filterungs-Statistik pro Trial und gesamt!
        
        Returns:
            DataFrame mit erweiterten Spalten inkl. Filterungs-Statistik
        """
        
        # Config laden
        try:
            from config import (ENABLE_BLINK_FILTERING_IN_ANALYSIS, 
                               ENABLE_OUTSIDE_MONITOR_FILTERING,
                               POST_BLINK_RECOVERY_MS,
                               MIN_VALID_SAMPLE_RATIO)
        except ImportError:
            ENABLE_BLINK_FILTERING_IN_ANALYSIS = True
            ENABLE_OUTSIDE_MONITOR_FILTERING = True
            POST_BLINK_RECOVERY_MS = 150
            MIN_VALID_SAMPLE_RATIO = 0.5
        
        # Extrahiere Trial-Liste
        if phases is not None:
            trial_list = self._extract_trials_from_phases(phases)
        else:
            trial_list = mediapipe_data['trial_number'].dropna().unique()
            trial_list = sorted([int(t) for t in trial_list if t > 0])
        
        print(f"\n{'='*70}")
        print("3-WEGE-VERGLEICH MIT FILTERUNG")
        print(f"{'='*70}\n")
        print(f"✓ Trials: {len(trial_list)}")
        print(f"✓ Blink-Filterung: {'AN' if ENABLE_BLINK_FILTERING_IN_ANALYSIS else 'AUS'}")
        print(f"✓ Outside-Monitor-Filterung: {'AN' if ENABLE_OUTSIDE_MONITOR_FILTERING else 'AUS'}")
        print(f"✓ Post-Blink-Recovery: {POST_BLINK_RECOVERY_MS}ms")
        print(f"✓ Min Valid Sample Ratio: {MIN_VALID_SAMPLE_RATIO*100:.0f}%")
        
        results = []
        
        # Statistik-Zähler
        total_filtered_blink_mp = 0
        total_filtered_outside_mp = 0
        total_filtered_blink_pt = 0
        total_filtered_outside_pt = 0
        total_samples_mp = 0
        total_samples_pt = 0
        rejected_trials_mp = 0
        rejected_trials_pt = 0
        
        print(f"\n{'─'*70}")
        print("TRIAL-WEISE FILTERUNG")
        print(f"{'─'*70}\n")
        
        for trial_num in trial_list:
            # MediaPipe
            mp_result = calculate_trial_mean(
                mediapipe_data, trial_num, 'mediapipe',
                filter_blinks=ENABLE_BLINK_FILTERING_IN_ANALYSIS,
                filter_outside_monitor=ENABLE_OUTSIDE_MONITOR_FILTERING
            )
            
            # ptgaze
            pt_result = calculate_trial_mean(
                ptgaze_data, trial_num, 'ptgaze',
                filter_blinks=ENABLE_BLINK_FILTERING_IN_ANALYSIS,
                filter_outside_monitor=ENABLE_OUTSIDE_MONITOR_FILTERING
            )
            
            # EyeLink (keine Blink-Filterung - EyeLink hat eigene!)
            if eyelink_data is not None:
                et_result = calculate_trial_mean(
                    eyelink_data, trial_num, 'eyelink', 
                    phases=phases,
                    filter_blinks=False,
                    filter_outside_monitor=False
                )
            else:
                et_result = {'mean_x': np.nan, 'mean_y': np.nan, 'n_samples': 0,
                            'n_filtered_blink': 0, 'n_filtered_outside': 0, 'valid_ratio': 1.0}
            
            # Statistik sammeln
            mp_blink = mp_result.get('n_filtered_blink', 0)
            mp_outside = mp_result.get('n_filtered_outside', 0)
            pt_blink = pt_result.get('n_filtered_blink', 0)
            pt_outside = pt_result.get('n_filtered_outside', 0)
            
            total_filtered_blink_mp += mp_blink
            total_filtered_outside_mp += mp_outside
            total_filtered_blink_pt += pt_blink
            total_filtered_outside_pt += pt_outside
            total_samples_mp += mp_result['n_samples'] + mp_blink + mp_outside
            total_samples_pt += pt_result['n_samples'] + pt_blink + pt_outside
            
            # Prüfe ob Trial rejected wurde
            mp_rejected = 'rejected_reason' in mp_result
            pt_rejected = 'rejected_reason' in pt_result
            
            if mp_rejected:
                rejected_trials_mp += 1
            if pt_rejected:
                rejected_trials_pt += 1
            
            # ═══════════════════════════════════════════════════════════════
            # IMMER Ausgabe pro Trial mit Mittelwerten
            # ═══════════════════════════════════════════════════════════════
            mp_x_str = f"{mp_result['mean_x']:>+7.2f}°" if not np.isnan(mp_result['mean_x']) else "    N/A"
            pt_x_str = f"{pt_result['mean_x']:>+7.2f}°" if not np.isnan(pt_result['mean_x']) else "    N/A"
            et_x_str = f"{et_result['mean_x']:>+7.2f}°" if not np.isnan(et_result['mean_x']) else "    N/A"
            
            # Status-Marker
            mp_status = "✗" if mp_rejected else "✓"
            pt_status = "✗" if pt_rejected else "✓"
            
            print(f"Trial {trial_num:2d}: MP={mp_x_str} [{mp_status}]  PT={pt_x_str} [{pt_status}]  ET={et_x_str}  "
                  f"(n: MP={mp_result['n_samples']:>3}, PT={pt_result['n_samples']:>3}, ET={et_result['n_samples']:>3})")
            
            # Optional: Filterdetails wenn gefiltert wurde
            if mp_blink > 0 or mp_outside > 0 or pt_blink > 0 or pt_outside > 0:
                filter_info = []
                if mp_blink > 0 or mp_outside > 0:
                    filter_info.append(f"MP: -{mp_blink}B/-{mp_outside}O")
                if pt_blink > 0 or pt_outside > 0:
                    filter_info.append(f"PT: -{pt_blink}B/-{pt_outside}O")
                print(f"         └─ Gefiltert: {', '.join(filter_info)}")
            
            if mp_rejected:
                print(f"         └─ ⚠ MP REJECTED: {mp_result.get('rejected_reason', 'unknown')}")
            if pt_rejected:
                print(f"         └─ ⚠ PT REJECTED: {pt_result.get('rejected_reason', 'unknown')}")
            
            results.append({
                'trial_number': trial_num,
                
                # MediaPipe
                'mediapipe_mean_x': mp_result['mean_x'],
                'mediapipe_mean_y': mp_result.get('mean_y', np.nan),
                'mediapipe_n_samples': mp_result['n_samples'],
                'mediapipe_n_filtered_blink': mp_blink,
                'mediapipe_n_filtered_outside': mp_outside,
                'mediapipe_valid_ratio': mp_result.get('valid_ratio', 1.0),
                'mediapipe_rejected': mp_rejected,
                
                # ptgaze
                'ptgaze_mean_x': pt_result['mean_x'],
                'ptgaze_mean_y': pt_result.get('mean_y', np.nan),
                'ptgaze_n_samples': pt_result['n_samples'],
                'ptgaze_n_filtered_blink': pt_blink,
                'ptgaze_n_filtered_outside': pt_outside,
                'ptgaze_valid_ratio': pt_result.get('valid_ratio', 1.0),
                'ptgaze_rejected': pt_rejected,
                
                # EyeLink
                'eyelink_mean_x': et_result['mean_x'],
                'eyelink_n_samples': et_result['n_samples']
            })
        
        # ═══════════════════════════════════════════════════════════════
        # GESAMTMITTELWERTE ÜBER ALLE TRIALS
        # ═══════════════════════════════════════════════════════════════
        print(f"\n{'─'*70}")
        print("MITTELWERTE ÜBER ALLE TRIALS (nur Stimulus-Phase)")
        print(f"{'─'*70}")
        
        # Berechne Mittelwerte (nur nicht-rejected Trials)
        valid_mp_x = [r['mediapipe_mean_x'] for r in results if not r.get('mediapipe_rejected', False) and not np.isnan(r['mediapipe_mean_x'])]
        valid_pt_x = [r['ptgaze_mean_x'] for r in results if not r.get('ptgaze_rejected', False) and not np.isnan(r['ptgaze_mean_x'])]
        valid_et_x = [r['eyelink_mean_x'] for r in results if not np.isnan(r['eyelink_mean_x'])]
        
        mean_mp_x = np.nanmean(valid_mp_x) if valid_mp_x else np.nan
        mean_pt_x = np.nanmean(valid_pt_x) if valid_pt_x else np.nan
        mean_et_x = np.nanmean(valid_et_x) if valid_et_x else np.nan
        
        std_mp_x = np.nanstd(valid_mp_x) if valid_mp_x else np.nan
        std_pt_x = np.nanstd(valid_pt_x) if valid_pt_x else np.nan
        std_et_x = np.nanstd(valid_et_x) if valid_et_x else np.nan
        
        print(f"\n┌─────────────────────────────────────────────────────────────────────┐")
        print(f"│ METHODE         MEAN_X (°)      STD (°)       VALIDE TRIALS        │")
        print(f"├─────────────────────────────────────────────────────────────────────┤")
        print(f"│ MediaPipe       {mean_mp_x:>+7.3f}        {std_mp_x:>6.3f}        {len(valid_mp_x):>3} / {len(trial_list)}              │")
        print(f"│ ptgaze          {mean_pt_x:>+7.3f}        {std_pt_x:>6.3f}        {len(valid_pt_x):>3} / {len(trial_list)}              │")
        print(f"│ EyeLink         {mean_et_x:>+7.3f}        {std_et_x:>6.3f}        {len(valid_et_x):>3} / {len(trial_list)}              │")
        print(f"└─────────────────────────────────────────────────────────────────────┘")
        
        df = pd.DataFrame(results)
        
        # ══════════════════════════════════════════════════════════════════
        # ZUSAMMENFASSENDE FILTERUNGS-STATISTIK
        # ══════════════════════════════════════════════════════════════════
        
        print(f"\n{'='*70}")
        print("FILTERUNGS-STATISTIK (GESAMT)")
        print(f"{'='*70}")
        
        print(f"\n┌─────────────────────────────────────────────────────────────────────┐")
        print(f"│ MEDIAPIPE                                                           │")
        print(f"├─────────────────────────────────────────────────────────────────────┤")
        print(f"│  Gesamt-Frames:        {total_samples_mp:>8}                                   │")
        print(f"│  Gefiltert (Blinks):   {total_filtered_blink_mp:>8} ({total_filtered_blink_mp/max(total_samples_mp,1)*100:>5.1f}%)                        │")
        print(f"│  Gefiltert (Outside):  {total_filtered_outside_mp:>8} ({total_filtered_outside_mp/max(total_samples_mp,1)*100:>5.1f}%)                        │")
        print(f"│  Verbleibend:          {total_samples_mp - total_filtered_blink_mp - total_filtered_outside_mp:>8} ({(total_samples_mp - total_filtered_blink_mp - total_filtered_outside_mp)/max(total_samples_mp,1)*100:>5.1f}%)                        │")
        print(f"│  Abgelehnte Trials:    {rejected_trials_mp:>8} / {len(trial_list)}                                   │")
        print(f"└─────────────────────────────────────────────────────────────────────┘")
        
        print(f"\n┌─────────────────────────────────────────────────────────────────────┐")
        print(f"│ PTGAZE                                                              │")
        print(f"├─────────────────────────────────────────────────────────────────────┤")
        print(f"│  Gesamt-Frames:        {total_samples_pt:>8}                                   │")
        print(f"│  Gefiltert (Blinks):   {total_filtered_blink_pt:>8} ({total_filtered_blink_pt/max(total_samples_pt,1)*100:>5.1f}%)                        │")
        print(f"│  Gefiltert (Outside):  {total_filtered_outside_pt:>8} ({total_filtered_outside_pt/max(total_samples_pt,1)*100:>5.1f}%)                        │")
        print(f"│  Verbleibend:          {total_samples_pt - total_filtered_blink_pt - total_filtered_outside_pt:>8} ({(total_samples_pt - total_filtered_blink_pt - total_filtered_outside_pt)/max(total_samples_pt,1)*100:>5.1f}%)                        │")
        print(f"│  Abgelehnte Trials:    {rejected_trials_pt:>8} / {len(trial_list)}                                   │")
        print(f"└─────────────────────────────────────────────────────────────────────┘")
        
        # ══════════════════════════════════════════════════════════════════
        # FIXATIONSZEIT-ANALYSE (EyeLink)
        # ══════════════════════════════════════════════════════════════════
        
        if eyelink_data is not None and phases is not None:
            print(f"\n{'='*70}")
            print("FIXATIONSZEIT-ANALYSE (EyeLink)")
            print(f"{'='*70}")
            
            # Lade Fixationen aus debug_3
            fixations_csv = os.path.join(OUTPUT_BASE_DIR, "debug_3_fixations.csv")
            blocks_csv = os.path.join(OUTPUT_BASE_DIR, "debug_3_blocks_overview.csv")
            
            total_trial_duration_ms = 0
            total_fixation_duration_ms = 0
            fixation_data_available = False
            
            # ─────────────────────────────────────────────────────────────
            # METHODE 1: Direkt aus debug_3_fixations.csv (falls vorhanden)
            # ─────────────────────────────────────────────────────────────
            
            if os.path.exists(fixations_csv):
                fixations_df = pd.read_csv(fixations_csv)
                fixation_data_available = True
                
                print(f"\n[>] Fixationen geladen: {len(fixations_df)} aus debug_3_fixations.csv")
                
                # Berechne Gesamtdauer aller experimentellen Trials
                for block_key in ['experiment_block1', 'experiment_block2']:
                    if block_key not in phases.get('phases', {}):
                        continue
                    
                    for trial in phases['phases'][block_key]['trials']:
                        if trial.get('is_practice', False):
                            continue
                        
                        trial_num = trial['trial_number']
                        
                        # Trial-Dauer aus phases
                        if 'stimulus' in trial:
                            start_ms = trial['fixation']['start_eyelink_ms']
                            end_ms = trial['stimulus']['end_eyelink_ms']
                            trial_duration = end_ms - start_ms
                            total_trial_duration_ms += trial_duration
                            
                            # Fixationen in diesem Zeitbereich
                            trial_fixations = fixations_df[
                                (fixations_df['start_time'] >= start_ms) &
                                (fixations_df['end_time'] <= end_ms)
                            ]
                            
                            if 'duration' in trial_fixations.columns:
                                total_fixation_duration_ms += trial_fixations['duration'].sum()
            
            # ─────────────────────────────────────────────────────────────
            # METHODE 2: Aus debug_3_blocks_overview.csv (Fallback)
            # ─────────────────────────────────────────────────────────────
            
            elif os.path.exists(blocks_csv):
                blocks_df = pd.read_csv(blocks_csv)
                
                print(f"\n[>] Block-Übersicht geladen (Fallback): debug_3_blocks_overview.csv")
                
                # Nur Trial-Blöcke (keine Fixationskreuz-Blöcke)
                trial_blocks = blocks_df[
                    (blocks_df['block_type'] == 'trial') & 
                    (blocks_df['is_practice'] == False)
                ]
                
                if len(trial_blocks) > 0:
                    total_trial_duration_ms = trial_blocks['duration_ms'].sum()
                    
                    # n_fixations ist Anzahl, nicht Dauer - wir brauchen die Rohdaten
                    print(f"  ⚠ Fixations-Dauern nicht verfügbar (nur Anzahl)")
                    print(f"    >> Für genaue Analyse: SAVE_EYELINK_FIXATIONS = True in config.py")
                    fixation_data_available = False
            
            # ─────────────────────────────────────────────────────────────
            # AUSGABE
            # ─────────────────────────────────────────────────────────────
            
            if total_trial_duration_ms > 0:
                print(f"\n┌─────────────────────────────────────────────────────────────────────┐")
                print(f"│ FIXATIONSZEIT-STATISTIK                                             │")
                print(f"├─────────────────────────────────────────────────────────────────────┤")
                print(f"│                                                                     │")
                print(f"│  Logik: Summe aller Fixationsdauern / Gesamtdauer aller 24 Trials   │")
                print(f"│         (nur experimentelle Trials, ohne Practice)                  │")
                print(f"│                                                                     │")
                print(f"├─────────────────────────────────────────────────────────────────────┤")
                print(f"│  Gesamtdauer (24 Trials):  {total_trial_duration_ms/1000:>8.2f} s ({total_trial_duration_ms/60000:.1f} min)              │")
                
                if fixation_data_available and total_fixation_duration_ms > 0:
                    fixation_ratio = total_fixation_duration_ms / total_trial_duration_ms
                    saccade_blink_ratio = 1 - fixation_ratio
                    
                    print(f"│  Fixationszeit (Summe):    {total_fixation_duration_ms/1000:>8.2f} s                            │")
                    print(f"│  Sakkaden/Blinks/Andere:   {(total_trial_duration_ms - total_fixation_duration_ms)/1000:>8.2f} s                            │")
                    print(f"│                                                                     │")
                    print(f"│  ► Fixationsanteil:        {fixation_ratio*100:>8.1f} %                            │")
                    print(f"│  ► Sakkaden/Blinks:        {saccade_blink_ratio*100:>8.1f} %                            │")
                    
                    # Qualitätsbewertung
                    print(f"│                                                                     │")
                    if fixation_ratio >= 0.85:
                        print(f"│  ✓ Sehr gute Datenqualität (>85% Fixationen)                        │")
                    elif fixation_ratio >= 0.70:
                        print(f"│  ✓ Gute Datenqualität (70-85% Fixationen)                           │")
                    elif fixation_ratio >= 0.50:
                        print(f"│  ⚠ Mittlere Datenqualität (50-70% Fixationen)                       │")
                    else:
                        print(f"│  ✗ Schlechte Datenqualität (<50% Fixationen)                        │")
                else:
                    print(f"│  Fixationszeit:            N/A (Daten nicht verfügbar)             │")
                    print(f"│                                                                     │")
                    print(f"│  💡 Aktiviere SAVE_EYELINK_FIXATIONS = True in config.py            │")
                
                print(f"└─────────────────────────────────────────────────────────────────────┘")
            
            # ─────────────────────────────────────────────────────────────
            # FIXATIONSANTEIL NUR FÜR KORRELIERTE TRIALS
            # ─────────────────────────────────────────────────────────────
            
            if fixation_data_available and total_fixation_duration_ms > 0:
                # Berechne nur für valide (nicht-rejected) Trials
                valid_trials = df[
                    (df['mediapipe_rejected'] == False) & 
                    (df['ptgaze_rejected'] == False)
                ]['trial_number'].tolist()
                
                if len(valid_trials) > 0:
                    valid_trial_duration_ms = 0
                    valid_fixation_duration_ms = 0
                    
                    fixations_df = pd.read_csv(fixations_csv) if os.path.exists(fixations_csv) else None
                    
                    if fixations_df is not None:
                        for block_key in ['experiment_block1', 'experiment_block2']:
                            if block_key not in phases.get('phases', {}):
                                continue
                            
                            for trial in phases['phases'][block_key]['trials']:
                                trial_num = trial['trial_number']
                                
                                if trial_num not in valid_trials:
                                    continue
                                
                                if 'stimulus' in trial:
                                    start_ms = trial['fixation']['start_eyelink_ms']
                                    end_ms = trial['stimulus']['end_eyelink_ms']
                                    valid_trial_duration_ms += (end_ms - start_ms)
                                    
                                    trial_fixations = fixations_df[
                                        (fixations_df['start_time'] >= start_ms) &
                                        (fixations_df['end_time'] <= end_ms)
                                    ]
                                    
                                    if 'duration' in trial_fixations.columns:
                                        valid_fixation_duration_ms += trial_fixations['duration'].sum()
                    
                    if valid_trial_duration_ms > 0:
                        valid_fixation_ratio = valid_fixation_duration_ms / valid_trial_duration_ms
                        
                        print(f"\n┌─────────────────────────────────────────────────────────────────────┐")
                        print(f"│ FIXATIONSZEIT (NUR KORRELIERTE TRIALS: {len(valid_trials)})                         │")
                        print(f"├─────────────────────────────────────────────────────────────────────┤")
                        print(f"│                                                                     │")
                        print(f"│  Logik: Nur Trials die für Korrelationsberechnung verwendet werden  │")
                        print(f"│         (ohne rejected Trials durch Blink/Outside-Filterung)        │")
                        print(f"│                                                                     │")
                        print(f"├─────────────────────────────────────────────────────────────────────┤")
                        print(f"│  Valide Trials:            {len(valid_trials):>8} / {len(trial_list)}                             │")
                        print(f"│  Gesamtdauer (valide):     {valid_trial_duration_ms/1000:>8.2f} s                            │")
                        print(f"│  Fixationszeit (valide):   {valid_fixation_duration_ms/1000:>8.2f} s                            │")
                        print(f"│                                                                     │")
                        print(f"│  ► Fixationsanteil:        {valid_fixation_ratio*100:>8.1f} %                            │")
                        print(f"└─────────────────────────────────────────────────────────────────────┘")

        # Valide Trials
        if eyelink_data is not None:
            valid = df.dropna(subset=['mediapipe_mean_x', 'ptgaze_mean_x', 'eyelink_mean_x'])
        else:
            valid = df.dropna(subset=['mediapipe_mean_x', 'ptgaze_mean_x'])
        
        rejected = len(df) - len(valid)
        
        print(f"\n{'─'*70}")
        print(f"✓ Valide Trials für Vergleich: {len(valid)}/{len(df)}")
        if rejected > 0:
            print(f"⚠ Nicht vergleichbar (NaN in einer Methode): {rejected}")
        print(f"{'─'*70}\n")
        
        return df
    
    def _extract_trials_from_phases(self, phases: dict) -> list:
        """Extrahiert Trial-Liste aus phases_detected.json"""
        
        trial_list = []
        
        for block_key in ['experiment_block1', 'experiment_block2']:
            if block_key not in phases['phases']:
                continue
            
            for trial in phases['phases'][block_key]['trials']:
                if not trial.get('is_practice', False):
                    trial_list.append(trial['trial_number'])
        
        return sorted(trial_list)

# ==================== METRIKEN ====================

def calculate_pairwise_metrics(df: pd.DataFrame, method_a: str, method_b: str) -> dict:
    """
    Berechnet Metriken fuer ein Methodenpaar.
    
    Args:
        df: Vergleichs-DataFrame (von ThreeWayComparison)
        method_a, method_b: 'mediapipe', 'ptgaze', 'eyelink'
    
    Returns:
        dict mit Korrelationen, MAE, Bias, LoA
    """
    
    col_a = f'{method_a}_mean_x'
    col_b = f'{method_b}_mean_x'
    
    valid = df.dropna(subset=[col_a, col_b])
    
    if len(valid) < 3:
        return {
            'n_trials': len(valid),
            'error': 'Zu wenige Trials'
        }
    
    a_vals = valid[col_a].values
    b_vals = valid[col_b].values
    
    # Korrelationen
    pearson_r, pearson_p = stats.pearsonr(a_vals, b_vals)
    spearman_rho, spearman_p = stats.spearmanr(a_vals, b_vals)
    
    # Fehler
    diff = a_vals - b_vals
    mae = np.abs(diff).mean()
    rmse = np.sqrt((diff ** 2).mean())
    bias = diff.mean()
    std_err = diff.std()
    
    # Bland-Altman LoA
    loa_lower = bias - 1.96 * std_err
    loa_upper = bias + 1.96 * std_err
    
    # Lateralization Agreement
    lat_a = ['left' if x < LEFT_THRESHOLD_DEG else ('right' if x > RIGHT_THRESHOLD_DEG else 'middle') for x in a_vals]
    lat_b = ['left' if x < LEFT_THRESHOLD_DEG else ('right' if x > RIGHT_THRESHOLD_DEG else 'middle') for x in b_vals]
    
    lat_agreement = sum([1 for a, b in zip(lat_a, lat_b) if a == b]) / len(lat_a)
    
    return {
        'n_trials': len(valid),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_rho': spearman_rho,
        'spearman_p': spearman_p,
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
        'std_error': std_err,
        'loa_lower': loa_lower,
        'loa_upper': loa_upper,
        'lateralization_agreement': lat_agreement
    }

# ==================== VISUALISIERUNGEN ====================

def create_three_way_visualization(comparison_df: pd.DataFrame, 
                                   metrics: dict,
                                   output_dir: str):
    """
    Erstellt umfassende 3-Wege-Visualisierung.
    
    Layout (2x3):
    - Row 1: Scatter Plots (MediaPipe vs. EyeLink, ptgaze vs. EyeLink, MediaPipe vs. ptgaze)
    - Row 2: Bland-Altman, Fehler-Verteilung, Lateralization Agreement
    """
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # ═════════════════════════════════════════════════════════════════
    # ROW 1: SCATTER PLOTS
    # ═════════════════════════════════════════════════════════════════
    
    # Panel 1: MediaPipe vs. EyeLink
    ax1 = fig.add_subplot(gs[0, 0])
    
    valid_mp_et = comparison_df.dropna(subset=['mediapipe_mean_x', 'eyelink_mean_x'])
    
    if len(valid_mp_et) > 0:
        ax1.scatter(valid_mp_et['eyelink_mean_x'], valid_mp_et['mediapipe_mean_x'],
                   s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
        
        lim_min = min(valid_mp_et['eyelink_mean_x'].min(), valid_mp_et['mediapipe_mean_x'].min()) - 1
        lim_max = max(valid_mp_et['eyelink_mean_x'].max(), valid_mp_et['mediapipe_mean_x'].max()) + 1
        ax1.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', alpha=0.5, linewidth=2, label='Perfect Agreement')
        
        mp_et_metrics = metrics.get('mediapipe_vs_eyelink', {})
        r = mp_et_metrics.get('pearson_r', 0)
        mae = mp_et_metrics.get('mae', 0)
        
        ax1.set_xlabel('EyeLink Gaze X (deg)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('MediaPipe Gaze X (deg)', fontsize=11, fontweight='bold')
        ax1.set_title(f'MediaPipe vs. EyeLink\nr={r:.3f}, MAE={mae:.2f}deg', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'Keine EyeLink-Daten', ha='center', va='center', fontsize=14, transform=ax1.transAxes)
        ax1.axis('off')
    
    # Panel 2: ptgaze vs. EyeLink
    ax2 = fig.add_subplot(gs[0, 1])
    
    valid_pt_et = comparison_df.dropna(subset=['ptgaze_mean_x', 'eyelink_mean_x'])
    
    if len(valid_pt_et) > 0:
        ax2.scatter(valid_pt_et['eyelink_mean_x'], valid_pt_et['ptgaze_mean_x'],
                   s=100, alpha=0.7, c='orange', edgecolors='black', linewidth=1.5)
        
        lim_min = min(valid_pt_et['eyelink_mean_x'].min(), valid_pt_et['ptgaze_mean_x'].min()) - 1
        lim_max = max(valid_pt_et['eyelink_mean_x'].max(), valid_pt_et['ptgaze_mean_x'].max()) + 1
        ax2.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', alpha=0.5, linewidth=2, label='Perfect Agreement')
        
        pt_et_metrics = metrics.get('ptgaze_vs_eyelink', {})
        r = pt_et_metrics.get('pearson_r', 0)
        mae = pt_et_metrics.get('mae', 0)
        
        ax2.set_xlabel('EyeLink Gaze X (deg)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('ptgaze Gaze X (deg)', fontsize=11, fontweight='bold')
        ax2.set_title(f'ptgaze vs. EyeLink\nr={r:.3f}, MAE={mae:.2f}deg', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Keine EyeLink-Daten', ha='center', va='center', fontsize=14, transform=ax2.transAxes)
        ax2.axis('off')
    
    # Panel 3: MediaPipe vs. ptgaze (IMMER verfuegbar!)
    ax3 = fig.add_subplot(gs[0, 2])
    
    valid_mp_pt = comparison_df.dropna(subset=['mediapipe_mean_x', 'ptgaze_mean_x'])
    
    ax3.scatter(valid_mp_pt['mediapipe_mean_x'], valid_mp_pt['ptgaze_mean_x'],
               s=100, alpha=0.7, c='green', edgecolors='black', linewidth=1.5)
    
    lim_min = min(valid_mp_pt['mediapipe_mean_x'].min(), valid_mp_pt['ptgaze_mean_x'].min()) - 1
    lim_max = max(valid_mp_pt['mediapipe_mean_x'].max(), valid_mp_pt['ptgaze_mean_x'].max()) + 1
    ax3.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', alpha=0.5, linewidth=2, label='Perfect Agreement')
    
    mp_pt_metrics = metrics.get('mediapipe_vs_ptgaze', {})
    r = mp_pt_metrics.get('pearson_r', 0)
    mae = mp_pt_metrics.get('mae', 0)
    
    ax3.set_xlabel('MediaPipe Gaze X (deg)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('ptgaze Gaze X (deg)', fontsize=11, fontweight='bold')
    ax3.set_title(f'MediaPipe vs. ptgaze\nr={r:.3f}, MAE={mae:.2f}deg', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # ═════════════════════════════════════════════════════════════════
    # ROW 2: BLAND-ALTMAN & FEHLER
    # ═════════════════════════════════════════════════════════════════
    
    # Panel 4: Bland-Altman (MediaPipe vs. EyeLink)
    ax4 = fig.add_subplot(gs[1, 0])
    
    if len(valid_mp_et) > 0:
        mean_vals = (valid_mp_et['mediapipe_mean_x'] + valid_mp_et['eyelink_mean_x']) / 2
        diff_vals = valid_mp_et['mediapipe_mean_x'] - valid_mp_et['eyelink_mean_x']
        
        ax4.scatter(mean_vals, diff_vals, alpha=0.6, s=100, edgecolors='black', linewidth=1)
        
        bias = mp_et_metrics.get('bias', 0)
        loa_lower = mp_et_metrics.get('loa_lower', 0)
        loa_upper = mp_et_metrics.get('loa_upper', 0)
        
        ax4.axhline(bias, color='red', linestyle='-', linewidth=2, label=f'Bias: {bias:.2f}deg')
        ax4.axhline(loa_lower, color='red', linestyle='--', linewidth=1.5, label=f'LoA: [{loa_lower:.2f}, {loa_upper:.2f}]deg')
        ax4.axhline(loa_upper, color='red', linestyle='--', linewidth=1.5)
        ax4.axhline(0, color='gray', linestyle=':', alpha=0.5)
        
        ax4.set_xlabel('Mean [(MediaPipe + EyeLink)/2] (deg)', fontsize=10)
        ax4.set_ylabel('Diff (MediaPipe - EyeLink) (deg)', fontsize=10)
        ax4.set_title('Bland-Altman: MediaPipe vs. EyeLink', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.axis('off')
    
    # Panel 5: Bland-Altman (ptgaze vs. EyeLink)
    ax5 = fig.add_subplot(gs[1, 1])
    
    if len(valid_pt_et) > 0:
        mean_vals = (valid_pt_et['ptgaze_mean_x'] + valid_pt_et['eyelink_mean_x']) / 2
        diff_vals = valid_pt_et['ptgaze_mean_x'] - valid_pt_et['eyelink_mean_x']
        
        ax5.scatter(mean_vals, diff_vals, alpha=0.6, s=100, c='orange', edgecolors='black', linewidth=1)
        
        bias = pt_et_metrics.get('bias', 0)
        loa_lower = pt_et_metrics.get('loa_lower', 0)
        loa_upper = pt_et_metrics.get('loa_upper', 0)
        
        ax5.axhline(bias, color='red', linestyle='-', linewidth=2, label=f'Bias: {bias:.2f}deg')
        ax5.axhline(loa_lower, color='red', linestyle='--', linewidth=1.5)
        ax5.axhline(loa_upper, color='red', linestyle='--', linewidth=1.5)
        ax5.axhline(0, color='gray', linestyle=':', alpha=0.5)
        
        ax5.set_xlabel('Mean [(ptgaze + EyeLink)/2] (deg)', fontsize=10)
        ax5.set_ylabel('Diff (ptgaze - EyeLink) (deg)', fontsize=10)
        ax5.set_title('Bland-Altman: ptgaze vs. EyeLink', fontsize=11, fontweight='bold')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
    else:
        ax5.axis('off')
    
    # Panel 6: Fehler-Vergleich (alle Paare)
    ax6 = fig.add_subplot(gs[1, 2])
    
    mae_values = []
    labels = []
    colors = []
    
    for pair, color in [('mediapipe_vs_eyelink', 'blue'), 
                        ('ptgaze_vs_eyelink', 'orange'),
                        ('mediapipe_vs_ptgaze', 'green')]:
        if pair in metrics and 'mae' in metrics[pair]:
            mae_values.append(metrics[pair]['mae'])
            labels.append(pair.replace('_', ' ').replace('vs', 'vs.').title())
            colors.append(color)
    
    if mae_values:
        bars = ax6.bar(range(len(mae_values)), mae_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        for i, (bar, val) in enumerate(zip(bars, mae_values)):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2f}deg', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax6.set_xticks(range(len(labels)))
        ax6.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
        ax6.set_ylabel('MAE (deg)', fontsize=11, fontweight='bold')
        ax6.set_title('Mean Absolute Error Vergleich', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.set_ylim(0, max(mae_values) * 1.2)
    else:
        ax6.text(0.5, 0.5, 'Keine Metriken', ha='center', va='center', fontsize=14, transform=ax6.transAxes)
        ax6.axis('off')
    
    # Suptitle
    n_trials = len(comparison_df)
    plt.suptitle(f'3-Wege-Vergleich: MediaPipe vs. ptgaze vs. EyeLink | {n_trials} Trials', 
                fontsize=16, fontweight='bold', y=0.995)
    
    output_path = os.path.join(output_dir, "debug_6_three_way_comparison.png")
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    print(f" 3-Wege-Visualisierung: {output_path}")
    plt.close()

# ==================== MAIN ====================

if __name__ == "__main__":
    mode = ANALYSIS_MODE
    
    print(f"\n{'='*70}")
    print(f"DEBUG 6 EXTENDED: 3-WEGE-VERGLEICH v1.0 (MODUS {mode})")
    print(f"{'='*70}\n")
    
    # ══════════════════════════════════════════════════════════════════
    # Lade Daten
    # ══════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print("DATEN LADEN")
    print(f"{'='*70}\n")
    
    # MediaPipe
    mediapipe_data = load_gaze_data('mediapipe')
    if mediapipe_data is None:
        exit(1)
    
    # ptgaze
    ptgaze_data = load_gaze_data('ptgaze')
    if ptgaze_data is None:
        exit(1)
    
    # EyeLink (optional, nur Modus 2/3)
    eyelink_data = None
    
    if mode in [2, 3]:
        et_csv = os.path.join(OUTPUT_BASE_DIR, "debug_3_eyetracker_data.csv")
        
        if os.path.exists(et_csv):
            eyelink_data = pd.read_csv(et_csv)
            print(f" EYELINK Daten geladen: {len(eyelink_data)} Samples")
        else:
            print(f" WARNUNG: EyeLink-Daten nicht verfuegbar (Modus {mode})")
            print(f"   >> Vergleich nur: MediaPipe vs. ptgaze")
    
    # phases_detected.json
    phases_json = os.path.join(OUTPUT_BASE_DIR, "phases_detected.json")
    
    if os.path.exists(phases_json):
        with open(phases_json, 'r') as f:
            phases = json.load(f)
        print(f" phases_detected.json geladen")
    else:
        phases = None
        print(f" WARNUNG: phases_detected.json fehlt (nutze Trial-Nummern aus Daten)")
    
    # ══════════════════════════════════════════════════════════════════
    # 3-Wege-Vergleich
    # ══════════════════════════════════════════════════════════════════
    
    comparator = ThreeWayComparison(mode=mode)
    
    comparison_df = comparator.compare(
        mediapipe_data=mediapipe_data,
        ptgaze_data=ptgaze_data,
        eyelink_data=eyelink_data,
        phases=phases
    )
    
    # ══════════════════════════════════════════════════════════════════
    # Metriken berechnen
    # ══════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print("METRIKEN BERECHNEN")
    print(f"{'='*70}\n")
    
    metrics = {}
    
    # MediaPipe vs. EyeLink
    if eyelink_data is not None:
        print(" MediaPipe vs. EyeLink...")
        metrics['mediapipe_vs_eyelink'] = calculate_pairwise_metrics(
            comparison_df, 'mediapipe', 'eyelink'
        )
        
        m = metrics['mediapipe_vs_eyelink']
        if 'error' not in m:
            print(f"   r={m['pearson_r']:.3f} (p={m['pearson_p']:.4f})")
            print(f"   MAE={m['mae']:.2f}deg, Bias={m['bias']:.2f}deg +/- {m['std_error']:.2f}deg")
            print(f"   Lat. Agreement: {m['lateralization_agreement']*100:.1f}%")
    
    # ptgaze vs. EyeLink
    if eyelink_data is not None:
        print("\n ptgaze vs. EyeLink...")
        metrics['ptgaze_vs_eyelink'] = calculate_pairwise_metrics(
            comparison_df, 'ptgaze', 'eyelink'
        )
        
        m = metrics['ptgaze_vs_eyelink']
        if 'error' not in m:
            print(f"   r={m['pearson_r']:.3f} (p={m['pearson_p']:.4f})")
            print(f"   MAE={m['mae']:.2f}deg, Bias={m['bias']:.2f}deg +/- {m['std_error']:.2f}deg")
            print(f"   Lat. Agreement: {m['lateralization_agreement']*100:.1f}%")
    
    # MediaPipe vs. ptgaze (immer!)
    print("\n MediaPipe vs. ptgaze...")
    metrics['mediapipe_vs_ptgaze'] = calculate_pairwise_metrics(
        comparison_df, 'mediapipe', 'ptgaze'
    )
    
    m = metrics['mediapipe_vs_ptgaze']
    if 'error' not in m:
        print(f"   r={m['pearson_r']:.3f} (p={m['pearson_p']:.4f})")
        print(f"   MAE={m['mae']:.2f}deg, Bias={m['bias']:.2f}deg +/- {m['std_error']:.2f}deg")
        print(f"   Lat. Agreement: {m['lateralization_agreement']*100:.1f}%")
    
    # ══════════════════════════════════════════════════════════════════
    # Visualisierungen
    # ══════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print("VISUALISIERUNGEN")
    print(f"{'='*70}\n")
    
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    create_three_way_visualization(comparison_df, metrics, OUTPUT_BASE_DIR)
    
    # ══════════════════════════════════════════════════════════════════
    # Export
    # ══════════════════════════════════════════════════════════════════
    
    comparison_csv = os.path.join(OUTPUT_BASE_DIR, "debug_6_three_way_comparison.csv")
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\n Comparison CSV: {comparison_csv}")
    
    metrics_json = os.path.join(OUTPUT_BASE_DIR, "debug_6_three_way_metrics.json")
    with open(metrics_json, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f" Metrics JSON: {metrics_json}")
    
    # ══════════════════════════════════════════════════════════════════
    # Zusammenfassung
    # ══════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print("ZUSAMMENFASSUNG")
    print(f"{'='*70}\n")
    
    print(f" Trials analysiert: {len(comparison_df)}")
    
    valid_all = comparison_df.dropna(subset=['mediapipe_mean_x', 'ptgaze_mean_x'])
    print(f" Valide (beide Methoden): {len(valid_all)}")
    
    if eyelink_data is not None:
        valid_with_et = comparison_df.dropna(subset=['mediapipe_mean_x', 'ptgaze_mean_x', 'eyelink_mean_x'])
        print(f" Valide (alle 3 Methoden): {len(valid_with_et)}")
    
    print(f"\n Beste Methode (vs. EyeLink):")
    if eyelink_data is not None:
        mp_mae = metrics.get('mediapipe_vs_eyelink', {}).get('mae', np.inf)
        pt_mae = metrics.get('ptgaze_vs_eyelink', {}).get('mae', np.inf)
        
        if mp_mae < pt_mae:
            print(f"   >> MediaPipe (MAE={mp_mae:.2f}deg)")
        else:
            print(f"   >> ptgaze (MAE={pt_mae:.2f}deg)")
        
        print(f"\n Differenz: {abs(mp_mae - pt_mae):.2f}deg")
    else:
        print(f"   (Kein EyeLink zum Vergleich)")
    
    print(f"\n{'='*70}")
    print("ERFOLGREICH!")
    print(f"{'='*70}")