"""
=================================================================================
DEBUG_5_PTGAZE.PY - Kalibrierungs-Anwendung fuer ptgaze v1.1
=================================================================================
Analog zu debug_5_partial_calibration.py, aber fuer ptgaze Gaze Angles:

Input:
- Gaze Angles (pitch, yaw) in GRAD (von debug_1_ptgaze.py oder debug_4)
- calibration_ptgaze_*.pkl (von offline_calibration_ptgaze.py)

Output:
- debug_5_ptgaze_calibrated.csv (mit Screen-Pixel + Gradsehwinkel)

Unterschiede zu MediaPipe-Version:
- Input: (pitch, yaw) statt (pupil_x, pupil_y)
- Polynomial: (pitch, yaw) -> (screen_x, screen_y)
- Keine Pupillen-Outlier-Filter (Gaze Angles sind bereits "sauber")

NEU in v1.1:
------------
- Package-Struktur (debug/)
- Import aus shared.shared_coordinate_utils

Version: v1.1 (basierend auf debug_5_partial_calibration v2.4)
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
import pickle
import json
from typing import Tuple, Dict

from config import *

from shared.shared_coordinate_utils import (
    CoordinateTransformer,
    ScreenParameters,
    CalibrationMetadata,
    create_coordinate_transformer
)

# Environment-Variable-Overrides
if 'PIPELINE_OUTPUT_BASE_DIR' in os.environ:
    OUTPUT_BASE_DIR = Path(os.environ['PIPELINE_OUTPUT_BASE_DIR'])

if 'PIPELINE_CALIBRATION_PKL_PATH' in os.environ:
    CALIBRATION_PKL_PATH = Path(os.environ['PIPELINE_CALIBRATION_PKL_PATH'])

# ==================== KALIBRIERUNGS-MANAGER ====================

class PtgazeCalibrationManager:
    """
    Verwaltet Kalibrierung fuer ptgaze Gaze Angles.
    
    Analog zu CalibrationManager (MediaPipe), aber:
    - Input: (pitch, yaw) in GRAD
    - Nutzt calibration_ptgaze_*.pkl
    """
    
    def __init__(self, mode: int):
        self.mode = mode
        
        screen_params = {
            'width_px': SCREEN_WIDTH_PX,
            'height_px': SCREEN_HEIGHT_PX,
            'center_x': SCREEN_WIDTH_PX // 2,
            'center_y': SCREEN_HEIGHT_PX // 2,
            'width_cm': SCREEN_WIDTH_CM,
            'height_cm': SCREEN_HEIGHT_CM,
            'viewing_distance_cm': VIEWING_DISTANCE_CM
        }
        
        self.transformer = CoordinateTransformer(screen_params)
    
    def calibrate(self, gaze_data: pd.DataFrame):
        """
        Wendet Kalibrierung auf ptgaze Gaze Angles an.
        
        Args:
            gaze_data: DataFrame mit gaze_pitch_deg, gaze_yaw_deg
        
        Returns:
            (gaze_calibrated, cal_info)
        """
        
        mode_name = "STANDALONE" if self.mode == 1 else "COMPARISON"
        print(f"\n{'='*70}")
        print(f"PTGAZE KALIBRIERUNG - MODUS {self.mode}: {mode_name}")
        print(f"{'='*70}\n")
        
        # ======================================================================
        # SCHRITT 1: LADE .PKL
        # ======================================================================
        
        # Suche nach calibration_ptgaze_*.pkl
        pkl_files = list(Path(OUTPUT_BASE_DIR).glob("calibration_ptgaze_*.pkl"))
        
        if not pkl_files:
            # Fallback: Nutze Standard-Name aus config
            pkl_path = Path(CALIBRATION_PKL_PATH)
            if not pkl_path.exists():
                print(f"[!] FEHLER: Keine calibration_ptgaze_*.pkl gefunden!")
                print(f"   Gesucht in: {OUTPUT_BASE_DIR}")
                print(f"   Erwarte: calibration_ptgaze_*.pkl")
                print(f"\n[>] Workflow:")
                print(f"   1. Laufe offline_calibration_ptgaze.py zuerst")
                print(f"   2. Dann debug_5_ptgaze.py (dieser)")
                return gaze_data, {'success': False, 'reason': 'calibration_file_not_found'}
        else:
            # Nutze neueste Datei
            pkl_path = max(pkl_files, key=lambda p: p.stat().st_mtime)
            print(f"[>] Kalibrierung gefunden: {pkl_path.name}")
        
        try:
            with open(pkl_path, 'rb') as f:
                calibration_data = pickle.load(f)
            
            method = calibration_data.get('method', 'unknown')
            poly_degree = calibration_data.get('polynomial_degree', 1)
            n_points = calibration_data.get('n_training_points', 0)
            r2_scores = calibration_data.get('r2_scores', (0, 0))
            input_type = calibration_data.get('input_type', 'unknown')
            
            print(f"   Methode: {method}")
            print(f"   Input-Type: {input_type}")
            print(f"   Polynomial-Grad: {poly_degree}")
            print(f"   Trainings-Punkte: {n_points}")
            print(f"   R2: X={r2_scores[0]:.3f}, Y={r2_scores[1]:.3f}")
            
            # Validierung: Ist das wirklich ein ptgaze PKL?
            if input_type != 'gaze_angles':
                print(f"\n[!] WARNUNG: Kalibrierung scheint NICHT fuer ptgaze zu sein!")
                print(f"   Input-Type: {input_type} (erwartet: 'gaze_angles')")
                print(f"   >> Nutze offline_calibration_ptgaze.py fuer ptgaze-Kalibrierung!")
                return gaze_data, {'success': False, 'reason': 'wrong_calibration_type'}
            
            if r2_scores[0] < MIN_CALIBRATION_R2 or r2_scores[1] < MIN_CALIBRATION_R2:
                print(f"\n[!] Niedrige R2 (<{MIN_CALIBRATION_R2})")
        
        except Exception as e:
            print(f"[!] Laden fehlgeschlagen: {e}")
            return gaze_data, {'success': False, 'reason': 'calibration_load_error', 'error': str(e)}
        
        # ======================================================================
        # SCHRITT 2: SPALTEN-VALIDIERUNG
        # ======================================================================
        
        print(f"\n{'='*70}")
        print("SPALTEN-VALIDIERUNG")
        print(f"{'='*70}\n")
        
        if 'gaze_pitch_deg' not in gaze_data.columns or 'gaze_yaw_deg' not in gaze_data.columns:
            print(f"[!] FEHLER: Keine Gaze Angles in CSV!")
            print(f"   Erwarte: gaze_pitch_deg, gaze_yaw_deg")
            print(f"   Gefunden: {list(gaze_data.columns)}")
            return gaze_data, {'success': False, 'reason': 'invalid_columns'}
        
        print(f"[>] Gaze Angles gefunden")
        print(f"   Input: gaze_pitch_deg, gaze_yaw_deg")
        
        # ======================================================================
        # SCHRITT 3: POLYNOMIAL-PREDICTION
        # ======================================================================
        
        print(f"\n{'='*70}")
        print("KALIBRIERUNGS-ANWENDUNG")
        print(f"{'='*70}\n")
        
        model_x = calibration_data.get('model_x')
        model_y = calibration_data.get('model_y')
        poly_transformer = calibration_data.get('poly_transformer')
        
        # Extrahiere valide Samples
        valid_mask = (
            gaze_data['gaze_pitch_deg'].notna() &
            gaze_data['gaze_yaw_deg'].notna() &
            (gaze_data.get('confidence', 1.0) >= MIN_CONFIDENCE)
        )
        
        # WICHTIG: Input ist (pitch, yaw) in GRAD!
        X = gaze_data.loc[valid_mask, ['gaze_pitch_deg', 'gaze_yaw_deg']].values
        
        print(f"[>] Input-Statistik:")
        print(f"   Valide Samples: {valid_mask.sum()}/{len(gaze_data)} ({valid_mask.sum()/len(gaze_data)*100:.1f}%)")
        print(f"   Pitch: {X[:, 0].min():.2f} - {X[:, 0].max():.2f} deg")
        print(f"   Yaw: {X[:, 1].min():.2f} - {X[:, 1].max():.2f} deg")
        
        # Transformiere Input (falls Polynomial)
        if poly_degree > 1 and poly_transformer is not None:
            print(f"\n[>] Polynomial-Transformation (Grad {poly_degree})")
            X_transformed = poly_transformer.transform(X)
            print(f"   Features: {X.shape[1]} -> {X_transformed.shape[1]}")
        else:
            X_transformed = X
            print(f"\n[>] Linear (keine Transformation)")
        
        # Prediction
        try:
            pred_x = model_x.predict(X_transformed)
            pred_y = model_y.predict(X_transformed)
            
            print(f"\n[>] Prediction erfolgreich")
            print(f"   Screen X: {pred_x.min():.1f} - {pred_x.max():.1f} px")
            print(f"   Screen Y: {pred_y.min():.1f} - {pred_y.max():.1f} px")
        
        except Exception as e:
            print(f"[!] Prediction Error: {e}")
            return gaze_data, {'success': False, 'reason': 'prediction_error', 'error': str(e)}
        
        # Clipping
        pred_x_clipped = np.clip(pred_x, 0, SCREEN_WIDTH_PX)
        pred_y_clipped = np.clip(pred_y, 0, SCREEN_HEIGHT_PX)
        
        # Speichere in DataFrame
        gaze_calibrated = gaze_data.copy()
        if 'timestamp_ms_synced' in gaze_data.columns:
            gaze_calibrated['timestamp_ms_synced'] = gaze_data['timestamp_ms_synced']
        gaze_calibrated.loc[valid_mask, 'gaze_screen_x_calib_px'] = pred_x_clipped
        gaze_calibrated.loc[valid_mask, 'gaze_screen_y_calib_px'] = pred_y_clipped
        
        # ======================================================================
        # SCHRITT 4: GRADSEHWINKEL-TRANSFORMATION
        # ======================================================================
        
        print(f"\n{'='*70}")
        print("GRADSEHWINKEL-TRANSFORMATION")
        print(f"{'='*70}\n")
        
        deg_x_list = []
        deg_y_list = []
        plausibility_list = []
        
        for _, row in gaze_calibrated.iterrows():
            if pd.notna(row.get('gaze_screen_x_calib_px')) and pd.notna(row.get('gaze_screen_y_calib_px')):
                deg_x, deg_y = self.transformer.pixels_to_degrees(
                    row['gaze_screen_x_calib_px'],
                    row['gaze_screen_y_calib_px']
                )
                
                plausible = (abs(deg_x) <= MAX_PLAUSIBLE_DEG_X and 
                            abs(deg_y) <= MAX_PLAUSIBLE_DEG_Y)
            else:
                deg_x, deg_y = np.nan, np.nan
                plausible = False
            
            deg_x_list.append(deg_x)
            deg_y_list.append(deg_y)
            plausibility_list.append(plausible)
        
        gaze_calibrated['gaze_deg_x_calib'] = deg_x_list
        gaze_calibrated['gaze_deg_y_calib'] = deg_y_list
        gaze_calibrated['plausibility_check'] = plausibility_list
        
        # Statistik
        valid_deg = gaze_calibrated[gaze_calibrated['plausibility_check']].dropna(subset=['gaze_deg_x_calib'])
        
        print(f"[>] Gradsehwinkel:")
        print(f"   Gueltig: {len(valid_deg)}/{len(gaze_calibrated)} ({len(valid_deg)/len(gaze_calibrated)*100:.1f}%)")
        
        if len(valid_deg) > 0:
            print(f"   X: {valid_deg['gaze_deg_x_calib'].min():.2f} - {valid_deg['gaze_deg_x_calib'].max():.2f} deg")
            print(f"   Y: {valid_deg['gaze_deg_y_calib'].min():.2f} - {valid_deg['gaze_deg_y_calib'].max():.2f} deg")
            print(f"   Mean: ({valid_deg['gaze_deg_x_calib'].mean():.2f}, {valid_deg['gaze_deg_y_calib'].mean():.2f}) deg")
        
        n_implausible = (~gaze_calibrated['plausibility_check']).sum()
        if n_implausible > 0:
            print(f"\n[!] {n_implausible} ausserhalb plausiblem Bereich (>{MAX_PLAUSIBLE_DEG_X:.0f} deg X, >{MAX_PLAUSIBLE_DEG_Y:.0f} deg Y)")
        
        # ══════════════════════════════════════════════════════════════════
        # SCHRITT 4b: OUTSIDE-MONITOR DETECTION (NEU!)
        # ══════════════════════════════════════════════════════════════════
        
        print(f"\n{'='*70}")
        print("OUTSIDE-MONITOR DETECTION")
        print(f"{'='*70}\n")
        
        # Berechne maximale Sehwinkel aus Screen-Parametern
        max_deg_x = np.arctan2(SCREEN_WIDTH_CM / 2, VIEWING_DISTANCE_CM) * 180 / np.pi
        max_deg_y = np.arctan2(SCREEN_HEIGHT_CM / 2, VIEWING_DISTANCE_CM) * 180 / np.pi
        
        print(f"[>] Monitor-Grenzen (basierend auf {VIEWING_DISTANCE_CM}cm Abstand):")
        print(f"    X: ±{max_deg_x:.1f}° (Screen: {SCREEN_WIDTH_CM}cm breit)")
        print(f"    Y: ±{max_deg_y:.1f}° (Screen: {SCREEN_HEIGHT_CM}cm hoch)")
        
        # Markiere Outside-Monitor
        gaze_calibrated['outside_monitor'] = (
            (gaze_calibrated['gaze_deg_x_calib'].abs() > max_deg_x) |
            (gaze_calibrated['gaze_deg_y_calib'].abs() > max_deg_y)
        )
        
        n_outside = gaze_calibrated['outside_monitor'].sum()
        n_total_valid = gaze_calibrated['gaze_deg_x_calib'].notna().sum()
        
        if n_outside > 0:
            print(f"\n {n_outside}/{n_total_valid} Samples außerhalb Monitor ({n_outside/n_total_valid*100:.1f}%)")
        else:
            print(f"\n✓ Alle Samples innerhalb Monitor-Grenzen")
        
        # ══════════════════════════════════════════════════════════════════
        # SCHRITT 4c: BLINK-SPALTEN VALIDIERUNG (NEU!)
        # ══════════════════════════════════════════════════════════════════
        
        print(f"\n{'='*70}")
        print("BLINK-DATEN VALIDIERUNG")
        print(f"{'='*70}\n")
        
        # Prüfe ob Blink-Spalten vorhanden sind
        blink_columns = ['is_blink', 'eyes_closed', 'avg_ear', 'left_ear', 'right_ear', 'blink_count']
        missing_blink_cols = [col for col in blink_columns if col not in gaze_calibrated.columns]
        
        if missing_blink_cols:
            print(f"Fehlende Blink-Spalten: {missing_blink_cols}")
            print(f"  >> Blink-Filterung in debug_6 nicht möglich!")
            
            # Füge leere Spalten hinzu (für Kompatibilität)
            for col in missing_blink_cols:
                if col == 'is_blink' or col == 'eyes_closed':
                    gaze_calibrated[col] = False
                elif col == 'blink_count':
                    gaze_calibrated[col] = 0
                else:
                    gaze_calibrated[col] = np.nan
        else:
            # Statistik
            n_eyes_closed = gaze_calibrated['eyes_closed'].sum()
            n_blinks = gaze_calibrated['is_blink'].sum()
            total_blink_count = gaze_calibrated['blink_count'].max()
            
            print(f"✓ Alle Blink-Spalten vorhanden")
            print(f"  • Frames mit geschlossenen Augen: {n_eyes_closed} ({n_eyes_closed/len(gaze_calibrated)*100:.1f}%)")
            print(f"  • Blink-Events detektiert: {n_blinks}")
            print(f"  • Gesamt-Blink-Count: {total_blink_count}")
            
            # EAR-Statistik
            valid_ear = gaze_calibrated['avg_ear'].dropna()
            if len(valid_ear) > 0:
                print(f"  • EAR: Mean={valid_ear.mean():.3f}, Min={valid_ear.min():.3f}, Max={valid_ear.max():.3f}")

        # ======================================================================
        # SCHRITT 5: QUALITAETS-METRIKEN
        # ======================================================================
        
        gaze_calibrated['calibration_method'] = f"polynomial_degree_{poly_degree}_ptgaze"
        gaze_calibrated['calibration_r2_x'] = r2_scores[0]
        gaze_calibrated['calibration_r2_y'] = r2_scores[1]
        gaze_calibrated['calibration_source'] = str(pkl_path.name)
        
        # Metadaten
        cal_info = {
            'success': True,
            'calibration_type': f'polynomial_degree_{poly_degree}_ptgaze',
            'mode': self.mode,
            'input_type': 'gaze_angles',
            'n_calibration_points': n_points,
            'r2_scores': list(r2_scores),
            'method': method,
            'pkl_file': str(pkl_path.name),
            'n_samples_total': len(gaze_calibrated),
            'n_samples_valid': len(valid_deg),
            'n_samples_implausible': int(n_implausible)
        }
        
        print(f"\n[>] Kalibrierung erfolgreich!")
        
        return gaze_calibrated, cal_info

# ==================== MAIN ====================

if __name__ == "__main__":
    mode = ANALYSIS_MODE
    
    print(f"\n{'='*70}")
    print(f"DEBUG 5 PTGAZE: KALIBRIERUNG v1.0 (MODUS {mode})")
    print(f"{'='*70}\n")
    
    # ======================================================================
    # Lade Gaze-Daten
    # ======================================================================
    
    if mode in [2, 3]:
        # Nach Synchronisation
        gaze_csv = os.path.join(OUTPUT_BASE_DIR, "debug_4_ptgaze_gaze_synced.csv")
        
        # Fallback: Falls debug_4 nicht gelaufen (ptgaze ist generisch!)
        if not os.path.exists(gaze_csv):
            print(f"[!] debug_4_ptgaze_gaze_synced.csv nicht gefunden")
            print(f"   >> Versuche debug_1_ptgaze_data.csv (Fallback)")
            gaze_csv = os.path.join(OUTPUT_BASE_DIR, "debug_1_ptgaze_data.csv")
        
        if not os.path.exists(gaze_csv):
            print(f"[!] FEHLER: Weder debug_4 noch debug_1 Output gefunden!")
            exit(1)
        
        print(f"[>] Input (Modus {mode}): {Path(gaze_csv).name}")
    
    else:
        # Modus 1: Direkt von debug_1
        gaze_csv = os.path.join(OUTPUT_BASE_DIR, "debug_1_ptgaze_data.csv")
        
        if not os.path.exists(gaze_csv):
            print(f"[!] FEHLER: debug_1_ptgaze.py muss zuerst laufen!")
            exit(1)
        
        print(f"[>] Input (Modus 1): debug_1_ptgaze_data.csv")
    
    gaze_data = pd.read_csv(gaze_csv)
    print(f"[>] Geladen: {len(gaze_data)} Samples\n")
    
    # ======================================================================
    # Kalibriere
    # ======================================================================
    
    manager = PtgazeCalibrationManager(mode=mode)
    gaze_calibrated, cal_info = manager.calibrate(gaze_data)
    
    # ======================================================================
    # Export
    # ======================================================================
    
    if cal_info['success']:
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        
        # ══════════════════════════════════════════════════════════════════════
        # DEDUPLIZIERUNG (Failsafe gegen debug_1 Overlap-Bug)
        # ══════════════════════════════════════════════════════════════════════

        time_col = 'timestamp_ms_synced' if 'timestamp_ms_synced' in gaze_calibrated.columns else 'timestamp_ms'

        print(f"\n{'='*70}")
        print("DEDUPLIZIERUNG (Failsafe)")
        print(f"{'='*70}\n")

        n_before = len(gaze_calibrated)
        n_duplicates = gaze_calibrated.duplicated(subset=[time_col]).sum()

        if n_duplicates > 0:
            print(f"️ {n_duplicates} Duplikate bei {time_col} gefunden!")
            print(f"   Ursache: Vermutlich debug_1 Trial-Overlap (0.5s Buffer)")
            print(f"   Aktion: Entferne Duplikate (behalte ersten Eintrag)...")
            
            # Sortiere nach Zeit
            gaze_calibrated = gaze_calibrated.sort_values(time_col).reset_index(drop=True)
            
            # Entferne Duplikate
            gaze_calibrated = gaze_calibrated.drop_duplicates(subset=[time_col], keep='first')
            gaze_calibrated = gaze_calibrated.reset_index(drop=True)
            
            n_after = len(gaze_calibrated)
            print(f"   ✓ Nach Bereinigung: {n_after} Frames (entfernt: {n_before - n_after})")
        else:
            print(f"✓ Keine Duplikate gefunden")

        print(f"{'='*70}\n")

        output_path = os.path.join(OUTPUT_BASE_DIR, "debug_5_ptgaze_calibrated.csv")
        gaze_calibrated.to_csv(output_path, index=False)
        
        info_path = os.path.join(OUTPUT_BASE_DIR, "debug_5_ptgaze_calibration_info.json")
        with open(info_path, 'w') as f:
            json.dump(cal_info, f, indent=2)
        
        print(f"\n{'='*70}")
        print("ERFOLGREICH!")
        print(f"{'='*70}")
        print(f"\n[>] Gespeichert:")
        print(f"   - {output_path}")
        print(f"   - {info_path}")
        
        print(f"\n[>] CSV-Spalten ({len(gaze_calibrated.columns)}):")
        print(f"   - gaze_pitch_deg, gaze_yaw_deg (Original ptgaze)")
        print(f"   - gaze_screen_x_calib_px, gaze_screen_y_calib_px (Kalibriert)")
        print(f"   - gaze_deg_x_calib, gaze_deg_y_calib (Finale Gradsehwinkel)")
        print(f"   - plausibility_check, calibration_method, ...")
        
        print(f"\n[>] Naechster Schritt: debug_6_trial_comparison.py (erweitert)")
    
    else:
        print(f"\n{'='*70}")
        print("FEHLER BEI KALIBRIERUNG")
        print(f"{'='*70}")
        print(f"\n[!] Grund: {cal_info.get('reason', 'unknown')}")
