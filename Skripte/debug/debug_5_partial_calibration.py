"""
DEBUG 5: KALIBRIERUNGS-ANWENDUNG v2.4
=====================================
Wendet Kalibrierung auf Pupillendaten an (Modus 1 oder 2).

NEU in v2.4:
------------
- Package-Struktur (debug/)
- Modus 3 ENTFERNT (Legacy)
- Import aus shared.shared_coordinate_utils

NEU in v2.3:
------------
- Kompatibel mit debug_1 v2.2 (*_raw Spalten)
- Kompatibel mit debug_4 v2.3 (timestamp_ms_synced, sync_confidence)
- Konsistente Pipeline mit offline_calibration (shared_pupil_detection)
- Qualitaets-Metriken im CSV (calibration_r2, prediction_confidence)
- Plausibilitaets-Checks (dynamisch aus config)
- PKL-Validierung (prueft Trainings-Spalten)

Modus-Unterschiede:
-------------------
Modus 1 & 2: IDENTISCH - Polynomial Regression (.pkl)
  - Input: debug_1 (Modus 1) oder debug_4 (Modus 2)
  - Kalibrierung: CALIBRATION_PKL_PATH (9-16 Punkt)
  - Output: Praezise Gradsehwinkel (+-1-2 Grad Ziel)

Version: v2.4
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
from typing import List, Tuple, Dict

from config import *

from shared.shared_coordinate_utils import (
    CoordinateTransformer,
    ScreenParameters,
    CalibrationMetadata,
    create_coordinate_transformer
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

# ==================== DOUBLE EXPONENTIAL FILTER (NEU v3.1) ====================

class DoubleExponentialFilter:
    """
    Double Exponential Smoothing (Holt's Linear Trend) für Gaze-Koordinaten.
    
    Wissenschaftliche Basis:
    - Holt, C. C. (1957): Forecasting trends and seasonals
    - Brown, R. G. (1963): Smoothing, Forecasting and Prediction
    
    Empirisch optimiert (aus live_calibration.py v2.1):
    - alpha=0.3: Balance zwischen Reaktivität und Glättung
    - beta=0.1: Trend-Stabilität
    - Jitter-Reduktion: ~50% (validiert)
    - Lag: ~30ms @ 30 FPS (akzeptabel)
    
    Version: v3.1 (aus Live-Tests übernommen)
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        """
        Args:
            alpha: Level-Smoothing (0.1-0.5)
                   0.1 = sehr glatt, 0.5 = sehr reaktiv
            beta: Trend-Smoothing (0.05-0.2)
                  0.05 = stabiler Trend, 0.2 = adaptiv
        """
        self.alpha = alpha
        self.beta = beta
        self.level = None  # Initialisiert beim ersten Wert
        self.trend = 0.0
    
    def __call__(self, x: float) -> float:
        """
        Filtert einen Wert.
        
        Args:
            x: Neuer Wert (z.B. gaze_x in Pixel)
        
        Returns:
            Gefilterter Wert
        """
        if self.level is None:
            # Erster Wert: Initialisiere
            self.level = x
            return x
        
        # Holt's Formeln
        prev_level = self.level
        self.level = self.alpha * x + (1 - self.alpha) * (prev_level + self.trend)
        self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend
        
        return self.level
    
    def reset(self):
        """Reset für neuen Trial/Session"""
        self.level = None
        self.trend = 0.0

# ==================== MANUELLE HEAD-POSE-KORREKTUR (NEU v3.1.1) ====================

def apply_manual_gaze_offset(pupil_calibrated: pd.DataFrame, 
                             offset_x_deg: float, 
                             offset_y_deg: float) -> pd.DataFrame:
    """
    OPTION A: Addiert manuellen Offset zu kalibrierten Gaze-Werten (NACH Kalibrierung).
    
    Einfachste Lösung für systematische Offsets!
    
    Args:
        pupil_calibrated: DataFrame mit 'gaze_deg_x_calib', 'gaze_deg_y_calib'
        offset_x_deg: Offset in Grad (negativ = nach links)
        offset_y_deg: Offset in Grad (negativ = nach oben)
    
    Returns:
        DataFrame mit korrigierten Werten
    
    Beispiel:
        CoC = +0.3 >> ca. 8° zu weit rechts >> offset_x_deg = -8.0
    """
    
    if offset_x_deg == 0 and offset_y_deg == 0:
        return pupil_calibrated
    
    print(f"\n{'='*70}")
    print("MANUELLE GAZE-OFFSET-KORREKTUR (Option A)")
    print(f"{'='*70}\n")
    print(f"Angewandte Offsets:")
    print(f"  X: {offset_x_deg:+.1f}° ({'nach links' if offset_x_deg < 0 else 'nach rechts'})")
    print(f"  Y: {offset_y_deg:+.1f}° ({'nach oben' if offset_y_deg < 0 else 'nach unten'})")
    
    # Speichere Original (für Vergleich)
    if 'gaze_deg_x_calib' in pupil_calibrated.columns:
        pupil_calibrated['gaze_deg_x_calib_raw'] = pupil_calibrated['gaze_deg_x_calib'].copy()
        pupil_calibrated['gaze_deg_y_calib_raw'] = pupil_calibrated['gaze_deg_y_calib'].copy()
        
        # Anwenden
        pupil_calibrated['gaze_deg_x_calib'] += offset_x_deg
        pupil_calibrated['gaze_deg_y_calib'] += offset_y_deg
        
        # Statistik
        valid_before = pupil_calibrated.dropna(subset=['gaze_deg_x_calib_raw'])
        valid_after = pupil_calibrated.dropna(subset=['gaze_deg_x_calib'])
        
        if len(valid_before) > 0:
            print(f"\nVorher:")
            print(f"  X: {valid_before['gaze_deg_x_calib_raw'].min():.1f} bis {valid_before['gaze_deg_x_calib_raw'].max():.1f}°")
            print(f"  Ø: {valid_before['gaze_deg_x_calib_raw'].mean():.1f}°")
            
            print(f"\nNachher:")
            print(f"  X: {valid_after['gaze_deg_x_calib'].min():.1f} bis {valid_after['gaze_deg_x_calib'].max():.1f}°")
            print(f"  Ø: {valid_after['gaze_deg_x_calib'].mean():.1f}°")
            
            print(f"\n💡 Erwartete Verbesserung:")
            print(f"   • Lateralisierung sollte jetzt ausgeglichener sein")
            print(f"   • CoC sollte näher an 0 liegen")
    
    return pupil_calibrated


def apply_manual_head_pose_correction(pupil_data: pd.DataFrame,
                                      yaw_deg: float,
                                      pitch_deg: float,
                                      x_col: str,
                                      y_col: str,
                                      pixels_per_degree: float = HEAD_POSE_CORRECTION_PIXELS_PER_DEGREE) -> tuple:
    """
    OPTION B: Korrigiert Pupillen-Rohdaten VOR Kalibrierung (physikalisch korrekter).
    
    Simuliert andere Head-Pose während Kalibrierung.
    
    Args:
        pupil_data: DataFrame mit Pupillen-Rohdaten
        yaw_deg: Yaw-Korrektur (positiv = Kopf war nach RECHTS, korrigiere nach LINKS)
        pitch_deg: Pitch-Korrektur (positiv = Kopf war nach UNTEN, korrigiere nach OBEN)
        x_col, y_col: Spalten-Namen der Pupillen-Koordinaten
        pixels_per_degree: Umrechnungsfaktor (empirisch, abhängig von Setup)
    
    Returns:
        (pupil_data_corrected, neue_spalten_namen)
    
    Geometrie:
        Yaw +5° (Kopf rechts) >> Pupillen erscheinen LINKS im Bild >> Addiere +X Pixel
        Pitch +5° (Kopf unten) >> Pupillen erscheinen OBEN im Bild >> Addiere -Y Pixel
    """
    
    if yaw_deg == 0 and pitch_deg == 0:
        return pupil_data, (x_col, y_col)
    
    print(f"\n{'='*70}")
    print("MANUELLE HEAD-POSE-KORREKTUR (Option B)")
    print(f"{'='*70}\n")
    print(f"Head-Pose-Offsets (während Kalibrierung):")
    print(f"  Yaw: {yaw_deg:+.1f}° ({'Kopf rechts' if yaw_deg > 0 else 'Kopf links'})")
    print(f"  Pitch: {pitch_deg:+.1f}° ({'Kopf unten' if pitch_deg > 0 else 'Kopf oben'})")
    print(f"\nUmrechnung: {pixels_per_degree:.1f} Pixel/Grad")
    
    # Berechne Pixel-Offsets (Geometrie beachten!)
    offset_x_px = yaw_deg * pixels_per_degree      # Positiv Yaw >> Addiere rechts
    offset_y_px = -pitch_deg * pixels_per_degree   # Positiv Pitch >> Subtrahiere (nach unten)
    
    print(f"\nAngewandte Pixel-Offsets:")
    print(f"  X: {offset_x_px:+.1f} px")
    print(f"  Y: {offset_y_px:+.1f} px")
    
    # Erstelle korrigierte Spalten
    x_col_corrected = f'{x_col}_head_corrected'
    y_col_corrected = f'{y_col}_head_corrected'
    
    pupil_data[x_col_corrected] = pupil_data[x_col] + offset_x_px
    pupil_data[y_col_corrected] = pupil_data[y_col] + offset_y_px
    
    # Statistik
    print(f"\nPupillen-Rohdaten (Vorher >> Nachher):")
    print(f"  X: {pupil_data[x_col].mean():.1f} px >> {pupil_data[x_col_corrected].mean():.1f} px")
    print(f"  Y: {pupil_data[y_col].mean():.1f} px >> {pupil_data[y_col_corrected].mean():.1f} px")
    
    print(f"\n💡 Diese korrigierten Werte werden für Kalibrierung genutzt")
    
    return pupil_data, (x_col_corrected, y_col_corrected)

# ==================== ADAPTIVE HEAD-POSE-KORREKTUR (NEU v3.2) ====================

def apply_adaptive_head_pose_correction_on_pupil_data(
    pupil_data: pd.DataFrame,
    reference_head_pose: Dict,
    x_col: str,
    y_col: str,
    correction_method: str = 'linear_approximation'
) -> pd.DataFrame:
    """
    Adaptive Head-Pose-Korrektur auf PUPILLEN-ROHDATEN (v3.2.2).
    
    WICHTIG: Arbeitet im VIDEO-PIXEL-SPACE (50-80 px Range), NICHT Screen-Space!
    
    Geometrie:
    - Pupillen-Bewegung im Video: ~50-80 px (bei 1920x1080 Webcam, 50cm Abstand)
    - Gaze-Bewegung auf Screen: ~2560 px
    - Kompressionsfaktor: ~30-50x
    - >> Head-Pose-Korrektur muss ENTSPRECHEND SKALIERT werden!
    
    Args:
        pupil_data: DataFrame mit Pupillen-Rohdaten + Head-Pose
        reference_head_pose: Referenz aus .pkl
        x_col, y_col: Spalten der Pupillen-Koordinaten (z.B. 'avg_pupil_x_px_raw')
        correction_method: 'linear_approximation'
    
    Returns:
        DataFrame mit korrigierten Pupillen-Koordinaten (IN-PLACE auf x_col, y_col!)
    """
    
    if not reference_head_pose.get('enabled', False):
        print(f"\n Keine Referenz-Head-Pose >> Korrektur nicht möglich")
        return pupil_data
    
    print(f"\n{'='*70}")
    print("ADAPTIVE HEAD-POSE-KORREKTUR (auf Pupillen-Rohdaten!)")
    print(f"{'='*70}\n")
    print(f"Methode: {correction_method}")
    print(f"Referenz-Pose: Yaw={reference_head_pose['yaw']:+.1f}°, "
          f"Pitch={reference_head_pose['pitch']:+.1f}°")
    
    # Prüfe ob Head-Pose-Daten vorhanden
    if 'head_yaw' not in pupil_data.columns:
        print(f" Keine Head-Pose in Daten >> Korrektur nicht möglich")
        return pupil_data
    
    # Berechne Abweichungen
    ref_yaw = reference_head_pose['yaw']
    ref_pitch = reference_head_pose['pitch']
    
    pupil_data['head_yaw_deviation'] = pupil_data['head_yaw'] - ref_yaw
    pupil_data['head_pitch_deviation'] = pupil_data['head_pitch'] - ref_pitch
    
    # Filtere Frames mit gültigen Daten
    valid_mask = (
        pupil_data[x_col].notna() &
        pupil_data['head_yaw'].notna() &
        (pupil_data.get('head_pose_confidence', 1.0) >= HEAD_POSE_CORRECTION_CONFIDENCE_THRESHOLD)
    )
    
    n_total = len(pupil_data)
    n_valid = valid_mask.sum()
    
    print(f"\nFrames:")
    print(f"  Gesamt: {n_total}")
    print(f"  Mit Head-Pose: {n_valid} ({n_valid/n_total*100:.1f}%)")
    
    # Statistik über Abweichungen
    deviations_yaw = pupil_data.loc[valid_mask, 'head_yaw_deviation']
    deviations_pitch = pupil_data.loc[valid_mask, 'head_pitch_deviation']
    
    print(f"\nAbweichungen von Referenz:")
    print(f"  Yaw:   Mean={deviations_yaw.mean():+.2f}°, Std={deviations_yaw.std():.2f}°, "
          f"Range=[{deviations_yaw.min():+.1f}, {deviations_yaw.max():+.1f}]")
    print(f"  Pitch: Mean={deviations_pitch.mean():+.2f}°, Std={deviations_pitch.std():.2f}°, "
          f"Range=[{deviations_pitch.min():+.1f}, {deviations_pitch.max():+.1f}]")
    
    # Entscheide ob Korrektur sinnvoll
    needs_correction_yaw = deviations_yaw.abs().mean() > HEAD_POSE_MIN_DEVIATION_FOR_CORRECTION_DEG
    needs_correction_pitch = deviations_pitch.abs().mean() > HEAD_POSE_MIN_DEVIATION_FOR_CORRECTION_DEG
    
    if not (needs_correction_yaw or needs_correction_pitch):
        print(f"\n💡 Abweichungen <{HEAD_POSE_MIN_DEVIATION_FOR_CORRECTION_DEG}° >> Korrektur nicht nötig")
        return pupil_data
    
    # ═════════════════════════════════════════════════════════════════════
    # ANWENDUNG DER KORREKTUR (IM PUPILLEN-PIXEL-SPACE!)
    # ═════════════════════════════════════════════════════════════════════
    
    # Speichere Original
    pupil_data[f'{x_col}_before_head_corr'] = pupil_data[x_col].copy()
    pupil_data[f'{y_col}_before_head_corr'] = pupil_data[y_col].copy()
    
    corrections_x = []
    corrections_y = []
    
    #  WICHTIG: Korrektur-Faktor für PUPILLEN-EBENE (nicht Screen-Ebene!)
    # Pupillen-Bewegung ist ~30-50x kleiner als Screen-Bewegung
    # >> Faktor muss DURCH Kompressionsfaktor geteilt werden!
    
    # Empirisch: Bei HEAD_POSE_CORRECTION_FACTOR_YAW_PX_PER_DEG = 2.8 (Screen-Ebene)
    # >> Pupillen-Ebene: 2.8 / 30 ≈ 0.09 px/Grad
    
    #  DYNAMISCHE BERECHNUNG (v3.2.3): Kompression aus Daten ableiten
    valid_pupil = pupil_data[valid_mask]
    
    if len(valid_pupil) > 100:
        pupil_x_range = valid_pupil[x_col].max() - valid_pupil[x_col].min()
        
        # Erwartete Screen-Range bei typischem Blickverhalten
        # Bei ±15° Blickwinkel auf 1920px Bildschirm
        max_gaze_deg = np.arctan2(SCREEN_WIDTH_CM / 2, VIEWING_DISTANCE_CM) * 180 / np.pi
        screen_range_for_gaze = SCREEN_WIDTH_PX * 0.8  # 80% der Bildschirmbreite (konservativ)
        
        # Tatsächliche Kompression
        actual_compression = screen_range_for_gaze / pupil_x_range if pupil_x_range > 0 else 30.0
        
        # Begrenze auf plausiblen Bereich (20-50x)
        actual_compression = np.clip(actual_compression, 20.0, 50.0)
        
        print(f"\n Dynamische Kompressions-Analyse:")
        print(f"   Pupillen-Range: {pupil_x_range:.1f} px")
        print(f"   Screen-Range (80%): {screen_range_for_gaze:.0f} px")
        print(f"   Kompression: {actual_compression:.1f}x (statt hardcoded 30x)")
        
        pupil_correction_factor_yaw = HEAD_POSE_CORRECTION_FACTOR_YAW_PX_PER_DEG / actual_compression
        pupil_correction_factor_pitch = HEAD_POSE_CORRECTION_FACTOR_PITCH_PX_PER_DEG / actual_compression
    else:
        print(f"\n Zu wenig Daten für dynamische Berechnung, nutze 30x Fallback")
        pupil_correction_factor_yaw = HEAD_POSE_CORRECTION_FACTOR_YAW_PX_PER_DEG / 30.0
        pupil_correction_factor_pitch = HEAD_POSE_CORRECTION_FACTOR_PITCH_PX_PER_DEG / 30.0
    
    print(f"\n Korrektur-Faktoren (Pupillen-Ebene):")
    print(f"   Yaw: {pupil_correction_factor_yaw:.3f} px/Grad (Screen: {HEAD_POSE_CORRECTION_FACTOR_YAW_PX_PER_DEG} / 30)")
    print(f"   Pitch: {pupil_correction_factor_pitch:.3f} px/Grad")
    
    for idx, row in pupil_data.iterrows():
        if not valid_mask.loc[idx]:
            corrections_x.append(0.0)
            corrections_y.append(0.0)
            continue
        
        yaw_dev = row['head_yaw_deviation']
        pitch_dev = row['head_pitch_deviation']
        
        # Begrenze auf max. plausible Abweichung
        yaw_dev = np.clip(yaw_dev, -HEAD_POSE_MAX_PLAUSIBLE_DEVIATION_DEG, 
                         HEAD_POSE_MAX_PLAUSIBLE_DEVIATION_DEG)
        pitch_dev = np.clip(pitch_dev, -HEAD_POSE_MAX_PLAUSIBLE_DEVIATION_DEG, 
                           HEAD_POSE_MAX_PLAUSIBLE_DEVIATION_DEG)
        
        # Linear Approximation (auf Pupillen-Ebene!)
        if correction_method == 'linear_approximation':
            # Geometrie (gleich wie bei Screen, aber kleinerer Faktor):
            # Positiv Yaw (Kopf rechts) >> Pupillen links >> Korrigiere rechts (+X)
            corr_x_px = yaw_dev * pupil_correction_factor_yaw
            corr_y_px = pitch_dev * pupil_correction_factor_pitch
        else:
            corr_x_px = 0.0
            corr_y_px = 0.0
        
        corrections_x.append(corr_x_px)
        corrections_y.append(corr_y_px)
    
    pupil_data['head_correction_x_px'] = corrections_x
    pupil_data['head_correction_y_px'] = corrections_y
    
    #  Anwenden (IN-PLACE auf Pupillen-Koordinaten!)
    pupil_data[x_col] = pupil_data[x_col] + pupil_data['head_correction_x_px']
    pupil_data[y_col] = pupil_data[y_col] + pupil_data['head_correction_y_px']
    
    # Statistik
    mean_corr_x = np.mean([c for c in corrections_x if c != 0])
    mean_corr_y = np.mean([c for c in corrections_y if c != 0])
    
    print(f"\nKorrektur angewendet (Pupillen-Ebene):")
    print(f"  X: {mean_corr_x:+.3f} px (Mean)")
    print(f"  Y: {mean_corr_y:+.3f} px (Mean)")
    print(f"\n💡 Diese Korrektur wird VOR Kalibrierung angewendet!")
    print(f"   >> Polynomial-Prediction nutzt korrigierte Pupillen-Positionen")
    print(f"   >> Effekt auf Screen: ~{mean_corr_x * 30:+.1f} px (nach Kalibrierung)")
    
        # ═══════════════════════════════════════════════════════════════════
    #  VALIDIERUNG DER KORREKTUR (v3.2.3)
    # ═══════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print("KORREKTUR-VALIDIERUNG")
    print(f"{'='*70}\n")
    
    corrected_frames = pupil_data[pupil_data['head_correction_x_px'] != 0]
    
    if len(corrected_frames) > 0:
        print(f"Korrigierte Frames: {len(corrected_frames)}/{len(pupil_data)} ({len(corrected_frames)/len(pupil_data)*100:.1f}%)")
        
        # Vor vs. Nach
        before_mean_x = corrected_frames[f'{x_col}_before_head_corr'].mean()
        after_mean_x = corrected_frames[x_col].mean()
        shift_px = after_mean_x - before_mean_x
        
        print(f"\nPupillen-Mittelwert (korrigierte Frames):")
        print(f"  Vorher:  {before_mean_x:.2f} px")
        print(f"  Nachher: {after_mean_x:.2f} px")
        print(f"  Shift:   {shift_px:+.2f} px")
        
        # Erwartete Wirkung auf Screen (nach Kalibrierung)
        expected_screen_shift_px = shift_px * actual_compression if 'actual_compression' in locals() else shift_px * 30
        expected_screen_shift_deg = expected_screen_shift_px / (SCREEN_WIDTH_PX / (2 * MAX_PLAUSIBLE_DEG_X))
        
        print(f"\nErwartete Wirkung auf Screen (nach Kalibrierung):")
        print(f"  ~{expected_screen_shift_px:+.0f} px")
        print(f"  ~{expected_screen_shift_deg:+.1f}°")
        
        # Warnung bei zu großer Korrektur
        if abs(shift_px) > 5.0:  # >5 px Shift im Pupillen-Space ist verdächtig
            print(f"\n WARNUNG: Korrektur-Shift ist sehr groß ({shift_px:+.2f} px)!")
            print(f"   Mögliche Probleme:")
            print(f"   • Korrektur-Faktoren falsch")
            print(f"   • Referenz-Head-Pose fehlerhaft")
            print(f"   >> Prüfe debug_5_head_pose_corrections_pupil_level.csv")
    else:
        print(f" Keine Frames korrigiert (HEAD_POSE_MIN_DEVIATION nicht erreicht?)")
        deviations_yaw = pupil_data['head_yaw_deviation'].abs() if 'head_yaw_deviation' in pupil_data.columns else pd.Series([0])
        print(f"   Mittlere Abweichung: {deviations_yaw.mean():.2f}° (Schwelle: {HEAD_POSE_MIN_DEVIATION_FOR_CORRECTION_DEG}°)")

    if SAVE_HEAD_POSE_DEBUG_CSV:
        debug_csv = os.path.join(OUTPUT_BASE_DIR, "debug_5_head_pose_corrections_pupil_level.csv")
        debug_df = pupil_data[['timestamp_ms', 'head_yaw', 'head_pitch', 
                                'head_yaw_deviation', 'head_pitch_deviation',
                                'head_correction_x_px', 'head_correction_y_px',
                                f'{x_col}_before_head_corr', x_col]].copy()
        debug_df.to_csv(debug_csv, index=False)
        print(f"\n Debug-CSV: {debug_csv}")
    
    return pupil_data

# ==================== KALIBRIERUNGS-MANAGER ====================

class CalibrationManager:
    """Verwaltet Kalibrierung für verschiedene Modi (v2.3)"""
    
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
    
    def calibrate(self, pupil_data: pd.DataFrame, et_blocks_df: pd.DataFrame = None):
        """Hauptmethode: Ruft modus-spezifische Kalibrierung auf"""
        
        if self.mode in [1, 2]:
            return self._calibrate_from_pkl(pupil_data)
        elif self.mode == 3:
            # Modus 3 wurde entfernt (v2.4 - Legacy-Code)
            print(f"\n[!] FEHLER: Modus 3 ist nicht mehr unterstuetzt!")

            return pupil_data, {'success': False, 'reason': 'mode3_removed'}
        else:
            raise ValueError(f"Ungueltiger Modus: {self.mode}")
    
    # ═════════════════════════════════════════════════════════════════════
    # MODUS 1 & 2: PKL-KALIBRIERUNG (IDENTISCH, MIT VALIDIERUNG)
    # ═════════════════════════════════════════════════════════════════════
    
    def _calibrate_from_pkl(self, pupil_data: pd.DataFrame):
        """
        Modus 1 & 2: PKL-Kalibrierung mit KORREKTER Korrektur-Reihenfolge (v3.2.2).
        
        RICHTIGE REIHENFOLGE (v3.2.2):
        1. Spalten-Erkennung
        2.  HEAD-POSE-KORREKTUREN auf PUPILLEN-ROHDATEN
        3.  Outlier-Filter (auf korrigierten Rohdaten!)
        4. Polynomial-Prediction
        5. Double Exp Filter
        6. Gradsehwinkel-Transformation
        7. Plausibilitäts-Check
        8. (Optional) Manueller Offset als Fallback
        """
        
        mode_name = "STANDALONE" if self.mode == 1 else "COMPARISON"
        print(f"\n{'='*70}")
        print(f"MODUS {self.mode}: {mode_name} (v3.2.2 - Head-Pose auf Rohdaten-Ebene)")
        print(f"{'='*70}\n")
        
        # ══════════════════════════════════════════════════════════════════
        # SCHRITT 1: LADE .PKL
        # ══════════════════════════════════════════════════════════════════
        
        if not os.path.exists(CALIBRATION_PKL_PATH):
            print(f" FEHLER: {CALIBRATION_PKL_PATH}")
            return pupil_data, {'success': False, 'reason': 'calibration_file_not_found'}
        
        try:
            with open(CALIBRATION_PKL_PATH, 'rb') as f:
                calibration_data = pickle.load(f)
            
            print(f" Kalibrierung geladen: {Path(CALIBRATION_PKL_PATH).name}")
            method = calibration_data.get('method', 'unknown')
            poly_degree = calibration_data.get('polynomial_degree', 1)
            n_points = len(calibration_data.get('calibration_points', []))
            r2_scores = calibration_data.get('r2_scores', (0, 0))
            
            print(f"  Methode: {method}, Grad: {poly_degree}, Punkte: {n_points}")
            print(f"  R²: X={r2_scores[0]:.3f}, Y={r2_scores[1]:.3f}")
            
            if r2_scores[0] < MIN_CALIBRATION_R2 or r2_scores[1] < MIN_CALIBRATION_R2:
                print(f"   Niedrige R² (<{MIN_CALIBRATION_R2})")
        
        except Exception as e:
            print(f" Laden fehlgeschlagen: {e}")
            return pupil_data, {'success': False, 'reason': 'calibration_load_error', 'error': str(e)}
        
        # ══════════════════════════════════════════════════════════════════
        # SCHRITT 2: SPALTEN-ERKENNUNG
        # ══════════════════════════════════════════════════════════════════
        
        print(f"\n{'='*70}")
        print("SPALTEN-VALIDIERUNG")
        print(f"{'='*70}\n")
        
        if 'avg_pupil_x_px_raw' in pupil_data.columns:
            x_col = 'avg_pupil_x_px_raw'
            y_col = 'avg_pupil_y_px_raw'
            print(f" CSV v2.2 (shared pipeline)")
        elif 'avg_pupil_x_px_final' in pupil_data.columns:
            x_col = 'avg_pupil_x_px_final'
            y_col = 'avg_pupil_y_px'
            print(f" CSV v2.1 (legacy)")
        else:
            print(f" Keine gültigen Spalten: {list(pupil_data.columns)}")
            return pupil_data, {'success': False, 'reason': 'invalid_columns'}
        
        print(f"  Nutze: {x_col}, {y_col}")
        
        # ══════════════════════════════════════════════════════════════════
        # SCHRITT 3: HEAD-POSE-KORREKTUREN (AUF PUPILLEN-ROHDATEN!)
        # ══════════════════════════════════════════════════════════════════
        
        print(f"\n{'='*70}")
        print("HEAD-POSE-KORREKTUREN (auf Pupillen-Rohdaten, VOR Outlier-Filter!)")
        print(f"{'='*70}")
        
        # Erstelle Arbeitskopie
        pupil_corrected = pupil_data.copy()
        
        # ─────────────────────────────────────────────────────────────────
        # 3A: Option B - Manuelle statische Korrektur
        # ─────────────────────────────────────────────────────────────────
        
        if ENABLE_MANUAL_HEAD_POSE_CORRECTION:
            pupil_corrected, (x_col, y_col) = apply_manual_head_pose_correction(
                pupil_corrected,
                yaw_deg=MANUAL_HEAD_POSE_YAW_DEG,
                pitch_deg=MANUAL_HEAD_POSE_PITCH_DEG,
                x_col=x_col,
                y_col=y_col,
                pixels_per_degree=HEAD_POSE_CORRECTION_PIXELS_PER_DEGREE
            )
            print(f"\n  >> Spalten nach manueller Korrektur: {x_col}, {y_col}")
        else:
            print(f"\n💡 Manuelle Head-Pose-Korrektur deaktiviert")
        
        # ─────────────────────────────────────────────────────────────────
        # 3B: Adaptive frame-für-frame Korrektur (auf Pupillen-Ebene!)
        # ─────────────────────────────────────────────────────────────────
        
        if ENABLE_HEAD_POSE_CORRECTION and 'reference_head_pose' in calibration_data:
            pupil_corrected = apply_adaptive_head_pose_correction_on_pupil_data(
                pupil_data=pupil_corrected,
                reference_head_pose=calibration_data['reference_head_pose'],
                x_col=x_col,
                y_col=y_col,
                correction_method=HEAD_POSE_CORRECTION_METHOD
            )
            print(f"\n  >> Adaptive Korrektur angewendet auf {x_col}, {y_col}")
        else:
            print(f"\n💡 Adaptive Head-Pose-Korrektur nicht verfügbar")
            if not ENABLE_HEAD_POSE_CORRECTION:
                print(f"   (ENABLE_HEAD_POSE_CORRECTION=False)")
            elif 'reference_head_pose' not in calibration_data:
                print(f"   (Kein reference_head_pose im PKL)")
        
        # ══════════════════════════════════════════════════════════════════
        # SCHRITT 4: OUTLIER-FILTER (AUF KORRIGIERTEN ROHDATEN!)
        # ══════════════════════════════════════════════════════════════════
        
        n_outliers = 0
        outlier_mask = pd.Series(False, index=pupil_corrected.index)
        
        if ENABLE_PRECALIBRATION_OUTLIER_FILTER:
            print(f"\n{'='*70}")
            print("PRE-CALIBRATION OUTLIER-FILTER (auf HEAD-POSE-korrigierten Daten!)")
            print(f"{'='*70}\n")
            
            x_low = pupil_corrected[x_col].quantile(PUPIL_OUTLIER_PERCENTILE_LOW / 100)
            x_high = pupil_corrected[x_col].quantile(PUPIL_OUTLIER_PERCENTILE_HIGH / 100)
            y_low = pupil_corrected[y_col].quantile(PUPIL_OUTLIER_PERCENTILE_LOW / 100)
            y_high = pupil_corrected[y_col].quantile(PUPIL_OUTLIER_PERCENTILE_HIGH / 100)
            
            print(f"Grenzen (auf korrigierten Daten):")
            print(f"  X: {x_low:.1f} - {x_high:.1f} px")
            print(f"  Y: {y_low:.1f} - {y_high:.1f} px")
            
            outlier_mask = (
                (pupil_corrected[x_col] < x_low) |
                (pupil_corrected[x_col] > x_high) |
                (pupil_corrected[y_col] < y_low) |
                (pupil_corrected[y_col] > y_high)
            )
            
            n_outliers = outlier_mask.sum()
            if n_outliers > 0:
                print(f"\n {n_outliers}/{len(pupil_corrected)} Outliers ({n_outliers/len(pupil_corrected)*100:.1f}%)")
                pupil_corrected.loc[outlier_mask, x_col] = np.nan
                pupil_corrected.loc[outlier_mask, y_col] = np.nan
            else:
                print(f" Keine Outliers")
        
        # ══════════════════════════════════════════════════════════════════
        # SCHRITT 5: POLYNOMIAL-PREDICTION
        # ══════════════════════════════════════════════════════════════════
        
        print(f"\n{'='*70}")
        print("KALIBRIERUNGS-ANWENDUNG (Polynomial auf korrigierte Pupillen)")
        print(f"{'='*70}\n")
        
        model_x = calibration_data.get('model_x')
        model_y = calibration_data.get('model_y')
        poly_transformer = calibration_data.get('poly_transformer')

        valid_mask = pupil_corrected[x_col].notna() & pupil_corrected[y_col].notna()
        X = pupil_corrected.loc[valid_mask, [x_col, y_col]].values
        
        print(f"Eingabe (korrigierte Pupillen): X={X[:, 0].min():.1f}-{X[:, 0].max():.1f} px")
        
        #  AUTO-DETECT: Prüfe Model-Features
        try:
            n_features_expected = model_x.n_features_in_
            print(f"  Model erwartet: {n_features_expected} Features")
        except AttributeError:
            n_features_expected = None
        
        #  ROBUSTE TRANSFORMATION
        if poly_degree > 1 and poly_transformer is not None:
            print(f"  >> Polynomial Grad {poly_degree} (laut PKL-Metadaten)")
            X_transformed = poly_transformer.transform(X)
            
            # Validierung
            if n_features_expected and X_transformed.shape[1] != n_features_expected:
                print(f"   WARNUNG: Feature-Mismatch!")
                print(f"     Transformiert: {X_transformed.shape[1]} Features")
                print(f"     Erwartet: {n_features_expected} Features")
                print(f"  >> Versuche Auto-Korrektur...")
                
                # Auto-Korrektur: Erstelle passenden Transformer
                actual_degree = 2 if n_features_expected == 6 else 1
                print(f"  >> Nutze Grad {actual_degree} (basierend auf Model)")
                
                from sklearn.preprocessing import PolynomialFeatures
                poly_corrected = PolynomialFeatures(degree=actual_degree, include_bias=True)
                X_transformed = poly_corrected.fit_transform(X)
                
                print(f"   Korrigiert: {X_transformed.shape[1]} Features")
        else:
            X_transformed = X
            
            # Prüfe ob Model eigentlich Polynomial erwartet
            if n_features_expected and n_features_expected > 2:
                print(f"   WARNUNG: Model erwartet {n_features_expected} Features, aber poly_transformer fehlt!")
                print(f"  >> Erstelle Transformer (Grad 2)...")
                
                from sklearn.preprocessing import PolynomialFeatures
                poly_emergency = PolynomialFeatures(degree=2, include_bias=True)
                X_transformed = poly_emergency.fit_transform(X)
                
                print(f"   Notfall-Transformation: {X_transformed.shape[1]} Features")
        
        try:
            pred_x = model_x.predict(X_transformed)
            pred_y = model_y.predict(X_transformed)
            print(f"Prediction: X={pred_x.min():.1f}-{pred_x.max():.1f} px")
        except Exception as e:
            print(f" Prediction Error: {e}")
            return pupil_data, {'success': False, 'reason': 'prediction_error', 'error': str(e)}
        
        # ══════════════════════════════════════════════════════════════════
        # SCHRITT 6: DOUBLE EXPONENTIAL FILTER (auf Gaze-Pixels)
        # ══════════════════════════════════════════════════════════════════
        
        jitter_reduction_x = 0
        jitter_reduction_y = 0
        
        if ENABLE_DOUBLE_EXPONENTIAL_FILTER:
            print(f"\n  >> Double Exponential Filter (α={DOUBLE_EXP_ALPHA}, β={DOUBLE_EXP_BETA})")
            
            filter_x = DoubleExponentialFilter(alpha=DOUBLE_EXP_ALPHA, beta=DOUBLE_EXP_BETA)
            filter_y = DoubleExponentialFilter(alpha=DOUBLE_EXP_ALPHA, beta=DOUBLE_EXP_BETA)
            
            gaze_x_raw = pred_x.copy()
            gaze_y_raw = pred_y.copy()
            
            gaze_x_filtered = np.array([filter_x(x) for x in pred_x])
            gaze_y_filtered = np.array([filter_y(y) for y in pred_y])
            
            jitter_reduction_x = 1 - (np.std(gaze_x_filtered) / np.std(pred_x)) if np.std(pred_x) > 0 else 0
            jitter_reduction_y = 1 - (np.std(gaze_y_filtered) / np.std(pred_y)) if np.std(pred_y) > 0 else 0
            
            print(f"     Jitter: X={jitter_reduction_x*100:.1f}%, Y={jitter_reduction_y*100:.1f}%")
            
            pred_x_final = gaze_x_filtered
            pred_y_final = gaze_y_filtered
        else:
            print(f"\n   Filter deaktiviert")
            gaze_x_raw = pred_x.copy()
            gaze_y_raw = pred_y.copy()
            pred_x_final = pred_x
            pred_y_final = pred_y
        
        # Clipping + Speichern
        pred_x_clipped = np.clip(pred_x_final, 0, SCREEN_WIDTH_PX)
        pred_y_clipped = np.clip(pred_y_final, 0, SCREEN_HEIGHT_PX)
        
        pupil_calibrated = pupil_corrected.copy()  # ← Nutze korrigierte Version!
        pupil_calibrated.loc[valid_mask, 'gaze_screen_x_calib_px'] = pred_x_clipped
        pupil_calibrated.loc[valid_mask, 'gaze_screen_y_calib_px'] = pred_y_clipped
        
        # Optional: Speichere Raw
        if SAVE_RAW_AND_FILTERED_GAZE and ENABLE_DOUBLE_EXPONENTIAL_FILTER:
            pupil_calibrated.loc[valid_mask, 'gaze_screen_x_raw_px'] = np.clip(gaze_x_raw, 0, SCREEN_WIDTH_PX)
            pupil_calibrated.loc[valid_mask, 'gaze_screen_y_raw_px'] = np.clip(gaze_y_raw, 0, SCREEN_HEIGHT_PX)
            print(f"\n💡 Beide gespeichert: *_raw_px + *_calib_px")
        
        # ══════════════════════════════════════════════════════════════════
        # SCHRITT 7: GRADSEHWINKEL-TRANSFORMATION
        # ══════════════════════════════════════════════════════════════════
        
        print(f"\n{'='*70}")
        print("GRADSEHWINKEL-TRANSFORMATION (finale Werte)")
        print(f"{'='*70}\n")
        
        deg_x_list = []
        deg_y_list = []
        plausibility_list = []
        
        for _, row in pupil_calibrated.iterrows():
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
        
        pupil_calibrated['gaze_deg_x_calib'] = deg_x_list
        pupil_calibrated['gaze_deg_y_calib'] = deg_y_list
        pupil_calibrated['plausibility_check'] = plausibility_list
        
        valid_deg = pupil_calibrated[pupil_calibrated['plausibility_check']].dropna(subset=['gaze_deg_x_calib'])
        
        print(f"Gradsehwinkel:")
        print(f"  Gültig: {len(valid_deg)}/{len(pupil_calibrated)} ({len(valid_deg)/len(pupil_calibrated)*100:.1f}%)")
        if len(valid_deg) > 0:
            print(f"  X: {valid_deg['gaze_deg_x_calib'].min():.2f} - {valid_deg['gaze_deg_x_calib'].max():.2f}°")
            print(f"  Y: {valid_deg['gaze_deg_y_calib'].min():.2f} - {valid_deg['gaze_deg_y_calib'].max():.2f}°")
        
        n_implausible = (~pupil_calibrated['plausibility_check']).sum()
        if n_implausible > 0:
            print(f"\n   {n_implausible} außerhalb Bereich (>{MAX_PLAUSIBLE_DEG_X:.0f}° X, >{MAX_PLAUSIBLE_DEG_Y:.0f}° Y)")
        
        # ══════════════════════════════════════════════════════════════════
        # SCHRITT 8: MANUELLER GAZE-OFFSET (nur Fallback!)
        # ══════════════════════════════════════════════════════════════════
        
        if ENABLE_MANUAL_GAZE_OFFSET:
            print(f"\n{'='*70}")
            print("MANUELLER GAZE-OFFSET (Fallback-Korrektur)")
            print(f"{'='*70}")
            pupil_calibrated = apply_manual_gaze_offset(
                pupil_calibrated,
                offset_x_deg=MANUAL_GAZE_OFFSET_X_DEG,
                offset_y_deg=MANUAL_GAZE_OFFSET_Y_DEG
            )
        
        # ══════════════════════════════════════════════════════════════════
        # SCHRITT 8b: OUTSIDE-MONITOR DETECTION
        # ══════════════════════════════════════════════════════════════════
        
        print(f"\n{'='*70}")
        print("OUTSIDE-MONITOR DETECTION")
        print(f"{'='*70}\n")
        
        # Berechne maximale Sehwinkel aus Screen-Parametern
        max_deg_x = np.arctan2(SCREEN_WIDTH_CM / 2, VIEWING_DISTANCE_CM) * 180 / np.pi
        max_deg_y = np.arctan2(SCREEN_HEIGHT_CM / 2, VIEWING_DISTANCE_CM) * 180 / np.pi
        
        print(f"Monitor-Grenzen (basierend auf {VIEWING_DISTANCE_CM}cm Abstand):")
        print(f"  X: ±{max_deg_x:.1f}° (Screen: {SCREEN_WIDTH_CM}cm breit)")
        print(f"  Y: ±{max_deg_y:.1f}° (Screen: {SCREEN_HEIGHT_CM}cm hoch)")
        
        # Markiere Outside-Monitor
        pupil_calibrated['outside_monitor'] = (
            (pupil_calibrated['gaze_deg_x_calib'].abs() > max_deg_x) |
            (pupil_calibrated['gaze_deg_y_calib'].abs() > max_deg_y)
        )
        
        n_outside = pupil_calibrated['outside_monitor'].sum()
        n_total_valid = pupil_calibrated['gaze_deg_x_calib'].notna().sum()
        
        if n_outside > 0:
            print(f"\n⚠ {n_outside}/{n_total_valid} Samples außerhalb Monitor ({n_outside/n_total_valid*100:.1f}%)")
        else:
            print(f"\n✓ Alle Samples innerhalb Monitor-Grenzen")
        
        # ══════════════════════════════════════════════════════════════════
        # SCHRITT 8c: BLINK-SPALTEN VALIDIERUNG
        # ══════════════════════════════════════════════════════════════════
        
        print(f"\n{'='*70}")
        print("BLINK-DATEN VALIDIERUNG")
        print(f"{'='*70}\n")
        
        # Prüfe ob Blink-Spalten vorhanden sind
        blink_columns = ['is_blink', 'eyes_closed', 'avg_ear', 'left_ear', 'right_ear', 'blink_count']
        missing_blink_cols = [col for col in blink_columns if col not in pupil_calibrated.columns]
        
        if missing_blink_cols:
            print(f"⚠ Fehlende Blink-Spalten: {missing_blink_cols}")
            print(f"  >> Blink-Filterung in debug_6 nicht möglich!")
            
            # Füge leere Spalten hinzu (für Kompatibilität)
            for col in missing_blink_cols:
                if col == 'is_blink' or col == 'eyes_closed':
                    pupil_calibrated[col] = False
                elif col == 'blink_count':
                    pupil_calibrated[col] = 0
                else:
                    pupil_calibrated[col] = np.nan
        else:
            # Statistik
            n_eyes_closed = pupil_calibrated['eyes_closed'].sum()
            n_blinks = pupil_calibrated['is_blink'].sum()
            total_blink_count = pupil_calibrated['blink_count'].max()
            
            print(f"✓ Alle Blink-Spalten vorhanden")
            print(f"  • Frames mit geschlossenen Augen: {n_eyes_closed} ({n_eyes_closed/len(pupil_calibrated)*100:.1f}%)")
            print(f"  • Blink-Events detektiert: {n_blinks}")
            print(f"  • Gesamt-Blink-Count: {total_blink_count}")
            
            # EAR-Statistik
            valid_ear = pupil_calibrated['avg_ear'].dropna()
            if len(valid_ear) > 0:
                print(f"  • EAR: Mean={valid_ear.mean():.3f}, Min={valid_ear.min():.3f}, Max={valid_ear.max():.3f}")

        # ══════════════════════════════════════════════════════════════════
        # SCHRITT 9: QUALITÄTS-METRIKEN
        # ══════════════════════════════════════════════════════════════════
        
        pupil_calibrated['calibration_method'] = f"polynomial_degree_{poly_degree}"
        pupil_calibrated['calibration_r2_x'] = r2_scores[0]
        pupil_calibrated['calibration_r2_y'] = r2_scores[1]
        
        if COMBINE_SYNC_AND_DETECTION_CONFIDENCE and self.mode in [2, 3]:
            if 'sync_confidence' in pupil_calibrated.columns and 'confidence' in pupil_calibrated.columns:
                pupil_calibrated['prediction_confidence'] = (
                    pupil_calibrated['sync_confidence'] * pupil_calibrated['confidence']
                )
            else:
                pupil_calibrated['prediction_confidence'] = pupil_calibrated.get('confidence', 1.0)
        else:
            pupil_calibrated['prediction_confidence'] = pupil_calibrated.get('confidence', 1.0)
        
        pupil_calibrated['outlier_filtered'] = outlier_mask
        
        # Metadaten
        cal_info = {
            'success': True,
            'calibration_type': f'polynomial_degree_{poly_degree}',
            'mode': self.mode,
            'filter_applied': ENABLE_DOUBLE_EXPONENTIAL_FILTER,
            'filter_params': {
                'alpha': DOUBLE_EXP_ALPHA,
                'beta': DOUBLE_EXP_BETA
            } if ENABLE_DOUBLE_EXPONENTIAL_FILTER else None,
            'jitter_reduction_x_percent': float(jitter_reduction_x * 100),
            'jitter_reduction_y_percent': float(jitter_reduction_y * 100),
            'head_pose_correction_applied': ENABLE_HEAD_POSE_CORRECTION and 'reference_head_pose' in calibration_data,
            'n_calibration_points': n_points,
            'r2_scores': list(r2_scores),
            'method': method,
            'input_columns': {'x': x_col, 'y': y_col},
            'n_samples_total': len(pupil_calibrated),
            'n_samples_valid': len(valid_deg),
            'n_samples_implausible': int(n_implausible),
            'n_samples_outlier_filtered': int(n_outliers)
        }
        
        print(f"\n Kalibrierung erfolgreich!")
        
        return pupil_calibrated, cal_info

# ==================== MAIN ====================

if __name__ == "__main__":
    mode = ANALYSIS_MODE
    
    print(f"\n{'='*70}")
    print(f"DEBUG 5: KALIBRIERUNG v2.3 (MODUS {mode})")
    print(f"{'='*70}\n")
    
    # ══════════════════════════════════════════════════════════════════════
    # Lade Pupillendaten
    # ══════════════════════════════════════════════════════════════════════
    
    if mode in [2, 3]:
        pupil_csv = os.path.join(OUTPUT_BASE_DIR, "debug_4_pupil_data_synced.csv")
        if not os.path.exists(pupil_csv):
            print(f" FEHLER: debug_4_synchronization.py muss für Modus {mode} zuerst laufen!")
            exit(1)
        print(f"Input (Modus {mode}): debug_4_pupil_data_synced.csv (synchronisiert)")
    else:
        pupil_csv = os.path.join(OUTPUT_BASE_DIR, "debug_1_pupil_data.csv")
        if not os.path.exists(pupil_csv):
            print(f" FEHLER: debug_1_video_analysis.py muss zuerst laufen!")
            exit(1)
        print(f"Input (Modus 1): debug_1_pupil_data.csv (roh)")
    
    pupil_data = pd.read_csv(pupil_csv)
    print(f" Geladen: {len(pupil_data)} Pupillen-Samples\n")
    
    # ══════════════════════════════════════════════════════════════════════
    # EyeLink-Bloecke (Modus 3 entfernt in v2.4)
    # ══════════════════════════════════════════════════════════════════════
    
    et_blocks_df = None  # Nicht mehr benoetigt (Modus 3 entfernt)
    
    if mode == 3:
        print(f"\n[!] FEHLER: Modus 3 ist nicht mehr unterstuetzt!")
        exit(1)
    
    # ══════════════════════════════════════════════════════════════════════
    # Kalibriere
    # ══════════════════════════════════════════════════════════════════════
    
    manager = CalibrationManager(mode=mode)
    pupil_calibrated, cal_info = manager.calibrate(pupil_data, et_blocks_df)
    
    # ══════════════════════════════════════════════════════════════════════
    # Export
    # ══════════════════════════════════════════════════════════════════════
    
    if cal_info['success']:
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        
        # ══════════════════════════════════════════════════════════════════════
        # DEDUPLIZIERUNG (Failsafe gegen debug_1 Overlap-Bug)
        # ══════════════════════════════════════════════════════════════════════

        time_col = 'timestamp_ms_synced' if 'timestamp_ms_synced' in pupil_calibrated.columns else 'timestamp_ms'

        print(f"\n{'='*70}")
        print("DEDUPLIZIERUNG (Failsafe)")
        print(f"{'='*70}\n")

        n_before = len(pupil_calibrated)
        n_duplicates = pupil_calibrated.duplicated(subset=[time_col]).sum()

        if n_duplicates > 0:
            print(f"️ {n_duplicates} Duplikate bei {time_col} gefunden!")
            print(f"   Ursache: Vermutlich debug_1 Trial-Overlap (0.5s Buffer)")
            print(f"   Aktion: Entferne Duplikate (behalte ersten Eintrag)...")
            
            # Sortiere nach Zeit
            pupil_calibrated = pupil_calibrated.sort_values(time_col).reset_index(drop=True)
            
            # Entferne Duplikate (keep='first' behält ältesten Eintrag)
            pupil_calibrated = pupil_calibrated.drop_duplicates(subset=[time_col], keep='first')
            pupil_calibrated = pupil_calibrated.reset_index(drop=True)
            
            n_after = len(pupil_calibrated)
            print(f"   ✓ Nach Bereinigung: {n_after} Frames (entfernt: {n_before - n_after})")
        else:
            print(f"✓ Keine Duplikate gefunden")

        print(f"{'='*70}\n")

        output_path = os.path.join(OUTPUT_BASE_DIR, "debug_5_pupil_data_calibrated.csv")
        pupil_calibrated.to_csv(output_path, index=False)
        
        info_path = os.path.join(OUTPUT_BASE_DIR, "debug_5_calibration_info.json")
        with open(info_path, 'w') as f:
            json.dump(cal_info, f, indent=2)
        
        print(f"\n{'='*70}")
        print("ERFOLGREICH!")
        print(f"{'='*70}")
        print(f"\nGespeichert:")
        print(f"  • {output_path}")
        print(f"  • {info_path}")
        
        print(f"\nCSV-Spalten:")
        for col in pupil_calibrated.columns:
            print(f"  - {col}")
        
        print(f"\nNächster Schritt: debug_6_trial_comparison.py")
    else:
        print(f"\n{'='*70}")
        print("FEHLER BEI KALIBRIERUNG")
        print(f"{'='*70}")
        print(f"\nGrund: {cal_info.get('reason', 'unknown')}")