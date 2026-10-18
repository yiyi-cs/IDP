"""
=================================================================================
BATCH-SKRIPT: Kalibrierungs-Test R²-Vergleich
=================================================================================
Führt debug_0 + offline_calibration für alle 8 Test-VPs aus und
extrahiert die R²-Werte für den Vergleich.

Ordnerstruktur:
---------------
Videos:     .../Videos_25/kly1.mp4, kly2.mp4, ...
VP-Ordner:  .../kly/kly1/, kly2/, ...
Calib-JSON: .../kly/kly1/Calibration/calibration_timing_*.json
Sync-Log:   .../kly/kly1/experiment_sync_log.json

Usage:
    python run_calibration_test_batch.py

Output:
    - calibration_test_results.csv (R²-Werte aller VPs)
    - Console-Ausgabe mit Vergleichstabelle
=================================================================================
"""

import os
import sys
import subprocess
import pickle
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

# Pandas optional (für schöne Tabellen)
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("[INFO] pandas nicht installiert - nutze einfache Ausgabe")

# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES FÜR PKL-DESERIALISIERUNG
# (Müssen IDENTISCH mit offline_calibration.py sein!)
# ══════════════════════════════════════════════════════════════════════════════

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
    Kalibrierungs-Metadaten (aus phases_detected.json).
    """
    timepoints: List[str] = None  # ['beg'] oder ['beg', 'mid', 'end']
    n_points_total: int = 0
    source: str = 'phases_detected.json'
    mode: str = 'unknown'  # Wird dynamisch gesetzt
    
    def __post_init__(self):
        if self.timepoints is None:
            self.timepoints = []


# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURATION - ANPASSEN!
# ══════════════════════════════════════════════════════════════════════════════

# Basis-Ordner für Ergebnisse
BASE_FOLDER = Path(r"C:\Users\imanu\Documents\Imanuel\Studium\Psy\Master\Pupillendetektion\Ergebnisse")

# Video-Ordner (separat!)
VIDEO_FOLDER = BASE_FOLDER / "Videos_25"

# VP-Ordner (kly1, kly2, ...)
VP_BASE_FOLDER = BASE_FOLDER / "kly"

# Skript-Ordner
SCRIPTS_FOLDER = Path(r"C:\Users\imanu\Documents\Imanuel\Studium\Psy\Master\Pupillendetektion\Skripte\Debug\v3.0.9")

# Live-Output aktivieren? (True = zeigt kompletten Output von debug_0/offline_calibration)
SHOW_LIVE_OUTPUT = False

# VP-Codes und Metadaten
# kly1-4: Start links oben (wie mid-Kalibrierung)
# kly5-8: Start rechts unten (wie end-Kalibrierung)
VP_INFO = {
    'kly1': {'direction': 'links_oben', 'speed': 'speed_1', 'calib_type': 'mid'},
    'kly2': {'direction': 'links_oben', 'speed': 'speed_2', 'calib_type': 'mid'},
    'kly3': {'direction': 'links_oben', 'speed': 'speed_3', 'calib_type': 'mid'},
    'kly4': {'direction': 'links_oben', 'speed': 'speed_4', 'calib_type': 'mid'},
    'kly5': {'direction': 'rechts_unten', 'speed': 'speed_1', 'calib_type': 'end'},
    'kly6': {'direction': 'rechts_unten', 'speed': 'speed_2', 'calib_type': 'end'},
    'kly7': {'direction': 'rechts_unten', 'speed': 'speed_3', 'calib_type': 'end'},
    'kly8': {'direction': 'rechts_unten', 'speed': 'speed_4', 'calib_type': 'end'},
}

# ══════════════════════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN
# ══════════════════════════════════════════════════════════════════════════════

def find_video(vp_code: str) -> Path:
    """Findet Video für VP (in Videos_25 Ordner)"""
    
    patterns = [
        f"{vp_code}.mp4",
        f"{vp_code}.avi",
        f"{vp_code}_*.mp4",
        f"*{vp_code}*.mp4",
    ]
    
    for pattern in patterns:
        matches = list(VIDEO_FOLDER.glob(pattern))
        if matches:
            return matches[0]
    
    return None


def find_calibration_json(vp_folder: Path) -> Path:
    """Findet calibration_timing_*.json im Calibration-Unterordner"""
    
    calib_folder = vp_folder / "Calibration"
    
    if not calib_folder.exists():
        calib_folder = vp_folder
    
    matches = list(calib_folder.glob("calibration_timing_*.json"))
    
    if matches:
        return max(matches, key=lambda p: p.stat().st_mtime)
    
    return None


def find_sync_log(vp_folder: Path) -> Path:
    """Findet experiment_sync_log.json"""
    
    sync_log = vp_folder / "experiment_sync_log.json"
    
    if sync_log.exists():
        return sync_log
    
    matches = list(vp_folder.glob("*sync*.json"))
    if matches:
        return matches[0]
    
    return None


def run_script_with_live_output(script_path: Path, env: dict, cwd: Path) -> tuple:
    """
    Führt ein Python-Skript aus und zeigt Output LIVE an.
    
    Returns:
        (success: bool, output: str)
    """
    
    process = subprocess.Popen(
        [sys.executable, str(script_path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Kombiniere stderr in stdout
        text=True,
        cwd=str(cwd),
        encoding='utf-8',
        errors='replace',
        bufsize=1  # Line-buffered für Live-Output
    )
    
    output_lines = []
    
    # Lese Output Zeile für Zeile (LIVE!)
    for line in process.stdout:
        line_stripped = line.rstrip()
        
        if SHOW_LIVE_OUTPUT:
            print(f"      {line_stripped}")
        
        output_lines.append(line_stripped)
    
    process.wait()
    
    return process.returncode == 0, '\n'.join(output_lines)


# ══════════════════════════════════════════════════════════════════════════════
# HAUPTFUNKTION PRO VP
# ══════════════════════════════════════════════════════════════════════════════

def run_calibration_test_for_vp(vp_code: str) -> dict:
    """
    Führt debug_0 + offline_calibration (MediaPipe + ptgaze) für eine VP aus.
    
    Returns:
        dict mit r2_x, r2_y für BEIDE Methoden
    """
    
    print(f"\n{'='*70}")
    print(f" VP: {vp_code}")
    print(f"{'='*70}")
    
    # ══════════════════════════════════════════════════════════════════
    # PFADE FINDEN
    # ══════════════════════════════════════════════════════════════════
    
    vp_folder = VP_BASE_FOLDER / vp_code
    
    if not vp_folder.exists():
        print(f"   [!] VP-Ordner nicht gefunden: {vp_folder}")
        return {'error': 'VP-Ordner nicht gefunden', 'vp_code': vp_code}
    
    # Video
    video_path = find_video(vp_code)
    if video_path is None:
        print(f"   [!] Video nicht gefunden für {vp_code}")
        return {'error': 'Video nicht gefunden', 'vp_code': vp_code}
    print(f"   Video: {video_path.name}")
    
    # Sync-Log
    sync_log = find_sync_log(vp_folder)
    if sync_log is None:
        print(f"   [!] experiment_sync_log.json nicht gefunden")
        return {'error': 'Sync-Log nicht gefunden', 'vp_code': vp_code}
    print(f"   Sync-Log: {sync_log.name}")
    
    # Calibration JSON
    calib_json = find_calibration_json(vp_folder)
    if calib_json is None:
        print(f"   [!] calibration_timing_*.json nicht gefunden")
        return {'error': 'Calibration-JSON nicht gefunden', 'vp_code': vp_code}
    print(f"   Calib-JSON: {calib_json.name}")
    
    # Analyse-Ordner erstellen
    analyse_folder = vp_folder / "Analyse"
    analyse_folder.mkdir(exist_ok=True)
    print(f"   Output: {analyse_folder}")
    
    # ══════════════════════════════════════════════════════════════════
    # ENVIRONMENT-VARIABLEN SETZEN
    # ══════════════════════════════════════════════════════════════════
    
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    env['PIPELINE_OUTPUT_BASE_DIR'] = str(analyse_folder)
    env['PIPELINE_MAIN_VIDEO_PATH'] = str(video_path)
    env['PIPELINE_EXPERIMENT_SYNC_JSON_PATH'] = str(sync_log)
    env['PIPELINE_CALIBRATION_TIMING_JSON_DIR'] = str(calib_json.parent)
    env['PIPELINE_CALIB_MODE'] = 'CalibrationTest'
    
    # ══════════════════════════════════════════════════════════════════
    # SCHRITT 1: debug_0_phase_detection.py
    # ══════════════════════════════════════════════════════════════════
    
    print(f"\n   [1/3] debug_0_phase_detection.py...")
    print(f"   {'-'*60}")
    
    debug_0_script = SCRIPTS_FOLDER / "debug" / "debug_0_phase_detection.py"
    
    if not debug_0_script.exists():
        print(f"   [!] Skript nicht gefunden: {debug_0_script}")
        return {'error': 'debug_0 nicht gefunden', 'vp_code': vp_code}
    
    success, output = run_script_with_live_output(debug_0_script, env, SCRIPTS_FOLDER)
    
    print(f"   {'-'*60}")
    
    # Prüfe auf OUTPUT-DATEI
    phases_json = analyse_folder / "phases_detected.json"
    
    if not phases_json.exists():
        print(f"   [!] debug_0 fehlgeschlagen - keine phases_detected.json!")
        return {'error': 'Keine phases_detected.json erstellt', 'vp_code': vp_code}
    
    print(f"   [OK] phases_detected.json erstellt")
    
    # ══════════════════════════════════════════════════════════════════
    # SCHRITT 2: offline_calibration.py (MEDIAPIPE)
    # ══════════════════════════════════════════════════════════════════
    
    print(f"\n   [2/3] offline_calibration.py (MediaPipe)...")
    print(f"   {'-'*60}")
    
    calib_script = SCRIPTS_FOLDER / "calibration" / "offline_calibration.py"
    
    if not calib_script.exists():
        print(f"   [!] Skript nicht gefunden: {calib_script}")
        return {'error': 'offline_calibration nicht gefunden', 'vp_code': vp_code}
    
    success, output = run_script_with_live_output(calib_script, env, SCRIPTS_FOLDER)
    
    print(f"   {'-'*60}")
    
    # ══════════════════════════════════════════════════════════════════
    # SCHRITT 3: offline_calibration_ptgaze.py (PTGAZE)
    # ══════════════════════════════════════════════════════════════════
    
    print(f"\n   [3/3] offline_calibration_ptgaze.py (ptgaze)...")
    print(f"   {'-'*60}")
    
    calib_ptgaze_script = SCRIPTS_FOLDER / "calibration" / "offline_calibration_ptgaze.py"
    
    ptgaze_success = False
    if calib_ptgaze_script.exists():
        success, output = run_script_with_live_output(calib_ptgaze_script, env, SCRIPTS_FOLDER)
        ptgaze_success = True
    else:
        print(f"   [!] Skript nicht gefunden: {calib_ptgaze_script}")
        print(f"   [INFO] ptgaze-Kalibrierung wird übersprungen")
    
    print(f"   {'-'*60}")
    
    # ══════════════════════════════════════════════════════════════════
    # SCHRITT 4: R²-WERTE EXTRAHIEREN (BEIDE METHODEN!)
    # ══════════════════════════════════════════════════════════════════
    
    result = {'vp_code': vp_code, 'error': None}
    
    # --- MediaPipe PKL ---
    mp_pkl_files = [p for p in analyse_folder.glob("calibration_*.pkl") 
                   if 'ptgaze' not in p.name]
    
    if mp_pkl_files:
        pkl_path = max(mp_pkl_files, key=lambda p: p.stat().st_mtime)
        try:
            with open(pkl_path, 'rb') as f:
                model = pickle.load(f)
            
            result['mp_r2_x'] = model['r2_scores'][0]
            result['mp_r2_y'] = model['r2_scores'][1]
            result['mp_r2_mean'] = (result['mp_r2_x'] + result['mp_r2_y']) / 2
            result['mp_n_points'] = model.get('n_training_points', 0)
            result['mp_pkl_path'] = str(pkl_path)
            
            print(f"\n   [OK] MediaPipe ERGEBNIS:")
            print(f"       R2 X: {result['mp_r2_x']:.4f}")
            print(f"       R2 Y: {result['mp_r2_y']:.4f}")
            print(f"       R2 Mean: {result['mp_r2_mean']:.4f}")
            print(f"       Punkte: {result['mp_n_points']}")
        except Exception as e:
            print(f"   [!] MediaPipe PKL Fehler: {e}")
            result['mp_error'] = str(e)
    else:
        print(f"   [!] Keine MediaPipe PKL gefunden!")
        result['mp_error'] = 'Keine PKL'
    
    # --- ptgaze PKL ---
    pt_pkl_files = list(analyse_folder.glob("calibration_ptgaze_*.pkl"))
    
    if pt_pkl_files:
        pkl_path = max(pt_pkl_files, key=lambda p: p.stat().st_mtime)
        try:
            with open(pkl_path, 'rb') as f:
                model = pickle.load(f)
            
            result['pt_r2_x'] = model['r2_scores'][0]
            result['pt_r2_y'] = model['r2_scores'][1]
            result['pt_r2_mean'] = (result['pt_r2_x'] + result['pt_r2_y']) / 2
            result['pt_n_points'] = model.get('n_training_points', 0)
            result['pt_pkl_path'] = str(pkl_path)
            
            print(f"\n   [OK] ptgaze ERGEBNIS:")
            print(f"       R2 X: {result['pt_r2_x']:.4f}")
            print(f"       R2 Y: {result['pt_r2_y']:.4f}")
            print(f"       R2 Mean: {result['pt_r2_mean']:.4f}")
            print(f"       Punkte: {result['pt_n_points']}")
        except Exception as e:
            print(f"   [!] ptgaze PKL Fehler: {e}")
            result['pt_error'] = str(e)
    else:
        print(f"   [!] Keine ptgaze PKL gefunden!")
        result['pt_error'] = 'Keine PKL'
    
    return result

# ══════════════════════════════════════════════════════════════════════════════
# HAUPTPROGRAMM
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*70)
    print(" KALIBRIERUNGS-TEST BATCH-ANALYSE")
    print("="*70)
    print(f"\n   Video-Ordner: {VIDEO_FOLDER}")
    print(f"   VP-Ordner: {VP_BASE_FOLDER}")
    print(f"   Skripte: {SCRIPTS_FOLDER}")
    print(f"   Live-Output: {'Ja' if SHOW_LIVE_OUTPUT else 'Nein'}")
    print(f"   VPs: {', '.join(VP_INFO.keys())}")
    
    # Prüfe ob Ordner existieren
    if not VIDEO_FOLDER.exists():
        print(f"\n   [!] Video-Ordner nicht gefunden: {VIDEO_FOLDER}")
        return
    
    if not VP_BASE_FOLDER.exists():
        print(f"\n   [!] VP-Ordner nicht gefunden: {VP_BASE_FOLDER}")
        return
    
    results = []
    
    for vp_code, info in VP_INFO.items():
        result = run_calibration_test_for_vp(vp_code)
        result['direction'] = info['direction']
        result['speed'] = info['speed']
        result['calib_type'] = info['calib_type']
        results.append(result)
    
    # ══════════════════════════════════════════════════════════════════
    # ERGEBNIS-TABELLE (BEIDE METHODEN!)
    # ══════════════════════════════════════════════════════════════════
    
    print("\n\n" + "="*70)
    print(" ERGEBNIS-UEBERSICHT")
    print("="*70)
    
    # MediaPipe Ergebnisse
    print(f"\n[MEDIAPIPE]")
    print("-" * 90)
    print(f"{'VP':<8} {'Richtung':<15} {'Speed':<10} {'R2 X':<10} {'R2 Y':<10} {'R2 Mean':<10} {'N':<5} {'Status':<10}")
    print("-" * 90)
    
    for r in results:
        if 'mp_r2_x' in r:
            print(f"{r['vp_code']:<8} {r['direction']:<15} {r['speed']:<10} "
                  f"{r['mp_r2_x']:.4f}     {r['mp_r2_y']:.4f}     {r['mp_r2_mean']:.4f}     "
                  f"{r['mp_n_points']:<5} OK")
        else:
            print(f"{r['vp_code']:<8} {r['direction']:<15} {r['speed']:<10} "
                  f"{'--':^10} {'--':^10} {'--':^10} {'--':^5} {r.get('mp_error', r.get('error', '?'))}")
    
    # ptgaze Ergebnisse
    print(f"\n[PTGAZE]")
    print("-" * 90)
    print(f"{'VP':<8} {'Richtung':<15} {'Speed':<10} {'R2 X':<10} {'R2 Y':<10} {'R2 Mean':<10} {'N':<5} {'Status':<10}")
    print("-" * 90)
    
    for r in results:
        if 'pt_r2_x' in r:
            print(f"{r['vp_code']:<8} {r['direction']:<15} {r['speed']:<10} "
                  f"{r['pt_r2_x']:.4f}     {r['pt_r2_y']:.4f}     {r['pt_r2_mean']:.4f}     "
                  f"{r['pt_n_points']:<5} OK")
        else:
            print(f"{r['vp_code']:<8} {r['direction']:<15} {r['speed']:<10} "
                  f"{'--':^10} {'--':^10} {'--':^10} {'--':^5} {r.get('pt_error', 'Nicht verfuegbar')}")
    
    # ══════════════════════════════════════════════════════════════════
    # GRUPPENVERGLEICH (BEIDE METHODEN!)
    # ══════════════════════════════════════════════════════════════════
    
    print("\n\n" + "="*70)
    print(" GRUPPENVERGLEICH")
    print("="*70)
    
    import statistics
    
    for method, prefix in [('MediaPipe', 'mp'), ('ptgaze', 'pt')]:
        print(f"\n   [{method.upper()}]")
        
        for direction in ['links_oben', 'rechts_unten']:
            group = [r for r in results 
                    if r['direction'] == direction and f'{prefix}_r2_mean' in r]
            
            if group:
                r2_x_vals = [r[f'{prefix}_r2_x'] for r in group]
                r2_y_vals = [r[f'{prefix}_r2_y'] for r in group]
                r2_mean_vals = [r[f'{prefix}_r2_mean'] for r in group]
                
                print(f"\n      {direction.upper()} (n={len(group)}):")
                print(f"         R2 X:    {statistics.mean(r2_x_vals):.4f} +/- {statistics.stdev(r2_x_vals) if len(r2_x_vals) > 1 else 0:.4f}")
                print(f"         R2 Y:    {statistics.mean(r2_y_vals):.4f} +/- {statistics.stdev(r2_y_vals) if len(r2_y_vals) > 1 else 0:.4f}")
                print(f"         R2 Mean: {statistics.mean(r2_mean_vals):.4f} +/- {statistics.stdev(r2_mean_vals) if len(r2_mean_vals) > 1 else 0:.4f}")
    
    # ══════════════════════════════════════════════════════════════════
    # CSV SPEICHERN (MIT BEIDEN METHODEN!)
    # ══════════════════════════════════════════════════════════════════
    
    csv_path = VP_BASE_FOLDER / "calibration_test_results.csv"
    
    headers = [
        'vp_code', 'direction', 'speed', 'calib_type',
        'mp_r2_x', 'mp_r2_y', 'mp_r2_mean', 'mp_n_points', 'mp_error',
        'pt_r2_x', 'pt_r2_y', 'pt_r2_mean', 'pt_n_points', 'pt_error',
        'error'
    ]
    
    if PANDAS_AVAILABLE:
        df = pd.DataFrame(results)
        # Reorder columns
        existing_cols = [c for c in headers if c in df.columns]
        df = df[existing_cols]
        df.to_csv(csv_path, index=False)
    else:
        with open(csv_path, 'w') as f:
            f.write(','.join(headers) + '\n')
            for r in results:
                row = [str(r.get(h, '')) for h in headers]
                f.write(','.join(row) + '\n')
    
    print(f"\n\n   [OK] Ergebnisse gespeichert: {csv_path}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Abgebrochen durch Nutzer")
    except Exception as e:
        print(f"\n[!] Fehler: {e}")
        import traceback
        traceback.print_exc()
