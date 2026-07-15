"""
=================================================================================
VISUALISIERUNG: Blickbewegungen aus Video (MediaPipe + ptgaze)
=================================================================================
Extrahiert Pupillenpositionen aus Videos, wendet Kalibrierung an,
und erstellt Visualisierungen für beide Methoden nebeneinander.

Layout pro Seite:
- 2 Zeilen (2 VPs)
- Pro Zeile: MediaPipe (links) | ptgaze (rechts)
- Eine Colorbar pro Zeile

Output:
    - kly_video_seite1-4.png (4 Seiten mit je 2 VPs × 2 Methoden)
=================================================================================
"""

# ══════════════════════════════════════════════════════════════════════════════
# WICHTIG: OpenMP Konflikt vermeiden (MUSS VOR ALLEN IMPORTS STEHEN!)
# ══════════════════════════════════════════════════════════════════════════════
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import cv2
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
import sys

# ══════════════════════════════════════════════════════════════════════════════
# PATH SETUP
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Shared Pupil Detection (MediaPipe)
try:
    from shared.shared_pupil_detection import RobustPupilDetector
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("[!] shared_pupil_detection nicht gefunden!")
    MEDIAPIPE_AVAILABLE = False

# ptgaze Detection
try:
    from shared.shared_gaze_detection_ptgaze import PtgazeGazeDetector
    PTGAZE_AVAILABLE = True
except ImportError:
    print("[!] shared_gaze_detection_ptgaze nicht gefunden!")
    PTGAZE_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES FÜR PKL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScreenParameters:
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
    timepoints: List[str] = None
    n_points_total: int = 0
    source: str = 'phases_detected.json'
    mode: str = 'unknown'
    
    def __post_init__(self):
        if self.timepoints is None:
            self.timepoints = []


# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BASE_FOLDER = Path(r"C:\Users\imanu\Documents\Imanuel\Studium\Psy\Master\Pupillendetektion\Ergebnisse")
VIDEO_FOLDER = BASE_FOLDER / "Videos_25"
VP_BASE_FOLDER = BASE_FOLDER / "kly"
OUTPUT_FOLDER = VP_BASE_FOLDER / "Visualisierungen"
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Bildschirm-Parameter
RES_X = 1920
RES_Y = 1080
MARGIN_PERCENT_X = 0.17
MARGIN_PERCENT_Y = 0.20

# Extraktion
MIN_CONFIDENCE = 0.7
SUBSAMPLE_FACTOR = 3  # Nur jeden N-ten Frame
MAX_FRAMES = None     # None = alle

# VP-Info
VP_INFO = {
    'kly1': {'speed': 400, 'direction': 'links_oben', 'reverse': False},
    'kly2': {'speed': 300, 'direction': 'links_oben', 'reverse': False},
    'kly3': {'speed': 500, 'direction': 'links_oben', 'reverse': False},
    'kly4': {'speed': 200, 'direction': 'links_oben', 'reverse': False},
    'kly5': {'speed': 200, 'direction': 'rechts_unten', 'reverse': True},
    'kly6': {'speed': 500, 'direction': 'rechts_unten', 'reverse': True},
    'kly7': {'speed': 300, 'direction': 'rechts_unten', 'reverse': True},
    'kly8': {'speed': 400, 'direction': 'rechts_unten', 'reverse': True},
}


# ══════════════════════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN
# ══════════════════════════════════════════════════════════════════════════════

def calc_calibration_points():
    """Berechnet die 10 Kalibrierungspunkte"""
    
    margin_x = RES_X * MARGIN_PERCENT_X
    margin_y = RES_Y * MARGIN_PERCENT_Y
    
    def calc_pos(col, row):
        x = margin_x + (RES_X - 2 * margin_x) * col
        y = margin_y + (RES_Y - 2 * margin_y) * row
        return (x, y)
    
    points = [
        (RES_X / 2, RES_Y / 2),  # 0/5: Center
        calc_pos(0.0, 0.0),      # 1
        calc_pos(0.5, 0.0),      # 2
        calc_pos(1.0, 0.0),      # 3
        calc_pos(1.0, 0.5),      # 4
        calc_pos(0.5, 0.5),      # 5
        calc_pos(0.0, 0.5),      # 6
        calc_pos(0.0, 1.0),      # 7
        calc_pos(0.5, 1.0),      # 8
        calc_pos(1.0, 1.0),      # 9
    ]
    
    return points


def load_calibration_model(vp_code: str, method: str = 'mediapipe') -> Optional[dict]:
    """
    Lädt das Kalibrierungsmodell (PKL) für eine VP.
    
    Args:
        vp_code: VP-Code (z.B. 'kly1')
        method: 'mediapipe' oder 'ptgaze'
    """
    
    analyse_folder = VP_BASE_FOLDER / vp_code / "Analyse"
    
    if method == 'ptgaze':
        pkl_files = list(analyse_folder.glob("calibration_ptgaze_*.pkl"))
    else:
        # MediaPipe: calibration_*.pkl aber NICHT ptgaze
        pkl_files = [p for p in analyse_folder.glob("calibration_*.pkl") 
                    if 'ptgaze' not in p.name]
    
    if not pkl_files:
        return None
    
    pkl_path = max(pkl_files, key=lambda p: p.stat().st_mtime)
    
    with open(pkl_path, 'rb') as f:
        model = pickle.load(f)
    
    return model


def apply_calibration(positions: np.ndarray, model: dict) -> np.ndarray:
    """Wendet Kalibrierungsmodell auf Positionen an."""
    
    if len(positions) == 0:
        return np.array([])
    
    if model.get('polynomial_degree', 1) > 1:
        poly = model['poly_transformer']
        X_poly = poly.transform(positions)
        screen_x = model['model_x'].predict(X_poly)
        screen_y = model['model_y'].predict(X_poly)
    else:
        screen_x = model['model_x'].predict(positions)
        screen_y = model['model_y'].predict(positions)
    
    return np.column_stack([screen_x, screen_y])


def extract_mediapipe_positions(video_path: Path, detector, 
                                subsample: int = 3,
                                max_frames: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Extrahiert Pupillenpositionen mit MediaPipe."""
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.array([]), np.array([])
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    positions = []
    timestamps = []
    frame_idx = 0
    
    progress_interval = max(1, total_frames // 20)
    
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_idx >= max_frames):
            break
        
        if frame_idx % subsample == 0:
            result = detector.extract_from_frame(frame)
            if result['position'] is not None and result['confidence'] >= MIN_CONFIDENCE:
                positions.append(result['position'])
                timestamps.append(frame_idx / fps)
        
        if frame_idx % progress_interval == 0:
            pct = frame_idx / total_frames * 100
            print(f"         {pct:5.1f}%", end='\r')
        
        frame_idx += 1
    
    cap.release()
    print(f"         100% - {len(positions)} Punkte")
    
    return np.array(positions), np.array(timestamps)


def extract_ptgaze_positions(video_path: Path, detector,
                             subsample: int = 3,
                             max_frames: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Extrahiert Gaze Angles mit ptgaze."""
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.array([]), np.array([])
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    positions = []  # (pitch, yaw)
    timestamps = []
    frame_idx = 0
    
    progress_interval = max(1, total_frames // 20)
    
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_idx >= max_frames):
            break
        
        if frame_idx % subsample == 0:
            result = detector.extract_from_frame(frame)
            if result['detected'] and result['confidence'] >= MIN_CONFIDENCE:
                positions.append([result['gaze_pitch_deg'], result['gaze_yaw_deg']])
                timestamps.append(frame_idx / fps)
        
        if frame_idx % progress_interval == 0:
            pct = frame_idx / total_frames * 100
            print(f"         {pct:5.1f}%", end='\r')
        
        frame_idx += 1
    
    cap.release()
    print(f"         100% - {len(positions)} Punkte")
    
    return np.array(positions), np.array(timestamps)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT-FUNKTION
# ══════════════════════════════════════════════════════════════════════════════
def plot_gaze_data(ax, vp_code: str, gaze_positions: np.ndarray, 
                   timestamps: np.ndarray, model: dict, method: str,
                   show_ylabel: bool = True):
    """
    Plottet Blickdaten auf Bildschirm.
    
    Args:
        ax: Matplotlib Axes
        vp_code: VP-Code
        gaze_positions: Bildschirmkoordinaten
        timestamps: Zeitstempel
        model: Kalibrierungsmodell
        method: 'MediaPipe' oder 'ptgaze'
        show_ylabel: Y-Achsen-Label anzeigen (False für ptgaze)
    """
    
    info = VP_INFO.get(vp_code, {})
    
    # Bildschirmrahmen
    ax.add_patch(Rectangle((0, 0), RES_X, RES_Y, 
                           fill=False, edgecolor=[0.6, 0.6, 0.6], linewidth=1))
    
    # Blickdaten (farbcodiert nach Zeit)
    scatter = None
    n_valid = 0
    
    if len(gaze_positions) > 0:
        valid_mask = (
            (gaze_positions[:, 0] >= -50) & 
            (gaze_positions[:, 0] <= RES_X + 50) &
            (gaze_positions[:, 1] >= -50) & 
            (gaze_positions[:, 1] <= RES_Y + 50)
        )
        
        gaze_valid = gaze_positions[valid_mask]
        time_valid = timestamps[valid_mask]
        n_valid = len(gaze_valid)
        
        if len(time_valid) > 0:
            t_norm = (time_valid - time_valid.min()) / (time_valid.max() - time_valid.min() + 1e-6)
            scatter = ax.scatter(gaze_valid[:, 0], gaze_valid[:, 1],
                               c=t_norm, cmap='turbo', s=4, alpha=0.7)
    
    # Kalibrierungspunkte (GRÖSSERE Marker und Schrift!)
    target_points = calc_calibration_points()
    
    if info.get('reverse', False):
        seq = [0, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    else:
        seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    center_drawn = False
    for i, idx in enumerate(seq):
        x, y = target_points[idx]
        is_center = (abs(x - RES_X/2) < 10) and (abs(y - RES_Y/2) < 10)
        
        if is_center and not center_drawn:
            ax.scatter(x, y, s=150, facecolors='white', edgecolors=[0.5, 0.5, 0.5],
                      linewidth=1.0, alpha=0.6, zorder=10)
            ax.text(x, y, '0/5', ha='center', va='center', fontsize=7, 
                   color='black', fontweight='bold', zorder=11)
            center_drawn = True
        elif not is_center:
            ax.scatter(x, y, s=150, facecolors='white', edgecolors=[0.5, 0.5, 0.5],
                      linewidth=1.0, alpha=0.6, zorder=10)
            ax.text(x, y, str(i), ha='center', va='center', fontsize=7, 
                   color='black', fontweight='bold', zorder=11)
    
    # Achsen
    ax.set_xlim(-50, RES_X + 50)
    ax.set_ylim(-50, RES_Y + 50)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xlabel('X (px)', fontsize=9)
    
    # Y-Label nur wenn gewünscht (MediaPipe ja, ptgaze nein)
    if show_ylabel:
        ax.set_ylabel('Y (px)', fontsize=9)
    else:
        ax.set_ylabel('')
    
    ax.tick_params(labelsize=8)
    
    # Titel: NUR Methode + Speed + Richtung (OHNE R2 und n!)
    speed = info.get('speed', '?')
    direction = 'Start: links oben' if not info.get('reverse') else 'Start: rechts unten'
    
    title = f"{method}\n{speed} px/s | {direction}"
    ax.set_title(title, fontsize=10, fontweight='normal')
    
    return scatter

# ══════════════════════════════════════════════════════════════════════════════
# HAUPT-VERARBEITUNG
# ══════════════════════════════════════════════════════════════════════════════

def process_single_vp(vp_code: str, mp_detector, pt_detector) -> dict:
    """
    Verarbeitet eine VP mit beiden Methoden.
    
    Returns:
        dict mit Daten für MediaPipe und ptgaze
    """
    
    print(f"\n   {'='*60}")
    print(f"   VP: {vp_code}")
    print(f"   {'='*60}")
    
    video_path = VIDEO_FOLDER / f"{vp_code}.mp4"
    if not video_path.exists():
        print(f"      [!] Video nicht gefunden")
        return None
    
    result = {'vp_code': vp_code}
    
    # ══════════════════════════════════════════════════════════════════
    # MEDIAPIPE
    # ══════════════════════════════════════════════════════════════════
    
    print(f"      MediaPipe...")
    model_mp = load_calibration_model(vp_code, 'mediapipe')
    
    if model_mp and mp_detector:
        positions_mp, timestamps_mp = extract_mediapipe_positions(
            video_path, mp_detector, SUBSAMPLE_FACTOR, MAX_FRAMES)
        
        if len(positions_mp) > 0:
            gaze_mp = apply_calibration(positions_mp, model_mp)
            result['mediapipe'] = {
                'gaze': gaze_mp, 
                'timestamps': timestamps_mp, 
                'model': model_mp
            }
    else:
        print(f"         [!] Kein MediaPipe-Modell oder Detektor")
    
    # ══════════════════════════════════════════════════════════════════
    # PTGAZE
    # ══════════════════════════════════════════════════════════════════
    
    print(f"      ptgaze...")
    model_pt = load_calibration_model(vp_code, 'ptgaze')
    
    if model_pt and pt_detector:
        positions_pt, timestamps_pt = extract_ptgaze_positions(
            video_path, pt_detector, SUBSAMPLE_FACTOR, MAX_FRAMES)
        
        if len(positions_pt) > 0:
            gaze_pt = apply_calibration(positions_pt, model_pt)
            result['ptgaze'] = {
                'gaze': gaze_pt, 
                'timestamps': timestamps_pt, 
                'model': model_pt
            }
    else:
        print(f"         [!] Kein ptgaze-Modell oder Detektor")
    
    return result

def create_overview_pages(all_data: dict):
    """
    Erstellt Übersichts-Seiten (2 VPs × 2 Methoden pro Seite).
    
    Layout pro Seite (mit gutem Abstand):
    - Zeile 0: VP1 (MediaPipe + ptgaze)
    - Zeile 1: VP2 (MediaPipe + ptgaze)  
    - Zeile 2: Horizontale Colorbar (zentral unten)
    """
    
    print("\n" + "="*70)
    print(" ERSTELLE VISUALISIERUNGEN")
    print("="*70)
    
    vp_codes = list(VP_INFO.keys())
    
    for page in range(4):
        # Figure mit gutem Seitenverhältnis
        fig = plt.figure(figsize=(12, 9))
        
        # GridSpec: 3 Zeilen (VP1 | VP2 | Colorbar), 2 Spalten (MP | ptgaze)
        gs = GridSpec(3, 2, figure=fig, 
                     height_ratios=[1, 1, 0.05],  # Colorbar etwas größer für Lesbarkeit
                     hspace=0.40,                  # vertikaler Abstand
                     wspace=0.12,                  # horizontaler Abstand
                     left=0.08, right=0.98, top=0.92, bottom=0.08)
        
        # ══════════════════════════════════════════════════════════════════
        # ZEILE 0: Erste VP
        # ══════════════════════════════════════════════════════════════════
        
        vp_idx_0 = page * 2
        if vp_idx_0 < len(vp_codes):
            vp_code_0 = vp_codes[vp_idx_0]
            data_0 = all_data.get(vp_code_0)
            
            # MediaPipe (links, MIT Y-Label)
            ax_mp_0 = fig.add_subplot(gs[0, 0])
            if data_0 and 'mediapipe' in data_0:
                plot_gaze_data(
                    ax_mp_0, vp_code_0,
                    data_0['mediapipe']['gaze'],
                    data_0['mediapipe']['timestamps'],
                    data_0['mediapipe']['model'],
                    'MediaPipe',
                    show_ylabel=True
                )
            else:
                ax_mp_0.text(0.5, 0.5, f'{vp_code_0}\nMediaPipe: Keine Daten',
                            ha='center', va='center', fontsize=10, transform=ax_mp_0.transAxes)
                ax_mp_0.set_xlim(0, 1)
                ax_mp_0.set_ylim(0, 1)
            
            # ptgaze (rechts, OHNE Y-Label)
            ax_pt_0 = fig.add_subplot(gs[0, 1])
            if data_0 and 'ptgaze' in data_0:
                plot_gaze_data(
                    ax_pt_0, vp_code_0,
                    data_0['ptgaze']['gaze'],
                    data_0['ptgaze']['timestamps'],
                    data_0['ptgaze']['model'],
                    'ptgaze',
                    show_ylabel=False
                )
            else:
                ax_pt_0.text(0.5, 0.5, f'{vp_code_0}\nptgaze: Keine Daten',
                            ha='center', va='center', fontsize=10, transform=ax_pt_0.transAxes)
                ax_pt_0.set_xlim(0, 1)
                ax_pt_0.set_ylim(0, 1)
        
        # ══════════════════════════════════════════════════════════════════
        # ZEILE 1: Zweite VP
        # ══════════════════════════════════════════════════════════════════
        
        vp_idx_1 = page * 2 + 1
        if vp_idx_1 < len(vp_codes):
            vp_code_1 = vp_codes[vp_idx_1]
            data_1 = all_data.get(vp_code_1)
            
            # MediaPipe (links, MIT Y-Label)
            ax_mp_1 = fig.add_subplot(gs[1, 0])
            if data_1 and 'mediapipe' in data_1:
                plot_gaze_data(
                    ax_mp_1, vp_code_1,
                    data_1['mediapipe']['gaze'],
                    data_1['mediapipe']['timestamps'],
                    data_1['mediapipe']['model'],
                    'MediaPipe',
                    show_ylabel=True
                )
            else:
                ax_mp_1.text(0.5, 0.5, f'{vp_code_1}\nMediaPipe: Keine Daten',
                            ha='center', va='center', fontsize=10, transform=ax_mp_1.transAxes)
                ax_mp_1.set_xlim(0, 1)
                ax_mp_1.set_ylim(0, 1)
            
            # ptgaze (rechts, OHNE Y-Label)
            ax_pt_1 = fig.add_subplot(gs[1, 1])
            if data_1 and 'ptgaze' in data_1:
                plot_gaze_data(
                    ax_pt_1, vp_code_1,
                    data_1['ptgaze']['gaze'],
                    data_1['ptgaze']['timestamps'],
                    data_1['ptgaze']['model'],
                    'ptgaze',
                    show_ylabel=False
                )
            else:
                ax_pt_1.text(0.5, 0.5, f'{vp_code_1}\nptgaze: Keine Daten',
                            ha='center', va='center', fontsize=10, transform=ax_pt_1.transAxes)
                ax_pt_1.set_xlim(0, 1)
                ax_pt_1.set_ylim(0, 1)
        
        # ══════════════════════════════════════════════════════════════════
        # ZEILE 2: HORIZONTALE COLORBAR (zentral unten)
        # ══════════════════════════════════════════════════════════════════
        
        ax_cb = fig.add_subplot(gs[2, :])
        sm = plt.cm.ScalarMappable(cmap='turbo', norm=Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax_cb, orientation='horizontal')
        cbar.set_label('Zeit (normalisiert)', fontsize=10)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        cbar.ax.tick_params(labelsize=9)
        
        # Speichern
        output_path = OUTPUT_FOLDER / f"kly_video_seite{page + 1}.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   -> Gespeichert: {output_path.name}")
        
        plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# HAUPTPROGRAMM
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*70)
    print(" VIDEO-BLICKDATEN VISUALISIERUNG (MediaPipe + ptgaze)")
    print("="*70)
    print(f"\n   Video-Ordner: {VIDEO_FOLDER}")
    print(f"   VP-Ordner: {VP_BASE_FOLDER}")
    print(f"   Output: {OUTPUT_FOLDER}")
    print(f"   Subsample: jeden {SUBSAMPLE_FACTOR}. Frame")
    print(f"   MediaPipe: {'Verfuegbar' if MEDIAPIPE_AVAILABLE else 'NICHT verfuegbar'}")
    print(f"   ptgaze: {'Verfuegbar' if PTGAZE_AVAILABLE else 'NICHT verfuegbar'}")
    
    # Detektoren initialisieren
    mp_detector = None
    pt_detector = None
    
    if MEDIAPIPE_AVAILABLE:
        print("\n   Initialisiere MediaPipe...")
        mp_detector = RobustPupilDetector()
    
    if PTGAZE_AVAILABLE:
        print("   Initialisiere ptgaze...")
        # Video-Auflösung für ptgaze
        sample_video = VIDEO_FOLDER / "kly1.mp4"
        if sample_video.exists():
            cap = cv2.VideoCapture(str(sample_video))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            pt_detector = PtgazeGazeDetector(video_width=w, video_height=h)
    
    if mp_detector is None and pt_detector is None:
        print("\n   [!] Keine Detektoren verfuegbar!")
        return
    
    # Alle VPs verarbeiten
    all_data = {}
    
    for vp_code in VP_INFO.keys():
        try:
            result = process_single_vp(vp_code, mp_detector, pt_detector)
            all_data[vp_code] = result
        except Exception as e:
            print(f"      [!] Fehler bei {vp_code}: {e}")
            import traceback
            traceback.print_exc()
            all_data[vp_code] = None
    
    # Detektoren schließen
    if mp_detector:
        mp_detector.close()
    if pt_detector:
        pt_detector.close()
    
    # Visualisierungen erstellen
    create_overview_pages(all_data)
    
    print("\n" + "="*70)
    print(" FERTIG!")
    print("="*70)
    print(f"\n   Output: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Abgebrochen")
    except Exception as e:
        print(f"\n[!] Fehler: {e}")
        import traceback
        traceback.print_exc()
