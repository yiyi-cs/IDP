"""
=================================================================================
DEBUG_1_PTGAZE.PY - ptgaze Gaze Detection & Video Analysis v1.0
=================================================================================
Analog zu debug_1_video_analysis.py, aber mit ptgaze (ETH-XGaze):

Output:
- CSV: gaze_pitch_deg, gaze_yaw_deg (statt pupil_x_px, pupil_y_px)
- QS-Video: ptgaze Visualizer (Face BBox, Gaze Vector, Head Pose)
- Blink Detection: MediaPipe EAR
- Bad Sample Detection: Outside Monitor, Corrupt Samples

CSV-Struktur:
- frame, timestamp_ms, trial_number
- gaze_pitch_deg, gaze_yaw_deg (DIREKT von ptgaze, in GRAD)
- gaze_vector_x, gaze_vector_y, gaze_vector_z (3D Unit Vector)
- confidence, detected
- is_blink, left_ear, right_ear, avg_ear (Blink Detection)
- head_yaw, head_pitch, head_roll (Head Pose)
- outside_monitor, corrupt_sample (Bad Sample Flags)

Version: v1.0 (basierend auf debug_1_video_analysis v2.2)
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

import cv2
import pandas as pd
import numpy as np
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict

# Config import
from config import *

# ptgaze imports
from shared.shared_gaze_detection_ptgaze import PtgazeGazeDetector
from omegaconf import OmegaConf
from ptgaze.common import Visualizer

# Environment-Variable-Overrides (fuer master_cli.py)
if 'PIPELINE_OUTPUT_BASE_DIR' in os.environ:
    OUTPUT_BASE_DIR = Path(os.environ['PIPELINE_OUTPUT_BASE_DIR'])

if 'PIPELINE_MAIN_VIDEO_PATH' in os.environ:
    VIDEO_PATH = Path(os.environ['PIPELINE_MAIN_VIDEO_PATH'])

# ==================== BAD SAMPLE DETECTION ====================

class BadSampleDetector:
    """
    Detektiert schlechte Samples (Ehinger et al. 2019):
    - Outside Monitor (NUR NACH KALIBRIERUNG!)
    - Corrupt Samples (niedrige Confidence)
    
    KORRIGIERT v1.1: Outside Monitor Check DEAKTIVIERT für uncalibrated data!
    """
    
    def __init__(self, check_outside_monitor: bool = False):
        """
        Args:
            check_outside_monitor: True = Aktiviere Outside-Check (NUR für calibrated data!)
                                   False = Nur Confidence-Check (für uncalibrated)
        """
        
        self.check_outside_monitor = check_outside_monitor
        
        if check_outside_monitor:
            # Aus config.py
            screen_width_cm = SCREEN_WIDTH_CM
            screen_height_cm = SCREEN_HEIGHT_CM
            viewing_distance_cm = VIEWING_DISTANCE_CM
            
            # Berechne maximale Winkel (in Grad)
            self.max_gaze_x = np.arctan((screen_width_cm / 2) / viewing_distance_cm) * 180 / np.pi
            self.max_gaze_y = np.arctan((screen_height_cm / 2) / viewing_distance_cm) * 180 / np.pi
            
            print(f"   BadSampleDetector initialisiert:")
            print(f"     Max Gaze X: +/- {self.max_gaze_x:.1f} deg")
            print(f"     Max Gaze Y: +/- {self.max_gaze_y:.1f} deg")
        else:
            print(f"   BadSampleDetector initialisiert (Outside Monitor: DEAKTIVIERT)")
            self.max_gaze_x = None
            self.max_gaze_y = None
        
        # Confidence-Schwelle (aus config.py oder Default)
        self.min_confidence = getattr(globals(), 'MIN_GAZE_CONFIDENCE', 0.7)
        print(f"     Min Confidence: {self.min_confidence}")
    
    def check_sample(self, gaze_pitch_deg: float, gaze_yaw_deg: float, 
                    confidence: float) -> Dict[str, bool]:
        """
        Prueft ob Sample schlecht ist.
        
        Returns:
            dict mit:
            - 'outside_monitor': True/False (immer False wenn check_outside_monitor=False)
            - 'corrupt_sample': True/False
        """
        
        # 1. Outside Monitor (NUR wenn aktiviert!)
        if self.check_outside_monitor and self.max_gaze_x is not None:
            outside_monitor = (
                abs(gaze_pitch_deg) > self.max_gaze_y or
                abs(gaze_yaw_deg) > self.max_gaze_x
            )
        else:
            outside_monitor = False  # ← DEAKTIVIERT für uncalibrated!
        
        # 2. Corrupt Sample (niedrige Confidence)
        corrupt_sample = (confidence < self.min_confidence)
        
        return {
            'outside_monitor': outside_monitor,
            'corrupt_sample': corrupt_sample
        }

# ==================== MULTI-VIDEO HELPER ====================

def detect_block_videos(video_folder: Path) -> Optional[Dict[str, str]]:
    """
    Erkennt automatisch block1.mp4 und block2.mp4 aus Ordner.
    
    Sucht nach Pattern:
    - *_block1*.mp4
    - *_block2*.mp4
    
    Returns:
        {'block1': path, 'block2': path} oder None
    """
    
    if not video_folder.exists():
        return None
    
    print(f"\n[>] Suche Block-Videos in: {video_folder}")
    
    block_patterns = {
        'block1': ['*_block1*.mp4', '*block_1*.mp4'],
        'block2': ['*_block2*.mp4', '*block_2*.mp4']
    }
    
    video_paths = {}
    
    for block, patterns in block_patterns.items():
        for pattern in patterns:
            videos = list(video_folder.glob(pattern))
            if videos:
                video_paths[block] = str(max(videos, key=lambda p: p.stat().st_mtime))
                print(f"    {block.upper()}: {Path(video_paths[block]).name}")
                break
    
    if len(video_paths) == 2:
        return video_paths
    else:
        return None

# ==================== PTGAZE GAZE ANALYZER ====================

class PtgazeGazeAnalyzer:
    """
    Hauptklasse fuer ptgaze-basierte Gaze-Analyse.
    
    Analog zu PupilDetector aus debug_1_video_analysis.py
    """
    
    def __init__(self, video_width: int = 1440, video_height: int = 1080):
        """
        Args:
            video_width, video_height: Video-Aufloesung
        """
        
        print(f"\n{'='*70}")
        print("PTGAZE GAZE ANALYZER")
        print(f"{'='*70}")
        
        # Initialisiere Detector
        self.detector = PtgazeGazeDetector(
            video_width=video_width,
            video_height=video_height,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            model='eth-xgaze'
        )
        
        # Bad Sample Detector
        self.bad_sample_detector = BadSampleDetector(check_outside_monitor=False)
        
        # Visualizer (ptgaze)
        # Wird in process_video() initialisiert (braucht Camera)
        self.visualizer = None
        
        self.fps = None
        
        print(f"\n CSV-Struktur:")
        print(f"   - gaze_pitch_deg, gaze_yaw_deg (DIREKT von ptgaze)")
        print(f"   - gaze_vector_x/y/z (3D Unit Vector)")
        print(f"   - confidence, detected")
        print(f"   - is_blink, left/right/avg_ear (Blink Detection)")
        print(f"   - head_yaw/pitch/roll (Head Pose)")
        print(f"   - outside_monitor, corrupt_sample (Bad Samples)")
    
    def process_video(self, video_path: str, video_path_block2: Optional[str] = None):
        """
        Verarbeitet Video(s) Frame-by-Frame.
        
        [PHASE 1]: Nutzt phases_detected.json fuer gezielte Extraktion
        [FALLBACK]: Verarbeitet ganzes Video (Legacy)
        """
        
        # ======================================================================
        # PRUEFE: Automatische Phasen-Detektion aktiviert?
        # ======================================================================
        
        from config import ENABLE_AUTOMATIC_PHASE_DETECTION
        
        phases_json = os.path.join(OUTPUT_BASE_DIR, "phases_detected.json")
        
        if ENABLE_AUTOMATIC_PHASE_DETECTION and os.path.exists(phases_json):
            print(f"\n{'='*70}")
            print("[PHASE 1]: GEZIELTE TRIAL-EXTRAKTION")
            print(f"{'='*70}")
            print(f"   Nutze phases_detected.json fuer Trial-Grenzen")
            
            return self._process_video_with_phases(video_path, phases_json)
        
        else:
            # ==================================================================
            # FALLBACK: Verarbeite ganzes Video (Legacy)
            # ==================================================================
            
            print(f"\n[!] phases_detected.json nicht gefunden")
            print(f"   >> Verarbeite GANZES Video (Legacy)")
            
            return self._process_video_legacy(video_path, video_path_block2)
    
    def _process_video_with_phases(self, video_path: str, phases_json: str) -> pd.DataFrame:
        """
        [NEU]: Verarbeitet nur Trial-Phasen aus phases_detected.json.
        
        Vorteile:
        - Spart Zeit (nur relevante Phasen)
        - Automatische Trial-Segmentierung
        - Konsistent mit debug_0
        """
        
        # Lade Phasen
        with open(phases_json, 'r') as f:
            phases = json.load(f)
        
        # Extrahiere alle Trials
        all_trials = []
        for block_key in ['experiment_block1', 'experiment_block2']:
            if block_key in phases['phases']:
                all_trials.extend(phases['phases'][block_key]['trials'])
        
        print(f"\n    {len(all_trials)} Trials gefunden")
        
        video_path_from_json = phases.get('video_path', None)
        
        if video_path_from_json and os.path.exists(video_path_from_json):
            print(f"    Nutze Video aus phases_detected.json: {Path(video_path_from_json).name}")
            video_path = video_path_from_json
        else:
            print(f"    Video-Pfad aus JSON nicht gefunden, nutze Parameter: {Path(video_path).name}")
        
        # Oeffne Video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("    [X] Video konnte nicht geoeffnet werden!")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = fps
        
        print(f"   Video: {width}x{height} @ {fps:.1f} FPS")
        
        # Initialisiere ptgaze Visualizer
        self._init_visualizer(width, height)
        
        # QS-Video vorbereiten
        output_video = os.path.join(OUTPUT_BASE_DIR, "debug_1_ptgaze_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        # Sammle alle Daten
        all_data = []
        
        # ======================================================================
        # Iteriere ueber Trials
        # ======================================================================
        
        for trial in tqdm(all_trials, desc="Verarbeite Trials"):
            trial_num = trial['trial_number']
            
            # Zeitfenster (Fixation + Stimulus + Buffer)
            start_s = trial['fixation']['start_video_s'] - 0.5  # 0.5s Buffer
            end_s = trial['stimulus']['end_video_s']
            
            start_frame = int(start_s * fps)
            end_frame = int(end_s * fps)
            
            # Springe zu Start
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Verarbeite Frames in diesem Trial
            for frame_num in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detektiere Gaze
                result = self._detect_frame(frame, frame_num, timestamp_offset_ms=0)
                result['trial_number'] = trial_num  # Fuege Trial-Nummer hinzu
                all_data.append(result)
                
                # QS-Video
                vis_frame = self._create_visualization(frame, result, width, height)
                out.write(vis_frame)
        
        cap.release()
        out.release()
        
        # Erstelle DataFrame
        df = pd.DataFrame(all_data)
        
        print(f"\n    {len(df)} Frames verarbeitet")
        print(f"    QS-Video: {output_video}")
        
        # Statistik
        self._print_statistics(df)
        
        return df
    
    def _process_video_legacy(self, video_path: str, video_path_block2: Optional[str] = None):
        """
        Legacy: Verarbeitet ganzes Video (wie bisher).
        
        Wird genutzt wenn:
        - ENABLE_AUTOMATIC_PHASE_DETECTION = False
        - phases_detected.json fehlt
        - Multi-Video Modus (block1/block2)
        """
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("[X] Video konnte nicht geoeffnet werden!")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = fps
        
        print(f"Video: {Path(video_path).name}")
        print(f"  {width}x{height} @ {fps:.1f} FPS")
        print(f"  {total_frames} Frames ({total_frames/fps:.1f}s)\n")
        
        # Initialisiere Visualizer
        self._init_visualizer(width, height)
        
        # QS-Video
        output_video = os.path.join(OUTPUT_BASE_DIR, "debug_1_ptgaze_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        data = []
        frame_number = 0
        
        print("Verarbeite Frames...")
        with tqdm(total=total_frames, desc="Progress", ncols=80) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                result = self._detect_frame(frame, frame_number, 0)
                data.append(result)
                
                # Visualisiere Frame
                vis_frame = self._create_visualization(frame, result, width, height)
                out.write(vis_frame)
                
                frame_number += 1
                pbar.update(1)
        
        cap.release()
        out.release()
        
        print(f"\n[OK] {frame_number} Frames verarbeitet")
        print(f"[OK] QS-Video: {output_video}")
        
        # Erstelle DataFrame
        df = pd.DataFrame(data)
        
        # Statistik
        self._print_statistics(df)
        
        return df
    
    def _init_visualizer(self, width: int, height: int):
        """Initialisiert ptgaze Visualizer"""
        
        # Visualizer braucht Camera + nose_index
        # Camera ist in detector.gaze_estimator.camera
        # nose_index: MediaPipe = 1, dlib = 30
        
        camera = self.detector.gaze_estimator.camera
        nose_index = 1  # MediaPipe Face Mesh Nose Tip
        
        self.visualizer = Visualizer(camera, nose_index)
        
        print(f"   [OK] ptgaze Visualizer initialisiert")
    
    def _detect_frame(self, frame, frame_number, timestamp_offset_ms):
        """Detektiert Gaze + Blink + Bad Samples"""
        
        result_dict = self.detector.extract_from_frame(frame)
        
        if result_dict['detected']:
            # Bad Sample Detection
            bad_sample_flags = self.bad_sample_detector.check_sample(
                gaze_pitch_deg=result_dict['gaze_pitch_deg'],
                gaze_yaw_deg=result_dict['gaze_yaw_deg'],
                confidence=result_dict['confidence']
            )
            
            result = {
                'frame': frame_number,
                'timestamp_ms': frame_number * (1000 / self.fps) + timestamp_offset_ms,
                
                # trial_number IMMER vorhanden (auch Legacy!)
                'trial_number': None,  # Wird in Phase-Modus ueberschrieben
                
                # Gaze Angles (DIREKT von ptgaze, in GRAD)
                'gaze_pitch_deg': result_dict['gaze_pitch_deg'],
                'gaze_yaw_deg': result_dict['gaze_yaw_deg'],
                
                # Gaze Vector (3D Unit Vector)
                'gaze_vector_x': result_dict['gaze_vector'][0],
                'gaze_vector_y': result_dict['gaze_vector'][1],
                'gaze_vector_z': result_dict['gaze_vector'][2],
                
                # Quality
                'confidence': result_dict['confidence'],
                'detected': True,
                
                # Blink Detection
                'is_blink': result_dict['is_blink'],
                'left_ear': result_dict['left_ear'],
                'right_ear': result_dict['right_ear'],
                'avg_ear': result_dict['avg_ear'],
                'eyes_closed': result_dict.get('eyes_closed', False),
                'blink_count': result_dict.get('blink_count', 0),
                
                # Head Pose
                'head_yaw': result_dict['head_yaw'],
                'head_pitch': result_dict['head_pitch'],
                'head_roll': result_dict['head_roll'],
                
                # Bad Sample Flags
                'outside_monitor': bad_sample_flags['outside_monitor'],
                'corrupt_sample': bad_sample_flags['corrupt_sample'],
                
                # Face (fuer Visualizer, nicht in CSV)
                '_face_object': result_dict['face']  # Internes Flag
            }
        else:
            result = {
                'frame': frame_number,
                'timestamp_ms': frame_number * (1000 / self.fps),
                'trial_number': None,
                'gaze_pitch_deg': np.nan,
                'gaze_yaw_deg': np.nan,
                'gaze_vector_x': np.nan,
                'gaze_vector_y': np.nan,
                'gaze_vector_z': np.nan,
                'confidence': 0.0,
                'detected': False,
                'is_blink': False,
                'left_ear': None,
                'right_ear': None,
                'avg_ear': None,
                'eyes_closed': False,
                'blink_count': 0,
                'head_yaw': np.nan,
                'head_pitch': np.nan,
                'head_roll': np.nan,
                'outside_monitor': False,
                'corrupt_sample': True,  # Keine Detection = corrupt
                '_face_object': None
            }
        
        return result
    
    def _create_visualization(self, frame, row, frame_width, frame_height):
        """
        QS-Video: Nutzt ptgaze Visualizer.
        
        Zeigt:
        - Face Bounding Box
        - Gaze Vector (3D Line)
        - Head Pose Axes
        - Landmarks (optional)
        - Overlays (Text)
        """
        
        vis = frame.copy()
        
        if not row['detected']:
            cv2.putText(vis, "NO DETECTION", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return vis
        
        # ======================================================================
        # ptgaze Visualizer nutzen
        # ======================================================================
        
        face = row['_face_object']
        
        if face is not None:
            self.visualizer.set_image(vis.copy())
            
            # Face BBox
            self.visualizer.draw_bbox(face.bbox)
            
            # Head Pose Axes
            length = 0.05  # 5cm (wie in ptgaze demo.py)
            self.visualizer.draw_model_axes(face, length=length, lw=2)
            
            # Gaze Vector (3D Line)
            gaze_length = 0.05  # 5cm
            self.visualizer.draw_3d_line(
                face.center,
                face.center + gaze_length * face.gaze_vector
            )
            
            # Hole visualisiertes Bild zurueck
            vis = self.visualizer.image
        
        # ======================================================================
        # EINHEITLICHES INFO-OVERLAY (konsistent mit debug_1_video_analysis)
        # ======================================================================
        
        # Konstanten
        LEFT_MARGIN = 10
        LINE_HEIGHT = 22
        SECTION_GAP = 8
        BAR_HEIGHT = 15
        BOX_WIDTH = 450
        BOX_PADDING_TOP = 10
        BOX_PADDING_BOTTOM = 15
        
        # Erst Höhe berechnen
        overlay_y = BOX_PADDING_TOP + 15
        
        overlay_y += LINE_HEIGHT  # Frame
        overlay_y += LINE_HEIGHT  # Gaze
        overlay_y += LINE_HEIGHT  # Head
        overlay_y += LINE_HEIGHT  # Confidence
        overlay_y += SECTION_GAP
        
        if row.get('avg_ear') is not None:
            overlay_y += LINE_HEIGHT  # EAR Text
            overlay_y += BAR_HEIGHT + SECTION_GAP  # EAR Bar
        
        overlay_y += LINE_HEIGHT  # Blinks
        
        if row['outside_monitor']:
            overlay_y += LINE_HEIGHT
        if row['corrupt_sample']:
            overlay_y += LINE_HEIGHT
        if row.get('eyes_closed', False):
            overlay_y += LINE_HEIGHT + 3
        if row.get('is_blink', False):
            overlay_y += LINE_HEIGHT + 3
        
        box_height = overlay_y + BOX_PADDING_BOTTOM
        
        # Kasten zeichnen
        cv2.rectangle(vis, (5, 5), (BOX_WIDTH, box_height), (0, 0, 0), -1)
        cv2.rectangle(vis, (5, 5), (BOX_WIDTH, box_height), (255, 255, 255), 2)
        
        # Inhalt zeichnen
        overlay_y = BOX_PADDING_TOP + 15
        
        cv2.putText(vis, f"Frame: {int(row['frame'])}", (LEFT_MARGIN, overlay_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        overlay_y += LINE_HEIGHT
        
        cv2.putText(vis, f"Gaze: Pitch {row['gaze_pitch_deg']:.2f}° Yaw {row['gaze_yaw_deg']:.2f}°", 
                   (LEFT_MARGIN, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        overlay_y += LINE_HEIGHT
        
        cv2.putText(vis, f"Head: Y:{row.get('head_yaw', 0):.1f} P:{row.get('head_pitch', 0):.1f} R:{row.get('head_roll', 0):.1f}", 
                   (LEFT_MARGIN, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        overlay_y += LINE_HEIGHT
        
        cv2.putText(vis, f"Confidence: {row['confidence']:.2f}", 
                   (LEFT_MARGIN, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        overlay_y += LINE_HEIGHT + SECTION_GAP
        
        # EAR + Balken
        if row.get('avg_ear') is not None:
            ear_value = row['avg_ear']
            threshold = 0.22
            eyes_closed = row.get('eyes_closed', False)
            ear_color = (0, 0, 255) if eyes_closed else (0, 255, 0)
            
            cv2.putText(vis, f"EAR: {ear_value:.3f} (Thresh: {threshold})", 
                       (LEFT_MARGIN, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ear_color, 1)
            overlay_y += LINE_HEIGHT
            
            ear_normalized = min(ear_value / 0.4, 1.0)
            bar_width = int(400 * ear_normalized)
            
            cv2.rectangle(vis, (LEFT_MARGIN, overlay_y), (LEFT_MARGIN + 400, overlay_y + BAR_HEIGHT), (50, 50, 50), -1)
            cv2.rectangle(vis, (LEFT_MARGIN, overlay_y), (LEFT_MARGIN + bar_width, overlay_y + BAR_HEIGHT), ear_color, -1)
            threshold_x = LEFT_MARGIN + int(400 * (threshold / 0.4))
            cv2.line(vis, (threshold_x, overlay_y), (threshold_x, overlay_y + BAR_HEIGHT), (0, 255, 255), 2)
            cv2.rectangle(vis, (LEFT_MARGIN, overlay_y), (LEFT_MARGIN + 400, overlay_y + BAR_HEIGHT), (255, 255, 255), 1)
            overlay_y += BAR_HEIGHT + SECTION_GAP + 14
        
        # Blink Counter
        cv2.putText(vis, f"Blinks: {row.get('blink_count', 0)}", 
                   (LEFT_MARGIN, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        overlay_y += LINE_HEIGHT
        
        # Warnings
        if row['outside_monitor']:
            cv2.putText(vis, "[!] OUTSIDE MONITOR", (LEFT_MARGIN, overlay_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            overlay_y += LINE_HEIGHT
        
        if row['corrupt_sample']:
            cv2.putText(vis, "[!] LOW CONFIDENCE", (LEFT_MARGIN, overlay_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            overlay_y += LINE_HEIGHT
        
        if row.get('eyes_closed', False):
            cv2.putText(vis, ">>> EYES CLOSED <<<", (LEFT_MARGIN, overlay_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            overlay_y += LINE_HEIGHT + 3
        
        if row.get('is_blink', False):
            cv2.putText(vis, ">>> BLINK DETECTED <<<", (LEFT_MARGIN, overlay_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return vis
    
    def _print_statistics(self, df):
        """Finale Statistik"""
        
        print(f"\n{'='*70}")
        print("STATISTIK")
        print(f"{'='*70}\n")
        
        total = len(df)
        detected = df['detected'].sum()
        rate = (detected / total) * 100
        
        print(f"Detection Rate: {detected}/{total} ({rate:.1f}%)\n")
        
        if detected > 0:
            print(f"Gaze Angles (in Grad):")
            print(f"  Mean Pitch: {df['gaze_pitch_deg'].mean():.2f} deg")
            print(f"  Std Pitch:  {df['gaze_pitch_deg'].std():.2f} deg")
            print(f"  Mean Yaw:   {df['gaze_yaw_deg'].mean():.2f} deg")
            print(f"  Std Yaw:    {df['gaze_yaw_deg'].std():.2f} deg\n")
            
            print(f"Quality:")
            print(f"  Avg Confidence: {df['confidence'].mean():.2f}\n")
            
            # Blink Statistics
            blink_count = df['is_blink'].sum()
            blink_rate = (blink_count / detected) * 100
            print(f"Blinks: {blink_count} ({blink_rate:.1f}%)")
            
            if df['avg_ear'].notna().any():
                print(f"  Avg EAR: {df['avg_ear'].mean():.3f}\n")
            
            # Bad Samples
            outside = df['outside_monitor'].sum()
            corrupt = df['corrupt_sample'].sum()

            print(f"Bad Samples:")
            if self.bad_sample_detector.check_outside_monitor:
                print(f"  Outside Monitor: {outside} ({(outside/total)*100:.1f}%)")
            else:
                print(f"  Outside Monitor: Check deaktiviert (uncalibrated data)")
            print(f"  Corrupt Samples: {corrupt} ({(corrupt/total)*100:.1f}%)")
            
            # Good Samples
            good = (~df['outside_monitor'] & ~df['corrupt_sample'] & ~df['is_blink']).sum()
            print(f"\n[>] Good Samples: {good}/{total} ({(good/total)*100:.1f}%)")
    
    def close(self):
        """Schliesst Detektoren"""
        self.detector.close()

# ==================== MAIN ====================

if __name__ == "__main__":
    # Import torch (lazy import)
    try:
        import torch
    except ImportError:
        print("[X] PyTorch nicht installiert!")
        print("   Installiere mit: pip install torch torchvision")
        exit(1)
    
    print(f"\n{'='*70}")
    print("DEBUG 1: PTGAZE GAZE ANALYSIS v1.0")
    print(f"{'='*70}\n")
    
    print(f"Model: ETH-XGaze (Pre-trained)")
    print(f"Output: Gaze Angles (pitch, yaw) in Grad")
    
    # ======================================================================
    # AUTO-DETECTION: Multi-Video oder Single-Video?
    # ======================================================================
    
    from config import ENABLE_TEMPORAL_POOLING
    
    if ENABLE_TEMPORAL_POOLING:
        # Suche Block-Videos
        block_videos = detect_block_videos(Path(EXPERIMENTAL_VIDEOS_FOLDER))
        
        if block_videos is not None:
            video_path_block1 = block_videos['block1']
            video_path_block2 = block_videos['block2']
            
            print(f"\n[>] Multi-Video Modus")
            print(f"   Block 1: {Path(video_path_block1).name}")
            print(f"   Block 2: {Path(video_path_block2).name}")
        else:
            print(f"\n[X] Block-Videos nicht gefunden in: {EXPERIMENTAL_VIDEOS_FOLDER}")
            print(f"   Erwarte: *_block1.mp4 und *_block2.mp4")
            exit(1)
        
        analyzer = PtgazeGazeAnalyzer(video_width=1440, video_height=1080)
        df = analyzer.process_video(video_path_block1, video_path_block2)
    
    else:
        # Single-Video (Standard)
        video_path = str(VIDEO_PATH)
        
        print(f"\n[>] Single-Video Modus")
        print(f"   Video: {Path(VIDEO_PATH).name}")
        
        if not os.path.exists(VIDEO_PATH):
            print(f"\n[X] Video nicht gefunden: {VIDEO_PATH}")
            exit(1)
        
        analyzer = PtgazeGazeAnalyzer(video_width=1440, video_height=1080)
        df = analyzer.process_video(VIDEO_PATH)
    
    if df is not None:
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        
        # Entferne interne Spalte (_face_object) vor CSV-Export
        df_export = df.drop(columns=['_face_object'], errors='ignore')
        
        output_csv = os.path.join(OUTPUT_BASE_DIR, "debug_1_ptgaze_data.csv")
        df_export.to_csv(output_csv, index=False)
        
        print(f"\n{'='*70}")
        print("ERFOLGREICH!")
        print(f"{'='*70}")
        print(f"\nOutput: {output_csv}")
        print(f"Frames: {len(df)}")
        print(f"\nCSV-Spalten ({len(df_export.columns)}):")
        print(f"  - frame, timestamp_ms, trial_number")
        print(f"  - gaze_pitch_deg, gaze_yaw_deg (ptgaze Output)")
        print(f"  - gaze_vector_x/y/z (3D Unit Vector)")
        print(f"  - confidence, detected")
        print(f"  - is_blink, left/right/avg_ear (Blink Detection)")
        print(f"  - head_yaw/pitch/roll (Head Pose)")
        print(f"  - outside_monitor, corrupt_sample (Bad Samples)")
        
        # Konditional basierend auf Workflow
        from config import ENABLE_AUTOMATIC_PHASE_DETECTION
        if ENABLE_AUTOMATIC_PHASE_DETECTION:
            print(f"\nNaechster Schritt: offline_calibration_ptgaze.py")
        else:
            print(f"\nNaechster Schritt: debug_2_audio_detection.py (falls noetig)")
        
        analyzer.close()
