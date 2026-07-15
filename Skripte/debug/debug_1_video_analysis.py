"""
DEBUG 1: VIDEO-ANALYSE & PUPILLENDETEKTION (ERWEITERT)
Version: v2.2 - Variante B (Umfassende Analyse)

CSV-Struktur:
- *_px: ROH (ungespiegelt, ungeglättet)
- *_smooth: ROH + Geglättet
- *_final: ROH + Geglättet + Gespiegelt (falls Flag gesetzt)

QS-Video:
- Grün: Einzelne Pupillen (links/rechts)
- Gelb: Mittelwert ROH
- Orange: Mittelwert GEGLÄTTET
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
import mediapipe as mp
import pandas as pd
import numpy as np
import json
from pathlib import Path
import os
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
from config import *
from shared.shared_pupil_detection import RobustPupilDetector

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

# ==================== MULTI-VIDEO HELPER (NEU v2.3) ====================

def detect_block_videos(video_folder: Path) -> Optional[Dict[str, str]]:
    """
    🆕 v2.3: Erkennt automatisch block1.mp4 und block2.mp4 aus Ordner.
    
    Sucht nach Pattern:
    - *_block1*.mp4
    - *_block2*.mp4
    
    Returns:
        {'block1': path, 'block2': path} oder None
    """
    
    if not video_folder.exists():
        return None
    
    print(f"\n Suche Block-Videos in: {video_folder}")
    
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

# ==================== PUPILLEN-DETEKTOR (NEU: nutzt shared_pupil_detection) ====================

class PupilDetector:
    def __init__(self):
        """Initialisiert mit shared RobustPupilDetector"""
        self.detector = RobustPupilDetector()
        fps = None
        
        print(f"\n{'='*70}")
        print("PUPILLENDETEKTION (SHARED PIPELINE)")
        print(f"{'='*70}")
        print(f" Nutzt shared_pupil_detection.py")
        print(f" Ebene 1-3: Detektion + Plausibilität + Qualität")
        print(f" Keine Spiegelung (nur ROH-Daten!)")
        print(f"\nCSV-Struktur: REDUZIERT (12 Spalten statt 26)")
        print(f"  - *_raw: Ungefiltert, ungeglättet")
        print(f"  - left/right_quality: Landmark-Qualität")
        print(f"  - confidence: Gesamt-Confidence")
    
    def process_video(self, video_path: str, video_path_block2: Optional[str] = None):
        """
        Verarbeitet Video(s) Frame-by-Frame.
        
         PHASE 1: Nutzt phases_detected.json für gezielte Extraktion
         FALLBACK: Verarbeitet ganzes Video (Legacy)
        """
        
        # ═══════════════════════════════════════════════════════════════
        # PRÜFE: Automatische Phasen-Detektion aktiviert?
        # ═══════════════════════════════════════════════════════════════
        
        from config import ENABLE_AUTOMATIC_PHASE_DETECTION
        
        phases_json = os.path.join(OUTPUT_BASE_DIR, "phases_detected.json")
        
        if ENABLE_AUTOMATIC_PHASE_DETECTION and os.path.exists(phases_json):
            print(f"\n{'='*70}")
            print(" PHASE 1: GEZIELTE TRIAL-EXTRAKTION")
            print(f"{'='*70}")
            print(f"   Nutze phases_detected.json für Trial-Grenzen")
            
            return self._process_video_with_phases(video_path, phases_json)
        
        else:
            # ═══════════════════════════════════════════════════════════
            # FALLBACK: Verarbeite ganzes Video (Legacy)
            # ═══════════════════════════════════════════════════════════
            
            print(f"\n phases_detected.json nicht gefunden")
            print(f"   >> Verarbeite GANZES Video (Legacy)")
            
            return self._process_video_legacy(video_path, video_path_block2)
    
    def _process_video_with_phases(self, video_path: str, phases_json: str) -> pd.DataFrame:
        """
         NEU: Verarbeitet nur Trial-Phasen aus phases_detected.json.
        
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
        
        # Öffne Video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("    Video konnte nicht geöffnet werden!")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = fps
        
        print(f"   Video: {width}x{height} @ {fps:.1f} FPS")
        
        # QS-Video vorbereiten
        output_video = os.path.join(OUTPUT_BASE_DIR, "debug_1_pupil_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        # Sammle alle Daten
        all_data = []
        
        # ═══════════════════════════════════════════════════════════════
        # Iteriere über Trials
        # ═══════════════════════════════════════════════════════════════
        
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
                
                # Detektiere Pupille
                result = self._detect_frame(frame, frame_num, timestamp_offset_ms=0)
                result['trial_number'] = trial_num  #  Füge Trial-Nummer hinzu
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
        
        # ═══════════════════════════════════════════════════════════════
        # HIER KOMMT DER BISHERIGE CODE aus process_video()
        # ═══════════════════════════════════════════════════════════════
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(" Video konnte nicht geöffnet werden!")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = fps
        
        print(f"Video: {Path(video_path).name}")
        print(f"  {width}x{height} @ {fps:.1f} FPS")
        print(f"  {total_frames} Frames ({total_frames/fps:.1f}s)\n")
        
        # QS-Video
        output_video = os.path.join(OUTPUT_BASE_DIR, "debug_1_pupil_video.mp4")
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
        
        print(f"\n {frame_number} Frames verarbeitet")
        print(f" QS-Video: {output_video}")
        
        # Erstelle DataFrame
        df = pd.DataFrame(data)
        
        # Statistik
        self._print_statistics(df)
        
        return df
    
    def _detect_frame(self, frame, frame_number, timestamp_offset_ms):
        """Detektiert Pupillen + Blink - NUR ROH-WERTE (keine Glättung, keine Spiegelung!)"""
        
        result_dict = self.detector.extract_from_frame(frame)
        
        if result_dict['position'] is not None:
            result = {
                'frame': frame_number,
                'timestamp_ms': frame_number * (1000 / self.fps) + timestamp_offset_ms,
                
                # trial_number IMMER vorhanden (auch Legacy!)
                'trial_number': None,  # Wird in Phase-Modus überschrieben

                # ROH-Positionen (einzelne Augen)
                'left_pupil_x_px_raw': result_dict['left_position'][0] if result_dict['left_position'] else np.nan,
                'left_pupil_y_px_raw': result_dict['left_position'][1] if result_dict['left_position'] else np.nan,
                'right_pupil_x_px_raw': result_dict['right_position'][0] if result_dict['right_position'] else np.nan,
                'right_pupil_y_px_raw': result_dict['right_position'][1] if result_dict['right_position'] else np.nan,
                
                # ROH-Mittelwert (qualitätsgewichtet!)
                'avg_pupil_x_px_raw': result_dict['position'][0],
                'avg_pupil_y_px_raw': result_dict['position'][1],
                
                # Qualitäts-Metriken
                'left_quality': result_dict['left_quality'],
                'right_quality': result_dict['right_quality'],
                'confidence': result_dict['confidence'],

                # Head-Pose
                'head_yaw': result_dict.get('head_yaw'),
                'head_pitch': result_dict.get('head_pitch'),
                'head_roll': result_dict.get('head_roll'),
                'head_pose_confidence': result_dict.get('head_pose_confidence'),
                
                # Blink Detection
                'is_blink': result_dict.get('is_blink', False),
                'eyes_closed': result_dict.get('eyes_closed', False),
                'left_ear': result_dict.get('left_ear'),
                'right_ear': result_dict.get('right_ear'),
                'avg_ear': result_dict.get('avg_ear'),
                'blink_count': result_dict.get('blink_count', 0),
                
                'detected': True
            }
        else:
            result = {
                'frame': frame_number,
                'timestamp_ms': frame_number * (1000 / self.fps),
                'trial_number': None,
                'left_pupil_x_px_raw': np.nan,
                'left_pupil_y_px_raw': np.nan,
                'right_pupil_x_px_raw': np.nan,
                'right_pupil_y_px_raw': np.nan,
                'avg_pupil_x_px_raw': np.nan,
                'avg_pupil_y_px_raw': np.nan,
                'left_quality': 0.0,
                'right_quality': 0.0,
                'confidence': 0.0,
                'head_yaw': np.nan,
                'head_pitch': np.nan,
                'head_roll': np.nan,
                'head_pose_confidence': 0.0,
                # Blink Detection (auch bei fehlgeschlagener Detektion)
                'is_blink': False,
                'eyes_closed': False,
                'left_ear': None,
                'right_ear': None,
                'avg_ear': None,
                'blink_count': 0,
                'detected': False
            }
        
        return result
    
    def _create_visualization(self, frame, row, frame_width, frame_height):
        """
        QS-Video: Einheitliches Overlay mit Blink-Detection.
        
        LAYOUT-SYSTEM:
        - overlay_y: Aktuelle Y-Position für nächstes Element
        - Zeilenabstand: 22px (Text), 25-30px (nach Abschnitten)
        - Balken-Höhe: 15px
        - Kasten-Höhe: Dynamisch berechnet am Ende
        """
        
        vis = frame.copy()
        
        if not row['detected']:
            cv2.putText(vis, "NO DETECTION", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return vis
        
        # ===== EINZELNE PUPILLEN (GRÜN) =====
        if not np.isnan(row['left_pupil_x_px_raw']):
            left_x = int(row['left_pupil_x_px_raw'])
            left_y = int(row['left_pupil_y_px_raw'])
            cv2.circle(vis, (left_x, left_y), 5, (0, 255, 0), 2)
            cv2.putText(vis, f"L", (left_x + 8, left_y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        if not np.isnan(row['right_pupil_x_px_raw']):
            right_x = int(row['right_pupil_x_px_raw'])
            right_y = int(row['right_pupil_y_px_raw'])
            cv2.circle(vis, (right_x, right_y), 5, (0, 255, 0), 2)
            cv2.putText(vis, f"R", (right_x + 8, right_y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # ===== MITTELWERT (GELB, GRÖSSER) =====
        avg_x = int(row['avg_pupil_x_px_raw'])
        avg_y = int(row['avg_pupil_y_px_raw'])
        cv2.circle(vis, (avg_x, avg_y), 8, (0, 255, 255), 2)
        cv2.circle(vis, (avg_x, avg_y), 2, (0, 255, 255), -1)
        
        # ═══════════════════════════════════════════════════════════════
        # INFO-OVERLAY - Erst Inhalt berechnen, dann Kasten zeichnen
        # ═══════════════════════════════════════════════════════════════
        
        # Konstanten für Layout
        LEFT_MARGIN = 10
        LINE_HEIGHT = 22        # Standard-Zeilenabstand
        SECTION_GAP = 8         # Extra-Abstand nach Abschnitten
        BAR_HEIGHT = 15
        BOX_WIDTH = 350
        BOX_PADDING_TOP = 10
        BOX_PADDING_BOTTOM = 15
        
        # Starte Inhalts-Berechnung
        overlay_y = BOX_PADDING_TOP + 15  # Erste Zeile
        
        # ───────────────────────────────────────────────────────────────
        # BLOCK 1: Frame + Position
        # ───────────────────────────────────────────────────────────────
        content_start_y = overlay_y
        
        # Zeile 1: Frame
        frame_text = f"Frame: {int(row['frame'])}"
        overlay_y += LINE_HEIGHT
        
        # Zeile 2: Pupil Position
        pupil_text = f"Pupil: ({row['avg_pupil_x_px_raw']:.1f}, {row['avg_pupil_y_px_raw']:.1f})"
        overlay_y += LINE_HEIGHT
        
        # Zeile 3: Quality
        quality_text = f"Quality L: {row['left_quality']:.2f}  R: {row['right_quality']:.2f}"
        overlay_y += LINE_HEIGHT + SECTION_GAP
        
        # ───────────────────────────────────────────────────────────────
        # BLOCK 2: EAR + Balken
        # ───────────────────────────────────────────────────────────────
        
        ear_value = row.get('avg_ear')
        threshold = EAR_BLINK_THRESHOLD
        
        if ear_value is not None:
            # EAR Text
            overlay_y += LINE_HEIGHT
            
            # EAR Balken
            bar_y = overlay_y
            overlay_y += BAR_HEIGHT + SECTION_GAP
        
        # ───────────────────────────────────────────────────────────────
        # BLOCK 3: Blink Counter
        # ───────────────────────────────────────────────────────────────
        
        overlay_y += LINE_HEIGHT
        
        # ───────────────────────────────────────────────────────────────
        # BLOCK 4: Head-Pose (optional)
        # ───────────────────────────────────────────────────────────────
        
        has_head_pose = (row.get('head_yaw') is not None and 
                        not np.isnan(row.get('head_yaw', np.nan)))
        if has_head_pose:
            overlay_y += LINE_HEIGHT
        
        # ───────────────────────────────────────────────────────────────
        # BLOCK 5: Status-Warnungen (nur wenn aktiv)
        # ───────────────────────────────────────────────────────────────
        
        if row.get('eyes_closed', False):
            overlay_y += LINE_HEIGHT + 3
        
        if row.get('is_blink', False):
            overlay_y += LINE_HEIGHT + 3
        
        # ═══════════════════════════════════════════════════════════════
        # JETZT KASTEN ZEICHNEN (mit berechneter Höhe)
        # ═══════════════════════════════════════════════════════════════
        
        box_height = overlay_y + BOX_PADDING_BOTTOM
        
        cv2.rectangle(vis, (5, 5), (BOX_WIDTH, box_height), (0, 0, 0), -1)
        cv2.rectangle(vis, (5, 5), (BOX_WIDTH, box_height), (255, 255, 255), 2)
        
        # ═══════════════════════════════════════════════════════════════
        # JETZT INHALT ZEICHNEN
        # ═══════════════════════════════════════════════════════════════
        
        overlay_y = BOX_PADDING_TOP + 15
        
        # Block 1: Frame + Position
        cv2.putText(vis, frame_text, (LEFT_MARGIN, overlay_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        overlay_y += LINE_HEIGHT
        
        cv2.putText(vis, pupil_text, (LEFT_MARGIN, overlay_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        overlay_y += LINE_HEIGHT
        
        cv2.putText(vis, quality_text, (LEFT_MARGIN, overlay_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        overlay_y += LINE_HEIGHT + SECTION_GAP
        
        # Block 2: EAR + Balken
        if ear_value is not None:
            eyes_closed = row.get('eyes_closed', False)
            ear_color = (0, 0, 255) if eyes_closed else (0, 255, 0)
            
            cv2.putText(vis, f"EAR: {ear_value:.3f} (Thresh: {threshold})", 
                       (LEFT_MARGIN, overlay_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, ear_color, 1)
            overlay_y += LINE_HEIGHT
            
            # EAR-Balken
            ear_normalized = min(ear_value / 0.4, 1.0)
            bar_width = int(300 * ear_normalized)
            
            cv2.rectangle(vis, (LEFT_MARGIN, overlay_y), 
                         (LEFT_MARGIN + 300, overlay_y + BAR_HEIGHT), (50, 50, 50), -1)
            cv2.rectangle(vis, (LEFT_MARGIN, overlay_y), 
                         (LEFT_MARGIN + bar_width, overlay_y + BAR_HEIGHT), ear_color, -1)
            
            # Threshold-Linie
            threshold_x = LEFT_MARGIN + int(300 * (threshold / 0.4))
            cv2.line(vis, (threshold_x, overlay_y), 
                    (threshold_x, overlay_y + BAR_HEIGHT), (0, 255, 255), 2)
            cv2.rectangle(vis, (LEFT_MARGIN, overlay_y), 
                         (LEFT_MARGIN + 300, overlay_y + BAR_HEIGHT), (255, 255, 255), 1)
            
            overlay_y += BAR_HEIGHT + SECTION_GAP + 14
        
        # Block 3: Blink Counter
        blink_count = row.get('blink_count', 0)
        cv2.putText(vis, f"Blinks: {blink_count}", 
                   (LEFT_MARGIN, overlay_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        overlay_y += LINE_HEIGHT
        
        # Block 4: Head-Pose
        if has_head_pose:
            cv2.putText(vis, f"Head: Y:{row['head_yaw']:.1f} P:{row['head_pitch']:.1f} R:{row['head_roll']:.1f}", 
                       (LEFT_MARGIN, overlay_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            overlay_y += LINE_HEIGHT
        
        # Block 5: Status-Warnungen
        if row.get('eyes_closed', False):
            cv2.putText(vis, ">>> EYES CLOSED <<<", 
                       (LEFT_MARGIN, overlay_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            overlay_y += LINE_HEIGHT + 3
        
        if row.get('is_blink', False):
            cv2.putText(vis, ">>> BLINK DETECTED <<<", 
                       (LEFT_MARGIN, overlay_y),
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
            print(f"ROH-Daten (ungefiltert):")
            print(f"  Mean X: {df['avg_pupil_x_px_raw'].mean():.1f} px")
            print(f"  Std X:  {df['avg_pupil_x_px_raw'].std():.2f} px")
            print(f"  Range:  {df['avg_pupil_x_px_raw'].max() - df['avg_pupil_x_px_raw'].min():.2f} px\n")
            
            print(f"Qualität:")
            print(f"  Avg Confidence: {df['confidence'].mean():.2f}")
            print(f"  Avg L Quality:  {df['left_quality'].mean():.2f}")
            print(f"  Avg R Quality:  {df['right_quality'].mean():.2f}")
    
    def close(self):
        """Schließt Detektor"""
        self.detector.close()

# ==================== MAIN ====================

if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("DEBUG 1: VIDEO-ANALYSE v2.3 (MULTI-VIDEO SUPPORT)")
    print(f"{'='*70}\n")
    
    print(f"Modus: {ANALYSIS_MODE}")
    print(f"Temporal Pooling: {' AKTIVIERT' if ENABLE_TEMPORAL_POOLING else ' Standard'}")
    
    # ══════════════════════════════════════════════════════════════
    # AUTO-DETECTION: Multi-Video oder Single-Video?
    # ══════════════════════════════════════════════════════════════
    
    if ENABLE_TEMPORAL_POOLING:
        # Suche Block-Videos
        block_videos = detect_block_videos(Path(EXPERIMENTAL_VIDEOS_FOLDER))
        
        if block_videos is not None:
            video_path_block1 = block_videos['block1']
            video_path_block2 = block_videos['block2']
            
            print(f"\n🎬 Multi-Video Modus")
            print(f"   Block 1: {Path(video_path_block1).name}")
            print(f"   Block 2: {Path(video_path_block2).name}")
        else:
            print(f"\n Block-Videos nicht gefunden in: {EXPERIMENTAL_VIDEOS_FOLDER}")
            print(f"   Erwarte: *_block1.mp4 und *_block2.mp4")
            exit(1)
    
        detector = PupilDetector()
        df = detector.process_video(video_path_block1, video_path_block2)

    else:
        # Single-Video (Standard)
        video_path_block1 = str(VIDEO_PATH)
        video_path_block2 = None
        
        print(f"\n Single-Video Modus")
        print(f"   Video: {Path(VIDEO_PATH).name}")
        
        if not os.path.exists(VIDEO_PATH):
            print(f"\n Video nicht gefunden: {VIDEO_PATH}")
            exit(1)
        detector = PupilDetector()
        df = detector.process_video(VIDEO_PATH)
        
    if df is not None:
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        
        output_csv = os.path.join(OUTPUT_BASE_DIR, "debug_1_pupil_data.csv")
        df.to_csv(output_csv, index=False)
        
        print(f"\n{'='*70}")
        print("ERFOLGREICH!")
        print(f"{'='*70}")
        print(f"\nOutput: {output_csv}")
        print(f"Frames: {len(df)}")
        print(f"\nCSV-Spalten ({len(df.columns)}):")
        print(f"  - frame, timestamp_ms")
        print(f"  - left/right_pupil_*_px_raw (einzelne Augen)")
        print(f"  - avg_pupil_*_px_raw (SINGLE SOURCE OF TRUTH)")
        print(f"  - left/right_quality (Landmark-Qualität)")
        print(f"  - confidence (gesamt)")
        print(f"  - detected")
        print(f"\n Keine Spiegelung, keine Glättung!")
        print(f"   >> Spätere Verarbeitung in debug_5/debug_6")
        
        # Konditional basierend auf Workflow
        if ENABLE_AUTOMATIC_PHASE_DETECTION:
            print(f"\nNächster Schritt: debug_4_synchronization.py")
        else:
            print(f"\nNächster Schritt: debug_2_audio_detection.py")

        
        detector.close()