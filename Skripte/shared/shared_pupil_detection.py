"""
=================================================================================
SHARED_PUPIL_DETECTION.PY - Vereinheitlichte Pupillen-Detektion v1.3
=================================================================================
Version: v1.3 - Einheitliche Blink-Detection + Package-Struktur

Enthaelt:
- RobustPupilDetector: Hauptklasse fuer Pupillen-Detektion
- HeadPoseEstimator: Kopfhaltungs-Schaetzung via solvePnP
- Qualitaetsgewichtete Durchschnittsberechnung
- IQR Outlier Removal
- Temporale Glaettung

Verwendet von:
- debug_1_video_analysis.py
- offline_calibration.py

Version: v1.3
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
import numpy as np
import pandas as pd
import mediapipe as mp
from typing import Tuple, Optional, Dict, List
from collections import deque
from shared.shared_blink_detection import BlinkDetector, create_blink_detector, ENABLE_BLINK_DETECTION

# ==================== CONFIG-IMPORT ====================

try:
    from config import *
except ImportError:
    # Fallback-Werte
    ENABLE_HEAD_POSE_EXTRACTION = False
    ENABLE_HEAD_POSE_TEMPORAL_SMOOTHING = False
    HEAD_POSE_SMOOTHING_WINDOW_FRAMES = 5
    HEAD_POSE_REFERENCE_LANDMARKS = [1, 33, 61, 199, 263, 291]
    CAMERA_FOCAL_LENGTH = 1000
    
    USE_QUALITY_WEIGHTED_AVERAGE = True
    SINGLE_EYE_PENALTY = 0.8
    MIN_LANDMARK_QUALITY = 0.4
    MIN_CONFIDENCE = 0.7
    EYE_DISTANCE_MIN_PX = 5
    EYE_DISTANCE_MAX_PX = 300
    EYE_DISTANCE_PENALTY = 0.5
    IQR_MULTIPLIER = 1.5
    MIN_SAMPLES_FOR_IQR = 10
    SMOOTHING_WINDOW_CALIBRATION = 7
    USE_MEDIAN_INSTEAD_OF_MEAN = True
    
    # P9: Blink Detection Fallback (falls shared_blink_detection auch fehlt)
    try:
        from shared_blink_detection import ENABLE_BLINK_DETECTION
    except ImportError:
        ENABLE_BLINK_DETECTION = True  # Standardmaessig aktiviert

# ==================== HEAD-POSE-ESTIMATOR ====================
class HeadPoseEstimator:
    """
    3D Head-Pose-Schätzung via solvePnP + RQDecomp3x3.
    
    KORRIGIERT v1.3:
    - Verwendet cv2.RQDecomp3x3 statt manueller Euler-Extraktion
    - Korrekte 3D-Modell-Punkt-Zuordnung zu Landmark-Indizes
    - Robuste Winkel-Berechnung ohne Gimbal-Lock-Probleme
    
    Erwartete Ausgabe (bei geradeaus schauendem Kopf):
    - Yaw ≈ 0° (links/rechts)
    - Pitch ≈ 0° (oben/unten)  
    - Roll ≈ 0° (Neigung)
    """
    
    def __init__(self, frame_width: int, frame_height: int,
                 focal_length: float = None):
        """
        Args:
            frame_width, frame_height: Video-Auflösung
            focal_length: Kamera-Brennweite (default: frame_width als Approximation)
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Focal Length: Wenn nicht angegeben, nutze Bildbreite als Approximation
        # (typisch für Webcams mit ~60° FOV)
        if focal_length is None:
            focal_length = frame_width
        
        # Kamera-Matrix (Standard-Webcam, kein Distortion)
        self.camera_matrix = np.array([
            [focal_length, 0, frame_width / 2],
            [0, focal_length, frame_height / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        
        # ═══════════════════════════════════════════════════════════════
        # 3D-MODELL-PUNKTE (KORRIGIERT v1.4)
        # ═══════════════════════════════════════════════════════════════
        # 
        # Zuordnung zu HEAD_POSE_REFERENCE_LANDMARKS = [1, 33, 61, 199, 263, 291]
        #
        # 3D-Koordinatensystem (Standard-Konvention):
        #   X: positiv = RECHTS (aus Kamera-Sicht)
        #   Y: positiv = OBEN (!) - NICHT wie Bildkoordinaten!
        #   Z: positiv = ZUR KAMERA (Gesicht zeigt in +Z)
        #
        # WICHTIG: Y im 3D-Modell ist INVERS zu Bild-Y!
        #   - Augen ÜBER Nase → Y POSITIV in 3D
        #   - Mund/Kinn UNTER Nase → Y NEGATIV in 3D
        #
        # Einheiten: Millimeter (relative Verhältnisse wichtiger als absolut)
        # Referenz: Durchschnittsgesicht (~170mm Höhe, ~140mm Breite)
        
        self.model_points_3d = np.array([
            # Index 0 → Landmark 1: Nose tip (ORIGIN)
            (0.0, 0.0, 0.0),
            
            # Index 1 → Landmark 33: Right eye outer corner
            # X: rechts (positiv), Y: ÜBER Nase (POSITIV!), Z: hinter Nase (negativ)
            (43.0, 32.0, -26.0),
            
            # Index 2 → Landmark 61: Right mouth corner
            # X: rechts (positiv), Y: UNTER Nase (NEGATIV!), Z: hinter Nase
            (29.0, -28.0, -24.0),
            
            # Index 3 → Landmark 199: Chin
            # X: Mitte, Y: weit UNTER Nase (stark NEGATIV!), Z: leicht hinter Nase
            (0.0, -63.0, -10.0),
            
            # Index 4 → Landmark 263: Left eye outer corner
            # X: links (negativ), Y: ÜBER Nase (POSITIV!), Z: hinter Nase
            (-43.0, 32.0, -26.0),
            
            # Index 5 → Landmark 291: Left mouth corner
            # X: links (negativ), Y: UNTER Nase (NEGATIV!), Z: hinter Nase
            (-29.0, -28.0, -24.0),
            
        ], dtype=np.float64)
        
        # Temporal Smoothing (für Stabilität)
        if ENABLE_HEAD_POSE_TEMPORAL_SMOOTHING:
            self.yaw_buffer = deque(maxlen=HEAD_POSE_SMOOTHING_WINDOW_FRAMES)
            self.pitch_buffer = deque(maxlen=HEAD_POSE_SMOOTHING_WINDOW_FRAMES)
            self.roll_buffer = deque(maxlen=HEAD_POSE_SMOOTHING_WINDOW_FRAMES)
        else:
            self.yaw_buffer = None
        
        print(f"   ✓ HeadPoseEstimator v1.4 initialisiert")
        print(f"     • Kamera: {frame_width}x{frame_height}, f={focal_length:.0f}px")
        print(f"     • Methode: cv2.RQDecomp3x3 (robust)")
        print(f"     • Temporal Smoothing: {'✓' if ENABLE_HEAD_POSE_TEMPORAL_SMOOTHING else '✗'}")
    
    def estimate_pose(self, face_landmarks, frame_shape: Tuple[int, int]) -> Optional[Dict]:
        """
        Schätzt Head-Pose aus MediaPipe Landmarks.
        
        Args:
            face_landmarks: MediaPipe FaceLandmarkList
            frame_shape: (height, width)
        
        Returns:
            dict mit:
            - 'yaw': Rotation um Y-Achse (links-/rechts-Drehung) in Grad
                     Positiv = Kopf dreht nach RECHTS (schaut nach rechts)
            - 'pitch': Rotation um X-Achse (oben/unten) in Grad
                       Positiv = Kopf kippt nach OBEN (schaut nach oben)
            - 'roll': Rotation um Z-Achse (Neigung) in Grad
                      Positiv = Kopf neigt nach RECHTS (rechtes Ohr zur Schulter)
            - 'confidence': 0.0-1.0 (basierend auf Reprojection Error)
        """
        
        height, width = frame_shape
        
        # ═══════════════════════════════════════════════════════════════
        # 1. EXTRAHIERE 2D-PUNKTE
        # ═══════════════════════════════════════════════════════════════
        
        image_points_2d = []
        for idx in HEAD_POSE_REFERENCE_LANDMARKS:  # [1, 33, 61, 199, 263, 291]
            landmark = face_landmarks.landmark[idx]
            x = landmark.x * width
            y = landmark.y * height
            image_points_2d.append([x, y])
        
        image_points_2d = np.array(image_points_2d, dtype=np.float64)
        
        # ═══════════════════════════════════════════════════════════════
        # 2. SOLVE PNP
        # ═══════════════════════════════════════════════════════════════
        
        success, rvec, tvec = cv2.solvePnP(
            self.model_points_3d,
            image_points_2d,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None
        
        # ═══════════════════════════════════════════════════════════════
        # 3. EULER-WINKEL VIA RQDecomp3x3 (ROBUST!)
        # ═══════════════════════════════════════════════════════════════
        
        # Konvertiere Rotation Vector zu Matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # RQDecomp3x3 gibt Euler-Winkel in Grad (!)
        # Rückgabe: (angles, Qx, Qy, Qz) wobei angles = [pitch, yaw, roll]
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
        
        pitch_raw = angles[0]  # X-Rotation
        yaw_raw = angles[1]    # Y-Rotation  
        roll_raw = angles[2]   # Z-Rotation
        
        # ═══════════════════════════════════════════════════════════════
        # 4. VORZEICHEN-KORREKTUR (für intuitive Interpretation)
        # ═══════════════════════════════════════════════════════════════
        #
        # cv2.RQDecomp3x3 gibt: [pitch, yaw, roll] in Grad
        #
        # Nach Korrektur (intuitive Richtungen):
        # - Yaw positiv = Kopf dreht nach RECHTS (schaut nach rechts)
        # - Pitch positiv = Kopf kippt nach OBEN (schaut nach oben)
        # - Roll positiv = Kopf neigt nach RECHTS (rechtes Ohr zur Schulter)
        
        yaw = float(yaw_raw)
        pitch = float(-pitch_raw)   # Invertiere Pitch für intuitive Richtung
        roll = float(-roll_raw)     # Invertiere Roll für intuitive Richtung
        
        # ═══════════════════════════════════════════════════════════════
        # 5. REPROJECTION ERROR → CONFIDENCE
        # ═══════════════════════════════════════════════════════════════
        
        projected_points, _ = cv2.projectPoints(
            self.model_points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        projected_points = projected_points.reshape(-1, 2)
        
        reprojection_error = np.mean(np.linalg.norm(image_points_2d - projected_points, axis=1))
        
        # Confidence: <5px Error = 1.0, >30px = 0.0
        confidence = np.clip(1.0 - (reprojection_error / 30.0), 0.0, 1.0)
        
        # ═══════════════════════════════════════════════════════════════
        # 6. TEMPORAL SMOOTHING (optional)
        # ═══════════════════════════════════════════════════════════════
        
        if self.yaw_buffer is not None:
            self.yaw_buffer.append(yaw)
            self.pitch_buffer.append(pitch)
            self.roll_buffer.append(roll)
            
            if len(self.yaw_buffer) >= 3:
                yaw_smooth = float(np.median(self.yaw_buffer))
                pitch_smooth = float(np.median(self.pitch_buffer))
                roll_smooth = float(np.median(self.roll_buffer))
            else:
                yaw_smooth, pitch_smooth, roll_smooth = yaw, pitch, roll
        else:
            yaw_smooth, pitch_smooth, roll_smooth = yaw, pitch, roll
        
        return {
            'yaw': yaw_smooth,
            'pitch': pitch_smooth,
            'roll': roll_smooth,
            'yaw_raw': yaw,
            'pitch_raw': pitch,
            'roll_raw': roll,
            'confidence': float(confidence),
            'reprojection_error_px': float(reprojection_error),
            'rvec': rvec,
            'tvec': tvec
        }
    
    def reset(self):
        """Reset Temporal Buffers"""
        if self.yaw_buffer is not None:
            self.yaw_buffer.clear()
            self.pitch_buffer.clear()
            self.roll_buffer.clear()

# ==================== ROBUSTER PUPILLEN-DETEKTOR (ERWEITERT) ====================

class RobustPupilDetector:
    """
    Robuste Pupillen-Detektion v1.2 (MIT HEAD-POSE).
    """
    
    def __init__(self):
        """Initialisiert MediaPipe Face Mesh + Head-Pose-Estimator + Blink Detector"""
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.left_iris_indices = [468, 469, 470, 471, 472]
        self.right_iris_indices = [473, 474, 475, 476, 477]
        
        # Head-Pose-Estimator (falls aktiviert)
        self.head_pose_estimator = None
        self.head_pose_enabled = ENABLE_HEAD_POSE_EXTRACTION
        
        # Blink Detector (aus shared_blink_detection.py)
        self.blink_detector = create_blink_detector()
        
        print(f"shared_blink_detection initialisiert")
        print(f"  • Qualitätsgewichtung: {'✓' if USE_QUALITY_WEIGHTED_AVERAGE else '✗'}")
        print(f"  • Confidence-Schwelle: {MIN_CONFIDENCE}")
        print(f"  • Head-Pose-Extraktion: {'✓' if self.head_pose_enabled else '✗'}")
        print(f"  • Blink Detection: {'✓' if self.blink_detector else '✗'}")
    
    def _initialize_head_pose_estimator(self, frame_width: int, frame_height: int):
        """Lazy Initialization (erst bei erstem Frame)"""
        if self.head_pose_enabled and self.head_pose_estimator is None:
            self.head_pose_estimator = HeadPoseEstimator(
                frame_width, frame_height, focal_length=CAMERA_FOCAL_LENGTH
            )
    
    # ═════════════════════════════════════════════════════════════════════
    # MODUS 1: FRAME-BY-FRAME (für debug_1) - ERWEITERT MIT HEAD-POSE
    # ═════════════════════════════════════════════════════════════════════
    
    def extract_from_frame(self, frame: np.ndarray) -> Dict:
        """
        Extrahiert Pupillenposition + Head-Pose aus EINEM Frame.
        
        Returns:
            dict mit (NEU: head_pose_*):
            - 'position': [x, y] in Pixel (oder None)
            - 'confidence': 0.0-1.0
            - 'left_position', 'right_position': [x, y] (oder None)
            - 'left_quality', 'right_quality': 0.0-1.0
            - 'plausibility_passed': True/False
            - 'eye_distance_px': float (oder None)
            - 'head_yaw': float (Grad, oder None)  # ← NEU!
            - 'head_pitch': float (Grad, oder None)
            - 'head_roll': float (Grad, oder None)
            - 'head_pose_confidence': 0.0-1.0 (oder None)
        """
        
        # ─────────────────────────────────────────────────────────────────
        # EBENE 1: MEDIAPIPE DETEKTION (wie vorher)
        # ─────────────────────────────────────────────────────────────────
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return self._create_empty_result()
        
        face_landmarks = results.multi_face_landmarks[0]
        height, width = frame.shape[:2]
        
        # ─────────────────────────────────────────────────────────────────
        # BLINK DETECTION
        # ─────────────────────────────────────────────────────────────────
        
        blink_data = None
        if self.blink_detector is not None:
            blink_data = self.blink_detector.detect_blink(
                face_landmarks, width, height
            )

        # ─────────────────────────────────────────────────────────────────
        #  HEAD-POSE-EXTRAKTION
        # ─────────────────────────────────────────────────────────────────
        
        head_pose_data = None
        
        if self.head_pose_enabled:
            # Lazy Init
            if self.head_pose_estimator is None:
                self._initialize_head_pose_estimator(width, height)
            
            head_pose_data = self.head_pose_estimator.estimate_pose(
                face_landmarks, (height, width)
            )
        
        # ─────────────────────────────────────────────────────────────────
        # EBENE 2+3: PUPILLEN-DETEKTION (wie vorher, unverändert)
        # ─────────────────────────────────────────────────────────────────
        
        # Linkes Auge
        left_coords = []
        for idx in self.left_iris_indices:
            landmark = face_landmarks.landmark[idx]
            x = landmark.x * width
            y = landmark.y * height
            if 0 <= x < width and 0 <= y < height:
                left_coords.append([x, y])
        
        left_quality = len(left_coords) / 5.0
        left_pupil = np.mean(left_coords, axis=0) if len(left_coords) > 0 else None
        
        # Rechtes Auge
        right_coords = []
        for idx in self.right_iris_indices:
            landmark = face_landmarks.landmark[idx]
            x = landmark.x * width
            y = landmark.y * height
            if 0 <= x < width and 0 <= y < height:
                right_coords.append([x, y])
        
        right_quality = len(right_coords) / 5.0
        right_pupil = np.mean(right_coords, axis=0) if len(right_coords) > 0 else None
        
        # Qualitäts-Check
        if left_quality < MIN_LANDMARK_QUALITY and right_quality < MIN_LANDMARK_QUALITY:
            return self._create_empty_result(head_pose=head_pose_data)
        
        # Qualitätsgewichteter Mittelwert (wie vorher)
        plausibility_passed = True
        eye_distance = None
        
        if left_pupil is not None and right_pupil is not None:
            if USE_QUALITY_WEIGHTED_AVERAGE:
                total_quality = left_quality + right_quality
                if total_quality > 0:
                    avg_pupil = (left_pupil * left_quality + 
                                right_pupil * right_quality) / total_quality
                    confidence = total_quality / 2.0
                else:
                    return self._create_empty_result(head_pose=head_pose_data)
            else:
                avg_pupil = (left_pupil + right_pupil) / 2
                confidence = (left_quality + right_quality) / 2.0
            
            # Augenabstand-Check
            eye_distance = np.linalg.norm(left_pupil - right_pupil)
            if not (EYE_DISTANCE_MIN_PX < eye_distance < EYE_DISTANCE_MAX_PX):
                confidence *= EYE_DISTANCE_PENALTY
                plausibility_passed = False
        
        elif left_pupil is not None:
            avg_pupil = left_pupil
            confidence = left_quality * SINGLE_EYE_PENALTY
        
        elif right_pupil is not None:
            avg_pupil = right_pupil
            confidence = right_quality * SINGLE_EYE_PENALTY
        
        else:
            return self._create_empty_result(head_pose=head_pose_data)
        
        # ─────────────────────────────────────────────────────────────────
        # RÜCKGABE (MIT HEAD-POSE)
        # ─────────────────────────────────────────────────────────────────
        
        result = {
            'position': avg_pupil.tolist(),
            'confidence': float(confidence),
            'left_position': left_pupil.tolist() if left_pupil is not None else None,
            'right_position': right_pupil.tolist() if right_pupil is not None else None,
            'left_quality': float(left_quality),
            'right_quality': float(right_quality),
            'plausibility_passed': plausibility_passed,
            'eye_distance_px': float(eye_distance) if eye_distance is not None else None
        }
        
        # Head-Pose
        if head_pose_data is not None:
            result['head_yaw'] = head_pose_data['yaw']
            result['head_pitch'] = head_pose_data['pitch']
            result['head_roll'] = head_pose_data['roll']
            result['head_pose_confidence'] = head_pose_data['confidence']
            result['head_pose_raw'] = {
                'yaw': head_pose_data['yaw_raw'],
                'pitch': head_pose_data['pitch_raw'],
                'roll': head_pose_data['roll_raw']
            }
        else:
            result['head_yaw'] = None
            result['head_pitch'] = None
            result['head_roll'] = None
            result['head_pose_confidence'] = None
        
        # Blink-Daten
        if blink_data is not None:
            # Original blink fields
            result["is_blink"] = blink_data.get("is_blink", False)
            result["eyes_closed"] = blink_data.get("eyes_closed", False)
            result["left_ear"] = blink_data.get("left_ear")
            result["right_ear"] = blink_data.get("right_ear")
            result["avg_ear"] = blink_data.get("avg_ear")
            result["blink_count"] = blink_data.get("blink_count", 0)
            result["frames_below_threshold"] = blink_data.get("frames_below_threshold", 0)

            # New filter fields
            result["valid_eye_frame"] = blink_data.get("valid_eye_frame", True)
            result["invalid_reason"] = blink_data.get("invalid_reason")
            result["is_suspected_blink"] = blink_data.get("is_suspected_blink", False)
            result["is_transition_frame"] = blink_data.get("is_transition_frame", False)

            # Useful debug fields
            result["left_threshold"] = blink_data.get("left_threshold")
            result["right_threshold"] = blink_data.get("right_threshold")
            result["ear_asymmetry"] = blink_data.get("ear_asymmetry")
            result["baseline_samples"] = blink_data.get("baseline_samples")

        else:
            # Blink detector disabled or not available:
            # do not invalidate the frame just because blink filtering is absent.
            result["is_blink"] = False
            result["eyes_closed"] = False
            result["left_ear"] = None
            result["right_ear"] = None
            result["avg_ear"] = None
            result["blink_count"] = 0
            result["frames_below_threshold"] = 0

            result["valid_eye_frame"] = True
            result["invalid_reason"] = None
            result["is_suspected_blink"] = False
            result["is_transition_frame"] = False

            result["left_threshold"] = None
            result["right_threshold"] = None
            result["ear_asymmetry"] = None
            result["baseline_samples"] = None

        
        return result
    
    # ═════════════════════════════════════════════════════════════════════
    # MODUS 2: MULTI-FRAME-AGGREGATION (für offline_calibration)
    # Ebene 1-6: Vollständige Pipeline
    # ═════════════════════════════════════════════════════════════════════
    
    def extract_from_window(self, video_path: str, timestamp: float,
                        window_start: float = 0.5, window_end: float = 3.0,
                        min_confidence: float = None) -> Dict:
        """
        Extrahiert Pupillenposition aus ZEITFENSTER.
        Nutzt ALLE 6 Ebenen inkl. IQR und Median-Aggregation!

        Args:
            video_path: Pfad zum Video
            timestamp: Marker-Zeitpunkt (Sekunden)
            window_start: Start-Offset (Sekunden NACH Marker)
            window_end: End-Offset (Sekunden NACH Marker)
            min_confidence: Mindest-Confidence (default: MIN_CONFIDENCE)

        Returns:
            dict mit:
            - 'success': True/False
            - 'position': [x, y] aggregiert (Median)
            - 'confidence': Durchschnittliche Confidence
            - 'n_frames_total': Anzahl Frames im Fenster
            - 'n_frames_valid': Anzahl nach Confidence/Plausibility/Blink-Filter
            - 'n_frames_after_iqr': Anzahl nach IQR
            - 'std_x', 'std_y': Streuung (Qualitätsmetrik)

            # new added:
            - 'n_frames_eye_valid': Anzahl Frames mit valid_eye_frame=True
            - 'n_frames_eye_invalid': Anzahl Frames mit valid_eye_frame=False
            - 'invalid_reason_counts': Gründe für ungültige Eye-Frames
        """

        if min_confidence is None:
            min_confidence = MIN_CONFIDENCE

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'success': False, 'error': 'Video nicht öffenbar'}

        fps = cap.get(cv2.CAP_PROP_FPS)

        start_time = timestamp + window_start
        end_time = timestamp + window_end

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # ──────────────────────────────────────────────────────────────
        # EBENE 1-3: Frame-by-Frame mit Qualitäts-Filtering
        # ──────────────────────────────────────────────────────────────

        positions = []
        confidences = []

        # new added: statistics for blink / eye-frame filter
        n_frames_eye_valid = 0
        n_frames_eye_invalid = 0
        invalid_reason_counts = {}

        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            result = self.extract_from_frame(frame)

            # new added: read eye-frame validity from blink filter
            # If the key is missing, default to True for backward compatibility.
            valid_eye_frame = result.get('valid_eye_frame', True)

            # new added: count valid / invalid eye frames
            if valid_eye_frame:
                n_frames_eye_valid += 1
            else:
                n_frames_eye_invalid += 1
                invalid_reason = result.get('invalid_reason', 'unknown')
                invalid_reason_counts[invalid_reason] = (
                    invalid_reason_counts.get(invalid_reason, 0) + 1
                )

            # modified: Ebene 3 now also filters by valid_eye_frame
            if (
                result['position'] is not None and
                result['confidence'] >= min_confidence and
                result['plausibility_passed'] and
                valid_eye_frame
            ):
                positions.append(result['position'])
                confidences.append(result['confidence'])

        cap.release()

        n_frames_total = end_frame - start_frame
        n_frames_valid = len(positions)

        if n_frames_valid < 5:
            return {
                'success': False,
                'error': f'Zu wenige valide Frames: {n_frames_valid}/5',
                'n_frames_total': n_frames_total,
                'n_frames_valid': n_frames_valid,

                # new added
                'n_frames_eye_valid': n_frames_eye_valid,
                'n_frames_eye_invalid': n_frames_eye_invalid,
                'invalid_reason_counts': invalid_reason_counts,
            }

        positions = np.array(positions)
        confidences = np.array(confidences)

        # ──────────────────────────────────────────────────────────────
        # EBENE 4: IQR OUTLIER-REMOVAL (nur bei ≥10 Frames!)
        # ──────────────────────────────────────────────────────────────

        if n_frames_valid >= MIN_SAMPLES_FOR_IQR:
            positions_clean = self._remove_outliers_iqr(positions)
        else:
            positions_clean = positions

        n_frames_after_iqr = len(positions_clean)

        if n_frames_after_iqr < 3:
            return {
                'success': False,
                'error': f'Zu wenige Frames nach IQR: {n_frames_after_iqr}/3',
                'n_frames_total': n_frames_total,
                'n_frames_valid': n_frames_valid,
                'n_frames_after_iqr': n_frames_after_iqr,

                # new added
                'n_frames_eye_valid': n_frames_eye_valid,
                'n_frames_eye_invalid': n_frames_eye_invalid,
                'invalid_reason_counts': invalid_reason_counts,
            }

        # ──────────────────────────────────────────────────────────────
        # EBENE 5: TEMPORAL SMOOTHING
        # ──────────────────────────────────────────────────────────────

        positions_smooth = self._temporal_smoothing(
            positions_clean,
            window=SMOOTHING_WINDOW_CALIBRATION
        )

        # ──────────────────────────────────────────────────────────────
        # EBENE 6: MEDIAN-AGGREGATION
        # ──────────────────────────────────────────────────────────────

        if USE_MEDIAN_INSTEAD_OF_MEAN:
            final_position = np.median(positions_smooth, axis=0)
        else:
            final_position = np.mean(positions_smooth, axis=0)

        # Durchschnittliche Confidence (begrenzt auf Anzahl smooth Frames)
        n_smooth = len(positions_smooth)
        avg_confidence = np.mean(confidences[:n_smooth])

        return {
            'success': True,
            'position': final_position.tolist(),
            'confidence': float(avg_confidence),
            'n_frames_total': n_frames_total,
            'n_frames_valid': n_frames_valid,
            'n_frames_after_iqr': n_frames_after_iqr,

            # new added
            'n_frames_eye_valid': n_frames_eye_valid,
            'n_frames_eye_invalid': n_frames_eye_invalid,
            'invalid_reason_counts': invalid_reason_counts,

            'std_x': float(np.std(positions_smooth[:, 0])),
            'std_y': float(np.std(positions_smooth[:, 1]))
        }
    
    # ═════════════════════════════════════════════════════════════════════
    # HELPER-FUNKTIONEN
    # ═════════════════════════════════════════════════════════════════════
    
    def _create_empty_result(self, head_pose=None) -> Dict:
        """Leeres Ergebnis bei fehlgeschlagener Detektion"""
        result = {
            'position': None,
            'confidence': 0.0,
            'left_position': None,
            'right_position': None,
            'left_quality': 0.0,
            'right_quality': 0.0,
            'plausibility_passed': False,
            'eye_distance_px': None
        }
        
        # Head-Pose (falls vorhanden)
        if head_pose is not None:
            result['head_yaw'] = head_pose['yaw']
            result['head_pitch'] = head_pose['pitch']
            result['head_roll'] = head_pose['roll']
            result['head_pose_confidence'] = head_pose['confidence']
        else:
            result['head_yaw'] = None
            result['head_pitch'] = None
            result['head_roll'] = None
            result['head_pose_confidence'] = None
        
        # Blink-Daten
        result['is_blink'] = False
        result['eyes_closed'] = False
        result['left_ear'] = None
        result['right_ear'] = None
        result['avg_ear'] = None
        result['blink_count'] = 0

        ## new added
        result["frames_below_threshold"] = 0
        result["valid_eye_frame"] = False
        result["invalid_reason"] = "no_valid_pupil_or_face"
        result["is_suspected_blink"] = False
        result["is_transition_frame"] = False
        result["left_threshold"] = None
        result["right_threshold"] = None
        result["ear_asymmetry"] = None
        result["baseline_samples"] = None

        return result
    
    def _remove_outliers_iqr(self, positions: np.ndarray) -> np.ndarray:
        """Ebene 4: IQR-basierte Outlier-Removal"""
        
        if len(positions) < 4:
            return positions
        
        # X-Koordinate
        Q1_x = np.percentile(positions[:, 0], 25)
        Q3_x = np.percentile(positions[:, 0], 75)
        IQR_x = Q3_x - Q1_x
        
        # Y-Koordinate
        Q1_y = np.percentile(positions[:, 1], 25)
        Q3_y = np.percentile(positions[:, 1], 75)
        IQR_y = Q3_y - Q1_y
        
        # Masken
        mask_x = ((positions[:, 0] >= Q1_x - IQR_MULTIPLIER*IQR_x) & 
                  (positions[:, 0] <= Q3_x + IQR_MULTIPLIER*IQR_x))
        mask_y = ((positions[:, 1] >= Q1_y - IQR_MULTIPLIER*IQR_y) & 
                  (positions[:, 1] <= Q3_y + IQR_MULTIPLIER*IQR_y))
        
        return positions[mask_x & mask_y]
    
    def _temporal_smoothing(self, positions: np.ndarray, window: int = 7) -> np.ndarray:
        """Ebene 5: Temporale Glättung"""
        
        if len(positions) < window:
            return positions
        
        df = pd.DataFrame(positions, columns=['x', 'y'])
        df_smooth = df.rolling(window=window, center=True, min_periods=1).mean()
        
        return df_smooth.values
    
    # new added
    def reset(self):
        """Reset stateful detectors for a new video or a new independent window."""
        if self.blink_detector is not None:
            self.blink_detector.reset()

        if self.head_pose_estimator is not None:
            self.head_pose_estimator.reset()    
    
    def close(self):
        """Schließt MediaPipe Face Mesh"""
        self.face_mesh.close()
        if self.head_pose_estimator is not None:
            self.head_pose_estimator.reset()


# ==================== CONVENIENCE-FUNKTIONEN ====================

def create_detector() -> RobustPupilDetector:
    """Erstellt neuen Detektor (für schnelles Testen)"""
    return RobustPupilDetector()


if __name__ == "__main__":
    # Test-Beispiel
    print("\n" + "="*70)
    print("SHARED_PUPIL_DETECTION - TEST")
    print("="*70 + "\n")
    
    detector = create_detector()
    
    # Beispiel: Frame-by-Frame
    print(" Detektor bereit für:")
    print("  • extract_from_frame() → debug_1")
    print("  • extract_from_window() → offline_calibration")
    
    detector.close()

# ==================== DEBUG-EXTRAKTOR ====================

class DetailedPupilDebugger:
    """
    Extrahiert ALLE Verarbeitungsschritte Frame-by-Frame für Debugging.
    
    Output-Format (CSV pro Kalibrierpunkt):
    - frame_number, timestamp_video, timestamp_relative
    - EBENE 1 (Rohdaten):
      - left_iris_0_x/y bis left_iris_4_x/y (5 Landmarks)
      - right_iris_0_x/y bis right_iris_4_x/y
      - left_landmark_quality, right_landmark_quality
    - EBENE 2 (Plausibility):
      - eye_distance_px
      - plausibility_passed (True/False)
    - EBENE 3 (Aggregation):
      - left_pupil_x/y, right_pupil_x/y
      - avg_pupil_x/y (qualitätsgewichtet)
      - confidence_frame
    - EBENE 4-6 (markiert nach Aggregation):
      - kept_after_confidence (True/False)
      - kept_after_iqr (True/False)
      - smoothed_x/y
      - is_final_training_value (True/False)
    """
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.detector = RobustPupilDetector()
        self.cap = None
        
    def extract_detailed_debug_data(self, 
                                   timestamp: float,
                                   window_start: float = 0.5,
                                   window_end: float = 3.0,
                                   min_confidence: float = MIN_CONFIDENCE,
                                   calibration_point_id: int = 0,
                                   screen_target: tuple = None) -> pd.DataFrame:
        """
        Extrahiert ALLE Verarbeitungsschritte für EIN Zeitfenster.
        
        Returns:
            DataFrame mit ~70 Spalten (alle Verarbeitungsebenen)
        """
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            return None
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        start_time = timestamp + window_start
        end_time = timestamp + window_end
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # ════════════════════════════════════════════════════════════
        # EBENE 1-3: Frame-by-Frame Daten sammeln
        # ════════════════════════════════════════════════════════════
        
        frame_data = []
        
        for frame_num in range(start_frame, end_frame):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            timestamp_video = frame_num / fps
            timestamp_relative = timestamp_video - timestamp
            
            # MediaPipe Detektion (Rohdaten)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.face_mesh.process(frame_rgb)
            
            # Initialisiere Frame-Daten-Dict
            row = {
                'calibration_point_id': calibration_point_id,
                'screen_target_x': screen_target[0] if screen_target else None,
                'screen_target_y': screen_target[1] if screen_target else None,
                'frame_number': frame_num,
                'timestamp_video': timestamp_video,
                'timestamp_relative': timestamp_relative,
                'audio_marker_time': timestamp,
                'face_detected': False
            }
            
            if not results.multi_face_landmarks:
                # Kein Gesicht → alle Werte None
                row.update(self._create_empty_frame_row())
                frame_data.append(row)
                continue
            
            row['face_detected'] = True
            
            face_landmarks = results.multi_face_landmarks[0]
            height, width = frame.shape[:2]
            
            # ──────────────────────────────────────────────────────────
            # EBENE 1: ROHE LANDMARK-DATEN
            # ──────────────────────────────────────────────────────────
            
            # Linkes Auge (5 Landmarks)
            left_coords = []
            for i, idx in enumerate(self.detector.left_iris_indices):
                landmark = face_landmarks.landmark[idx]
                x = landmark.x * width
                y = landmark.y * height
                row[f'left_iris_{i}_x'] = x
                row[f'left_iris_{i}_y'] = y
                
                if 0 <= x < width and 0 <= y < height:
                    left_coords.append([x, y])
            
            row['left_landmark_quality'] = len(left_coords) / 5.0
            
            # Rechtes Auge (5 Landmarks)
            right_coords = []
            for i, idx in enumerate(self.detector.right_iris_indices):
                landmark = face_landmarks.landmark[idx]
                x = landmark.x * width
                y = landmark.y * height
                row[f'right_iris_{i}_x'] = x
                row[f'right_iris_{i}_y'] = y
                
                if 0 <= x < width and 0 <= y < height:
                    right_coords.append([x, y])
            
            row['right_landmark_quality'] = len(right_coords) / 5.0
            
            # ──────────────────────────────────────────────────────────
            # EBENE 2: EINZELAUGEN-POSITIONEN
            # ──────────────────────────────────────────────────────────
            
            if len(left_coords) > 0:
                left_pupil = np.mean(left_coords, axis=0)
                row['left_pupil_x'] = left_pupil[0]
                row['left_pupil_y'] = left_pupil[1]
            else:
                row['left_pupil_x'] = None
                row['left_pupil_y'] = None
            
            if len(right_coords) > 0:
                right_pupil = np.mean(right_coords, axis=0)
                row['right_pupil_x'] = right_pupil[0]
                row['right_pupil_y'] = right_pupil[1]
            else:
                row['right_pupil_x'] = None
                row['right_pupil_y'] = None
            
            # ──────────────────────────────────────────────────────────
            # EBENE 2: PLAUSIBILITY CHECK (Augenabstand)
            # ──────────────────────────────────────────────────────────
            
            if row['left_pupil_x'] is not None and row['right_pupil_x'] is not None:
                left_p = np.array([row['left_pupil_x'], row['left_pupil_y']])
                right_p = np.array([row['right_pupil_x'], row['right_pupil_y']])
                eye_distance = np.linalg.norm(left_p - right_p)
                
                row['eye_distance_px'] = eye_distance
                row['plausibility_passed'] = (EYE_DISTANCE_MIN_PX < eye_distance < EYE_DISTANCE_MAX_PX)
            else:
                row['eye_distance_px'] = None
                row['plausibility_passed'] = False
            
            # ──────────────────────────────────────────────────────────
            # EBENE 3: QUALITÄTSGEWICHTETER DURCHSCHNITT
            # ──────────────────────────────────────────────────────────
            
            if row['left_pupil_x'] is not None and row['right_pupil_x'] is not None:
                if USE_QUALITY_WEIGHTED_AVERAGE:
                    left_q = row['left_landmark_quality']
                    right_q = row['right_landmark_quality']
                    total_q = left_q + right_q
                    
                    if total_q > 0:
                        avg_x = (left_p[0] * left_q + right_p[0] * right_q) / total_q
                        avg_y = (left_p[1] * left_q + right_p[1] * right_q) / total_q
                        confidence = total_q / 2.0
                    else:
                        avg_x = avg_y = confidence = None
                else:
                    avg_x = (left_p[0] + right_p[0]) / 2
                    avg_y = (left_p[1] + right_p[1]) / 2
                    confidence = (row['left_landmark_quality'] + row['right_landmark_quality']) / 2.0
                
                row['avg_pupil_x'] = avg_x
                row['avg_pupil_y'] = avg_y
                row['confidence_frame'] = confidence
                
                # Augenabstand-Penalty
                if not row['plausibility_passed']:
                    row['confidence_frame'] *= EYE_DISTANCE_PENALTY
                
            elif row['left_pupil_x'] is not None:
                row['avg_pupil_x'] = row['left_pupil_x']
                row['avg_pupil_y'] = row['left_pupil_y']
                row['confidence_frame'] = row['left_landmark_quality'] * SINGLE_EYE_PENALTY
                
            elif row['right_pupil_x'] is not None:
                row['avg_pupil_x'] = row['right_pupil_x']
                row['avg_pupil_y'] = row['right_pupil_y']
                row['confidence_frame'] = row['right_landmark_quality'] * SINGLE_EYE_PENALTY
            else:
                row['avg_pupil_x'] = None
                row['avg_pupil_y'] = None
                row['confidence_frame'] = 0.0
            
            # ──────────────────────────────────────────────────────────
            # EBENE 3: CONFIDENCE-FILTER (markieren)
            # ──────────────────────────────────────────────────────────
            
            row['kept_after_confidence'] = (
                row['confidence_frame'] is not None and 
                row['confidence_frame'] >= min_confidence and
                row['plausibility_passed']
            )
            
            frame_data.append(row)
        
        self.cap.release()
        
        df = pd.DataFrame(frame_data)
        
        # ════════════════════════════════════════════════════════════
        # EBENE 4: IQR OUTLIER-REMOVAL (markieren)
        # ════════════════════════════════════════════════════════════
        
        df['kept_after_iqr'] = False
        
        valid_rows = df[df['kept_after_confidence'] == True]
        
        if len(valid_rows) >= MIN_SAMPLES_FOR_IQR:
            positions = valid_rows[['avg_pupil_x', 'avg_pupil_y']].values
            
            # IQR-Filter (wie in shared_pupil_detection)
            Q1_x = np.percentile(positions[:, 0], 25)
            Q3_x = np.percentile(positions[:, 0], 75)
            IQR_x = Q3_x - Q1_x
            
            Q1_y = np.percentile(positions[:, 1], 25)
            Q3_y = np.percentile(positions[:, 1], 75)
            IQR_y = Q3_y - Q1_y
            
            mask_x = ((positions[:, 0] >= Q1_x - IQR_MULTIPLIER*IQR_x) & 
                      (positions[:, 0] <= Q3_x + IQR_MULTIPLIER*IQR_x))
            mask_y = ((positions[:, 1] >= Q1_y - IQR_MULTIPLIER*IQR_y) & 
                      (positions[:, 1] <= Q3_y + IQR_MULTIPLIER*IQR_y))
            
            iqr_mask = mask_x & mask_y
            
            # Markiere in Original-DF
            valid_indices = valid_rows.index[iqr_mask]
            df.loc[valid_indices, 'kept_after_iqr'] = True
        else:
            # Zu wenig Frames → alle behalten
            df.loc[df['kept_after_confidence'] == True, 'kept_after_iqr'] = True
        
        # ════════════════════════════════════════════════════════════
        # EBENE 5: TEMPORAL SMOOTHING
        # ════════════════════════════════════════════════════════════
        
        df['smoothed_x'] = None
        df['smoothed_y'] = None
        
        iqr_rows = df[df['kept_after_iqr'] == True].copy()
        
        if len(iqr_rows) >= SMOOTHING_WINDOW_CALIBRATION:
            smoothed = iqr_rows[['avg_pupil_x', 'avg_pupil_y']].rolling(
                window=SMOOTHING_WINDOW_CALIBRATION, 
                center=True, 
                min_periods=1
            ).mean()
            
            df.loc[iqr_rows.index, 'smoothed_x'] = smoothed['avg_pupil_x'].values
            df.loc[iqr_rows.index, 'smoothed_y'] = smoothed['avg_pupil_y'].values
        else:
            # Kein Smoothing möglich
            df.loc[iqr_rows.index, 'smoothed_x'] = iqr_rows['avg_pupil_x'].values
            df.loc[iqr_rows.index, 'smoothed_y'] = iqr_rows['avg_pupil_y'].values
        
        # ════════════════════════════════════════════════════════════
        # EBENE 6: FINAL VALUE (Median/Mean über alle smoothed)
        # ════════════════════════════════════════════════════════════
        
        final_rows = df[df['smoothed_x'].notna()]
        
        if len(final_rows) > 0:
            if USE_MEDIAN_INSTEAD_OF_MEAN:
                final_x = final_rows['smoothed_x'].median()
                final_y = final_rows['smoothed_y'].median()
            else:
                final_x = final_rows['smoothed_x'].mean()
                final_y = final_rows['smoothed_y'].mean()
            
            df['final_training_x'] = final_x
            df['final_training_y'] = final_y
            df['is_training_value'] = False
            
            # Markiere Frames die zum finalen Wert beigetragen haben
            df.loc[final_rows.index, 'is_training_value'] = True
        else:
            df['final_training_x'] = None
            df['final_training_y'] = None
            df['is_training_value'] = False
        
        return df
    
    def _create_empty_frame_row(self):
        """Leere Werte für fehlgeschlagene Detektion"""
        row = {}
        
        # Landmarks
        for i in range(5):
            row[f'left_iris_{i}_x'] = None
            row[f'left_iris_{i}_y'] = None
            row[f'right_iris_{i}_x'] = None
            row[f'right_iris_{i}_y'] = None
        
        # Quality
        row['left_landmark_quality'] = 0.0
        row['right_landmark_quality'] = 0.0
        
        # Pupil positions
        row['left_pupil_x'] = None
        row['left_pupil_y'] = None
        row['right_pupil_x'] = None
        row['right_pupil_y'] = None
        
        # Plausibility
        row['eye_distance_px'] = None
        row['plausibility_passed'] = False
        
        # Average
        row['avg_pupil_x'] = None
        row['avg_pupil_y'] = None
        row['confidence_frame'] = 0.0
        
        # Filters
        row['kept_after_confidence'] = False
        row['kept_after_iqr'] = False
        
        # Smoothing
        row['smoothed_x'] = None
        row['smoothed_y'] = None
        
        # Final
        row['is_training_value'] = False
        
        return row
    
    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.detector.close()
