"""
=================================================================================
SHARED_GAZE_DETECTION_PTGAZE.PY - ptgaze Integration v1.1
=================================================================================
Wrapper fuer ptgaze (ETH-XGaze Model) Gaze-Detektion.

Enthaelt:
- PtgazeGazeDetector: Hauptklasse fuer Gaze-Detektion
- BadSampleDetector: Erkennung ungueltiger Samples
- Face-Detection + Gaze-Estimation

Verwendet von:
- debug_1_ptgaze.py
- offline_calibration_ptgaze.py

Version: v1.1 - Einheitliche Blink-Detection + Package-Struktur
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
import tempfile
import pathlib
import yaml
from typing import Dict, Optional, Tuple
from collections import deque

# ptgaze imports
try:
    from omegaconf import DictConfig, OmegaConf
    from ptgaze.gaze_estimator import GazeEstimator
    from ptgaze.common import Face
    import torch
except ImportError as e:
    print(f"✗ ptgaze nicht korrekt installiert: {e}")
    print("  Installiere mit: pip install ptgaze")
    raise

# BLINK DETECTION IMPORT (NEU v1.1)
from shared.shared_blink_detection import BlinkDetector, create_blink_detector, ENABLE_BLINK_DETECTION

# Config-Import
try:
    from config import *
except ImportError:
    # Fallback
    print("[shared_gaze_detection_ptgaze] config.py nicht gefunden, nutze Default-Werte")
    MIN_GAZE_CONFIDENCE = 0.7
    CAMERA_FOCAL_LENGTH = 1440  # Typisch fuer 1440x1080
    SCREEN_WIDTH_CM = 53.3
    SCREEN_HEIGHT_CM = 30.0
    VIEWING_DISTANCE_CM = 70.0
    # ENABLE_BLINK_DETECTION kommt aus shared_blink_detection.py

# ==================== PTGAZE CONFIG GENERATOR ====================

class PtgazeConfigGenerator:
    """
    Erstellt ptgaze-kompatible Config (OmegaConf).
    
    Nutzt:
    - Auto Camera Params (aus Video-Auflösung)
    - ETH-XGaze Model (Pre-trained)
    - MediaPipe Face Detection
    """
    
    @staticmethod
    def create_config(video_width: int, video_height: int,
                    device: str = 'cuda', 
                    model: str = 'eth-xgaze') -> DictConfig:
        """
        Erstellt Config für ptgaze.
        
        Args:
            video_width, video_height: Video-Auflösung
            device: 'cuda' oder 'cpu'
            model: 'eth-xgaze', 'mpiigaze', oder 'mpiifacegaze'
        
        Returns:
            OmegaConf DictConfig
        """
        
        # ═══════════════════════════════════════════════════════════════
        # MODEL ARCHITECTURE MAPPING
        # ═══════════════════════════════════════════════════════════════
        
        if model == 'eth-xgaze':
            model_arch = 'resnet18'
            mode = 'ETH-XGaze'
        elif model == 'mpiigaze':
            model_arch = 'resnet_preact'
            mode = 'MPIIGaze'
        elif model == 'mpiifacegaze':
            model_arch = 'resnet_simple'
            mode = 'MPIIFaceGaze'
        else:
            raise ValueError(f"Unknown model: {model}")
        
        # Basis-Config (minimalistisch)
        config_dict = {
            'mode': mode,
            'device': device if torch.cuda.is_available() else 'cpu',
            
            # ✅ MODEL KEY (WICHTIG!)
            'model': {
                'name': model_arch
            },
            
            # Face Detector
            'face_detector': {
                'mode': 'mediapipe',  # ✅ Konsistent mit MediaPipe-Pipeline!
                'mediapipe_max_num_faces': 1,
                'mediapipe_static_image_mode': False
            },
            
            # Gaze Estimator
            'gaze_estimator': {
                'checkpoint': str(PtgazeConfigGenerator._get_model_path(model)),
                'camera_params': None,  # ← Wird automatisch generiert
                'normalized_camera_params': str(PtgazeConfigGenerator._get_normalized_camera_path(model)),
                'normalized_camera_distance': 0.8,  # Standard (60cm)
                'image_size': [224, 224]  # ETH-XGaze Input-Size
            },
            
            # Demo-Settings (für Visualizer)
            'demo': {
                'use_camera': False,
                'video_path': None,  # Wird später gesetzt
                'image_path': None,
                'display_on_screen': False,
                'output_dir': None,
                'output_file_extension': 'mp4',
                'head_pose_axis_length': 0.05,
                'gaze_visualization_length': 0.05,
                'show_bbox': True,
                'show_head_pose': True,
                'show_landmarks': False,
                'show_normalized_image': False,
                'show_template_model': False,
                'wait_time': 1
            }
        }
        
        config = OmegaConf.create(config_dict)
        
        # ═══════════════════════════════════════════════════════════════
        # AUTO CAMERA PARAMS (wie utils.py, Zeile 69-105)
        # ═══════════════════════════════════════════════════════════════
        
        camera_params_yaml = PtgazeConfigGenerator._generate_camera_params(
            video_width, video_height
        )
        config.gaze_estimator.camera_params = camera_params_yaml
        
        return config
    
    @staticmethod
    def _generate_camera_params(width: int, height: int) -> str:
        """
        Generiert dummy Camera Params YAML (wie ptgaze utils.py).
        
        Returns:
            Pfad zu temporärem YAML-File
        """
        
        camera_dict = {
            'image_width': width,
            'image_height': height,
            'camera_matrix': {
                'rows': 3,
                'cols': 3,
                'data': [
                    float(width), 0.0, float(width / 2),
                    0.0, float(width), float(height / 2),
                    0.0, 0.0, 1.0
                ]
            },
            'distortion_coefficients': {
                'rows': 1,
                'cols': 5,
                'data': [0.0, 0.0, 0.0, 0.0, 0.0]  # Keine Distortion
            }
        }
        
        # Speichere als temporäres YAML
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='_camera.yaml', delete=False
        )
        yaml.safe_dump(camera_dict, temp_file)
        temp_file.close()
        
        return temp_file.name
    
    @staticmethod
    def _get_model_path(model: str) -> pathlib.Path:
        """
        Pfad zu Pre-trained Model Weights.
        
        Lädt automatisch herunter falls nicht vorhanden (via ptgaze utils).
        """
        # Importiere ptgaze Download-Funktionen
        from ptgaze.utils import (
            download_ethxgaze_model,
            download_mpiigaze_model,
            download_mpiifacegaze_model
        )
        
        # Nutze offizielle Download-Funktionen
        # → Lädt automatisch herunter nach ~/.ptgaze/models/
        if model == 'eth-xgaze':
            return download_ethxgaze_model()
        elif model == 'mpiigaze':
            return download_mpiigaze_model()
        elif model == 'mpiifacegaze':
            return download_mpiifacegaze_model()
        else:
            raise ValueError(f"Unknown model: {model}")
        
    @staticmethod
    def _get_normalized_camera_path(model: str = 'eth-xgaze') -> pathlib.Path:
        """
        Pfad zu Normalized Camera Params (MODEL-SPEZIFISCH!).
        
        Sucht in ptgaze Package die RICHTIGEN Params für das gewählte Modell.
        
        Args:
            model: 'eth-xgaze', 'mpiigaze', oder 'mpiifacegaze'
        
        Returns:
            Path zu normalized_camera_params YAML
        """
        try:
            import ptgaze
            package_root = pathlib.Path(ptgaze.__file__).parent
            
            # ✅ KORRIGIERT: Model-spezifischer Pfad!
            if model == 'eth-xgaze':
                normalized_path = package_root / 'data' / 'normalized_camera_params' / 'eth-xgaze.yaml'
            elif model == 'mpiigaze':
                normalized_path = package_root / 'data' / 'normalized_camera_params' / 'mpiigaze.yaml'
            elif model == 'mpiifacegaze':
                normalized_path = package_root / 'data' / 'normalized_camera_params' / 'mpiifacegaze.yaml'
            else:
                raise ValueError(f"Unknown model: {model}")
            
            if normalized_path.exists():
                print(f"   ✓ Normalized Camera Params: {normalized_path.name}")
                return normalized_path
            else:
                print(f" Normalized Camera Params nicht gefunden: {normalized_path}")
                print(f"  >> Erstelle Fallback...")
        except Exception as e:
            print(f" Fehler beim Laden: {e}")
        
        # Fallback nur wenn File wirklich fehlt
        return PtgazeConfigGenerator._create_default_normalized_camera()
        
    @staticmethod
    def _create_default_normalized_camera() -> pathlib.Path:
        """Erstellt Default Normalized Camera (60cm Distanz, 1024×1024)"""
        
        normalized_dict = {
            'image_width': 1024,
            'image_height': 1024,
            'camera_matrix': {
                'rows': 3,
                'cols': 3,
                'data': [
                    1024.0, 0.0, 512.0,
                    0.0, 1024.0, 512.0,
                    0.0, 0.0, 1.0
                ]
            },
            'distortion_coefficients': {
                'rows': 1,
                'cols': 5,
                'data': [0.0, 0.0, 0.0, 0.0, 0.0]
            }
        }
        
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='_normalized_camera.yaml', delete=False
        )
        yaml.safe_dump(normalized_dict, temp_file)
        temp_file.close()
        
        return pathlib.Path(temp_file.name)

# ==================== PTGAZE GAZE DETECTOR ====================

class PtgazeGazeDetector:
    """
    Wrapper um ptgaze GazeEstimator.
    
    Analog zu RobustPupilDetector, aber Output:
    - Gaze Angles (pitch, yaw) in GRAD
    - Confidence (basierend auf Face Detection Quality)
    
    Version: v1.0
    """
    
    def __init__(self, video_width: int = 1440, video_height: int = 1080,
                 device: str = 'cuda', model: str = 'eth-xgaze'):
        """
        Args:
            video_width, video_height: Video-Auflösung (für Camera Params)
            device: 'cuda' oder 'cpu'
            model: 'eth-xgaze' (empfohlen), 'mpiigaze', 'faze' (später)
        """
        
        self.video_width = video_width
        self.video_height = video_height
        self.model = model
        
        print(f"\n{'='*70}")
        print("PTGAZE GAZE DETECTOR")
        print(f"{'='*70}")
        print(f" Model: {model}")
        print(f" Device: {device}")
        print(f" Video: {video_width}×{video_height}")
        
        # ═══════════════════════════════════════════════════════════════
        # CONFIG ERSTELLEN
        # ═══════════════════════════════════════════════════════════════
        
        config_generator = PtgazeConfigGenerator()
        self.config = config_generator.create_config(
            video_width, video_height, device, model
        )
        
        print(f" Config erstellt:")
        print(f"   • Face Detector: {self.config.face_detector.mode}")
        print(f"   • Camera Params: Auto-generiert")
        
        # ═══════════════════════════════════════════════════════════════
        # GAZE ESTIMATOR INITIALISIEREN
        # ═══════════════════════════════════════════════════════════════
        
        try:
            self.gaze_estimator = GazeEstimator(self.config)
            print(f" GazeEstimator initialisiert")
        except Exception as e:
            print(f"✗ GazeEstimator konnte nicht initialisiert werden: {e}")
            raise
        
        # ═══════════════════════════════════════════════════════════════
        # BLINK DETECTOR (aus shared_blink_detection.py)
        # ═══════════════════════════════════════════════════════════════
        
        self.blink_detector = create_blink_detector()
        if self.blink_detector is None:
            print("   [INFO] Blink Detection: Deaktiviert")
        
        print(f"{'='*70}\n")
    
    # ═══════════════════════════════════════════════════════════════════
    # MODUS 1: FRAME-BY-FRAME (für debug_1_ptgaze)
    # ═══════════════════════════════════════════════════════════════════
    
    def extract_from_frame(self, frame: np.ndarray) -> Dict:
        """
        Extrahiert Gaze Angles + Blink Status aus EINEM Frame.
        
        Returns:
            dict mit:
            - 'gaze_pitch_deg': float (vertikal, negativ=oben)
            - 'gaze_yaw_deg': float (horizontal, negativ=links)
            - 'gaze_vector': [x, y, z] (3D Unit Vector)
            - 'confidence': 0.0-1.0 (Face Detection Quality)
            - 'detected': True/False
            - 'face': Face-Objekt (für Visualizer)
            - 'is_blink': True/False
            - 'ear': float (Eye Aspect Ratio, oder None)
            - 'head_yaw', 'head_pitch', 'head_roll': float (Grad)
        """
        
        # ──────────────────────────────────────────────────────────────
        # 1. UNDISTORTION (wie ptgaze demo.py, Zeile 74)
        # ──────────────────────────────────────────────────────────────
        
        undistorted = cv2.undistort(
            frame,
            self.gaze_estimator.camera.camera_matrix,
            self.gaze_estimator.camera.dist_coefficients
        )
        
        # ──────────────────────────────────────────────────────────────
        # 2. FACE DETECTION (via ptgaze)
        # ──────────────────────────────────────────────────────────────
        
        faces = self.gaze_estimator.detect_faces(undistorted)
        
        if len(faces) == 0:
            return self._create_empty_result()
        
        face = faces[0]  # Erste Face (max_num_faces=1)
        
        # ──────────────────────────────────────────────────────────────
        # 3. GAZE ESTIMATION (via ptgaze)
        # ──────────────────────────────────────────────────────────────
        
        try:
            self.gaze_estimator.estimate_gaze(undistorted, face)
        except Exception as e:
            print(f" Gaze Estimation failed: {e}")
            return self._create_empty_result()
        
        # ──────────────────────────────────────────────────────────────
        # 4. EXTRAHIERE GAZE ANGLES (RADIANS → GRAD!)
        # ──────────────────────────────────────────────────────────────
        
        # Aus demo.py, Zeile 119:
        # pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
        
        pitch_rad, yaw_rad = face.normalized_gaze_angles  # RADIANS!
        pitch_deg = float(np.rad2deg(pitch_rad))
        yaw_deg = float(np.rad2deg(yaw_rad))
        
        gaze_vector = face.gaze_vector  # 3D Unit Vector
        
        # ──────────────────────────────────────────────────────────────
        # 5. HEAD-POSE (aus face.head_pose_rot)
        # ──────────────────────────────────────────────────────────────
        
        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        head_pitch, head_yaw, head_roll = face.change_coordinate_system(euler_angles)
        
        # ──────────────────────────────────────────────────────────────
        # 6. CONFIDENCE (Face Detection Quality)
        # ──────────────────────────────────────────────────────────────
        
        # ptgaze gibt keine explizite Confidence
        # → Schätze basierend auf Face-Distance und Landmark-Quality
        
        confidence = self._estimate_confidence(face)
        
        # ──────────────────────────────────────────────────────────────
        # 7. BLINK DETECTION 
        # ──────────────────────────────────────────────────────────────
        
        if self.blink_detector is not None:
            blink_result = self.blink_detector.detect_blink_from_face(
                face, 
                frame_height=self.video_height, 
                frame_width=self.video_width
            )
        else:
            blink_result = {
                'is_blink': False, 
                'eyes_closed': False,
                'left_ear': None, 
                'right_ear': None, 
                'avg_ear': None,
                'blink_count': 0
            }

        # ──────────────────────────────────────────────────────────────
        # RÜCKGABE
        # ──────────────────────────────────────────────────────────────
        
        return {
            # Gaze
            'gaze_pitch_deg': pitch_deg,
            'gaze_yaw_deg': yaw_deg,
            'gaze_vector': gaze_vector.tolist(),
            
            # Quality
            'confidence': confidence,
            'detected': True,
            
            # Face (für Visualizer)
            'face': face,
            
            # Blink
            'is_blink': blink_result['is_blink'],
            'eyes_closed': blink_result.get('eyes_closed', False),
            'left_ear': blink_result['left_ear'],
            'right_ear': blink_result['right_ear'],
            'avg_ear': blink_result['avg_ear'],
            'blink_count': blink_result.get('blink_count', 0),
            
            # Head-Pose
            'head_yaw': float(head_yaw),
            'head_pitch': float(head_pitch),
            'head_roll': float(head_roll)
        }
    
    def _estimate_confidence(self, face: Face) -> float:
        """
        Schätzt Confidence (ptgaze gibt keine explizite Confidence).
        
        Basierend auf:
        - Face Distance (näher = besser)
        - Landmark Count (mehr = besser)
        """
        
        # 1. Distance-basiert (optimal: 60cm)
        optimal_distance = 0.6  # 60cm (in Meter)
        distance_factor = np.exp(-((face.distance - optimal_distance) ** 2) / 0.1)
        
        # 2. Landmark-basiert (MediaPipe gibt 478 Landmarks)
        if face.landmarks is not None:
            landmark_factor = min(len(face.landmarks) / 478.0, 1.0)
        else:
            landmark_factor = 0.5
        
        # Kombiniere
        confidence = (distance_factor + landmark_factor) / 2.0
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _create_empty_result(self) -> Dict:
        """Leeres Ergebnis bei fehlgeschlagener Detektion"""
        return {
            'gaze_pitch_deg': np.nan,
            'gaze_yaw_deg': np.nan,
            'gaze_vector': [np.nan, np.nan, np.nan],
            'confidence': 0.0,
            'detected': False,
            'face': None,
            'is_blink': False,
            'left_ear': None,
            'right_ear': None,
            'avg_ear': None,
            'head_yaw': np.nan,
            'head_pitch': np.nan,
            'head_roll': np.nan
        }
    
    def close(self):
        """Schließt Detektoren"""
        if self.blink_detector is not None:
            self.blink_detector.close()


# ==================== CONVENIENCE-FUNKTIONEN ====================

def create_detector(video_width: int = 1440, video_height: int = 1080,
                   device: str = 'cuda', model: str = 'eth-xgaze') -> PtgazeGazeDetector:
    """Erstellt neuen ptgaze Detector (für schnelles Testen)"""
    return PtgazeGazeDetector(video_width, video_height, device, model)


if __name__ == "__main__":
    # Test-Beispiel
    print("\n" + "="*70)
    print("SHARED_GAZE_DETECTION_PTGAZE - TEST")
    print("="*70 + "\n")
    
    try:
        detector = create_detector(video_width=1440, video_height=1080, device='cuda')
        print(" Detektor bereit für:")
        print("  • extract_from_frame() → debug_1_ptgaze")
        
        detector.close()
        print("\n✓ Test erfolgreich!")
        
    except Exception as e:
        print(f"\n✗ Test fehlgeschlagen: {e}")
