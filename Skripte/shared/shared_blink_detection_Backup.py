"""
=================================================================================
SHARED_BLINK_DETECTION.PY - Einheitliche Blink-Detektion v1.0
=================================================================================
Zentrale Blink-Detection fuer BEIDE Pipelines (MediaPipe + ptgaze).

Basiert auf Eye Aspect Ratio (EAR) nach Soukupova & Cech (2016).
MediaPipe ist MASTER fuer Blink-Detection (konsistente Landmark-Indizes).

Verwendet von:
- shared_pupil_detection.py (MediaPipe Pipeline)
- shared_gaze_detection_ptgaze.py (ptgaze Pipeline)
- debug_1_video_analysis.py
- debug_1_ptgaze.py

Version: v1.0
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
import numpy as np
from typing import Dict, Optional

# ==================== CONFIG-IMPORT ====================

try:
    from config import (
        EAR_BLINK_THRESHOLD,
        EAR_CONSEC_FRAMES,
        ENABLE_BLINK_DETECTION
    )
except ImportError:
    # Fallback-Werte (P9: ENABLE_BLINK_DETECTION standardmaessig aktiviert)
    print("[shared_blink_detection] config.py nicht gefunden, nutze Default-Werte")
    EAR_BLINK_THRESHOLD = 0.15
    EAR_CONSEC_FRAMES = 2
    ENABLE_BLINK_DETECTION = True  # P9: Standardmaessig aktiviert


# ==================== BLINK DETECTOR ====================

class BlinkDetector:
    """
    EAR-basierte Blink-Detektion fuer BEIDE Pipelines (MediaPipe + ptgaze).
    
    Referenz: Soukupova & Cech (2016) - Real-Time Eye Blink Detection
    Formel: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    
    Landmark-Indizes: MediaPipe Face Mesh (478 Landmarks)
    - Verifiziert aus GitHub-Projekt (blink_counter_and_EAR_plot.py)
    
    Funktioniert mit:
    - MediaPipe FaceLandmarks (multi_face_landmarks[0])
    - NumPy Arrays (478, 2) oder (478, 3)
    - ptgaze Face.landmarks
    
    WICHTIG: MediaPipe ist MASTER fuer Blink-Detection!
    Beide Pipelines nutzen die gleichen Landmark-Indizes.
    """
    
    # ==================== LANDMARK-INDIZES ====================
    # MediaPipe Face Mesh Eye Landmarks
    # Reihenfolge: [p1, p2, p3, p4, p5, p6] fuer EAR-Formel
    # p1=aussen, p2=oben-aussen, p3=oben-innen, p4=innen, p5=unten-innen, p6=unten-aussen
    
    RIGHT_EYE_EAR = [33, 159, 158, 133, 153, 145]   # Rechtes Auge (Kamera-Perspektive)
    LEFT_EYE_EAR = [362, 380, 374, 263, 386, 385]   # Linkes Auge (Kamera-Perspektive)
    
    def __init__(self, threshold: float = None, consec_frames: int = None):
        """
        Args:
            threshold: EAR-Schwelle (< threshold = Augen geschlossen)
                       Default: EAR_BLINK_THRESHOLD aus config.py (0.294)
            consec_frames: Mindestanzahl Frames unter Schwelle fuer Blink
                          Default: EAR_CONSEC_FRAMES aus config.py (3)
        """
        self.threshold = threshold if threshold is not None else EAR_BLINK_THRESHOLD
        self.consec_frames = consec_frames if consec_frames is not None else EAR_CONSEC_FRAMES
        
        # State fuer consecutive frames
        self.frame_counter = 0
        self.blink_counter = 0
        
        # Flag ob aktiviert
        self.enabled = ENABLE_BLINK_DETECTION
        
        print(f"   [OK] BlinkDetector v1.0 initialisiert:")
        print(f"        Threshold: {self.threshold}")
        print(f"        Consecutive Frames: {self.consec_frames}")
        print(f"        Enabled: {self.enabled}")
    
    # ==================== EAR-BERECHNUNG ====================
    
    def _calculate_ear_from_points(self, eye_points: np.ndarray) -> float:
        """
        Berechnet Eye Aspect Ratio aus 6 Punkten.
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Args:
            eye_points: (6, 2) Array mit [p1, p2, p3, p4, p5, p6]
        
        Returns:
            EAR (0.0 = geschlossen, ~0.25-0.35 = offen)
        """
        if eye_points is None or len(eye_points) != 6:
            return 0.0
        
        try:
            # Vertikale Abstaende (Augenhoehe)
            A = np.linalg.norm(eye_points[1] - eye_points[5])  # p2-p6
            B = np.linalg.norm(eye_points[2] - eye_points[4])  # p3-p5
            
            # Horizontaler Abstand (Augenbreite)
            C = np.linalg.norm(eye_points[0] - eye_points[3])  # p1-p4
            
            if C == 0:
                return 0.0
            
            return float((A + B) / (2.0 * C))
            
        except (IndexError, TypeError):
            return 0.0
    
    def calculate_ear_from_landmarks(self, face_landmarks, frame_width: int,
                                     frame_height: int, eye_indices: list) -> float:
        """
        Berechnet EAR aus MediaPipe FaceLandmarks.
        
        Args:
            face_landmarks: MediaPipe multi_face_landmarks[0]
            frame_width, frame_height: Video-Aufloesung
            eye_indices: [p1, p2, p3, p4, p5, p6]
        
        Returns:
            EAR (0.0-0.5)
        """
        try:
            pts = []
            for idx in eye_indices:
                lm = face_landmarks.landmark[idx]
                pts.append([lm.x * frame_width, lm.y * frame_height])
            pts = np.array(pts)
            
            return self._calculate_ear_from_points(pts)
            
        except (IndexError, AttributeError, TypeError):
            return 0.0
    
    def calculate_ear_from_array(self, landmarks_array: np.ndarray,
                                 eye_indices: list) -> float:
        """
        Berechnet EAR aus NumPy-Array (fuer ptgaze Face.landmarks).
        
        Args:
            landmarks_array: (478, 2) oder (478, 3) Array mit Pixel-Koordinaten
            eye_indices: [p1, p2, p3, p4, p5, p6]
        
        Returns:
            EAR (0.0-0.5)
        """
        try:
            # Extrahiere nur x,y (ignoriere z falls vorhanden)
            pts = landmarks_array[eye_indices, :2]
            return self._calculate_ear_from_points(pts)
            
        except (IndexError, TypeError):
            return 0.0
    
    # ==================== BLINK-DETECTION METHODEN ====================
    
    def detect_blink(self, face_landmarks, frame_width: int,
                    frame_height: int) -> Dict:
        """
        Hauptmethode: Detektiert Blink aus MediaPipe FaceLandmarks.
        
        Args:
            face_landmarks: MediaPipe multi_face_landmarks[0]
            frame_width, frame_height: Video-Aufloesung
        
        Returns:
            dict mit:
            - 'is_blink': True wenn Blink ENDET (fuer Zaehlung)
            - 'eyes_closed': True wenn Augen AKTUELL geschlossen
            - 'left_ear', 'right_ear', 'avg_ear': EAR-Werte
            - 'blink_count': Gesamtzahl Blinks
            - 'frames_below_threshold': Aktuelle Dauer unter Threshold
        """
        
        if not self.enabled:
            return self._create_empty_result()
        
        # EAR fuer beide Augen
        left_ear = self.calculate_ear_from_landmarks(
            face_landmarks, frame_width, frame_height, self.LEFT_EYE_EAR
        )
        right_ear = self.calculate_ear_from_landmarks(
            face_landmarks, frame_width, frame_height, self.RIGHT_EYE_EAR
        )
        
        return self._process_ear_values(left_ear, right_ear)
    
    def detect_blink_from_array(self, landmarks_array: np.ndarray) -> Dict:
        """
        Detektiert Blink aus NumPy-Array (fuer ptgaze).
        
        Args:
            landmarks_array: (478, 2) Array mit Pixel-Koordinaten
                            (bereits skaliert auf Bildgroesse!)
        
        Returns:
            dict (wie detect_blink)
        """
        
        if not self.enabled:
            return self._create_empty_result()
        
        left_ear = self.calculate_ear_from_array(landmarks_array, self.LEFT_EYE_EAR)
        right_ear = self.calculate_ear_from_array(landmarks_array, self.RIGHT_EYE_EAR)
        
        return self._process_ear_values(left_ear, right_ear)
    
    def detect_blink_from_face(self, face, frame_height: int, frame_width: int) -> Dict:
        """
        Detektiert Blink aus ptgaze Face-Objekt.
        
        Args:
            face: ptgaze Face-Objekt (hat face.landmarks als NumPy Array)
            frame_height, frame_width: Video-Aufloesung
        
        Returns:
            dict (wie detect_blink)
        """
        
        if not self.enabled or face is None:
            return self._create_empty_result()
        
        # ptgaze Face hat .landmarks (NumPy Array mit shape (478, 3))
        if face.landmarks is None or len(face.landmarks) == 0:
            return self._create_empty_result()
        
        # Konvertiere zu Pixel-Koordinaten
        # face.landmarks ist (478, 3) mit [x, y, z] normalisiert (0-1)
        pts = face.landmarks[:, :2].copy()
        pts[:, 0] *= frame_width
        pts[:, 1] *= frame_height
        
        return self.detect_blink_from_array(pts)
    
    # ==================== HELPER METHODEN ====================
    
    def _process_ear_values(self, left_ear: float, right_ear: float) -> Dict:
        """
        Verarbeitet EAR-Werte und aktualisiert Blink-State.
        
        Args:
            left_ear, right_ear: EAR-Werte fuer beide Augen
        
        Returns:
            dict mit Blink-Informationen
        """
        
        if left_ear == 0.0 and right_ear == 0.0:
            return self._create_empty_result()
        
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Augen aktuell geschlossen?
        eyes_closed = (avg_ear < self.threshold)
        
        # Consecutive Frames Logic
        is_blink = False
        
        if eyes_closed:
            self.frame_counter += 1
        else:
            # Augen wieder offen -> War es ein Blink?
            if self.frame_counter >= self.consec_frames:
                self.blink_counter += 1
                is_blink = True  # Blink-Ende markieren
            self.frame_counter = 0
        
        return {
            'is_blink': is_blink,
            'eyes_closed': eyes_closed,
            'left_ear': float(left_ear),
            'right_ear': float(right_ear),
            'avg_ear': float(avg_ear),
            'blink_count': self.blink_counter,
            'frames_below_threshold': self.frame_counter
        }
    
    def _create_empty_result(self) -> Dict:
        """Leeres Ergebnis bei deaktivierter/fehlgeschlagener Detektion"""
        return {
            'is_blink': False,
            'eyes_closed': False,
            'left_ear': None,
            'right_ear': None,
            'avg_ear': None,
            'blink_count': self.blink_counter,
            'frames_below_threshold': 0
        }
    
    def reset(self):
        """Reset fuer neues Video"""
        self.frame_counter = 0
        self.blink_counter = 0
    
    def close(self):
        """Cleanup (fuer Konsistenz mit anderen Detektoren)"""
        pass


# ==================== FACTORY FUNCTION ====================

def create_blink_detector(threshold: float = None, consec_frames: int = None) -> Optional[BlinkDetector]:
    """
    Erstellt BlinkDetector falls aktiviert.
    
    Args:
        threshold: EAR-Schwelle (optional, Default aus config)
        consec_frames: Consecutive Frames (optional, Default aus config)
    
    Returns:
        BlinkDetector oder None (wenn ENABLE_BLINK_DETECTION=False)
    """
    if not ENABLE_BLINK_DETECTION:
        print("   [INFO] Blink Detection deaktiviert (ENABLE_BLINK_DETECTION=False)")
        return None
    
    return BlinkDetector(threshold=threshold, consec_frames=consec_frames)


# ==================== TEST ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SHARED_BLINK_DETECTION - TEST")
    print("="*70 + "\n")
    
    detector = create_blink_detector()
    
    if detector:
        print("\n[OK] BlinkDetector bereit fuer:")
        print("     - detect_blink() -> MediaPipe FaceLandmarks")
        print("     - detect_blink_from_array() -> NumPy Array")
        print("     - detect_blink_from_face() -> ptgaze Face")
        
        detector.close()
        print("\n[OK] Test erfolgreich!")
    else:
        print("\n[INFO] Blink Detection ist deaktiviert")
