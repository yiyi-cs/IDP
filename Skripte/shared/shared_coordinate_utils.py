"""
=================================================================================
SHARED_COORDINATE_UTILS.PY - Einheitliche Koordinaten-Transformation v1.0
=================================================================================
Zentrale Koordinaten-Utilities fuer BEIDE Pipelines (MediaPipe + ptgaze).

Enthaelt:
- CoordinateTransformer: Pixel <-> Gradsehwinkel Konvertierung
- ScreenParameters: Pickle-Kompatibilitaet (Dummy-Klasse)
- CalibrationMetadata: Pickle-Kompatibilitaet (Dummy-Klasse)
- Outside-Monitor Detection
- EyeLink-zu-Grad Konvertierung (fuer debug_6/7)

Verwendet von:
- debug_5_partial_calibration.py (MediaPipe)
- debug_5_ptgaze.py (ptgaze)
- debug_6_extended_comparison.py
- debug_7_interactive_timeline.py

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
from typing import Tuple, Dict, Optional
from dataclasses import dataclass, field
from typing import List

# ==================== CONFIG-IMPORT ====================

try:
    from config import (
        SCREEN_WIDTH_PX,
        SCREEN_HEIGHT_PX,
        SCREEN_WIDTH_CM,
        SCREEN_HEIGHT_CM,
        VIEWING_DISTANCE_CM,
        MAX_PLAUSIBLE_DEG_X,
        MAX_PLAUSIBLE_DEG_Y
    )
except ImportError:
    # Fallback-Werte
    print("[shared_coordinate_utils] config.py nicht gefunden, nutze Default-Werte")
    SCREEN_WIDTH_PX = 1920            
    SCREEN_HEIGHT_PX = 1080            
    SCREEN_WIDTH_CM = 53.2             
    SCREEN_HEIGHT_CM = 29.9            
    VIEWING_DISTANCE_CM = 70.0
    MAX_PLAUSIBLE_DEG_X = 30.0
    MAX_PLAUSIBLE_DEG_Y = 25.0


# ==================== PICKLE-KOMPATIBILITAET ====================

@dataclass
class CalibrationMetadata:
    """
    Dummy-Klasse fuer Pickle-Kompatibilitaet mit offline_calibration.py v3.2.
    
    **WICHTIG:** Diese Klasse wird NICHT aktiv genutzt!
    Sie existiert nur damit pickle.load() nicht fehlschlaegt, wenn die
    PKL-Datei von offline_calibration v3.2 erstellt wurde.
    
    Version: v1.0 (ausgelagert aus debug_5)
    """
    timepoints: List[str] = field(default_factory=list)
    n_points_total: int = 0
    source: str = 'phases_detected.json'


class ScreenParameters:
    """
    Dummy-Klasse fuer Pickle-Kompatibilitaet mit offline_calibration.py.
    
    **WICHTIG:** Diese Klasse wird NICHT aktiv genutzt!
    Sie existiert nur damit pickle.load() nicht fehlschlaegt, wenn die
    PKL-Datei von einer aelteren offline_calibration-Version erstellt wurde,
    die ScreenParameters serialisiert hat.
    
    Die tatsaechlichen Screen-Parameter kommen aus config.py!
    
    Version: v1.0 (ausgelagert aus debug_5)
    """
    def __init__(self, width_px=None, height_px=None, width_cm=None, 
                 height_cm=None, viewing_distance_cm=None, **kwargs):
        # Fallback zu config.py Werten
        self.width_px = width_px if width_px is not None else SCREEN_WIDTH_PX
        self.height_px = height_px if height_px is not None else SCREEN_HEIGHT_PX
        self.width_cm = width_cm if width_cm is not None else SCREEN_WIDTH_CM
        self.height_cm = height_cm if height_cm is not None else SCREEN_HEIGHT_CM
        self.viewing_distance_cm = viewing_distance_cm if viewing_distance_cm is not None else VIEWING_DISTANCE_CM
        self.center_x = self.width_px // 2
        self.center_y = self.height_px // 2
        
        # Zusaetzliche Attribute (dynamisch, falls in PKL vorhanden)
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self):
        return (f"ScreenParameters(width={self.width_px}x{self.height_px}px, "
                f"viewing_distance={self.viewing_distance_cm}cm)")


# ==================== KOORDINATEN-TRANSFORMER ====================

class CoordinateTransformer:
    """
    Konvertiert Koordinaten zwischen verschiedenen Systemen.
    
    Unterstuetzte Konvertierungen:
    - Pixel -> Gradsehwinkel
    - Gradsehwinkel -> Pixel
    - Pixel -> Relative (zur Bildschirmmitte)
    - Pixel -> Normalisiert [-1, +1]
    
    Geometrie:
    - Ursprung: Bildschirmmitte (center_x, center_y)
    - X positiv: rechts
    - Y positiv: unten (Pixel) / oben (Grad, je nach Konvention)
    
    Version: v1.0 (vereinheitlicht aus debug_5)
    """
    
    def __init__(self, screen_params: Dict = None):
        """
        Args:
            screen_params: Dictionary mit Screen-Parametern
                          Falls None, werden config.py Werte verwendet
        """
        if screen_params is None:
            screen_params = {
                'width_px': SCREEN_WIDTH_PX,
                'height_px': SCREEN_HEIGHT_PX,
                'center_x': SCREEN_WIDTH_PX // 2,
                'center_y': SCREEN_HEIGHT_PX // 2,
                'width_cm': SCREEN_WIDTH_CM,
                'height_cm': SCREEN_HEIGHT_CM,
                'viewing_distance_cm': VIEWING_DISTANCE_CM
            }
        
        self.screen = screen_params
        
        # Berechne abgeleitete Werte
        if 'center_x' not in self.screen:
            self.screen['center_x'] = self.screen['width_px'] // 2
        if 'center_y' not in self.screen:
            self.screen['center_y'] = self.screen['height_px'] // 2
    
    # ==================== PIXEL <-> GRAD ====================
    
    def pixels_to_degrees(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """
        Konvertiert Bildschirm-Pixel zu Gradsehwinkeln.
        
        Args:
            x_px, y_px: Position in Pixel (0,0 = oben-links)
        
        Returns:
            (deg_x, deg_y): Gradsehwinkel relativ zur Bildschirmmitte
                           Positiv X = rechts, Positiv Y = unten
        """
        # Relativ zur Bildschirmmitte (normalisiert auf Bildschirmgroesse)
        x_rel = (x_px - self.screen['center_x']) / self.screen['width_px']
        y_rel = (y_px - self.screen['center_y']) / self.screen['height_px']
        
        # In cm umrechnen
        x_cm = x_rel * self.screen['width_cm']
        y_cm = y_rel * self.screen['height_cm']
        
        # Gradsehwinkel (trigonometrisch)
        deg_x = np.arctan2(x_cm, self.screen['viewing_distance_cm']) * 180 / np.pi
        deg_y = np.arctan2(y_cm, self.screen['viewing_distance_cm']) * 180 / np.pi
        
        return deg_x, deg_y
    
    def degrees_to_pixels(self, deg_x: float, deg_y: float) -> Tuple[float, float]:
        """
        Konvertiert Gradsehwinkel zu Bildschirm-Pixel.
        
        Args:
            deg_x, deg_y: Gradsehwinkel relativ zur Bildschirmmitte
        
        Returns:
            (x_px, y_px): Position in Pixel (0,0 = oben-links)
        """
        # Grad zu cm
        x_cm = np.tan(deg_x * np.pi / 180) * self.screen['viewing_distance_cm']
        y_cm = np.tan(deg_y * np.pi / 180) * self.screen['viewing_distance_cm']
        
        # cm zu relative Position
        x_rel = x_cm / self.screen['width_cm']
        y_rel = y_cm / self.screen['height_cm']
        
        # Relative zu Pixel
        x_px = x_rel * self.screen['width_px'] + self.screen['center_x']
        y_px = y_rel * self.screen['height_px'] + self.screen['center_y']
        
        return x_px, y_px
    
    # ==================== RELATIVE KOORDINATEN ====================
    
    def pixels_to_relative(self, x_px: float, y_px: float, 
                          reference: str = 'center') -> Tuple[float, float]:
        """
        Konvertiert Pixel zu Relativmassen.
        
        Args:
            x_px, y_px: Position in Pixel
            reference: 'center' (relativ zur Mitte) oder 'screen' (absolut)
        
        Returns:
            (rel_x, rel_y): Relative Position in Pixel
        """
        if reference == 'center':
            rel_x = x_px - self.screen['center_x']
            rel_y = y_px - self.screen['center_y']
        else:
            rel_x = x_px
            rel_y = y_px
        
        return rel_x, rel_y
    
    def pixels_to_normalized(self, x_px: float, y_px: float) -> Tuple[float, float]:
        """
        Konvertiert Pixel zu normalisierten Werten [-1, +1].
        
        Args:
            x_px, y_px: Position in Pixel
        
        Returns:
            (norm_x, norm_y): Normalisierte Position
                             -1 = linker/oberer Rand
                             +1 = rechter/unterer Rand
                              0 = Bildschirmmitte
        """
        norm_x = (x_px - self.screen['center_x']) / (self.screen['width_px'] / 2)
        norm_y = (y_px - self.screen['center_y']) / (self.screen['height_px'] / 2)
        
        return norm_x, norm_y
    
    # ==================== VALIDIERUNG ====================
    
    def is_outside_monitor(self, x_px: float = None, y_px: float = None,
                          deg_x: float = None, deg_y: float = None) -> bool:
        """
        Prueft ob Position ausserhalb des Monitors liegt.
        
        Args:
            x_px, y_px: Position in Pixel (optional)
            deg_x, deg_y: Position in Grad (optional)
        
        Returns:
            True wenn ausserhalb Monitor
        """
        # Berechne maximale Sehwinkel
        max_deg_x = np.arctan2(self.screen['width_cm'] / 2, 
                              self.screen['viewing_distance_cm']) * 180 / np.pi
        max_deg_y = np.arctan2(self.screen['height_cm'] / 2, 
                              self.screen['viewing_distance_cm']) * 180 / np.pi
        
        # Falls Pixel gegeben, konvertiere zu Grad
        if deg_x is None and x_px is not None:
            deg_x, deg_y = self.pixels_to_degrees(x_px, y_px)
        
        if deg_x is None:
            return False
        
        return abs(deg_x) > max_deg_x or abs(deg_y) > max_deg_y
    
    def is_plausible(self, deg_x: float, deg_y: float,
                    max_x: float = None, max_y: float = None) -> bool:
        """
        Prueft ob Gradsehwinkel plausibel sind.
        
        Args:
            deg_x, deg_y: Position in Grad
            max_x, max_y: Maximale plausible Werte (Default aus config)
        
        Returns:
            True wenn plausibel
        """
        if max_x is None:
            max_x = MAX_PLAUSIBLE_DEG_X
        if max_y is None:
            max_y = MAX_PLAUSIBLE_DEG_Y
        
        return abs(deg_x) <= max_x and abs(deg_y) <= max_y
    
    def get_monitor_bounds_deg(self) -> Tuple[float, float]:
        """
        Gibt die Monitor-Grenzen in Grad zurueck.
        
        Returns:
            (max_deg_x, max_deg_y): Maximale Sehwinkel
        """
        max_deg_x = np.arctan2(self.screen['width_cm'] / 2, 
                              self.screen['viewing_distance_cm']) * 180 / np.pi
        max_deg_y = np.arctan2(self.screen['height_cm'] / 2, 
                              self.screen['viewing_distance_cm']) * 180 / np.pi
        return max_deg_x, max_deg_y


# ==================== EYELINK KONVERTIERUNG ====================

def convert_eyelink_to_degrees(gaze_x_px: float, gaze_y_px: float,
                               transformer: CoordinateTransformer = None) -> Tuple[float, float]:
    """
    Konvertiert EyeLink Gaze-Position (Pixel) zu Gradsehwinkeln.
    
    Fuer debug_6 und debug_7 (3-Wege-Vergleich).
    
    Args:
        gaze_x_px, gaze_y_px: EyeLink Gaze in Pixel (Screen-Koordinaten)
        transformer: CoordinateTransformer (optional, erstellt Default)
    
    Returns:
        (deg_x, deg_y): Gradsehwinkel
    """
    if transformer is None:
        transformer = CoordinateTransformer()
    
    return transformer.pixels_to_degrees(gaze_x_px, gaze_y_px)


def batch_convert_to_degrees(x_array, y_array,
                            transformer: CoordinateTransformer = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch-Konvertierung von Pixel zu Grad (fuer DataFrames).
    
    Args:
        x_array, y_array: Arrays/Series mit Pixel-Koordinaten
        transformer: CoordinateTransformer (optional)
    
    Returns:
        (deg_x_array, deg_y_array): Arrays mit Gradsehwinkeln
    """
    if transformer is None:
        transformer = CoordinateTransformer()
    
    x_array = np.asarray(x_array)
    y_array = np.asarray(y_array)
    
    deg_x = np.zeros_like(x_array, dtype=float)
    deg_y = np.zeros_like(y_array, dtype=float)
    
    valid_mask = ~(np.isnan(x_array) | np.isnan(y_array))
    
    for i in np.where(valid_mask)[0]:
        deg_x[i], deg_y[i] = transformer.pixels_to_degrees(x_array[i], y_array[i])
    
    deg_x[~valid_mask] = np.nan
    deg_y[~valid_mask] = np.nan
    
    return deg_x, deg_y


# ==================== FACTORY FUNCTION ====================

def create_coordinate_transformer(screen_params: Dict = None) -> CoordinateTransformer:
    """
    Factory-Funktion fuer CoordinateTransformer.
    
    Args:
        screen_params: Dictionary mit Screen-Parametern (optional)
    
    Returns:
        CoordinateTransformer Instanz
    """
    return CoordinateTransformer(screen_params)


# ==================== TEST ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SHARED_COORDINATE_UTILS - TEST")
    print("="*70 + "\n")
    
    transformer = create_coordinate_transformer()
    
    # Test Pixel -> Grad
    test_x_px = SCREEN_WIDTH_PX // 2 + 500  # 500px rechts von Mitte
    test_y_px = SCREEN_HEIGHT_PX // 2
    
    deg_x, deg_y = transformer.pixels_to_degrees(test_x_px, test_y_px)
    print(f"[TEST] Pixel ({test_x_px}, {test_y_px}) -> Grad ({deg_x:.2f}, {deg_y:.2f})")
    
    # Test Grad -> Pixel (Rueckkonvertierung)
    px_x, px_y = transformer.degrees_to_pixels(deg_x, deg_y)
    print(f"[TEST] Grad ({deg_x:.2f}, {deg_y:.2f}) -> Pixel ({px_x:.1f}, {px_y:.1f})")
    
    # Test Outside Monitor
    outside = transformer.is_outside_monitor(deg_x=25, deg_y=0)
    print(f"[TEST] 25 deg X outside monitor: {outside}")
    
    # Test Monitor Bounds
    max_x, max_y = transformer.get_monitor_bounds_deg()
    print(f"[TEST] Monitor Bounds: +/-{max_x:.1f} deg X, +/-{max_y:.1f} deg Y")
    
    print("\n[OK] Test erfolgreich!")
