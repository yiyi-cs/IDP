"""
=================================================================================
CALIBRATION MODE MANAGER v1.2 - Kalibrierungs-Varianten fuer Master-Pipeline
=================================================================================

Funktionen:
-----------
1. Definiert verschiedene Kalibrierungs-Strategien (Modi)
2. Filtert Kalibrierungspunkte nach Strategie
3. Kompatibel mit offline_calibration.py + offline_calibration_ptgaze.py
4. Erlaubt Experimente mit verschiedenen Modi

Verfuegbare Modi (v1.2):
------------------------
VOLLSTAENDIG:
  - FullCalib: Alle Punkte (beg + mid + end) + Fixationen

OHNE MID:
  - BegEnd: Nur beg + end (ohne Fixationen)
  - BegEndFix: beg + end + Fixationen

NUR ANFANG:
  - OnlyBeg: Alle 20 Punkte von beg
  - OnlyBegFix: Alle 20 Punkte von beg + Fixationen
  - BegFirst10: Nur erste 10 Punkte (1-10)
  - BegSecond10: Nur zweite 10 Punkte (11-20)

KOMBINATIONEN:
  - BegFirst10End: Erste 10 von beg + end
  - BegFirst10EndFix: Erste 10 von beg + end + Fixationen (NEU)
  - BegSecond10End: Zweite 10 von beg + end
  - BegSecond10Mid: Zweite 10 von beg (11-20) + Mid (NEU)

NUR ENDE:
  - OnlyEnd: Nur end-Kalibrierung

SPEZIAL:
  - First10: Erste 10 Punkte gesamt
  - OnlyCenter: Nur Mittenpunkte
  - Periphery: Nur periphere Punkte

NEU in v1.2:
------------
- beg_range: Punkt-Range fuer beg-Kalibrierung (1-10, 11-20, 1-20)
- use_fixations: Fixationskreuz-Phasen als zusaetzliche Punkte
- Erweiterte Modi (BegFirst10, BegSecond10, etc.)
- get_config_dict() fuer PKL-Speicherung

Version: 1.2
Datum: 2025-01
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
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Tuple
import json
import shutil

# =================================================================================
# DATENSTRUKTUREN
# =================================================================================

@dataclass
class CalibrationPoint:
    """
    Repräsentiert einen Kalibrierungspunkt aus phases_detected.json.
    """
    point_id: int
    position: tuple  # (x, y) in Pixel
    timepoint: str  # 'beg', 'mid', 'end'
    video_time_s: float
    eyelink_time_ms: float
    audio_found: bool
    confidence: float = 1.0
    
    def is_center(self, screen_width_px: int, screen_height_px: int, tolerance_px: int = 50) -> bool:
        """Prüft ob Punkt im Bildschirm-Zentrum liegt"""
        center_x = screen_width_px // 2
        center_y = screen_height_px // 2
        
        dist = ((self.position[0] - center_x)**2 + (self.position[1] - center_y)**2)**0.5
        return dist <= tolerance_px
    
    def is_peripheral(self, screen_width_px: int, screen_height_px: int, 
                     inner_margin_pct: float = 0.3) -> bool:
        """Prüft ob Punkt peripher liegt (nicht im inneren 30% Bereich)"""
        # Definiere inneren Bereich (z.B. 30% von Mitte aus)
        center_x = screen_width_px // 2
        center_y = screen_height_px // 2
        
        max_dist_x = screen_width_px * inner_margin_pct / 2
        max_dist_y = screen_height_px * inner_margin_pct / 2
        
        dist_x = abs(self.position[0] - center_x)
        dist_y = abs(self.position[1] - center_y)
        
        # Peripheral = außerhalb inneren Bereichs
        return dist_x > max_dist_x or dist_y > max_dist_y

@dataclass
class CalibrationMode:
    """
    Definiert eine Kalibrierungs-Strategie (v1.2: erweitert).
    
    Attributes:
        name: Eindeutiger Name (z.B. 'FullCalib')
        description: Beschreibung fuer User
        timepoints: Welche Zeitpunkte nutzen (['beg', 'mid', 'end'])
        point_filter: Optional - Funktion zum Filtern von Punkten
        min_points: Mindestanzahl Punkte (Sicherheitscheck)
        beg_range: Punkt-Range fuer beg (start, end) - NEU v1.2
        use_fixations: Fixationskreuz-Phasen verwenden? - NEU v1.2
    """
    name: str
    description: str
    timepoints: List[str]  # ['beg', 'mid', 'end']
    point_filter: Optional[Callable] = None
    min_points: int = 8
    beg_range: Tuple[int, int] = (1, 20)  # NEU v1.2: Standard alle 20 Punkte
    use_fixations: bool = True            # NEU v1.2: Standard mit Fixationen
    
    def __str__(self):
        range_info = ""
        if self.beg_range != (1, 20) and 'beg' in self.timepoints:
            range_info = f" [beg: {self.beg_range[0]}-{self.beg_range[1]}]"
        fix_info = " +Fix" if self.use_fixations else ""
        return f"{self.name}: {self.description}{range_info}{fix_info}"
    
    def get_config_dict(self) -> Dict:
        """Gibt Konfiguration als Dict zurueck (fuer PKL-Speicherung)."""
        return {
            'name': self.name,
            'description': self.description,
            'timepoints': self.timepoints,
            'beg_range': self.beg_range,
            'use_fixations': self.use_fixations,
            'min_points': self.min_points
        }

# =================================================================================
# VORDEFINIERTE MODI
# =================================================================================

class CalibrationModes:
    """
    Sammlung vordefinierter Kalibrierungs-Modi.
    
    Wissenschaftliche Begründung:
    ------------------------------
    - FullCalib: Gold-Standard (maximale Daten, temporaler Drift-Check)
    - BegEnd: Test ob Mid redundant (Drift-Analyse)
    - OnlyBeg: Minimal-Kalibrierung (Effizienz-Test)
    - First10: Ressourcen-Optimierung (Zeit vs. Genauigkeit)
    - OnlyCenter: Fovea-Fokus (zentrale Sehaufgaben)
    - Periphery: Extrapolation-Test (periphere Performance)
    """
    
    @staticmethod
    def get_all_modes() -> Dict[str, CalibrationMode]:
        """Gibt Dict mit allen vordefinierten Modi zurueck (v1.2: erweitert)"""
        
        from config import SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX
        
        return {
            # ══════════════════════════════════════════════════════════
            # VOLLSTAENDIG
            # ══════════════════════════════════════════════════════════
            
            'FullCalib': CalibrationMode(
                name='FullCalib',
                description='Vollstaendig (beg + mid + end + Fixationen)',
                timepoints=['beg', 'mid', 'end'],
                point_filter=None,
                min_points=8,
                beg_range=(1, 20),
                use_fixations=True
            ),
            
            # ══════════════════════════════════════════════════════════
            # OHNE MID
            # ══════════════════════════════════════════════════════════
            
            'BegEnd': CalibrationMode(
                name='BegEnd',
                description='Ohne Mitte (nur beg + end, ohne Fixationen)',
                timepoints=['beg', 'end'],
                point_filter=None,
                min_points=6,
                beg_range=(1, 20),
                use_fixations=False
            ),
            
            'BegEndFix': CalibrationMode(
                name='BegEndFix',
                description='beg + end + Fixationen',
                timepoints=['beg', 'end'],
                point_filter=None,
                min_points=6,
                beg_range=(1, 20),
                use_fixations=True
            ),
            
            # ══════════════════════════════════════════════════════════
            # NUR ANFANG (ALLE 20 PUNKTE)
            # ══════════════════════════════════════════════════════════
            
            'OnlyBeg': CalibrationMode(
                name='OnlyBeg',
                description='Nur Beginn (alle 20 Punkte)',
                timepoints=['beg'],
                point_filter=None,
                min_points=4,
                beg_range=(1, 20),
                use_fixations=False
            ),
            
            'OnlyBegFix': CalibrationMode(
                name='OnlyBegFix',
                description='Nur Beginn + Fixationen',
                timepoints=['beg'],
                point_filter=None,
                min_points=4,
                beg_range=(1, 20),
                use_fixations=True
            ),
            
            # ══════════════════════════════════════════════════════════
            # NUR ANFANG (TEIL-BEREICHE)
            # ══════════════════════════════════════════════════════════
            
            'BegFirst10': CalibrationMode(
                name='BegFirst10',
                description='Nur erste 10 Punkte von beg',
                timepoints=['beg'],
                point_filter=lambda points: [p for p in points if p.point_id <= 10],
                min_points=4,
                beg_range=(1, 10),
                use_fixations=False
            ),
            
            'BegSecond10': CalibrationMode(
                name='BegSecond10',
                description='Nur zweite 10 Punkte von beg',
                timepoints=['beg'],
                point_filter=lambda points: [p for p in points if p.point_id > 10],
                min_points=4,
                beg_range=(11, 20),
                use_fixations=False
            ),
            
            # ══════════════════════════════════════════════════════════
            # KOMBINATIONEN MIT TEIL-BEREICHEN
            # ══════════════════════════════════════════════════════════
            
            'BegFirst10End': CalibrationMode(
                name='BegFirst10End',
                description='Erste 10 von beg + end',
                timepoints=['beg', 'end'],
                point_filter=lambda points: (
                    [p for p in points if p.timepoint == 'beg' and p.point_id <= 10] +
                    [p for p in points if p.timepoint == 'end']
                ),
                min_points=6,
                beg_range=(1, 10),
                use_fixations=False
            ),
            
            'BegSecond10End': CalibrationMode(
                name='BegSecond10End',
                description='Zweite 10 von beg + end',
                timepoints=['beg', 'end'],
                point_filter=lambda points: (
                    [p for p in points if p.timepoint == 'beg' and p.point_id > 10] +
                    [p for p in points if p.timepoint == 'end']
                ),
                min_points=6,
                beg_range=(11, 20),
                use_fixations=False
            ),
            
            'BegFirst10EndFix': CalibrationMode(
                name='BegFirst10EndFix',
                description='Erste 10 von beg + end + Fixationen',
                timepoints=['beg', 'end'],
                point_filter=lambda points: (
                    [p for p in points if p.timepoint == 'beg' and p.point_id <= 10] +
                    [p for p in points if p.timepoint == 'end']
                ),
                min_points=6,
                beg_range=(1, 10),
                use_fixations=True
            ),
            
            'BegSecond10Mid': CalibrationMode(
                name='BegSecond10Mid',
                description='Zweite 10 von beg (11-20) + Mid',
                timepoints=['beg', 'mid'],
                point_filter=lambda points: (
                    [p for p in points if p.timepoint == 'beg' and p.point_id > 10] +
                    [p for p in points if p.timepoint == 'mid']
                ),
                min_points=6,
                beg_range=(11, 20),
                use_fixations=False
            ),

            # ══════════════════════════════════════════════════════════
            # NUR ENDE
            # ══════════════════════════════════════════════════════════
            
            'OnlyEnd': CalibrationMode(
                name='OnlyEnd',
                description='Nur Ende-Kalibrierung',
                timepoints=['end'],
                point_filter=None,
                min_points=4,
                beg_range=(1, 20),  # Nicht relevant
                use_fixations=False
            ),
            
            # ══════════════════════════════════════════════════════════
            # LEGACY MODI (fuer Rueckwaertskompatibilitaet)
            # ══════════════════════════════════════════════════════════
            
            'First10': CalibrationMode(
                name='First10',
                description='Erste 10 Punkte gesamt (beg + Anfang end)',
                timepoints=['beg', 'end'],
                point_filter=lambda points: points[:10],
                min_points=8,
                beg_range=(1, 20),
                use_fixations=False
            ),
            
            'OnlyCenter': CalibrationMode(
                name='OnlyCenter',
                description='Nur Mittenpunkte (Fovea-Fokus)',
                timepoints=['beg', 'mid', 'end'],
                point_filter=lambda points: [
                    p for p in points 
                    if p.is_center(SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX)
                ],
                min_points=3,
                beg_range=(1, 20),
                use_fixations=False
            ),
            
            'Periphery': CalibrationMode(
                name='Periphery',
                description='Nur periphere Punkte (Extrapolation)',
                timepoints=['beg', 'mid', 'end'],
                point_filter=lambda points: [
                    p for p in points 
                    if p.is_peripheral(SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX)
                ],
                min_points=6,
                beg_range=(1, 20),
                use_fixations=False
            ),
        }

    @staticmethod
    def get_mode(mode_name: str) -> CalibrationMode:
        """Gibt einen spezifischen Modus zurück"""
        modes = CalibrationModes.get_all_modes()
        
        if mode_name not in modes:
            raise ValueError(f"Unbekannter Modus: {mode_name}. "
                           f"Verfügbar: {', '.join(modes.keys())}")
        
        return modes[mode_name]


# =================================================================================
# CALIBRATION MODE MANAGER
# =================================================================================

class CalibrationModeManager:
    """
    Verwaltet Kalibrierungs-Modi und wendet sie auf phases_detected.json an.
    
    Workflow:
    ---------
    1. Lade phases_detected.json (Original)
    2. Extrahiere Kalibrierpunkte
    3. Wende Modus an (Filtering)
    4. Erstelle modifizierte phases_detected.json
    5. offline_calibration.py nutzt modifizierte Version
    """
    
    def __init__(self, phases_json_path: Path):
        """
        Args:
            phases_json_path: Pfad zu phases_detected.json (Original)
        """
        self.phases_json_path = Path(phases_json_path)
        
        if not self.phases_json_path.exists():
            raise FileNotFoundError(f"phases_detected.json nicht gefunden: {self.phases_json_path}")
        
        # Lade Original
        with open(self.phases_json_path, 'r') as f:
            self.phases_original = json.load(f)
        
        print(f"[OK] Calibration Mode Manager initialisiert")
        print(f"   Quelle: {self.phases_json_path.name}")
    
    # ═════════════════════════════════════════════════════════════════
    # HAUPT-METHODEN
    # ═════════════════════════════════════════════════════════════════
    
    def apply_mode(self, mode: CalibrationMode, output_path: Optional[Path] = None) -> Path:
        """
        Wendet einen Kalibrierungs-Modus an und erstellt modifizierte JSON.
        
        Args:
            mode: Kalibrierungs-Modus
            output_path: Optional - Pfad für modifizierte JSON
                        Falls None: Erstellt im gleichen Ordner mit Suffix
        
        Returns:
            Path zur modifizierten phases_detected.json
        """
        
        print(f"\n{'='*70}")
        print(f"WENDE KALIBRIERUNGS-MODUS AN: {mode.name}")
        print(f"{'='*70}")
        print(f"Beschreibung: {mode.description}")
        print(f"Zeitpunkte: {', '.join(mode.timepoints)}")
        print(f"Min. Punkte: {mode.min_points}")
        
        # ═════════════════════════════════════════════════════════════
        # SCHRITT 1: Extrahiere Original-Punkte
        # ═════════════════════════════════════════════════════════════
        
        original_points = self._extract_calibration_points(self.phases_original)
        
        print(f"\n[INFO] Original-Kalibrierung:")
        for timepoint in ['beg', 'mid', 'end']:
            tp_points = [p for p in original_points if p.timepoint == timepoint]
            print(f"   {timepoint.upper()}: {len(tp_points)} Punkte")
        
        # ═════════════════════════════════════════════════════════════
        # SCHRITT 2: Filtere nach Zeitpunkten
        # ═════════════════════════════════════════════════════════════
        
        filtered_by_timepoint = [
            p for p in original_points 
            if p.timepoint in mode.timepoints
        ]
        
        print(f"\n   Nach Zeitpunkt-Filter: {len(filtered_by_timepoint)} Punkte")
        
        # ═════════════════════════════════════════════════════════════
        # SCHRITT 3: Zusätzlicher Punkt-Filter (optional)
        # ═════════════════════════════════════════════════════════════
        
        if mode.point_filter is not None:
            try:
                filtered_points = mode.point_filter(filtered_by_timepoint)
                print(f"   Nach Punkt-Filter: {len(filtered_points)} Punkte")
            except Exception as e:
                print(f"   ️ Punkt-Filter fehlgeschlagen: {e}")
                print(f"   → Nutze nur Zeitpunkt-Filter")
                filtered_points = filtered_by_timepoint
        else:
            filtered_points = filtered_by_timepoint
        
        # ═════════════════════════════════════════════════════════════
        # SCHRITT 4: Validierung
        # ═════════════════════════════════════════════════════════════
        
        if len(filtered_points) < mode.min_points:
            print(f"\n[X] FEHLER: Zu wenig Punkte nach Filterung!")
            print(f"   Gefunden: {len(filtered_points)}")
            print(f"   Mindestens: {mode.min_points}")
            raise ValueError(f"Modus {mode.name} erfordert mindestens {mode.min_points} Punkte")
        
        print(f"\n[OK] Validierung erfolgreich: {len(filtered_points)}/{len(original_points)} Punkte")
        
        # ═════════════════════════════════════════════════════════════
        # SCHRITT 5: Erstelle modifizierte phases_detected.json
        # ═════════════════════════════════════════════════════════════
        
        modified_phases = self._create_modified_phases(
            self.phases_original, 
            filtered_points,
            mode
        )
        
        # ═════════════════════════════════════════════════════════════
        # SCHRITT 6: Speichern
        # ═════════════════════════════════════════════════════════════
        
        if output_path is None:
            # Auto-Name: phases_detected_BegEnd.json
            output_path = self.phases_json_path.parent / f"phases_detected_{mode.name}.json"
        
        with open(output_path, 'w') as f:
            json.dump(modified_phases, f, indent=2)
        
        print(f"\n[SAVE] Modifizierte JSON erstellt: {output_path.name}")
        
        return output_path
    
    def restore_original(self):
        """
        Stellt Original-phases_detected.json wieder her.
        
        Nützlich nach Experimenten mit verschiedenen Modi.
        """
        
        # Prüfe ob Backup existiert
        backup_path = self.phases_json_path.parent / "phases_detected_ORIGINAL.json"
        
        if backup_path.exists():
            shutil.copy(backup_path, self.phases_json_path)
            print(f"[OK] Original wiederhergestellt von: {backup_path.name}")
        else:
            print(f" Kein Backup gefunden (Original war nie überschrieben)")
    
    # ═════════════════════════════════════════════════════════════════
    # HELPER-METHODEN (PRIVATE)
    # ═════════════════════════════════════════════════════════════════
    
    def _extract_calibration_points(self, phases: Dict) -> List[CalibrationPoint]:
        """Extrahiert alle Kalibrierpunkte aus phases_detected.json"""
        
        points = []
        
        for timepoint in ['beg', 'mid', 'end']:
            calib_key = f'calibration_{timepoint}'
            
            if calib_key not in phases['phases']:
                continue
            
            calib_phase = phases['phases'][calib_key]
            
            if 'points' not in calib_phase:
                continue
            
            for point_dict in calib_phase['points']:
                point = CalibrationPoint(
                    point_id=point_dict['point_id'],
                    position=tuple(point_dict['position']),
                    timepoint=timepoint,
                    video_time_s=point_dict['video_time_s'],
                    eyelink_time_ms=point_dict['eyelink_time_ms'],
                    audio_found=point_dict.get('audio_found_s') is not None,
                    confidence=point_dict.get('confidence', 1.0)
                )
                
                points.append(point)
        
        return points
    
    def _create_modified_phases(self, original_phases: Dict, 
                                filtered_points: List[CalibrationPoint],
                                mode: CalibrationMode) -> Dict:
        """Erstellt modifizierte phases_detected.json mit gefilterten Punkten"""
        
        # Deep Copy Original
        modified = json.loads(json.dumps(original_phases))
        
        # Dokumentiere Modifikation
        modified['calibration_mode_applied'] = {
            'mode_name': mode.name,
            'description': mode.description,
            'timepoints_used': mode.timepoints,
            'total_points': len(filtered_points),
            'filter_applied': mode.point_filter is not None
        }
        
        # Ersetze Kalibrierpunkte pro Zeitpunkt
        for timepoint in ['beg', 'mid', 'end']:
            calib_key = f'calibration_{timepoint}'
            
            if calib_key not in modified['phases']:
                continue
            
            # Filtere Punkte für diesen Zeitpunkt
            tp_points = [p for p in filtered_points if p.timepoint == timepoint]
            
            if len(tp_points) == 0:
                # Zeitpunkt wird nicht genutzt → Entferne komplett
                del modified['phases'][calib_key]
                continue
            
            # Ersetze points-Liste
            modified['phases'][calib_key]['points'] = [
                self._point_to_dict(p) for p in tp_points
            ]
            
            # Update Statistik
            modified['phases'][calib_key]['n_points'] = len(tp_points)
        
        return modified
    
    def _point_to_dict(self, point: CalibrationPoint) -> Dict:
        """Konvertiert CalibrationPoint zurück zu Dict (für JSON)"""
        
        return {
            'point_id': point.point_id,
            'position': list(point.position),
            'video_time_s': point.video_time_s,
            'eyelink_time_ms': point.eyelink_time_ms,
            'audio_found_s': point.video_time_s if point.audio_found else None,
            'confidence': point.confidence
        }


# =================================================================================
# CONVENIENCE FUNCTIONS
# =================================================================================

def list_available_modes():
    """Gibt Uebersicht aller verfuegbaren Modi aus (v1.2: gruppiert)"""
    
    modes = CalibrationModes.get_all_modes()
    
    print(f"\n{'='*70}")
    print("VERFUEGBARE KALIBRIERUNGS-MODI")
    print(f"{'='*70}")
    
    # Gruppierte Ausgabe
    groups = {
        'VOLLSTAENDIG': ['FullCalib'],
        'OHNE MID': ['BegEnd', 'BegEndFix'],
        'NUR ANFANG': ['OnlyBeg', 'OnlyBegFix', 'BegFirst10', 'BegSecond10'],
        'KOMBINATIONEN': ['BegFirst10End', 'BegFirst10EndFix', 'BegSecond10End', 'BegSecond10Mid'],
        'NUR ENDE': ['OnlyEnd'],
        'SPEZIAL': ['First10', 'OnlyCenter', 'Periphery'],
    }
    
    for group_name, mode_names in groups.items():
        print(f"\n[{group_name}]")
        
        for name in mode_names:
            if name not in modes:
                continue
            
            mode = modes[name]
            
            # Formatiere Details
            range_info = ""
            if mode.beg_range != (1, 20) and 'beg' in mode.timepoints:
                range_info = f" | beg: {mode.beg_range[0]}-{mode.beg_range[1]}"
            
            fix_info = " | +Fixationen" if mode.use_fixations else ""
            
            print(f"  >> {name}")
            print(f"     {mode.description}")
            print(f"     Phasen: {', '.join(mode.timepoints)}{range_info}{fix_info}")
    
    print(f"\n{'='*70}")

def apply_calibration_mode(phases_json_path: Path, mode_name: str, 
                          output_path: Optional[Path] = None) -> Path:
    """
    Convenience-Funktion für schnelle Anwendung.
    
    Args:
        phases_json_path: Pfad zu phases_detected.json
        mode_name: Name des Modus (z.B. 'BegEnd')
        output_path: Optional - Output-Pfad
    
    Returns:
        Path zur modifizierten JSON
    """
    
    mode = CalibrationModes.get_mode(mode_name)
    manager = CalibrationModeManager(phases_json_path)
    
    return manager.apply_mode(mode, output_path)


# =================================================================================
# MAIN (TEST)
# =================================================================================

if __name__ == "__main__":
    """Test-Funktion für Calibration Mode Manager"""
    
    # Beispiel-Pfad (ANPASSEN!)
    PHASES_JSON = Path(r"C:\Users\imanu\Documents\Imanuel\Studium\Psy\Master\Pupillendetektion\Analyse\Debug_Output\phases_detected.json")
    
    print(f"\n{'='*70}")
    print("TEST: CALIBRATION MODE MANAGER")
    print(f"{'='*70}")
    
    # Test 1: Liste Modi
    list_available_modes()
    
    # Test 2: Wende Modus an (wenn JSON vorhanden)
    if PHASES_JSON.exists():
        print(f"\n{'='*70}")
        print("TEST: MODUS ANWENDEN")
        print(f"{'='*70}")
        
        manager = CalibrationModeManager(PHASES_JSON)
        
        # Teste BegEnd-Modus
        mode = CalibrationModes.get_mode('BegEnd')
        output_path = manager.apply_mode(mode)
        
        print(f"\n[OK] Test erfolgreich!")
        print(f"   Modifizierte JSON: {output_path}")
    else:
        print(f"\n️ phases_detected.json nicht gefunden")
        print(f"   Pfad: {PHASES_JSON}")
