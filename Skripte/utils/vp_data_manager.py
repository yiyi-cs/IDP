"""
=================================================================================
VP DATA MANAGER v1.1 - Automatische Datei-Erkennung fuer Master-Pipeline
=================================================================================

Funktionen:
-----------
1. Scannt Basis-Ordner nach VP-Ordnern (4-Zeichen-Code)
2. Erkennt automatisch:
   - ASC-Dateien (EyeLink)
   - experiment_sync_log.json
   - Videos (*_60.mp4, *_25.mp4)
   - Calibration/*.json (beg/mid/end)
3. Validiert Vollstaendigkeit
4. Erkennt neue VPs (ohne Analyse-Ordner)

NEU in v1.1:
------------
- Package-Struktur (utils/)
- Path-Setup fuer manuelle Ausfuehrung
- Video-FPS korrigiert (25hz statt 24hz)

Version: 1.1
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
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
import json

# =================================================================================
# DATENSTRUKTUREN
# =================================================================================

@dataclass
class VPData:
    """
    Repräsentiert eine Versuchsperson mit allen erkannten Dateien.
    """
    vp_code: str                              # z.B. 'bjs4'
    vp_folder: Path                           # Absoluter Pfad
    asc_file: Optional[Path] = None           # EyeLink ASC
    experiment_log: Optional[Path] = None     # experiment_sync_log.json
    videos: Dict[str, Path] = None            # {'60hz': Path, '25hz': Path}
    calibration_jsons: Dict[str, Path] = None # {'beg': Path, 'mid': Path, 'end': Path}
    analysis_folder: Optional[Path] = None    # Analyse-Unterordner (falls vorhanden)
    has_analysis: bool = False                # True wenn Analyse-Ordner existiert
    
    def __post_init__(self):
        if self.videos is None:
            self.videos = {}
        if self.calibration_jsons is None:
            self.calibration_jsons = {}


@dataclass
class ValidationResult:
    """
    Ergebnis der Validierung einer VP.
    """
    is_valid: bool
    vp_code: str
    errors: List[str]
    warnings: List[str]
    
    def __str__(self):
        status = "[OK] VALID" if self.is_valid else "[X] INVALID"
        msg = f"{status}: {self.vp_code}\n"
        
        if self.errors:
            msg += "  Errors:\n"
            for err in self.errors:
                msg += f"    - {err}\n"
        
        if self.warnings:
            msg += "  Warnings:\n"
            for warn in self.warnings:
                msg += f"    - {warn}\n"
        
        return msg

@dataclass
class RunConfig:
    """
    Konfiguration eines Pipeline-Runs (fuer Vergleiche).
    """
    video_fps: str                  # '60hz' oder '25hz'
    calib_mode: str                 # 'FullCalib', 'BegEnd', etc.
    analysis_mode: int              # 1 oder 2
    methods: str                    # 'mediapipe', 'ptgaze', 'both'
    
    def matches(self, other: 'RunConfig', ignore_fields: List[str] = None) -> bool:
        """
        Prueft ob zwei Configs uebereinstimmen.
        
        Args:
            other: Andere RunConfig
            ignore_fields: Felder die ignoriert werden sollen
        """
        if ignore_fields is None:
            ignore_fields = []
        
        if 'video_fps' not in ignore_fields and self.video_fps != other.video_fps:
            return False
        if 'calib_mode' not in ignore_fields and self.calib_mode != other.calib_mode:
            return False
        if 'analysis_mode' not in ignore_fields and self.analysis_mode != other.analysis_mode:
            return False
        if 'methods' not in ignore_fields and self.methods != other.methods:
            return False
        
        return True
    
    def __str__(self):
        return f"{self.video_fps}_{self.calib_mode}_mode{self.analysis_mode}_{self.methods}"


@dataclass
class CacheFile:
    """
    Repraesentiert eine einzelne Cache-Datei.
    """
    step: str                       # z.B. 'debug_0', 'debug_1_mediapipe'
    filename: str                   # z.B. 'debug_1_pupil_data.csv'
    path: Path                      # Absoluter Pfad
    source_run: str                 # Run-ID (z.B. 'Run_25hz_FullCalib_mode2_20250115')
    source_config: Optional[Dict]   # Config aus config_used.json
    is_compatible: bool             # True wenn mit aktueller Config kompatibel
    compatibility_reason: str       # Erklaerung warum kompatibel/inkompatibel


@dataclass
class CacheOverview:
    """
    Uebersicht aller verfuegbaren Cache-Dateien fuer eine VP.
    """
    vp_code: str
    target_config: RunConfig                        # Gewuenschte Config
    available_files: Dict[str, List[CacheFile]]     # Step -> Liste von CacheFiles
    compatible_files: Dict[str, CacheFile]          # Step -> Beste kompatible Datei
    missing_steps: List[str]                        # Steps ohne Cache
    
    def get_summary(self) -> str:
        """Gibt eine formatierte Zusammenfassung zurueck."""
        lines = []
        lines.append(f"Cache-Uebersicht fuer VP: {self.vp_code}")
        lines.append(f"Ziel-Config: {self.target_config}")
        lines.append(f"-" * 50)
        
        all_steps = [
            'debug_0', 'debug_1_mediapipe', 'debug_1_ptgaze',
            'offline_calibration', 'offline_calibration_ptgaze',
            'debug_3', 'debug_4',
            'debug_5_mediapipe', 'debug_5_ptgaze',
            'debug_6'
        ]
        
        for step in all_steps:
            if step in self.compatible_files:
                cf = self.compatible_files[step]
                lines.append(f"  [OK] {step}: {cf.filename}")
                lines.append(f"       Run: {cf.source_run}")
            elif step in self.available_files and self.available_files[step]:
                lines.append(f"  [!] {step}: Vorhanden aber INKOMPATIBEL")
                for cf in self.available_files[step][:2]:  # Max 2 anzeigen
                    lines.append(f"       - {cf.source_run}: {cf.compatibility_reason}")
            else:
                lines.append(f"  [ ] {step}: NICHT GEFUNDEN")
        
        return "\n".join(lines)


@dataclass 
class RunInfo:
    """
    Informationen ueber einen existierenden Run.
    """
    run_id: str
    run_folder: Path
    config: Optional[RunConfig]
    success: bool
    created_at: str
    completed_steps: List[str]
    error_message: Optional[str] = None


# =================================================================================
# CACHE-KRITERIEN
# =================================================================================

# Definiert welche Config-Felder fuer jeden Schritt relevant sind
CACHE_CRITERIA = {
    'debug_0': ['video_fps'],
    'debug_1_mediapipe': ['video_fps'],
    'debug_1_ptgaze': ['video_fps'],
    'offline_calibration': ['video_fps', 'calib_mode'],
    'offline_calibration_ptgaze': ['video_fps', 'calib_mode'],
    'debug_3': [],  # Keine Config-Abhaengigkeit (nur ASC-Datei)
    'debug_4': ['video_fps'],
    'debug_5_mediapipe': ['video_fps', 'calib_mode'],
    'debug_5_ptgaze': ['video_fps', 'calib_mode'],
    'debug_6': ['video_fps', 'calib_mode', 'analysis_mode'],
}

# Mapping: Step -> erwartete Output-Dateien
STEP_OUTPUT_FILES = {
    'debug_0': ['phases_detected.json'],
    'debug_1_mediapipe': ['debug_1_pupil_data.csv', 'debug_1_metadata.json'],
    'debug_1_ptgaze': ['debug_1_ptgaze_data.csv'],
    'offline_calibration': ['calibration_*.pkl'],  # Wildcard (ohne _ptgaze)
    'offline_calibration_ptgaze': ['calibration_ptgaze_*.pkl'],
    'debug_3': ['debug_3_eyetracker_data.csv', 'debug_3_blocks_overview.csv', 'debug_3_fixations.csv'],
    'debug_4': ['debug_4_pupil_data_synced.csv', 'debug_4_ptgaze_gaze_synced.csv', 'debug_4_sync_info.json'],
    'debug_5_mediapipe': ['debug_5_pupil_data_calibrated.csv'],
    'debug_5_ptgaze': ['debug_5_ptgaze_calibrated.csv'],
    'debug_6': ['debug_6_three_way_comparison.csv'],
}

# =================================================================================
# VP DATA MANAGER
# =================================================================================

class VPDataManager:
    """
    Automatische Erkennung und Verwaltung von VP-Daten.
    
    Workflow:
    ---------
    1. Scan: Finde alle VP-Ordner im Basis-Pfad
    2. Detect: Erkenne Dateien automatisch
    3. Validate: Prüfe Vollständigkeit
    """
    
    def __init__(self, base_folder: Path):
        """
        Args:
            base_folder: Basis-Ordner (z.B. .../Pupillendetektion/Ergebnisse)
        """
        self.base_folder = Path(base_folder)
        
        if not self.base_folder.exists():
            raise FileNotFoundError(f"Basis-Ordner nicht gefunden: {self.base_folder}")
        
        print(f"[OK] VP Data Manager initialisiert")
        print(f"   Basis-Ordner: {self.base_folder}")
    
    # ═════════════════════════════════════════════════════════════════
    # HAUPT-METHODEN
    # ═════════════════════════════════════════════════════════════════
    
    def scan_all_vps(self) -> List[str]:
        """
        Scannt Basis-Ordner nach VP-Codes (4-Zeichen-Ordner).
        
        Returns:
            Liste von VP-Codes (z.B. ['bjs4', 'ldj9', ...])
        """
        vp_codes = []
        
        for item in self.base_folder.iterdir():
            if item.is_dir() and self._is_valid_vp_code(item.name):
                vp_codes.append(item.name)
        
        return sorted(vp_codes)
    
    def scan_new_vps(self) -> List[str]:
        """
        Findet VPs OHNE Analyse-Ordner (neue VPs).
        
        Returns:
            Liste von VP-Codes ohne Analyse
        """
        all_vps = self.scan_all_vps()
        new_vps = []
        
        for vp_code in all_vps:
            vp_folder = self.base_folder / vp_code
            analysis_folder = vp_folder / 'Analyse'
            
            if not analysis_folder.exists():
                new_vps.append(vp_code)
        
        return new_vps
    
    def load_vp_data(self, vp_code: str) -> VPData:
        """
        Lädt Daten einer VP (auto-detect alle Dateien).
        
        Args:
            vp_code: VP-Code (z.B. 'bjs4')
        
        Returns:
            VPData mit erkannten Dateien
        
        Raises:
            ValueError: VP-Ordner existiert nicht
        """
        vp_folder = self.base_folder / vp_code
        
        if not vp_folder.exists():
            raise ValueError(f"VP-Ordner nicht gefunden: {vp_folder}")
        
        vp_data = VPData(
            vp_code=vp_code,
            vp_folder=vp_folder
        )
        
        # ═════════════════════════════════════════════════════════════
        # AUTO-DETECTION
        # ═════════════════════════════════════════════════════════════
        
        # 1. ASC-Datei
        vp_data.asc_file = self._find_asc_file(vp_folder, vp_code)
        
        # 2. experiment_sync_log.json
        vp_data.experiment_log = self._find_experiment_log(vp_folder)
        
        # 3. Videos
        vp_data.videos = self._find_videos(vp_folder, vp_code)
        
        # 4. Calibration-JSONs
        vp_data.calibration_jsons = self._find_calibration_jsons(vp_folder)
        
        # 5. Analyse-Ordner
        analysis_folder = vp_folder / 'Analyse'
        if analysis_folder.exists():
            vp_data.analysis_folder = analysis_folder
            vp_data.has_analysis = True
        
        return vp_data
    
    def validate_vp_data(self, vp_data: VPData) -> ValidationResult:
        """
        Validiert eine VP auf Vollständigkeit.
        
        Args:
            vp_data: VP-Daten
        
        Returns:
            ValidationResult mit Status
        """
        errors = []
        warnings = []
        
        # ═════════════════════════════════════════════════════════════
        # PFLICHT-DATEIEN
        # ═════════════════════════════════════════════════════════════
        
        # experiment_sync_log.json (PFLICHT!)
        if vp_data.experiment_log is None:
            errors.append("experiment_sync_log.json fehlt (PFLICHT)")
        
        # Mindestens 1 Video (PFLICHT!)
        if not vp_data.videos:
            errors.append("Kein Video gefunden (*_60.mp4 oder *_24.mp4)")
        
        # ═════════════════════════════════════════════════════════════
        # OPTIONAL (nur Warnung)
        # ═════════════════════════════════════════════════════════════
        
        # ASC (optional für Modus 1 Standalone)
        if vp_data.asc_file is None:
            warnings.append("Keine ASC-Datei (Modus 1: Standalone möglich)")
        
        # Calibration-JSONs (optional, können fehlen)
        if not vp_data.calibration_jsons:
            warnings.append("Keine calibration_timing_*.json (Fallback aktiv)")
        
        # Videos: Idealerweise beide vorhanden
        if '60hz' not in vp_data.videos:
            warnings.append("Kein 60Hz-Video gefunden")
        
        if '25hz' not in vp_data.videos:
            warnings.append("Kein 25Hz-Video gefunden")
        
        # ═════════════════════════════════════════════════════════════
        # ERGEBNIS
        # ═════════════════════════════════════════════════════════════
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            vp_code=vp_data.vp_code,
            errors=errors,
            warnings=warnings
        )
    
    # ═════════════════════════════════════════════════════════════════
    # HELPER-METHODEN (PRIVATE)
    # ═════════════════════════════════════════════════════════════════
    
    def _is_valid_vp_code(self, name: str) -> bool:
        """Prüft ob Ordnername ein valider VP-Code ist (4 Zeichen)"""
        return len(name) == 4 and name.isalnum()
    
    def _find_asc_file(self, vp_folder: Path, vp_code: str) -> Optional[Path]:
        """Findet ASC-Datei (VP-Code + 4 Ziffern.asc)"""
        
        # Pattern: bjs41511.asc (VP-Code + 4 Ziffern)
        pattern = f"{vp_code}*.asc"
        asc_files = list(vp_folder.glob(pattern))
        
        if len(asc_files) == 0:
            return None
        elif len(asc_files) == 1:
            return asc_files[0]
        else:
            # Multiple ASC: Nehme neueste
            return max(asc_files, key=lambda p: p.stat().st_mtime)
    
    def _find_experiment_log(self, vp_folder: Path) -> Optional[Path]:
        """Findet experiment_sync_log.json"""
        
        log_file = vp_folder / 'experiment_sync_log.json'
        
        if log_file.exists():
            return log_file
        
        # Fallback: Suche nach experiment_sync_log*.json
        logs = list(vp_folder.glob('experiment_sync_log*.json'))
        
        if logs:
            return logs[0]
        
        return None
    
    def _find_videos(self, vp_folder: Path, vp_code: str) -> Dict[str, Path]:
        """
        Findet Videos (*_60.mp4, *_25.mp4).
        
        Suchstrategie (Prioritaet):
        1. Im VP-Ordner selbst (z.B. ldj9/ldj9_60.mp4)
        2. Im Videos_60/Videos_25 Ordner auf Basis-Ebene (Fallback)
        
        Returns:
            {'60hz': Path, '25hz': Path}
        """
        
        videos = {}
        
        # Basis-Ordner (wo VP-Ordner liegen)
        base_folder = vp_folder.parent
        
        # ══════════════════════════════════════════════════════════════
        # 60 Hz Video
        # ══════════════════════════════════════════════════════════════
        
        # Prioritaet 1: Im VP-Ordner
        pattern_60 = f"{vp_code}_60.mp4"
        video_60 = vp_folder / pattern_60
        
        if video_60.exists():
            videos['60hz'] = video_60
        else:
            # Fallback: *_60.mp4 im VP-Ordner
            candidates_60 = list(vp_folder.glob('*_60.mp4')) + list(vp_folder.glob('*_60.MP4'))
            if candidates_60:
                videos['60hz'] = candidates_60[0]
            else:
                # Prioritaet 2: Im Videos_60 Ordner auf Basis-Ebene
                videos_60_folder = base_folder / 'Videos_60'
                if videos_60_folder.exists():
                    # Suche nach VP-Code im Dateinamen
                    for video_file in videos_60_folder.glob('*.mp4'):
                        if vp_code.lower() in video_file.name.lower():
                            videos['60hz'] = video_file
                            print(f"      [FALLBACK] 60Hz Video in Videos_60: {video_file.name}")
                            break
                    
                    # Falls nicht gefunden, auch in Gross-Schreibung suchen
                    if '60hz' not in videos:
                        for video_file in videos_60_folder.glob('*.MP4'):
                            if vp_code.lower() in video_file.name.lower():
                                videos['60hz'] = video_file
                                print(f"      [FALLBACK] 60Hz Video in Videos_60: {video_file.name}")
                                break
        
        # ══════════════════════════════════════════════════════════════
        # 25 Hz Video (Prioritaet: _25.mp4, Fallback: _24.mp4)
        # ══════════════════════════════════════════════════════════════
        
        # Prioritaet 1: Im VP-Ordner
        for suffix in ['_25.mp4', '_25.MP4', '_24.mp4', '_24.MP4']:
            pattern_25 = f"{vp_code}{suffix}"
            video_25 = vp_folder / pattern_25
            
            if video_25.exists():
                videos['25hz'] = video_25
                break
        
        # Fallback: *_25.mp4 oder *_24.mp4 im VP-Ordner
        if '25hz' not in videos:
            for suffix in ['_25.mp4', '_25.MP4', '_24.mp4', '_24.MP4']:
                candidates_25 = list(vp_folder.glob(f'*{suffix}'))
                if candidates_25:
                    videos['25hz'] = candidates_25[0]
                    break
        
        # Prioritaet 2: Im Videos_25 Ordner auf Basis-Ebene
        if '25hz' not in videos:
            videos_25_folder = base_folder / 'Videos_25'
            if videos_25_folder.exists():
                # Suche nach VP-Code im Dateinamen
                for ext in ['*.mp4', '*.MP4']:
                    for video_file in videos_25_folder.glob(ext):
                        if vp_code.lower() in video_file.name.lower():
                            videos['25hz'] = video_file
                            print(f"      [FALLBACK] 25Hz Video in Videos_25: {video_file.name}")
                            break
                    if '25hz' in videos:
                        break
        
        return videos
    
    def _find_calibration_jsons(self, vp_folder: Path) -> Dict[str, Path]:
        """
        Findet calibration_timing_*.json im Calibration-Unterordner.
        
        Returns:
            {'beg': Path, 'mid': Path, 'end': Path}
        """
        calib_jsons = {}
        
        calib_folder = vp_folder / 'Calibration'
        
        if not calib_folder.exists():
            return calib_jsons
        
        # Suche nach beg, mid, end
        for label in ['beg', 'mid', 'end']:
            pattern = f"calibration_timing*{label}*.json"
            matches = list(calib_folder.glob(pattern))
            
            if matches:
                # Nehme neueste (falls mehrere)
                calib_jsons[label] = max(matches, key=lambda p: p.stat().st_mtime)
        
        return calib_jsons
    
    # ═════════════════════════════════════════════════════════════════
    # AUSGABE-METHODEN
    # ═════════════════════════════════════════════════════════════════
    
    def print_vp_summary(self, vp_data: VPData):
        """Gibt Zusammenfassung einer VP aus"""
        
        print(f"\n{'-'*60}")
        print(f"VP: {vp_data.vp_code}")
        print(f"{'-'*60}")
        print(f"Ordner: {vp_data.vp_folder}")
        
        # ASC
        if vp_data.asc_file:
            print(f"[OK] ASC: {vp_data.asc_file.name}")
        else:
            print(f"[X] ASC: Nicht gefunden")
        
        # experiment_sync_log
        if vp_data.experiment_log:
            print(f"[OK] Experiment-Log: {vp_data.experiment_log.name}")
        else:
            print(f"[X] Experiment-Log: Nicht gefunden")
        
        # Videos
        if vp_data.videos:
            for fps, path in vp_data.videos.items():
                print(f"[OK] Video ({fps}): {path.name}")
        else:
            print(f"[X] Videos: Keine gefunden")
        
        # Calibration
        if vp_data.calibration_jsons:
            labels = ', '.join(vp_data.calibration_jsons.keys())
            print(f"[OK] Calibration: {labels}")
        else:
            print(f"[!] Calibration: Keine JSONs")
        
        # Analyse
        if vp_data.has_analysis:
            n_runs = len(list(vp_data.analysis_folder.glob('Run_*')))
            print(f"[i] Analyse: {n_runs} Runs vorhanden")
        else:
            print(f"[NEW] Analyse: Noch nicht analysiert")

    # ═════════════════════════════════════════════════════════════════
    # RUN-MANAGEMENT (NEU v1.2)
    # ═════════════════════════════════════════════════════════════════
    
    def get_run_config(self, run_folder: Path) -> Optional[RunConfig]:
        """
        Liest RunConfig aus config_used.json eines Runs.
        
        Args:
            run_folder: Pfad zum Run-Ordner
        
        Returns:
            RunConfig oder None wenn nicht lesbar
        """
        config_file = run_folder / 'config_used.json'
        
        if not config_file.exists():
            return None
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return RunConfig(
                video_fps=data.get('video_fps', ''),
                calib_mode=data.get('calib_mode', ''),
                analysis_mode=data.get('analysis_mode', 2),
                methods=data.get('methods', 'both')
            )
        except (json.JSONDecodeError, KeyError):
            return None
    
    def check_run_success(self, run_folder: Path) -> bool:
        """
        Prueft ob ein Run erfolgreich abgeschlossen wurde.
        
        Kriterien:
        1. config_used.json existiert
        2. Kein error_log.json ODER error_log.json mit success=True
        3. Mindestens debug_6 Output vorhanden
        
        Args:
            run_folder: Pfad zum Run-Ordner
        
        Returns:
            True wenn erfolgreich
        """
        # Check 1: config_used.json
        if not (run_folder / 'config_used.json').exists():
            return False
        
        # Check 2: error_log.json
        error_log = run_folder / 'error_log.json'
        if error_log.exists():
            try:
                with open(error_log, 'r', encoding='utf-8') as f:
                    error_data = json.load(f)
                # Wenn error_log existiert und failed_step gesetzt ist -> nicht erfolgreich
                if error_data.get('failed_step'):
                    return False
            except json.JSONDecodeError:
                pass
        
        # Check 3: Mindestens debug_5 oder debug_6 Output
        has_debug5 = (run_folder / 'debug_5_pupil_data_calibrated.csv').exists()
        has_debug6 = (run_folder / 'debug_6_three_way_comparison.csv').exists()
        
        return has_debug5 or has_debug6
    
    def get_run_info(self, run_folder: Path) -> Optional[RunInfo]:
        """
        Sammelt alle Informationen ueber einen Run.
        
        Args:
            run_folder: Pfad zum Run-Ordner
        
        Returns:
            RunInfo oder None
        """
        if not run_folder.exists():
            return None
        
        config = self.get_run_config(run_folder)
        success = self.check_run_success(run_folder)
        
        # Finde abgeschlossene Steps
        completed_steps = []
        for step, files in STEP_OUTPUT_FILES.items():
            step_complete = True
            for filename in files:
                if '*' in filename:
                    # Wildcard
                    if not list(run_folder.glob(filename)):
                        step_complete = False
                        break
                else:
                    if not (run_folder / filename).exists():
                        step_complete = False
                        break
            if step_complete:
                completed_steps.append(step)
        
        # Lese created_at aus config_used.json
        created_at = ""
        config_file = run_folder / 'config_used.json'
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                created_at = data.get('timestamp', '')
            except:
                pass
        
        # Lese error_message falls vorhanden
        error_message = None
        error_log = run_folder / 'error_log.json'
        if error_log.exists():
            try:
                with open(error_log, 'r', encoding='utf-8') as f:
                    error_data = json.load(f)
                error_message = error_data.get('error_message')
            except:
                pass
        
        return RunInfo(
            run_id=run_folder.name,
            run_folder=run_folder,
            config=config,
            success=success,
            created_at=created_at,
            completed_steps=completed_steps,
            error_message=error_message
        )
    
    def find_all_runs(self, vp_data: VPData) -> List[RunInfo]:
        """
        Findet alle Runs einer VP.
        
        Args:
            vp_data: VP-Daten
        
        Returns:
            Liste von RunInfo (sortiert nach Datum, neueste zuerst)
        """
        if not vp_data.has_analysis or vp_data.analysis_folder is None:
            return []
        
        runs = []
        for run_folder in vp_data.analysis_folder.glob('Run_*'):
            if run_folder.is_dir():
                run_info = self.get_run_info(run_folder)
                if run_info:
                    runs.append(run_info)
        
        # Sortiere nach Datum (neueste zuerst)
        runs.sort(key=lambda r: r.created_at, reverse=True)
        
        return runs
    
    def find_matching_runs(self, vp_data: VPData, target_config: RunConfig,
                          require_success: bool = True) -> List[RunInfo]:
        """
        Findet Runs die mit der Ziel-Config uebereinstimmen.
        
        Args:
            vp_data: VP-Daten
            target_config: Gewuenschte Konfiguration
            require_success: Nur erfolgreiche Runs (Default: True)
        
        Returns:
            Liste von passenden RunInfo
        """
        all_runs = self.find_all_runs(vp_data)
        matching = []
        
        for run in all_runs:
            if run.config is None:
                continue
            
            # Erfolgs-Check
            if require_success and not run.success:
                continue
            
            # Config-Match (exakt)
            if run.config.matches(target_config):
                matching.append(run)
        
        return matching
    
    # ═════════════════════════════════════════════════════════════════
    # CACHE-MANAGEMENT (NEU v1.2)
    # ═════════════════════════════════════════════════════════════════
    
    def _is_step_compatible(self, step: str, source_config: RunConfig, 
                           target_config: RunConfig) -> tuple:
        """
        Prueft ob ein Step-Output mit der Ziel-Config kompatibel ist.
        
        Args:
            step: Pipeline-Schritt
            source_config: Config des existierenden Runs
            target_config: Gewuenschte Config
        
        Returns:
            (is_compatible, reason)
        """
        criteria = CACHE_CRITERIA.get(step, [])
        
        if not criteria:
            # Keine Kriterien = immer kompatibel
            return True, "Keine Config-Abhaengigkeit"
        
        mismatches = []
        
        for field in criteria:
            source_val = getattr(source_config, field, None)
            target_val = getattr(target_config, field, None)
            
            if source_val != target_val:
                mismatches.append(f"{field}: {source_val} != {target_val}")
        
        if mismatches:
            return False, "; ".join(mismatches)
        else:
            return True, "Alle Kriterien erfuellt"
    
    def get_cache_overview(self, vp_data: VPData, target_config: RunConfig) -> CacheOverview:
        """
        Erstellt eine Uebersicht aller verfuegbaren Cache-Dateien.
        
        Args:
            vp_data: VP-Daten
            target_config: Gewuenschte Konfiguration
        
        Returns:
            CacheOverview mit allen verfuegbaren und kompatiblen Dateien
        """
        available_files = {}
        compatible_files = {}
        missing_steps = []
        
        # Sammle alle Runs
        all_runs = self.find_all_runs(vp_data)
        
        # Pruefe jeden Step
        for step, expected_files in STEP_OUTPUT_FILES.items():
            available_files[step] = []
            
            # Durchsuche alle Runs
            for run_info in all_runs:
                if run_info.config is None:
                    continue
                
                # Pruefe ob Step in diesem Run vorhanden
                has_files = True
                found_file = None
                
                for filename in expected_files:
                    if '*' in filename:
                        matches = list(run_info.run_folder.glob(filename))
                        if not matches:
                            has_files = False
                            break
                        found_file = matches[0]
                    else:
                        file_path = run_info.run_folder / filename
                        if not file_path.exists():
                            has_files = False
                            break
                        found_file = file_path
                
                if not has_files or found_file is None:
                    continue
                
                # Pruefe Kompatibilitaet
                is_compat, reason = self._is_step_compatible(step, run_info.config, target_config)
                
                cache_file = CacheFile(
                    step=step,
                    filename=found_file.name,
                    path=found_file,
                    source_run=run_info.run_id,
                    source_config=run_info.config.__dict__ if run_info.config else None,
                    is_compatible=is_compat,
                    compatibility_reason=reason
                )
                
                available_files[step].append(cache_file)
                
                # Speichere beste kompatible Datei (neueste)
                if is_compat and step not in compatible_files:
                    compatible_files[step] = cache_file
            
            # Pruefe ob Step fehlt
            if step not in compatible_files:
                missing_steps.append(step)
        
        return CacheOverview(
            vp_code=vp_data.vp_code,
            target_config=target_config,
            available_files=available_files,
            compatible_files=compatible_files,
            missing_steps=missing_steps
        )
    
    # ═════════════════════════════════════════════════════════════════
    # VP-FILTERUNG (NEU v1.2)
    # ═════════════════════════════════════════════════════════════════
    
    def scan_vps_without_matching_run(self, target_config: RunConfig,
                                      require_success: bool = True) -> List[str]:
        """
        Findet VPs die KEINEN Run mit der Ziel-Config haben.
        
        Args:
            target_config: Gewuenschte Konfiguration
            require_success: Nur erfolgreiche Runs zaehlen (Default: True)
        
        Returns:
            Liste von VP-Codes ohne passende Analyse
        """
        all_vps = self.scan_all_vps()
        vps_without_match = []
        
        for vp_code in all_vps:
            try:
                vp_data = self.load_vp_data(vp_code)
            except (ValueError, FileNotFoundError):
                # VP-Ordner existiert nicht mehr
                continue
            
            # Pruefe ob passendes Video vorhanden
            if target_config.video_fps not in vp_data.videos:
                # Kein passendes Video -> kann nicht analysiert werden
                continue
            
            # Suche passende Runs
            matching = self.find_matching_runs(vp_data, target_config, require_success)
            
            if not matching:
                vps_without_match.append(vp_code)
        
        return vps_without_match
    
    def scan_vps_with_failed_runs(self, target_config: RunConfig) -> List[str]:
        """
        Findet VPs die einen FEHLGESCHLAGENEN Run mit der Ziel-Config haben.
        
        Args:
            target_config: Gewuenschte Konfiguration
        
        Returns:
            Liste von VP-Codes mit fehlgeschlagenen Runs
        """
        all_vps = self.scan_all_vps()
        vps_with_failed = []
        
        for vp_code in all_vps:
            try:
                vp_data = self.load_vp_data(vp_code)
            except (ValueError, FileNotFoundError):
                continue
            
            if not vp_data.has_analysis:
                continue
            
            # Suche Runs mit passender Config (ohne Erfolgs-Filter)
            all_matching = self.find_matching_runs(vp_data, target_config, require_success=False)
            successful_matching = self.find_matching_runs(vp_data, target_config, require_success=True)
            
            # Wenn es matching Runs gibt, aber keinen erfolgreichen
            if all_matching and not successful_matching:
                vps_with_failed.append(vp_code)
        
        return vps_with_failed
    
    def get_vp_analysis_status(self, vp_code: str, target_config: RunConfig) -> Dict:
        """
        Gibt detaillierten Analyse-Status einer VP zurueck.
        
        Args:
            vp_code: VP-Code
            target_config: Gewuenschte Konfiguration
        
        Returns:
            Dict mit Status-Informationen
        """
        try:
            vp_data = self.load_vp_data(vp_code)
        except (ValueError, FileNotFoundError):
            return {
                'vp_code': vp_code,
                'exists': False,
                'status': 'not_found',
                'message': 'VP-Ordner nicht gefunden'
            }
        
        # Pruefe Video-Verfuegbarkeit
        if target_config.video_fps not in vp_data.videos:
            return {
                'vp_code': vp_code,
                'exists': True,
                'has_video': False,
                'status': 'no_video',
                'message': f'Kein {target_config.video_fps} Video vorhanden'
            }
        
        # Suche passende Runs
        successful_runs = self.find_matching_runs(vp_data, target_config, require_success=True)
        all_runs = self.find_matching_runs(vp_data, target_config, require_success=False)
        
        if successful_runs:
            latest = successful_runs[0]
            return {
                'vp_code': vp_code,
                'exists': True,
                'has_video': True,
                'status': 'completed',
                'message': f'Erfolgreicher Run vorhanden: {latest.run_id}',
                'latest_run': latest,
                'n_successful_runs': len(successful_runs)
            }
        elif all_runs:
            latest = all_runs[0]
            return {
                'vp_code': vp_code,
                'exists': True,
                'has_video': True,
                'status': 'failed',
                'message': f'Run fehlgeschlagen: {latest.error_message or "Unbekannter Fehler"}',
                'latest_run': latest,
                'n_failed_runs': len(all_runs)
            }
        else:
            return {
                'vp_code': vp_code,
                'exists': True,
                'has_video': True,
                'status': 'not_started',
                'message': 'Keine Analyse mit dieser Config vorhanden'
            }
    
    def print_cache_overview(self, cache_overview: CacheOverview):
        """
        Gibt eine formatierte Cache-Uebersicht aus.
        
        Args:
            cache_overview: CacheOverview Objekt
        """
        print(f"\n{'='*60}")
        print(f"CACHE-UEBERSICHT: VP {cache_overview.vp_code}")
        print(f"{'='*60}")
        print(f"Ziel-Config: {cache_overview.target_config}")
        print(f"{'-'*60}")
        
        all_steps = [
            ('debug_0', 'Phase Detection'),
            ('debug_1_mediapipe', 'MediaPipe Pupillen'),
            ('debug_1_ptgaze', 'ptgaze Gaze'),
            ('offline_calibration', 'MediaPipe PKL'),
            ('offline_calibration_ptgaze', 'ptgaze PKL'),
            ('debug_3', 'EyeLink Import'),
            ('debug_4', 'Synchronisation'),
            ('debug_5_mediapipe', 'MediaPipe Kalibriert'),
            ('debug_5_ptgaze', 'ptgaze Kalibriert'),
            ('debug_6', '3-Wege-Vergleich'),
        ]
        
        for step, description in all_steps:
            if step in cache_overview.compatible_files:
                cf = cache_overview.compatible_files[step]
                print(f"  [OK] {description}")
                print(f"       Datei: {cf.filename}")
                print(f"       Run: {cf.source_run}")
            elif step in cache_overview.available_files and cache_overview.available_files[step]:
                print(f"  [!] {description} (INKOMPATIBEL)")
                for cf in cache_overview.available_files[step][:2]:
                    print(f"       - {cf.source_run}: {cf.compatibility_reason}")
            else:
                print(f"  [ ] {description} (nicht vorhanden)")
        
        print(f"{'-'*60}")
        n_compat = len(cache_overview.compatible_files)
        n_missing = len(cache_overview.missing_steps)
        print(f"Kompatibel: {n_compat} | Fehlend: {n_missing}")
        print(f"{'='*60}")

# =================================================================================
# MAIN (TEST)
# =================================================================================

if __name__ == "__main__":
    """Test-Funktion fuer VP Data Manager"""
    
    # Beispiel-Pfad (ANPASSEN!)
    BASE_FOLDER = Path(r"C:\Users\imanu\Documents\Imanuel\Studium\Psy\Master\Pupillendetektion\Ergebnisse")
    
    manager = VPDataManager(BASE_FOLDER)
    
    print(f"\n{'='*70}")
    print("TEST: VP DATA MANAGER v1.2")
    print(f"{'='*70}")
    
    # Test 1: Alle VPs
    all_vps = manager.scan_all_vps()
    print(f"\n[1] Gefundene VPs: {len(all_vps)}")
    print(f"    {', '.join(all_vps)}")
    
    # Test 2: Neue VPs (alte Methode)
    new_vps = manager.scan_new_vps()
    print(f"\n[2] Neue VPs (ohne Analyse-Ordner): {len(new_vps)}")
    print(f"    {', '.join(new_vps) if new_vps else 'Keine'}")
    
    # Test 3: VPs ohne passende Config
    test_config = RunConfig(
        video_fps='25hz',
        calib_mode='FullCalib',
        analysis_mode=2,
        methods='both'
    )
    
    vps_without_config = manager.scan_vps_without_matching_run(test_config)
    print(f"\n[3] VPs ohne passende Analyse ({test_config}):")
    print(f"    {len(vps_without_config)} VPs: {', '.join(vps_without_config) if vps_without_config else 'Keine'}")
    
    # Test 4: VPs mit fehlgeschlagenen Runs
    vps_failed = manager.scan_vps_with_failed_runs(test_config)
    print(f"\n[4] VPs mit fehlgeschlagenen Runs:")
    print(f"    {len(vps_failed)} VPs: {', '.join(vps_failed) if vps_failed else 'Keine'}")
    
    # Test 5: Lade erste VP und zeige Cache-Uebersicht
    if all_vps:
        test_vp = all_vps[0]
        print(f"\n[5] Cache-Test fuer VP: {test_vp}")
        
        vp_data = manager.load_vp_data(test_vp)
        manager.print_vp_summary(vp_data)
        
        # Cache-Uebersicht
        cache_overview = manager.get_cache_overview(vp_data, test_config)
        manager.print_cache_overview(cache_overview)
        
        # Analyse-Status
        status = manager.get_vp_analysis_status(test_vp, test_config)
        print(f"\n[6] Analyse-Status: {status['status']}")
        print(f"    {status['message']}")
    
    print(f"\n{'='*70}")
    print("TEST ABGESCHLOSSEN")
    print(f"{'='*70}")
