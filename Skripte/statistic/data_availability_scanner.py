"""
=================================================================================
DATA AVAILABILITY SCANNER v1.0 - Prueft vorhandene Analysedaten
=================================================================================

Funktionen:
-----------
- Scannt alle VP-Ordner nach vorhandenen Analysen
- Prueft Vollstaendigkeit pro Konfiguration (video_fps x calib_mode)
- Identifiziert fehlende Daten fuer statistische Analyse
- Generiert Uebersichts-Report

Verwendung:
-----------
    scanner = DataAvailabilityScanner(base_folder)
    report = scanner.scan_all()
    report.print_summary()

Version: 1.0
Datum: 2025-01
=================================================================================
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import json
from datetime import datetime

# Path Setup
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.vp_data_manager import VPDataManager, VPData, RunConfig, STEP_OUTPUT_FILES

# =================================================================================
# KONFIGURATION
# =================================================================================

# Kalibrierungskonfigurationen fuer statistische Analyse (K1-K5)
STATISTICAL_CALIB_CONFIGS = {
    'K1': {
        'calib_mode': 'FullCalib',
        'description': 'Beg(20) + Mid(10) + End(10) + Fix(24) = 64 Punkte',
        'expected_points': 64
    },
    'K2': {
        'calib_mode': 'OnlyBegFix',
        'description': 'Beg(20) + Fix(24) = 44 Punkte',
        'expected_points': 44
    },
    'K3': {
        'calib_mode': 'BegEnd',
        'description': 'Beg(20) + End(10) = 30 Punkte',
        'expected_points': 30
    },
    'K4': {
        'calib_mode': 'BegFirst10EndFix',
        'description': 'BegFirst(10) + End(10) + Fix(24) = 44 Punkte',
        'expected_points': 44
    },
    'K5': {
        'calib_mode': 'BegFirst10End',
        'description': 'BegFirst(10) + End(10) = 20 Punkte',
        'expected_points': 20
    },
}

# Zusaetzliche Kalibrierungsmodi (fuer Vollstaendigkeit)
EXTRA_CALIB_CONFIGS = {
    'Extra1': {
        'calib_mode': 'OnlyBeg',
        'description': 'Nur Beg(20) = 20 Punkte',
        'expected_points': 20
    },
    'Extra2': {
        'calib_mode': 'BegSecond10Mid',
        'description': 'Beg(11-20) + Mid(10) = 20 Punkte',
        'expected_points': 20
    },
}

# Benoetigte Dateien fuer statistische Analyse
REQUIRED_FILES_FOR_STATISTICS = {
    'debug_5_mediapipe': 'debug_5_pupil_data_calibrated.csv',
    'debug_5_ptgaze': 'debug_5_ptgaze_calibrated.csv',
    'debug_3_eyelink': 'debug_3_eyetracker_data.csv',
}

# Framerates
FRAMERATES = ['25hz', '60hz']

# =================================================================================
# DATENSTRUKTUREN
# =================================================================================

@dataclass
class ConfigStatus:
    """Status einer einzelnen Konfiguration (video_fps x calib_mode)."""
    video_fps: str
    calib_mode: str
    calib_label: str  # K1, K2, etc.
    
    # Dateistatus
    has_mediapipe: bool = False
    has_ptgaze: bool = False
    has_eyelink: bool = False
    
    # Pfade (falls vorhanden)
    mediapipe_path: Optional[Path] = None
    ptgaze_path: Optional[Path] = None
    eyelink_path: Optional[Path] = None
    run_folder: Optional[Path] = None
    
    # Qualitaet
    run_successful: bool = False
    error_message: Optional[str] = None
    
    @property
    def is_complete(self) -> bool:
        """Prueft ob alle drei Datenquellen vorhanden sind."""
        return self.has_mediapipe and self.has_ptgaze and self.has_eyelink
    
    @property
    def is_partial(self) -> bool:
        """Prueft ob mindestens eine Datenquelle vorhanden ist."""
        return self.has_mediapipe or self.has_ptgaze or self.has_eyelink
    
    @property
    def missing_sources(self) -> List[str]:
        """Liste der fehlenden Datenquellen."""
        missing = []
        if not self.has_mediapipe:
            missing.append('MediaPipe')
        if not self.has_ptgaze:
            missing.append('ptgaze')
        if not self.has_eyelink:
            missing.append('EyeLink')
        return missing


@dataclass
class VPStatus:
    """Status einer VP ueber alle Konfigurationen."""
    vp_code: str
    vp_folder: Path
    
    # Videos vorhanden
    has_25hz: bool = False
    has_60hz: bool = False
    has_asc: bool = False
    
    # Status pro Konfiguration
    configs: Dict[str, ConfigStatus] = field(default_factory=dict)
    
    @property
    def available_framerates(self) -> List[str]:
        """Liste der verfuegbaren Framerates."""
        available = []
        if self.has_25hz:
            available.append('25hz')
        if self.has_60hz:
            available.append('60hz')
        return available
    
    @property
    def n_complete_configs(self) -> int:
        """Anzahl vollstaendiger Konfigurationen."""
        return sum(1 for c in self.configs.values() if c.is_complete)
    
    @property
    def n_partial_configs(self) -> int:
        """Anzahl teilweise vorhandener Konfigurationen."""
        return sum(1 for c in self.configs.values() if c.is_partial and not c.is_complete)


@dataclass
class DataAvailabilityReport:
    """Gesamtbericht ueber verfuegbare Daten."""
    scan_timestamp: str
    base_folder: Path
    
    # VP-Status
    vp_statuses: Dict[str, VPStatus] = field(default_factory=dict)
    
    # Aggregierte Statistiken
    n_vps_total: int = 0
    n_vps_with_25hz: int = 0
    n_vps_with_60hz: int = 0
    n_vps_with_both: int = 0
    
    # Config-basierte Statistiken
    configs_complete: Dict[str, int] = field(default_factory=dict)  # config_key -> Anzahl VPs
    configs_partial: Dict[str, int] = field(default_factory=dict)
    configs_missing: Dict[str, int] = field(default_factory=dict)
    
    def get_vps_for_config(self, video_fps: str, calib_mode: str, 
                          complete_only: bool = True) -> List[str]:
        """
        Gibt VPs zurueck, die eine bestimmte Konfiguration haben.
        
        Args:
            video_fps: '25hz' oder '60hz'
            calib_mode: z.B. 'FullCalib'
            complete_only: Nur vollstaendige Konfigurationen
        
        Returns:
            Liste von VP-Codes
        """
        config_key = f"{video_fps}_{calib_mode}"
        result = []
        
        for vp_code, vp_status in self.vp_statuses.items():
            if config_key in vp_status.configs:
                config_status = vp_status.configs[config_key]
                if complete_only:
                    if config_status.is_complete:
                        result.append(vp_code)
                else:
                    if config_status.is_partial:
                        result.append(vp_code)
        
        return result
    
    def get_vps_missing_config(self, video_fps: str, calib_mode: str) -> List[str]:
        """
        Gibt VPs zurueck, denen eine Konfiguration fehlt (aber Video vorhanden).
        """
        config_key = f"{video_fps}_{calib_mode}"
        result = []
        
        for vp_code, vp_status in self.vp_statuses.items():
            # Pruefe ob Framerate vorhanden
            if video_fps == '25hz' and not vp_status.has_25hz:
                continue
            if video_fps == '60hz' and not vp_status.has_60hz:
                continue
            
            # Pruefe ob Config fehlt oder unvollstaendig
            if config_key not in vp_status.configs:
                result.append(vp_code)
            elif not vp_status.configs[config_key].is_complete:
                result.append(vp_code)
        
        return result
    
    def get_analysis_feasibility(self) -> Dict[str, Dict]:
        """
        Prueft welche Forschungsfragen mit aktuellen Daten beantwortbar sind.
        
        Returns:
            Dict mit FF -> {feasible, n_vps, missing, message}
        """
        feasibility = {}
        
        # FF1/FF2: CV vs. EyeLink Uebereinstimmung, MediaPipe vs. ptgaze
        # Braucht: Mindestens eine vollstaendige Config
        vps_with_any_complete = set()
        for vp_code, vp_status in self.vp_statuses.items():
            if vp_status.n_complete_configs > 0:
                vps_with_any_complete.add(vp_code)
        
        feasibility['FF1_FF2'] = {
            'name': 'CV-EyeLink Uebereinstimmung & Methodenvergleich',
            'feasible': len(vps_with_any_complete) >= 5,
            'n_vps': len(vps_with_any_complete),
            'min_required': 5,
            'message': f"{len(vps_with_any_complete)} VPs mit vollstaendigen Daten"
        }
        
        # FF3: Framerate-Effekt
        # Braucht: VPs mit BEIDEN Framerates (25hz UND 60hz)
        vps_with_both = []
        for vp_code, vp_status in self.vp_statuses.items():
            has_25hz_complete = any(
                c.is_complete for k, c in vp_status.configs.items() if '25hz' in k
            )
            has_60hz_complete = any(
                c.is_complete for k, c in vp_status.configs.items() if '60hz' in k
            )
            if has_25hz_complete and has_60hz_complete:
                vps_with_both.append(vp_code)
        
        feasibility['FF3'] = {
            'name': 'Framerate-Effekt (25Hz vs. 60Hz)',
            'feasible': len(vps_with_both) >= 5,
            'n_vps': len(vps_with_both),
            'min_required': 5,
            'message': f"{len(vps_with_both)} VPs mit beiden Framerates"
        }
        
        # FF4: Kalibrierungsvergleich (TOST)
        # Braucht: Mehrere Kalibrierungskonfigurationen pro VP
        vps_with_multi_calib = []
        for vp_code, vp_status in self.vp_statuses.items():
            # Zaehle verschiedene calib_modes (bei gleicher framerate)
            for fps in ['25hz', '60hz']:
                calib_modes_complete = set()
                for config_key, config_status in vp_status.configs.items():
                    if fps in config_key and config_status.is_complete:
                        calib_modes_complete.add(config_status.calib_mode)
                if len(calib_modes_complete) >= 2:
                    vps_with_multi_calib.append(vp_code)
                    break
        
        feasibility['FF4'] = {
            'name': 'Kalibrierungsvergleich (K1-K5)',
            'feasible': len(vps_with_multi_calib) >= 5,
            'n_vps': len(vps_with_multi_calib),
            'min_required': 5,
            'message': f"{len(vps_with_multi_calib)} VPs mit mehreren Kalibrierungen"
        }
        
        # FF5: Trajektorienanalyse
        # Gleiche Anforderung wie FF1/FF2 (Frame-Level-Daten)
        feasibility['FF5'] = {
            'name': 'Trajektorienanalyse (RMSE, Korrelation)',
            'feasible': len(vps_with_any_complete) >= 5,
            'n_vps': len(vps_with_any_complete),
            'min_required': 5,
            'message': f"{len(vps_with_any_complete)} VPs mit Frame-Level-Daten"
        }
        
        return feasibility
    
    def print_summary(self):
        """Gibt formatierte Zusammenfassung aus."""
        print(f"\n{'='*70}")
        print("DATEN-VERFUEGBARKEITS-REPORT")
        print(f"{'='*70}")
        print(f"Scan: {self.scan_timestamp}")
        print(f"Basis: {self.base_folder}")
        
        print(f"\n[VP-UEBERSICHT]")
        print(f"  Gesamt:        {self.n_vps_total} VPs")
        print(f"  Mit 25Hz:      {self.n_vps_with_25hz}")
        print(f"  Mit 60Hz:      {self.n_vps_with_60hz}")
        print(f"  Mit beiden:    {self.n_vps_with_both}")
        
        print(f"\n[KONFIGURATIONEN]")
        
        # Tabelle: Config x Anzahl VPs
        print(f"\n  {'Config':<20} {'25Hz':<10} {'60Hz':<10}")
        print(f"  {'-'*40}")
        
        for k_label, k_info in STATISTICAL_CALIB_CONFIGS.items():
            calib_mode = k_info['calib_mode']
            n_25hz = len(self.get_vps_for_config('25hz', calib_mode))
            n_60hz = len(self.get_vps_for_config('60hz', calib_mode))
            print(f"  {k_label} ({calib_mode})"[:20].ljust(20) + f" {n_25hz:<10} {n_60hz:<10}")
        
        print(f"\n[FORSCHUNGSFRAGEN - MACHBARKEIT]")
        feasibility = self.get_analysis_feasibility()
        
        for ff_key, ff_info in feasibility.items():
            status = "[OK]" if ff_info['feasible'] else "[!!]"
            print(f"  {status} {ff_key}: {ff_info['name']}")
            print(f"       {ff_info['message']} (min. {ff_info['min_required']} benoetigt)")
        
        print(f"\n{'='*70}")
    
    def to_dict(self) -> Dict:
        """Konvertiert Report zu Dictionary (fuer JSON-Export)."""
        return {
            'scan_timestamp': self.scan_timestamp,
            'base_folder': str(self.base_folder),
            'summary': {
                'n_vps_total': self.n_vps_total,
                'n_vps_with_25hz': self.n_vps_with_25hz,
                'n_vps_with_60hz': self.n_vps_with_60hz,
                'n_vps_with_both': self.n_vps_with_both,
            },
            'feasibility': self.get_analysis_feasibility(),
            'vp_details': {
                vp_code: {
                    'has_25hz': vp_status.has_25hz,
                    'has_60hz': vp_status.has_60hz,
                    'has_asc': vp_status.has_asc,
                    'n_complete_configs': vp_status.n_complete_configs,
                    'configs': {
                        k: {
                            'is_complete': v.is_complete,
                            'has_mediapipe': v.has_mediapipe,
                            'has_ptgaze': v.has_ptgaze,
                            'has_eyelink': v.has_eyelink,
                        }
                        for k, v in vp_status.configs.items()
                    }
                }
                for vp_code, vp_status in self.vp_statuses.items()
            }
        }
    
    def save_json(self, output_path: Path):
        """Speichert Report als JSON."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# =================================================================================
# DATA AVAILABILITY SCANNER
# =================================================================================

class DataAvailabilityScanner:
    """
    Scannt Ergebnis-Ordner nach vorhandenen Analysedaten.
    
    Verwendung:
    -----------
        scanner = DataAvailabilityScanner(base_folder)
        report = scanner.scan_all()
        
        # Welche VPs haben 25Hz FullCalib?
        vps = report.get_vps_for_config('25hz', 'FullCalib')
        
        # Welche Forschungsfragen sind machbar?
        feasibility = report.get_analysis_feasibility()
    """
    
    def __init__(self, base_folder: Path):
        """
        Args:
            base_folder: Basis-Ordner (z.B. .../Ergebnisse)
        """
        self.base_folder = Path(base_folder)
        self.vp_manager = VPDataManager(base_folder)
        
        # Alle bekannten Kalibrierungsmodi
        self.all_calib_modes = {**STATISTICAL_CALIB_CONFIGS, **EXTRA_CALIB_CONFIGS}
    
    def scan_all(self) -> DataAvailabilityReport:
        """
        Scannt alle VPs und erstellt vollstaendigen Report.
        
        Returns:
            DataAvailabilityReport
        """
        report = DataAvailabilityReport(
            scan_timestamp=datetime.now().isoformat(),
            base_folder=self.base_folder
        )
        
        # Finde alle VP-Ordner
        all_vp_codes = self.vp_manager.scan_all_vps()
        report.n_vps_total = len(all_vp_codes)
        
        print(f"[SCAN] Scanne {len(all_vp_codes)} VPs...")
        
        for vp_code in all_vp_codes:
            try:
                vp_status = self._scan_vp(vp_code)
                report.vp_statuses[vp_code] = vp_status
                
                # Update Statistiken
                if vp_status.has_25hz:
                    report.n_vps_with_25hz += 1
                if vp_status.has_60hz:
                    report.n_vps_with_60hz += 1
                if vp_status.has_25hz and vp_status.has_60hz:
                    report.n_vps_with_both += 1
                    
            except Exception as e:
                print(f"  [!] Fehler bei VP {vp_code}: {e}")
        
        # Update Config-Statistiken
        for k_label, k_info in self.all_calib_modes.items():
            calib_mode = k_info['calib_mode']
            for fps in FRAMERATES:
                config_key = f"{fps}_{calib_mode}"
                
                n_complete = len(report.get_vps_for_config(fps, calib_mode, complete_only=True))
                n_partial = len(report.get_vps_for_config(fps, calib_mode, complete_only=False)) - n_complete
                n_missing = len(report.get_vps_missing_config(fps, calib_mode))
                
                report.configs_complete[config_key] = n_complete
                report.configs_partial[config_key] = n_partial
                report.configs_missing[config_key] = n_missing
        
        print(f"[SCAN] Abgeschlossen: {report.n_vps_total} VPs gescannt")
        
        return report
    
    def _scan_vp(self, vp_code: str) -> VPStatus:
        """Scannt eine einzelne VP."""
        
        vp_data = self.vp_manager.load_vp_data(vp_code)
        
        vp_status = VPStatus(
            vp_code=vp_code,
            vp_folder=vp_data.vp_folder,
            has_25hz=vp_data.videos.get('25hz') is not None,
            has_60hz=vp_data.videos.get('60hz') is not None,
            has_asc=vp_data.asc_file is not None
        )
        
        # Scanne Analyse-Ordner
        analysis_folder = vp_data.vp_folder / 'Analyse'
        
        if not analysis_folder.exists():
            return vp_status
        
        # Finde alle Run_* Ordner
        run_folders = sorted(analysis_folder.glob('Run_*'), key=lambda p: p.stat().st_mtime, reverse=True)
        
        for run_folder in run_folders:
            config = self._read_run_config(run_folder)
            
            if config is None:
                continue
            
            video_fps = config.get('video_fps', 'unknown')
            calib_mode = config.get('calib_mode', 'unknown')
            config_key = f"{video_fps}_{calib_mode}"
            
            # Finde K-Label
            k_label = self._get_k_label(calib_mode)
            
            # Pruefe ob wir diese Config schon haben (neueste gewinnt)
            if config_key in vp_status.configs:
                continue
            
            # Pruefe vorhandene Dateien
            config_status = self._check_run_files(run_folder, video_fps, calib_mode, k_label)
            
            vp_status.configs[config_key] = config_status
        
        return vp_status
    
    def _read_run_config(self, run_folder: Path) -> Optional[Dict]:
        """Liest config_used.json aus einem Run-Ordner."""
        config_file = run_folder / 'config_used.json'
        
        if not config_file.exists():
            return None
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def _check_run_files(self, run_folder: Path, video_fps: str, 
                        calib_mode: str, k_label: str) -> ConfigStatus:
        """Prueft welche Dateien in einem Run-Ordner vorhanden sind."""
        
        config_status = ConfigStatus(
            video_fps=video_fps,
            calib_mode=calib_mode,
            calib_label=k_label,
            run_folder=run_folder
        )
        
        # Pruefe MediaPipe
        mediapipe_file = run_folder / 'debug_5_pupil_data_calibrated.csv'
        if mediapipe_file.exists():
            config_status.has_mediapipe = True
            config_status.mediapipe_path = mediapipe_file
        
        # Pruefe ptgaze
        ptgaze_file = run_folder / 'debug_5_ptgaze_calibrated.csv'
        if ptgaze_file.exists():
            config_status.has_ptgaze = True
            config_status.ptgaze_path = ptgaze_file
        
        # Pruefe EyeLink
        eyelink_file = run_folder / 'debug_3_eyetracker_data.csv'
        if eyelink_file.exists():
            config_status.has_eyelink = True
            config_status.eyelink_path = eyelink_file
        
        # Pruefe Success-Marker
        success_file = run_folder / 'run_success.json'
        config_status.run_successful = success_file.exists()
        
        # Pruefe Error-Log
        error_file = run_folder / 'error_log.json'
        if error_file.exists():
            try:
                with open(error_file, 'r', encoding='utf-8') as f:
                    error_data = json.load(f)
                    config_status.error_message = error_data.get('error_message', 'Unbekannter Fehler')
            except:
                pass
        
        return config_status
    
    def _get_k_label(self, calib_mode: str) -> str:
        """Gibt das K-Label fuer einen calib_mode zurueck."""
        for k_label, k_info in self.all_calib_modes.items():
            if k_info['calib_mode'] == calib_mode:
                return k_label
        return 'Unknown'
    
    def get_missing_for_ff(self, ff: str) -> Dict[str, List[str]]:
        """
        Gibt fehlende Daten fuer eine Forschungsfrage zurueck.
        
        Args:
            ff: 'FF1_FF2', 'FF3', 'FF4', 'FF5'
        
        Returns:
            Dict mit 'vps_missing', 'configs_needed', 'actions'
        """
        report = self.scan_all()
        
        missing = {
            'vps_missing': [],
            'configs_needed': [],
            'actions': []
        }
        
        if ff in ['FF1_FF2', 'FF5']:
            # Braucht mindestens eine vollstaendige Config
            for vp_code, vp_status in report.vp_statuses.items():
                if vp_status.n_complete_configs == 0:
                    # Hat VP ueberhaupt Videos?
                    if vp_status.has_25hz or vp_status.has_60hz:
                        missing['vps_missing'].append(vp_code)
                        fps = '25hz' if vp_status.has_25hz else '60hz'
                        missing['configs_needed'].append(f"{vp_code}: {fps}_FullCalib")
                        missing['actions'].append(
                            f"Pipeline ausfuehren: {vp_code} mit {fps}, FullCalib"
                        )
        
        elif ff == 'FF3':
            # Braucht beide Framerates
            for vp_code, vp_status in report.vp_statuses.items():
                has_25hz_complete = any(
                    c.is_complete for k, c in vp_status.configs.items() if '25hz' in k
                )
                has_60hz_complete = any(
                    c.is_complete for k, c in vp_status.configs.items() if '60hz' in k
                )
                
                if vp_status.has_25hz and vp_status.has_60hz:
                    if not has_25hz_complete:
                        missing['configs_needed'].append(f"{vp_code}: 25hz_FullCalib")
                        missing['actions'].append(
                            f"Pipeline ausfuehren: {vp_code} mit 25hz, FullCalib"
                        )
                    if not has_60hz_complete:
                        missing['configs_needed'].append(f"{vp_code}: 60hz_FullCalib")
                        missing['actions'].append(
                            f"Pipeline ausfuehren: {vp_code} mit 60hz, FullCalib"
                        )
        
        elif ff == 'FF4':
            # Braucht mehrere Kalibrierungsmodi
            for vp_code, vp_status in report.vp_statuses.items():
                for fps in ['25hz', '60hz']:
                    if (fps == '25hz' and not vp_status.has_25hz) or \
                       (fps == '60hz' and not vp_status.has_60hz):
                        continue
                    
                    complete_modes = set()
                    for config_key, config_status in vp_status.configs.items():
                        if fps in config_key and config_status.is_complete:
                            complete_modes.add(config_status.calib_mode)
                    
                    # Fehlende Modi identifizieren
                    for k_label in ['K1', 'K2', 'K3', 'K4', 'K5']:
                        calib_mode = STATISTICAL_CALIB_CONFIGS[k_label]['calib_mode']
                        if calib_mode not in complete_modes:
                            missing['configs_needed'].append(f"{vp_code}: {fps}_{calib_mode}")
        
        return missing


# =================================================================================
# MAIN (TEST)
# =================================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DATA AVAILABILITY SCANNER v1.0 - Test")
    print("="*70)
    
    # Test mit Beispiel-Pfad
    test_folder = Path(r"C:\Users\imanu\Documents\Imanuel\Studium\Psy\Master\Pupillendetektion\Ergebnisse")
    
    if test_folder.exists():
        scanner = DataAvailabilityScanner(test_folder)
        report = scanner.scan_all()
        report.print_summary()
        
        # Speichere Report
        report_path = test_folder / 'data_availability_report.json'
        report.save_json(report_path)
        print(f"\n[OK] Report gespeichert: {report_path}")
    else:
        print(f"\n[!] Test-Ordner nicht gefunden: {test_folder}")
        print("    Bitte Pfad anpassen oder Scanner manuell verwenden.")
