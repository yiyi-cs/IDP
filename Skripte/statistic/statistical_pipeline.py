"""
=================================================================================
STATISTICAL_PIPELINE.PY v1.0 - Orchestrierung der statistischen Analyse
=================================================================================

Kernfunktionen:
---------------
1. Koordiniert data_availability_scanner und statistical_data_prep
2. Ruft bestehende Pipeline für fehlende Daten auf
3. Erstellt VP-übergreifenden Analysedatensatz
4. Generiert Lag-Korrektur-Report

Workflow:
---------
1. Scan: Welche Daten sind vorhanden?
2. Plan: Was fehlt für gewünschte Analyse?
3. Generate: Fehlende Daten mit bestehender Pipeline erzeugen
4. Prepare: Analysedatensatz erstellen
5. Export: CSV + R-ready Format

Version: 1.0
Datum: 2025-01
=================================================================================
"""

# =================================================================================
# PATH SETUP
# =================================================================================
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# =================================================================================
# IMPORTS
# =================================================================================
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import json
from datetime import datetime

from config import N_TRIALS, FIRST_TRIAL_IS_PRACTICE

from statistic.data_availability_scanner import (
    DataAvailabilityScanner,
    DataAvailabilityReport,
    STATISTICAL_CALIB_CONFIGS,
    FRAMERATES
)
from statistic.statistical_data_prep import (
    StatisticalDataPrep,
    DataPrepResult,
    LagCorrectionResult
)

from shared.shared_coordinate_utils import (
    batch_convert_to_degrees
)

from utils.vp_data_manager import VPDataManager, VPData, RunConfig
from utils.analysis_pipeline import AnalysisPipeline, RunConfiguration, RunResult

# =================================================================================
# HILFSFUNKTIONEN
# =================================================================================

def convert_numpy_types(obj):
    """
    Konvertiert NumPy-Typen zu nativen Python-Typen für JSON-Serialisierung.
    
    Löst: TypeError: Object of type int64 is not JSON serializable
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif hasattr(obj, 'item'):  # NumPy Skalar
        return obj.item()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj

# =================================================================================
# KONFIGURATION
# =================================================================================

# Practice-Trial ausschließen
EXCLUDE_PRACTICE_TRIAL = True
PRACTICE_TRIAL_NUMBER = 0

# Mindestanzahl VPs für Analyse
MIN_VPS_FOR_ANALYSIS = 5

# Erlaubte Kalibrierungsmodi für statistische Analyse
# Nur diese werden verarbeitet (Variante B: Filterung in Pipeline)
ALLOWED_CALIB_CONFIGS = ['K1', 'K4']  # FullCalib, BegFirst10EndFix
ALLOWED_CALIB_MODES = ['FullCalib', 'BegFirst10EndFix']  # Entsprechende calib_mode Werte

# Output-Dateien
OUTPUT_FRAME_LEVEL_CSV = 'analysis_frame_level.csv'
OUTPUT_TRIAL_LEVEL_CSV = 'analysis_trial_level.csv'
OUTPUT_LAG_REPORT_JSON = 'lag_correction_report.json'
OUTPUT_PREP_REPORT_JSON = 'data_preparation_report.json'

# =================================================================================
# DATENSTRUKTUREN
# =================================================================================

@dataclass
class AnalysisPlan:
    """Plan für statistische Analyse."""
    
    # Gewünschte Konfiguration
    framerates: List[str]           # ['25hz'], ['60hz'], oder ['25hz', '60hz']
    calib_configs: List[str]        # ['K1'], ['K1', 'K2'], etc.
    methods: List[str]              # ['mediapipe', 'ptgaze']
    
    # Verfügbare Daten
    available_vps: Dict[str, List[str]]   # config_key -> [vp_codes]
    
    # Fehlende Daten
    missing_data: Dict[str, List[str]]    # config_key -> [vp_codes die fehlen]
    
    # Machbare Forschungsfragen
    feasible_ffs: List[str]
    infeasible_ffs: Dict[str, str]        # FF -> Grund
    
    # Statistik
    n_vps_total: int = 0
    n_configs_complete: int = 0
    n_configs_missing: int = 0
    
    def get_summary(self) -> str:
        """Gibt formatierte Zusammenfassung zurück."""
        lines = [
            "ANALYSE-PLAN",
            "=" * 50,
            f"Framerates: {', '.join(self.framerates)}",
            f"Kalibrierungen: {', '.join(self.calib_configs)}",
            f"Methoden: {', '.join(self.methods)}",
            "",
            f"VPs gesamt: {self.n_vps_total}",
            f"Configs komplett: {self.n_configs_complete}",
            f"Configs fehlend: {self.n_configs_missing}",
            "",
            "Machbare Forschungsfragen:",
        ]
        
        for ff in self.feasible_ffs:
            lines.append(f"  [OK] {ff}")
        
        if self.infeasible_ffs:
            lines.append("\nNicht machbare Forschungsfragen:")
            for ff, reason in self.infeasible_ffs.items():
                lines.append(f"  [!!] {ff}: {reason}")
        
        return "\n".join(lines)


@dataclass
class PipelineResult:
    """Ergebnis der statistischen Pipeline."""
    
    success: bool
    
    # Output-Dateien
    frame_level_csv: Optional[Path] = None
    trial_level_csv: Optional[Path] = None
    lag_report_json: Optional[Path] = None
    prep_report_json: Optional[Path] = None
    
    # Statistik
    n_vps_processed: int = 0
    n_vps_failed: int = 0
    n_frames_total: int = 0
    n_trials_total: int = 0
    
    # Lag-Korrektur
    lag_results: List[LagCorrectionResult] = field(default_factory=list)
    
    # Fehler
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Timing
    duration_seconds: float = 0.0

# =================================================================================
# EYELINK CACHE SYSTEM
# =================================================================================

@dataclass
class CachedEyeLinkData:
    """Gecachte EyeLink-Daten für eine VP."""
    vp_id: str
    source_path: Path
    full_rate_df: pd.DataFrame
    downsampled_25hz: Optional[pd.DataFrame] = None
    downsampled_60hz: Optional[pd.DataFrame] = None
    n_samples_original: int = 0
    n_samples_valid: int = 0

class EyeLinkCache:
    """
    Zentraler Cache für EyeLink-Daten.
    
    Lädt und konvertiert EyeLink-Daten nur EINMAL pro VP,
    unabhängig davon wie viele Run-Konfigurationen verarbeitet werden.
    """
    
    def __init__(self, transformer):
        self.transformer = transformer
        self._cache: Dict[str, CachedEyeLinkData] = {}
        self._eyelink_source_paths: Dict[str, Path] = {}
    
    def get_for_vp(self, vp_id: str, run_folder: Path = None) -> Optional[CachedEyeLinkData]:
        """Holt EyeLink-Daten für eine VP (lädt bei Bedarf)."""
        
        if vp_id in self._cache:
            print(f"  [CACHE] EyeLink für {vp_id}: Aus Cache")
            return self._cache[vp_id]
        
        eyelink_path = self._find_eyelink_path(vp_id, run_folder)
        
        if eyelink_path is None:
            print(f"  [CACHE] EyeLink für {vp_id}: Nicht gefunden")
            return None
        
        print(f"  [CACHE] EyeLink für {vp_id}: Lade und konvertiere...")
        
        try:
            eyelink_df = pd.read_csv(eyelink_path)
            n_original = len(eyelink_df)
            
            deg_x, deg_y = batch_convert_to_degrees(
                eyelink_df['x_pos'].values,
                eyelink_df['y_pos'].values,
                self.transformer
            )
            
            eyelink_df['eyelink_deg_x'] = deg_x
            eyelink_df['eyelink_deg_y'] = deg_y
            
            n_valid = eyelink_df['eyelink_deg_x'].notna().sum()
            
            cached = CachedEyeLinkData(
                vp_id=vp_id,
                source_path=eyelink_path,
                full_rate_df=eyelink_df,
                n_samples_original=n_original,
                n_samples_valid=n_valid
            )
            
            self._cache[vp_id] = cached
            
            print(f"           Samples: {n_original}, Valide: {n_valid} ({n_valid/n_original*100:.1f}%)")
            
            return cached
            
        except Exception as e:
            print(f"  [CACHE] Fehler beim Laden für {vp_id}: {e}")
            return None
    
    def get_downsampled(self, vp_id: str, target_timestamps_ms: np.ndarray,
                       fps_key: str, run_folder: Path = None) -> Optional[pd.DataFrame]:
        """Holt downgesampelte EyeLink-Daten für eine VP."""
        
        cached = self.get_for_vp(vp_id, run_folder)
        
        if cached is None:
            return None
        
        if fps_key == '25hz' and cached.downsampled_25hz is not None:
            print(f"  [CACHE] Downsampled {fps_key} für {vp_id}: Aus Cache")
            return cached.downsampled_25hz
        elif fps_key == '60hz' and cached.downsampled_60hz is not None:
            print(f"  [CACHE] Downsampled {fps_key} für {vp_id}: Aus Cache")
            return cached.downsampled_60hz
        
        print(f"  [CACHE] Downsampled {fps_key} für {vp_id}: Erstelle neu...")
        
        downsampled = self._downsample(cached.full_rate_df, target_timestamps_ms)
        
        if fps_key == '25hz':
            cached.downsampled_25hz = downsampled
        elif fps_key == '60hz':
            cached.downsampled_60hz = downsampled
        
        return downsampled
    
    def _find_eyelink_path(self, vp_id: str, run_folder: Path = None) -> Optional[Path]:
        """Findet den Pfad zur EyeLink-Datei."""
        
        if vp_id in self._eyelink_source_paths:
            path = self._eyelink_source_paths[vp_id]
            if path.exists():
                return path
        
        if run_folder is not None:
            path = run_folder / 'debug_3_eyetracker_data.csv'
            if path.exists():
                self._eyelink_source_paths[vp_id] = path
                return path
            
            analysis_folder = run_folder.parent
            for other_run in analysis_folder.glob('Run_*'):
                path = other_run / 'debug_3_eyetracker_data.csv'
                if path.exists():
                    self._eyelink_source_paths[vp_id] = path
                    print(f"  [CACHE] EyeLink gefunden in: {other_run.name}")
                    return path
        
        return None
    
    def _downsample(self, full_df: pd.DataFrame, 
                   target_timestamps_ms: np.ndarray) -> pd.DataFrame:
        """Downsampled EyeLink-Daten auf Ziel-Timestamps (Nearest-Neighbor)."""
        
        sorted_df = full_df.sort_values('timestamp_ms').reset_index(drop=True)
        
        result = {
            'timestamp_ms': target_timestamps_ms,
            'eyelink_deg_x': np.full(len(target_timestamps_ms), np.nan),
            'eyelink_deg_y': np.full(len(target_timestamps_ms), np.nan),
            'eyelink_valid': np.zeros(len(target_timestamps_ms), dtype=bool)
        }
        
        eyelink_times = sorted_df['timestamp_ms'].values
        
        for i, target_ts in enumerate(target_timestamps_ms):
            idx = np.searchsorted(eyelink_times, target_ts)
            
            if idx == 0:
                nearest_idx = 0
            elif idx >= len(eyelink_times):
                nearest_idx = len(eyelink_times) - 1
            else:
                if abs(eyelink_times[idx] - target_ts) < abs(eyelink_times[idx-1] - target_ts):
                    nearest_idx = idx
                else:
                    nearest_idx = idx - 1
            
            if abs(eyelink_times[nearest_idx] - target_ts) <= 2.0:
                x_val = sorted_df.iloc[nearest_idx]['eyelink_deg_x']
                y_val = sorted_df.iloc[nearest_idx]['eyelink_deg_y']
                result['eyelink_deg_x'][i] = x_val
                result['eyelink_deg_y'][i] = y_val
                # FIX: Nur valide wenn Wert selbst nicht NaN ist
                result['eyelink_valid'][i] = not (pd.isna(x_val) or pd.isna(y_val))

        result_df = pd.DataFrame(result)
        n_valid = result_df['eyelink_valid'].sum()
        print(f"           Downsampled: {len(result_df)} Frames, Valide: {n_valid} ({n_valid/len(result_df)*100:.1f}%)")
        
        return result_df
    
    def get_trial_aggregates(self, vp_id: str, trial_boundaries: Dict[int, Tuple[float, float]],
                            run_folder: Path = None) -> pd.DataFrame:
        """
        Berechnet EyeLink Trial-Aggregate direkt aus Full-Rate-Daten.
        
        Dies garantiert identische EyeLink-Mittelwerte für 25Hz und 60Hz,
        da alle ~7000 Samples pro Trial verwendet werden (nicht nur 175/420).
        
        Args:
            vp_id: VP-ID
            trial_boundaries: Dict[trial_number] -> (start_ms, end_ms)
            run_folder: Fallback für Pfad-Suche
        
        Returns:
            DataFrame mit: trial, eyelink_mean_x, eyelink_mean_y, 
                          eyelink_std_x, eyelink_std_y, n_eyelink_samples
        """
        cached = self.get_for_vp(vp_id, run_folder)
        
        if cached is None:
            return pd.DataFrame()
        
        full_df = cached.full_rate_df
        
        results = []
        
        for trial_num, (start_ms, end_ms) in trial_boundaries.items():
            # Filtere EyeLink-Daten für dieses Trial
            mask = (
                (full_df['timestamp_ms'] >= start_ms) &
                (full_df['timestamp_ms'] <= end_ms) &
                (full_df['eyelink_deg_x'].notna()) &
                (full_df['eyelink_deg_y'].notna())
            )
            
            trial_data = full_df.loc[mask]
            n_samples = len(trial_data)
            
            if n_samples > 0:
                results.append({
                    'trial_assignment': trial_num,
                    'eyelink_mean_x_fullrate': trial_data['eyelink_deg_x'].mean(),
                    'eyelink_mean_y_fullrate': trial_data['eyelink_deg_y'].mean(),
                    'eyelink_std_x_fullrate': trial_data['eyelink_deg_x'].std(),
                    'eyelink_std_y_fullrate': trial_data['eyelink_deg_y'].std(),
                    'n_eyelink_samples_fullrate': n_samples
                })
            else:
                results.append({
                    'trial_assignment': trial_num,
                    'eyelink_mean_x_fullrate': np.nan,
                    'eyelink_mean_y_fullrate': np.nan,
                    'eyelink_std_x_fullrate': np.nan,
                    'eyelink_std_y_fullrate': np.nan,
                    'n_eyelink_samples_fullrate': 0
                })
        
        return pd.DataFrame(results)
    
    def extract_trial_boundaries(self, frame_df: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
        """
        Extrahiert Trial-Grenzen aus Frame-Level-Daten.
        
        Args:
            frame_df: Frame-Level DataFrame mit trial_assignment und timestamp_ms_synced
        
        Returns:
            Dict[trial_number] -> (start_ms, end_ms)
        """
        boundaries = {}
        
        for trial in frame_df['trial_assignment'].unique():
            trial_data = frame_df[frame_df['trial_assignment'] == trial]
            start_ms = trial_data['timestamp_ms_synced'].min()
            end_ms = trial_data['timestamp_ms_synced'].max()
            boundaries[trial] = (start_ms, end_ms)
        
        return boundaries


# =================================================================================
# STATISTICAL PIPELINE
# =================================================================================

class StatisticalPipeline:
    """
    Orchestriert die statistische Analyse.
    
    Workflow:
    ---------
    1. create_analysis_plan() - Plant die Analyse
    2. generate_missing_data() - Erzeugt fehlende Daten (optional)
    3. prepare_analysis_dataset() - Erstellt Analysedatensatz
    4. export_for_r() - Exportiert R-ready Format
    
    Kalibrierungsfilterung:
    -----------------------
    Nur K1 (FullCalib) und K4 (BegFirst10EndFix) werden verarbeitet.
    Andere Kalibrierungsmodi (K2, K3, K5) werden ignoriert.
    
    Phasen-Trennung:
    ----------------
    Die Daten enthalten eine `phase_type` Spalte:
    - 'fixation': Fixationskreuzphase (~1-3 Sekunden)
    - 'stimulus': Free Exploration (~7 Sekunden)
    - 'unassigned': Außerhalb der Trials
    
    Für die Hauptanalyse wird nur `phase_type == 'stimulus'` verwendet.
    Die Fixationsdaten werden separat deskriptiv ausgewertet.
    """
    
    def __init__(self, base_folder: Path, scripts_folder: Path):
        """
        Args:
            base_folder: Basis-Ordner (Ergebnisse/)
            scripts_folder: Ordner mit Pipeline-Skripten
        """
        self.base_folder = Path(base_folder)
        self.scripts_folder = Path(scripts_folder)
        
        # Manager initialisieren
        self.vp_manager = VPDataManager(base_folder)
        self.scanner = DataAvailabilityScanner(base_folder)
        self.data_prep = StatisticalDataPrep()
        
        # NEU: Zentraler EyeLink-Cache
        self.eyelink_cache = EyeLinkCache(self.data_prep.transformer)
        
        # Output-Ordner
        self.output_folder = base_folder / 'Statistical_Analysis'
        
        print(f"[INIT] Statistical Pipeline v1.1 (mit EyeLink-Cache)")
        print(f"       Basis: {base_folder}")
        print(f"       Output: {self.output_folder}")

    # =========================================================================
    # SCHRITT 1: ANALYSE PLANEN
    # =========================================================================
    
    def create_analysis_plan(self, 
                            framerates: List[str] = None,
                            calib_configs: List[str] = None,
                            methods: List[str] = None) -> AnalysisPlan:
        """
        Erstellt einen Plan für die statistische Analyse.
        
        Args:
            framerates: Gewünschte Framerates (Default: ['25hz', '60hz'])
            calib_configs: Gewünschte Kalibrierungen (Default: ALLOWED_CALIB_CONFIGS)
            methods: Gewünschte Methoden (Default: ['mediapipe', 'ptgaze'])
        
        Returns:
            AnalysisPlan
        
        Hinweis:
            Nur K1 (FullCalib) und K4 (BegFirst10EndFix) sind erlaubt.
            Andere Kalibrierungen werden ignoriert.
        """
        # Defaults
        if framerates is None:
            framerates = ['25hz', '60hz']  # Beide Framerates als Default
        if calib_configs is None:
            calib_configs = ALLOWED_CALIB_CONFIGS.copy()  # K1 und K4
        if methods is None:
            methods = ['mediapipe', 'ptgaze']
        
        # Filtere auf erlaubte Kalibrierungen
        original_configs = calib_configs.copy()
        calib_configs = [k for k in calib_configs if k in ALLOWED_CALIB_CONFIGS]
        
        filtered_out = set(original_configs) - set(calib_configs)
        if filtered_out:
            print(f"  [INFO] Ignorierte Kalibrierungen (nicht erlaubt): {filtered_out}")
            print(f"         Erlaubt sind nur: {ALLOWED_CALIB_CONFIGS}")
        
        if not calib_configs:
            print(f"  [!!] FEHLER: Keine erlaubten Kalibrierungen ausgewählt!")
            print(f"       Erlaubt sind: {ALLOWED_CALIB_CONFIGS}")
            calib_configs = ALLOWED_CALIB_CONFIGS.copy()  # Fallback
        
        print(f"\n[PLAN] Erstelle Analyse-Plan...")
        print(f"       Framerates: {framerates}")
        print(f"       Kalibrierungen: {calib_configs}")
        print(f"       Methoden: {methods}")
        
        # Scanne verfügbare Daten
        report = self.scanner.scan_all()
        
        # Sammle verfügbare und fehlende Daten
        available_vps = {}
        missing_data = {}
        
        for k_label in calib_configs:
            if k_label not in STATISTICAL_CALIB_CONFIGS:
                print(f"  [!] Unbekannte Kalibrierung: {k_label}")
                continue
            
            calib_mode = STATISTICAL_CALIB_CONFIGS[k_label]['calib_mode']
            
            for fps in framerates:
                config_key = f"{fps}_{k_label}"
                
                # Verfügbare VPs
                vps_complete = report.get_vps_for_config(fps, calib_mode, complete_only=True)
                available_vps[config_key] = vps_complete
                
                # Fehlende VPs (haben Video aber keine Analyse)
                vps_missing = report.get_vps_missing_config(fps, calib_mode)
                missing_data[config_key] = vps_missing
        
        # Prüfe Machbarkeit der Forschungsfragen
        feasible_ffs = []
        infeasible_ffs = {}
        
        # FF1/FF2: Braucht mindestens MIN_VPS_FOR_ANALYSIS VPs
        n_vps_any = len(set().union(*available_vps.values())) if available_vps else 0
        
        if n_vps_any >= MIN_VPS_FOR_ANALYSIS:
            feasible_ffs.extend(['FF1', 'FF2', 'FF5'])
        else:
            infeasible_ffs['FF1'] = f"Nur {n_vps_any} VPs (min. {MIN_VPS_FOR_ANALYSIS})"
            infeasible_ffs['FF2'] = f"Nur {n_vps_any} VPs (min. {MIN_VPS_FOR_ANALYSIS})"
            infeasible_ffs['FF5'] = f"Nur {n_vps_any} VPs (min. {MIN_VPS_FOR_ANALYSIS})"
        
        # FF3: Braucht beide Framerates
        if '25hz' in framerates and '60hz' in framerates:
            vps_25hz = set().union(*[v for k, v in available_vps.items() if '25hz' in k]) if available_vps else set()
            vps_60hz = set().union(*[v for k, v in available_vps.items() if '60hz' in k]) if available_vps else set()
            vps_both = vps_25hz & vps_60hz
            
            if len(vps_both) >= MIN_VPS_FOR_ANALYSIS:
                feasible_ffs.append('FF3')
            else:
                infeasible_ffs['FF3'] = f"Nur {len(vps_both)} VPs mit beiden Framerates"
        else:
            infeasible_ffs['FF3'] = "Nicht beide Framerates ausgewählt"
        
        # FF4: Braucht mehrere Kalibrierungen
        if len(calib_configs) >= 2:
            feasible_ffs.append('FF4')
        else:
            infeasible_ffs['FF4'] = "Nur eine Kalibrierung ausgewählt"
        
        # Statistik
        n_vps_total = report.n_vps_total
        n_configs_complete = sum(len(v) for v in available_vps.values())
        n_configs_missing = sum(len(v) for v in missing_data.values())
        
        plan = AnalysisPlan(
            framerates=framerates,
            calib_configs=calib_configs,
            methods=methods,
            available_vps=available_vps,
            missing_data=missing_data,
            feasible_ffs=feasible_ffs,
            infeasible_ffs=infeasible_ffs,
            n_vps_total=n_vps_total,
            n_configs_complete=n_configs_complete,
            n_configs_missing=n_configs_missing
        )
        
        print(f"\n{plan.get_summary()}")
        
        return plan
    
    # =========================================================================
    # SCHRITT 2: FEHLENDE DATEN GENERIEREN
    # =========================================================================
    
    def generate_missing_data(self, plan: AnalysisPlan,
                             vp_codes: List[str] = None,
                             config_keys: List[str] = None) -> Dict[str, RunResult]:
        """
        Generiert fehlende Daten mit der bestehenden Pipeline.
        
        Args:
            plan: Analyse-Plan
            vp_codes: Spezifische VPs (None = alle fehlenden)
            config_keys: Spezifische Configs (None = alle fehlenden)
        
        Returns:
            Dict[config_key, RunResult]
        """
        results = {}
        
        # Bestimme was generiert werden soll
        to_generate = []
        
        for config_key, missing_vps in plan.missing_data.items():
            if config_keys and config_key not in config_keys:
                continue
            
            # Parse config_key
            parts = config_key.split('_')
            fps = parts[0]
            k_label = parts[1]
            
            if k_label not in STATISTICAL_CALIB_CONFIGS:
                continue
            
            calib_mode = STATISTICAL_CALIB_CONFIGS[k_label]['calib_mode']
            
            for vp_code in missing_vps:
                if vp_codes and vp_code not in vp_codes:
                    continue
                
                to_generate.append({
                    'vp_code': vp_code,
                    'video_fps': fps,
                    'calib_mode': calib_mode,
                    'config_key': config_key
                })
        
        if not to_generate:
            print(f"[INFO] Keine fehlenden Daten zu generieren")
            return results
        
        print(f"\n[GENERATE] {len(to_generate)} Runs zu generieren...")
        
        for i, item in enumerate(to_generate, 1):
            vp_code = item['vp_code']
            config_key = item['config_key']
            
            print(f"\n[{i}/{len(to_generate)}] {vp_code} / {config_key}")
            
            try:
                # Lade VP-Daten
                vp_data = self.vp_manager.load_vp_data(vp_code)
                
                # Erstelle Pipeline
                pipeline = AnalysisPipeline(vp_data, self.scripts_folder)
                
                # Erstelle Config
                config = RunConfiguration(
                    vp_code=vp_code,
                    video_fps=item['video_fps'],
                    calib_mode=item['calib_mode'],
                    create_calibration=True,
                    analysis_mode=2,  # Comparison Mode
                    methods='both'
                )
                
                # Führe Pipeline aus
                result = pipeline.run_full_analysis(config)
                results[f"{vp_code}_{config_key}"] = result
                
                if result.success:
                    print(f"  [OK] Erfolgreich")
                else:
                    print(f"  [!!] Fehlgeschlagen: {result.error_message}")
                
            except Exception as e:
                print(f"  [!!] Fehler: {e}")
                results[f"{vp_code}_{config_key}"] = RunResult(
                    success=False,
                    run_id='error',
                    run_folder=None,
                    vp_code=vp_code,
                    video_fps=item['video_fps'],
                    calib_mode=item['calib_mode'],
                    analysis_mode=2,
                    duration_seconds=0,
                    completed_steps=[],
                    error_message=str(e)
                )
        
        return results
    
    # =========================================================================
    # SCHRITT 3: ANALYSEDATENSATZ ERSTELLEN
    # =========================================================================
    
    def prepare_analysis_dataset(self, plan: AnalysisPlan,
                                 output_folder: Path = None) -> PipelineResult:
        """
        Erstellt den VP-übergreifenden Analysedatensatz.
        
        Args:
            plan: Analyse-Plan
            output_folder: Output-Ordner (Default: self.output_folder)
        
        Returns:
            PipelineResult
        """
        import time
        start_time = time.time()
        
        if output_folder is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = self.output_folder / f"Analysis_{timestamp}"
        
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print("ERSTELLE ANALYSEDATENSATZ")
        print(f"{'='*70}")
        print(f"Output: {output_folder}")
        
        result = PipelineResult(success=False)
        
        all_frame_dfs = []
        all_lag_results = []
        prep_reports = []
        
        # Re-scan für aktuelle Daten
        report = self.scanner.scan_all()
        
        # Verarbeite alle verfügbaren Configs
        processed_vps = set()
        failed_vps = set()
        
        for k_label in plan.calib_configs:
            if k_label not in STATISTICAL_CALIB_CONFIGS:
                continue
            
            calib_mode = STATISTICAL_CALIB_CONFIGS[k_label]['calib_mode']
            
            for fps in plan.framerates:
                config_key = f"{fps}_{k_label}"
                
                # Hole VPs für diese Config
                vps = report.get_vps_for_config(fps, calib_mode, complete_only=True)
                
                print(f"\n[CONFIG] {config_key}: {len(vps)} VPs")
                
                for vp_code in vps:
                    try:
                        # Finde Run-Ordner
                        vp_data = self.vp_manager.load_vp_data(vp_code)
                        
                        # Suche passenden Run
                        target_config = RunConfig(
                            video_fps=fps,
                            calib_mode=calib_mode,
                            analysis_mode=2,
                            methods='both'
                        )
                        
                        matching_runs = self.vp_manager.find_matching_runs(
                            vp_data, target_config, require_success=True
                        )
                        
                        if not matching_runs:
                            print(f"  [!] {vp_code}: Kein passender Run gefunden")
                            continue
                        
                        run_folder = matching_runs[0].run_folder
                        
                        # Verarbeite VP (mit zentralem EyeLink-Cache)
                        prep_result = self.data_prep.process_vp(
                            run_folder, vp_code, fps, calib_mode,
                            eyelink_cache=self.eyelink_cache
                        )
                        
                        if prep_result.success:
                            # Füge K-Label hinzu
                            prep_result.frame_level_df['calib_label'] = k_label
                            
                            all_frame_dfs.append(prep_result.frame_level_df)
                            processed_vps.add(vp_code)
                            
                            if prep_result.lag_mediapipe:
                                all_lag_results.append(prep_result.lag_mediapipe)
                            if prep_result.lag_ptgaze:
                                all_lag_results.append(prep_result.lag_ptgaze)
                            
                            prep_reports.append({
                                'vp_id': vp_code,
                                'config_key': config_key,
                                'n_frames': prep_result.n_frames_total,
                                'n_valid': prep_result.n_frames_valid,
                                'n_trials': prep_result.n_trials,
                                'success': True
                            })
                        else:
                            failed_vps.add(vp_code)
                            result.errors.append(f"{vp_code}/{config_key}: {prep_result.error_message}")
                            
                            prep_reports.append({
                                'vp_id': vp_code,
                                'config_key': config_key,
                                'success': False,
                                'error': prep_result.error_message
                            })
                    
                    except Exception as e:
                        failed_vps.add(vp_code)
                        result.errors.append(f"{vp_code}/{config_key}: {str(e)}")
                        print(f"  [!!] {vp_code}: {e}")
        
        # Kombiniere alle DataFrames
        if not all_frame_dfs:
            result.errors.append("Keine Daten verarbeitet")
            return result
        
        print(f"\n[MERGE] Kombiniere {len(all_frame_dfs)} DataFrames...")
        
        combined_df = pd.concat(all_frame_dfs, ignore_index=True)
        
        # Practice-Trial ausschließen
        if EXCLUDE_PRACTICE_TRIAL:
            n_before = len(combined_df)
            combined_df = combined_df[combined_df['trial_assignment'] != PRACTICE_TRIAL_NUMBER]
            n_after = len(combined_df)
            print(f"  Practice-Trial entfernt: {n_before} -> {n_after} Zeilen")
        
        # Speichere Frame-Level CSV
        frame_csv_path = output_folder / OUTPUT_FRAME_LEVEL_CSV
        combined_df.to_csv(frame_csv_path, index=False)
        result.frame_level_csv = frame_csv_path
        print(f"  [OK] Frame-Level: {frame_csv_path.name} ({len(combined_df)} Zeilen)")
        
        # Erstelle Trial-Level Aggregation
        trial_df = self._create_trial_level(combined_df)
        trial_csv_path = output_folder / OUTPUT_TRIAL_LEVEL_CSV
        trial_df.to_csv(trial_csv_path, index=False)
        result.trial_level_csv = trial_csv_path
        print(f"  [OK] Trial-Level: {trial_csv_path.name} ({len(trial_df)} Zeilen)")
        
        # Speichere Lag-Report
        # Keine Valid/Invalid-Unterscheidung mehr - alle Lags werden dokumentiert
        mediapipe_lags = [r.optimal_lag_ms for r in all_lag_results if r.method == 'mediapipe']
        ptgaze_lags = [r.optimal_lag_ms for r in all_lag_results if r.method == 'ptgaze']
        
        lag_report = {
            'n_results': len(all_lag_results),
            'results': [r.to_dict() for r in all_lag_results],
            'summary': {
                'mean_lag_mediapipe_ms': float(np.mean(mediapipe_lags)) if mediapipe_lags else None,
                'std_lag_mediapipe_ms': float(np.std(mediapipe_lags)) if mediapipe_lags else None,
                'mean_lag_ptgaze_ms': float(np.mean(ptgaze_lags)) if ptgaze_lags else None,
                'std_lag_ptgaze_ms': float(np.std(ptgaze_lags)) if ptgaze_lags else None,
                'n_mediapipe': len(mediapipe_lags),
                'n_ptgaze': len(ptgaze_lags),
            }
        }
        
        lag_json_path = output_folder / OUTPUT_LAG_REPORT_JSON
        with open(lag_json_path, 'w', encoding='utf-8') as f:
            json.dump(lag_report, f, indent=2, ensure_ascii=False, default=str)
        result.lag_report_json = lag_json_path
        print(f"  [OK] Lag-Report: {lag_json_path.name}")
        
        # Speichere Prep-Report
        prep_report = {
            'timestamp': datetime.now().isoformat(),
            'n_vps_processed': len(processed_vps),
            'n_vps_failed': len(failed_vps),
            'processed_vps': list(processed_vps),
            'failed_vps': list(failed_vps),
            'configs': plan.calib_configs,
            'framerates': plan.framerates,
            'details': prep_reports
        }
        
        # FIX: Konvertiere NumPy-Typen vor JSON-Export
        prep_report = convert_numpy_types(prep_report)
        
        prep_json_path = output_folder / OUTPUT_PREP_REPORT_JSON
        with open(prep_json_path, 'w', encoding='utf-8') as f:
            json.dump(prep_report, f, indent=2, ensure_ascii=False)
        result.prep_report_json = prep_json_path
        print(f"  [OK] Prep-Report: {prep_json_path.name}")
        
        # Ergebnis zusammenfassen
        result.success = True
        result.n_vps_processed = len(processed_vps)
        result.n_vps_failed = len(failed_vps)
        result.n_frames_total = len(combined_df)
        result.n_trials_total = len(trial_df)
        result.lag_results = all_lag_results
        result.duration_seconds = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"[OK] ANALYSEDATENSATZ ERSTELLT")
        print(f"{'='*70}")
        print(f"VPs verarbeitet: {result.n_vps_processed}")
        print(f"VPs fehlgeschlagen: {result.n_vps_failed}")
        print(f"Frames gesamt: {result.n_frames_total}")
        print(f"Trials gesamt: {result.n_trials_total}")
        print(f"Dauer: {result.duration_seconds/60:.1f} Minuten")
        print(f"Output: {output_folder}")
        print(f"{'='*70}")
        
        return result
    
    def _create_trial_level(self, frame_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregiert Frame-Level zu Trial-Level.
        
        WICHTIG: EyeLink-Mittelwerte werden aus Full-Rate-Daten berechnet,
        nicht aus den downgesampelten Daten. Das garantiert identische
        EyeLink-Werte für 25Hz und 60Hz (statistisch sauberer).
        
        Berechnet pro Trial:
        - CV-Mittelwerte (aus Frame-Level, abhängig von Framerate)
        - EyeLink-Mittelwerte (aus Full-Rate, ~7000 Samples pro Trial)
        - Standardabweichungen
        - Ausschluss-Statistiken
        """
        # Nur nicht-ausgeschlossene Frames
        valid_df = frame_df[~frame_df['exclude_any']].copy()
        
        # Gruppierung
        group_cols = ['vp_id', 'trial_assignment', 'method', 'video_fps', 'calib_config', 'calib_label']
        
        # Aggregation für CV-Daten (aus Frame-Level)
        agg_dict = {
            'cv_deg_x': ['mean', 'std', 'count'],
            'cv_deg_y': ['mean', 'std'],
            'cv_confidence': ['mean'],
            'lag_applied_ms': ['first'],
            'timestamp_ms_synced': ['min', 'max']  # Für Trial-Grenzen
        }
        
        trial_df = valid_df.groupby(group_cols).agg(agg_dict).reset_index()
        
        # Flatten column names
        trial_df.columns = [
            '_'.join(col).strip('_') if isinstance(col, tuple) else col 
            for col in trial_df.columns
        ]
        
        # Rename CV-Spalten
        trial_df = trial_df.rename(columns={
            'cv_deg_x_mean': 'cv_mean_x',
            'cv_deg_x_std': 'cv_std_x',
            'cv_deg_x_count': 'n_valid_frames',
            'cv_deg_y_mean': 'cv_mean_y',
            'cv_deg_y_std': 'cv_std_y',
            'cv_confidence_mean': 'mean_confidence',
            'lag_applied_ms_first': 'lag_applied_ms',
            'timestamp_ms_synced_min': 'trial_start_ms',
            'timestamp_ms_synced_max': 'trial_end_ms'
        })
        
        # ─────────────────────────────────────────────────────────────────────
        # EyeLink-Aggregate aus Full-Rate-Daten (pro VP, unabhängig von FPS)
        # ─────────────────────────────────────────────────────────────────────
        
        print(f"\n  [>] Berechne EyeLink-Aggregate aus Full-Rate-Daten...")
        
        eyelink_aggregates_cache = {}  # vp_id -> DataFrame mit Trial-Aggregaten
        
        for vp_id in trial_df['vp_id'].unique():
            if vp_id in eyelink_aggregates_cache:
                continue
            
            # Extrahiere Trial-Grenzen für diese VP (aus allen Configs)
            vp_mask = trial_df['vp_id'] == vp_id
            trial_boundaries = {}
            
            for _, row in trial_df[vp_mask].iterrows():
                trial_num = row['trial_assignment']
                if trial_num not in trial_boundaries:
                    trial_boundaries[trial_num] = (row['trial_start_ms'], row['trial_end_ms'])
                else:
                    # Erweitere Grenzen falls nötig (verschiedene Configs können leicht variieren)
                    existing = trial_boundaries[trial_num]
                    trial_boundaries[trial_num] = (
                        min(existing[0], row['trial_start_ms']),
                        max(existing[1], row['trial_end_ms'])
                    )
            
            # Berechne Aggregate aus Full-Rate-Daten
            eyelink_agg = self.eyelink_cache.get_trial_aggregates(vp_id, trial_boundaries)
            eyelink_aggregates_cache[vp_id] = eyelink_agg
        
        # Merge EyeLink-Aggregate in trial_df
        # Erstelle Lookup: (vp_id, trial) -> EyeLink-Werte
        eyelink_lookup = {}
        for vp_id, agg_df in eyelink_aggregates_cache.items():
            for _, row in agg_df.iterrows():
                key = (vp_id, row['trial_assignment'])
                eyelink_lookup[key] = {
                    'eyelink_mean_x': row['eyelink_mean_x_fullrate'],
                    'eyelink_mean_y': row['eyelink_mean_y_fullrate'],
                    'eyelink_std_x': row['eyelink_std_x_fullrate'],
                    'eyelink_std_y': row['eyelink_std_y_fullrate'],
                    'n_eyelink_samples': row['n_eyelink_samples_fullrate']
                }
        
        # Füge EyeLink-Spalten hinzu
        trial_df['eyelink_mean_x'] = trial_df.apply(
            lambda r: eyelink_lookup.get((r['vp_id'], r['trial_assignment']), {}).get('eyelink_mean_x', np.nan),
            axis=1
        )
        trial_df['eyelink_mean_y'] = trial_df.apply(
            lambda r: eyelink_lookup.get((r['vp_id'], r['trial_assignment']), {}).get('eyelink_mean_y', np.nan),
            axis=1
        )
        trial_df['eyelink_std_x'] = trial_df.apply(
            lambda r: eyelink_lookup.get((r['vp_id'], r['trial_assignment']), {}).get('eyelink_std_x', np.nan),
            axis=1
        )
        trial_df['eyelink_std_y'] = trial_df.apply(
            lambda r: eyelink_lookup.get((r['vp_id'], r['trial_assignment']), {}).get('eyelink_std_y', np.nan),
            axis=1
        )
        trial_df['n_eyelink_samples'] = trial_df.apply(
            lambda r: eyelink_lookup.get((r['vp_id'], r['trial_assignment']), {}).get('n_eyelink_samples', 0),
            axis=1
        )
        
        # Entferne temporäre Spalten
        trial_df = trial_df.drop(columns=['trial_start_ms', 'trial_end_ms'], errors='ignore')
        
        # Berechne Differenzen (für schnelle Bias-Schätzung)
        trial_df['diff_x'] = trial_df['cv_mean_x'] - trial_df['eyelink_mean_x']
        trial_df['diff_y'] = trial_df['cv_mean_y'] - trial_df['eyelink_mean_y']
        
        # Statistik
        n_with_eyelink = trial_df['eyelink_mean_x'].notna().sum()
        print(f"      Trials mit EyeLink-Daten: {n_with_eyelink}/{len(trial_df)}")
        
        if n_with_eyelink > 0:
            avg_samples = trial_df['n_eyelink_samples'].mean()
            print(f"      Durchschnittliche EyeLink-Samples pro Trial: {avg_samples:.0f}")
        
        return trial_df

# =================================================================================
# MAIN (TEST)
# =================================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STATISTICAL PIPELINE v1.0 - TEST")
    print("="*70)
    
    # Test-Pfade
    base_folder = Path(r"C:\Users\imanu\Documents\Imanuel\Studium\Psy\Master\Pupillendetektion\Ergebnisse")
    scripts_folder = Path(r"C:\Users\imanu\Documents\Imanuel\Studium\Psy\Master\Pupillendetektion\Skripte\Debug\v3.0.9")
    
    if base_folder.exists():
        pipeline = StatisticalPipeline(base_folder, scripts_folder)
        
        # Erstelle Plan
        plan = pipeline.create_analysis_plan(
            framerates=['25hz'],
            calib_configs=['K1'],
            methods=['mediapipe', 'ptgaze']
        )
        
        print(f"\n[OK] Plan erstellt")
    else:
        print(f"\n[!] Test-Ordner nicht gefunden")
