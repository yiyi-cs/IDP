"""
=================================================================================
ANALYSIS PIPELINE v2.0 - Pipeline-Orchestrierung fuer VP-Analysen
=================================================================================

Aenderungen in v2.0:
--------------------
- Package-Struktur (utils/)
- ptgaze-Pipeline hinzugefuegt (debug_1_ptgaze, offline_calibration_ptgaze, debug_5_ptgaze)
- ANALYSIS_MODE Environment-Variable
- Cache-Mapping erweitert (ptgaze-Dateien)
- Video-FPS korrigiert (25hz statt 24hz)
- Skript-Namen korrigiert (debug_6_extended_comparison)

Pipeline-Schritte (Modus 2 - Comparison):
-----------------------------------------
debug_0 -> debug_1_mediapipe -> debug_1_ptgaze -> 
offline_calibration -> offline_calibration_ptgaze ->
debug_3 -> debug_4 -> 
debug_5_mediapipe -> debug_5_ptgaze -> 
debug_6

Version: 2.0
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
from typing import Optional, Dict, List, Tuple
import subprocess
import json
import shutil
import time
import os
from datetime import datetime

# Import VP Data Manager (aus utils/)
from utils.vp_data_manager import (
    VPData, 
    ValidationResult,
    RunConfig,
    CacheFile,
    CacheOverview,
    RunInfo,
    CACHE_CRITERIA,
    STEP_OUTPUT_FILES
)

# =================================================================================
# DATENSTRUKTUREN
# =================================================================================

@dataclass
class RunConfiguration:
    """Konfiguration fuer einen Pipeline-Run."""
    vp_code: str
    video_fps: str                              # '60hz' oder '25hz'
    calib_mode: str
    create_calibration: bool
    analysis_mode: int = 2                      # 1=Standalone, 2=Comparison
    methods: str = 'both'                       # 'mediapipe', 'ptgaze', 'both'
    calibration_pkl_path: Optional[Path] = None
    calibration_ptgaze_pkl_path: Optional[Path] = None
    
    # Cache-Entscheidungen (NEU v2.1)
    cache_decisions: Optional['CacheDecisions'] = None
    
    def __str__(self):
        calib_info = "Auto-Create" if self.create_calibration else f"Custom"
        return f"{self.vp_code} | {self.video_fps} | Mode {self.analysis_mode} | {self.calib_mode} | Methods: {self.methods}"
    
    def to_run_config(self) -> 'RunConfig':
        """Konvertiert zu RunConfig (fuer Cache-Vergleiche)."""
        return RunConfig(
            video_fps=self.video_fps,
            calib_mode=self.calib_mode,
            analysis_mode=self.analysis_mode,
            methods=self.methods
        )

@dataclass
class RunResult:
    """Ergebnis eines Pipeline-Runs."""
    success: bool
    run_id: str
    run_folder: Path
    vp_code: str
    video_fps: str
    calib_mode: str
    analysis_mode: int
    duration_seconds: float
    completed_steps: List[str]
    failed_step: Optional[str] = None
    error_message: Optional[str] = None
    output_files: Dict[str, Path] = None
    skipped: bool = False                       # NEU: VP wurde uebersprungen
    skip_reason: Optional[str] = None           # NEU: Grund fuer Ueberspringung
    
    def __post_init__(self):
        if self.output_files is None:
            self.output_files = {}


@dataclass
class StepCacheDecision:
    """
    Cache-Entscheidung fuer einen einzelnen Pipeline-Schritt.
    """
    step: str                           # z.B. 'debug_0', 'debug_1_mediapipe'
    use_cache: bool                     # Cache verwenden?
    source_run: Optional[str] = None    # Quell-Run (falls use_cache=True)
    source_path: Optional[Path] = None  # Pfad zur Cache-Datei
    reason: str = ""                    # Erklaerung


@dataclass
class CacheDecisions:
    """
    Gesammelte Cache-Entscheidungen fuer einen Pipeline-Run.
    """
    vp_code: str
    target_config: 'RunConfig'
    decisions: Dict[str, StepCacheDecision]     # Step -> Entscheidung
    
    # Statistik
    n_from_cache: int = 0
    n_fresh: int = 0
    
    def __post_init__(self):
        self.n_from_cache = sum(1 for d in self.decisions.values() if d.use_cache)
        self.n_fresh = sum(1 for d in self.decisions.values() if not d.use_cache)
    
    def use_cache_for(self, step: str) -> bool:
        """Prueft ob Cache fuer einen Step verwendet werden soll."""
        if step in self.decisions:
            return self.decisions[step].use_cache
        return False
    
    def get_source_run(self, step: str) -> Optional[str]:
        """Gibt den Quell-Run fuer einen Step zurueck."""
        if step in self.decisions and self.decisions[step].use_cache:
            return self.decisions[step].source_run
        return None
    
    def get_summary(self) -> str:
        """Gibt eine formatierte Zusammenfassung zurueck."""
        lines = []
        lines.append(f"Cache-Entscheidungen fuer VP: {self.vp_code}")
        lines.append(f"Aus Cache: {self.n_from_cache} | Neu berechnen: {self.n_fresh}")
        lines.append("-" * 50)
        
        for step, decision in self.decisions.items():
            if decision.use_cache:
                lines.append(f"  [CACHE] {step} <- {decision.source_run}")
            else:
                lines.append(f"  [NEU]   {step}: {decision.reason}")
        
        return "\n".join(lines)

# =================================================================================
# CROSS-RUN CACHE SCANNER (v2.1 - Granulare Cache-Logik)
# =================================================================================

class CrossRunCacheScanner:
    """
    Scannt vorherige Runs nach wiederverwendbaren Outputs.
    
    v2.1: Granulare Cache-Kriterien pro Schritt
    
    Strategie:
    ----------
    1. Finde alle Run_* Ordner der VP
    2. Pruefe Config-Match PRO SCHRITT (unterschiedliche Kriterien!)
    3. Validiere Output-Dateien
    4. Kopiere bei Match
    
    Cache-Kriterien (aus CACHE_CRITERIA):
    - debug_0: Nur video_fps muss matchen
    - debug_1_*: Nur video_fps muss matchen
    - offline_calibration: video_fps + calib_mode muessen matchen
    - debug_3: Keine Config-Abhaengigkeit (nur ASC-Datei)
    - debug_4: video_fps muss matchen
    - debug_5_*: video_fps + calib_mode muessen matchen
    - debug_6: video_fps + calib_mode + analysis_mode muessen matchen
    """
    
    def __init__(self, vp_folder: Path):
        self.vp_folder = vp_folder
        self.analysis_folder = vp_folder / 'Analyse'
        
        # Nutze STEP_OUTPUT_FILES aus vp_data_manager
        self.output_mapping = STEP_OUTPUT_FILES.copy()
        
        # Aliase fuer Kompatibilitaet
        if 'debug_1_mediapipe' not in self.output_mapping:
            self.output_mapping['debug_1_mediapipe'] = ['debug_1_pupil_data.csv', 'debug_1_metadata.json']
    
    def _get_run_config(self, run_folder: Path) -> Optional[Dict]:
        """Liest Config aus config_used.json."""
        config_file = run_folder / 'config_used.json'
        
        if not config_file.exists():
            return None
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def _is_step_compatible(self, step: str, cached_config: Dict, 
                           target_config: 'RunConfiguration') -> Tuple[bool, str]:
        """
        Prueft ob ein Step mit der Ziel-Config kompatibel ist.
        
        Nutzt CACHE_CRITERIA fuer granulare Pruefung.
        """
        criteria = CACHE_CRITERIA.get(step, [])
        
        if not criteria:
            return True, "Keine Config-Abhaengigkeit"
        
        mismatches = []
        
        for field in criteria:
            cached_val = cached_config.get(field)
            target_val = getattr(target_config, field, None)
            
            if cached_val != target_val:
                mismatches.append(f"{field}: {cached_val} != {target_val}")
        
        if mismatches:
            return False, "; ".join(mismatches)
        
        return True, "Alle Kriterien erfuellt"
    
    def find_compatible_run_for_step(self, config: 'RunConfiguration', 
                                     step: str,
                                     exclude_run: Optional[Path] = None) -> Optional[Path]:
        """
        Findet einen kompatiblen Run fuer einen bestimmten Step.
        
        Nutzt GRANULARE Kriterien (nicht globale Config-Match!).
        
        Args:
            config: Ziel-Konfiguration
            step: Pipeline-Schritt
            exclude_run: Run-Ordner der ausgeschlossen werden soll (aktueller Run)
        
        Returns:
            Path zum Run-Ordner oder None
        """
        if not self.analysis_folder.exists():
            return None
        
        # Finde alle Run-Ordner (neueste zuerst)
        run_folders = sorted(
            self.analysis_folder.glob('Run_*'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for run_folder in run_folders:
            # FIX v2.2: Aktuellen Run ausschliessen
            if exclude_run is not None:
                try:
                    if run_folder.resolve() == exclude_run.resolve():
                        continue
                except (OSError, ValueError):
                    if run_folder.name == exclude_run.name:
                        continue
            
            cached_config = self._get_run_config(run_folder)
            
            if cached_config is None:
                continue
            
            # VP-Code muss immer matchen
            if cached_config.get('vp_code') != config.vp_code:
                continue
            
            # Pruefe Step-spezifische Kompatibilitaet
            is_compat, reason = self._is_step_compatible(step, cached_config, config)
            
            if not is_compat:
                continue
            
            # Pruefe ob Step-Output existiert
            if self._has_step_outputs(run_folder, step):
                return run_folder
        
        return None

    def find_matching_run(self, config: 'RunConfiguration', 
                          required_step: str,
                          exclude_run: Optional[Path] = None) -> Optional[Path]:
        """
        Legacy-Methode fuer Rueckwaertskompatibilitaet.
        Nutzt jetzt find_compatible_run_for_step().
        """
        return self.find_compatible_run_for_step(config, required_step, exclude_run)

    def _has_step_outputs(self, run_folder: Path, step: str) -> bool:
        """Prueft ob alle Output-Dateien eines Schritts existieren."""
        
        if step not in self.output_mapping:
            return False
        
        filenames = self.output_mapping[step]
        
        for filename in filenames:
            if '*' in filename:
                # Wildcard
                matches = list(run_folder.glob(filename))
                if not matches:
                    return False
            else:
                file_path = run_folder / filename
                if not file_path.exists():
                    # Optionale Dateien ueberspringen
                    if any(opt in filename for opt in ['fixations', 'metadata', 'qc_', 'event_table']):
                        continue
                    return False
        
        return True
    
    def copy_step_outputs(self, source_run: Path, target_run: Path, 
                         step: str) -> bool:
        """Kopiert ALLE Output-Dateien eines Schritts."""
        
        if step not in self.output_mapping:
            return False
        
        # FIX v2.2: Pruefe ob source == target (verhindert "same file" Fehler)
        try:
            if source_run.resolve() == target_run.resolve():
                print(f"      [SKIP] {step}: Source und Target sind identisch")
                return False
        except (OSError, ValueError):
            pass  # Falls resolve() fehlschlaegt, weitermachen
        
        filenames = self.output_mapping[step]
        copied_count = 0
        skipped_count = 0
        
        for filename in filenames:
            if '*' in filename:
                for source_file in source_run.glob(filename):
                    dest_file = target_run / source_file.name
                    # FIX v2.2: Pruefe ob Dateien identisch sind
                    try:
                        if source_file.resolve() == dest_file.resolve():
                            skipped_count += 1
                            continue
                    except (OSError, ValueError):
                        pass
                    shutil.copy2(source_file, dest_file)
                    copied_count += 1
            else:
                source_file = source_run / filename
                if source_file.exists():
                    dest_file = target_run / filename
                    # FIX v2.2: Pruefe ob Dateien identisch sind
                    try:
                        if source_file.resolve() == dest_file.resolve():
                            skipped_count += 1
                            continue
                    except (OSError, ValueError):
                        pass
                    shutil.copy2(source_file, dest_file)
                    copied_count += 1
        
        if copied_count > 0:
            print(f"      [CACHE] {step}: {copied_count} Dateien kopiert von {source_run.name}")
        if skipped_count > 0:
            print(f"      [INFO] {step}: {skipped_count} Dateien uebersprungen (bereits vorhanden)")
        
        return copied_count > 0
  
    def get_all_available_cache(self, config: 'RunConfiguration') -> Dict[str, List[Tuple[Path, Dict]]]:
        """
        Sammelt ALLE verfuegbaren Cache-Dateien fuer alle Steps.
        
        Returns:
            Dict[step, List[(run_folder, config)]]
        """
        available = {}
        
        if not self.analysis_folder.exists():
            return available
        
        run_folders = sorted(
            self.analysis_folder.glob('Run_*'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for step in self.output_mapping.keys():
            available[step] = []
            
            for run_folder in run_folders:
                cached_config = self._get_run_config(run_folder)
                
                if cached_config is None:
                    continue
                
                if cached_config.get('vp_code') != config.vp_code:
                    continue
                
                if self._has_step_outputs(run_folder, step):
                    is_compat, reason = self._is_step_compatible(step, cached_config, config)
                    available[step].append({
                        'run_folder': run_folder,
                        'run_id': run_folder.name,
                        'config': cached_config,
                        'is_compatible': is_compat,
                        'reason': reason
                    })
        
        return available


# =================================================================================
# CACHE-HELPER-FUNKTIONEN
# =================================================================================

def create_default_cache_decisions(config: 'RunConfiguration', 
                                   scanner: CrossRunCacheScanner) -> CacheDecisions:
    """
    Erstellt Standard-Cache-Entscheidungen (alle kompatiblen aus Cache).
    
    Wird verwendet wenn kein interaktiver Dialog stattfindet.
    """
    decisions = {}
    
    all_steps = list(STEP_OUTPUT_FILES.keys())
    
    for step in all_steps:
        compatible_run = scanner.find_compatible_run_for_step(config, step)
        
        if compatible_run:
            decisions[step] = StepCacheDecision(
                step=step,
                use_cache=True,
                source_run=compatible_run.name,
                source_path=compatible_run,
                reason="Kompatible Daten gefunden"
            )
        else:
            decisions[step] = StepCacheDecision(
                step=step,
                use_cache=False,
                reason="Keine kompatiblen Daten"
            )
    
    return CacheDecisions(
        vp_code=config.vp_code,
        target_config=config.to_run_config(),
        decisions=decisions
    )


def create_no_cache_decisions(config: 'RunConfiguration') -> CacheDecisions:
    """
    Erstellt Cache-Entscheidungen die KEINEN Cache nutzen (alles neu).
    """
    decisions = {}
    
    all_steps = list(STEP_OUTPUT_FILES.keys())
    
    for step in all_steps:
        decisions[step] = StepCacheDecision(
            step=step,
            use_cache=False,
            reason="Benutzer hat Cache deaktiviert"
        )
    
    return CacheDecisions(
        vp_code=config.vp_code,
        target_config=config.to_run_config(),
        decisions=decisions
    )

# =================================================================================
# ANALYSIS PIPELINE
# =================================================================================

class AnalysisPipeline:
    """
    Orchestriert komplette Pipeline-Ausfuehrung fuer eine VP.
    
    v2.0: Unterstuetzt beide Pipelines (MediaPipe + ptgaze)
    """
    
    def __init__(self, vp_data: VPData, base_scripts_folder: Path):
        self.vp_data = vp_data
        self.scripts_folder = base_scripts_folder
        
        # Unterordner-Pfade (neue Struktur)
        self.debug_folder = base_scripts_folder / 'debug'
        self.calibration_folder = base_scripts_folder / 'calibration'
        self.shared_folder = base_scripts_folder / 'shared'
        self.utils_folder = base_scripts_folder / 'utils'
        
        print(f"\n[INIT] Analysis Pipeline v2.0 initialisiert")
        print(f"       VP: {vp_data.vp_code}")
        print(f"       Skripte: {base_scripts_folder}")
        print(f"       Debug-Ordner: {self.debug_folder}")
        print(f"       Calibration-Ordner: {self.calibration_folder}")

    def validate_prerequisites(self, config: RunConfiguration) -> Tuple[bool, Optional[str]]:
        """
        Prueft ob alle Voraussetzungen fuer die Pipeline erfuellt sind.
        
        Returns:
            (success, error_message) - success=False wenn VP uebersprungen werden soll
        """
        
        # Check 1: Video vorhanden?
        video_path = self.vp_data.videos.get(config.video_fps)
        if video_path is None:
            return False, f"Video {config.video_fps} nicht gefunden"
        
        if not video_path.exists():
            return False, f"Video-Datei existiert nicht: {video_path}"
        
        # Check 2: Experiment-Log vorhanden? (OPTIONAL - debug_0 hat Fallback!)
        if self.vp_data.experiment_log is None:
            print(f"    [INFO] experiment_sync_log.json nicht gefunden - debug_0 nutzt Fallback")
            # KEIN return False! debug_0 kann ohne JSON arbeiten
        elif not self.vp_data.experiment_log.exists():
            print(f"    [INFO] Experiment-Log existiert nicht - debug_0 nutzt Fallback")
            # KEIN return False! debug_0 kann ohne JSON arbeiten

        # Check 3: ASC-Datei (nur Modus 2)
        if config.analysis_mode == 2:
            if self.vp_data.asc_file is None:
                return False, "ASC-Datei nicht gefunden (benoetigt fuer Modus 2)"
            
            if not self.vp_data.asc_file.exists():
                return False, f"ASC-Datei existiert nicht: {self.vp_data.asc_file}"
        
        # Check 4: Custom PKL (wenn nicht auto-create)
        if not config.create_calibration:
            if config.calibration_pkl_path is None:
                return False, "Kein Kalibrierungs-PKL angegeben (create_calibration=False)"
            
            if not Path(config.calibration_pkl_path).exists():
                return False, f"Kalibrierungs-PKL existiert nicht: {config.calibration_pkl_path}"
        
        return True, None

    def run_full_analysis(self, config: RunConfiguration) -> RunResult:
        """Fuehrt komplette Pipeline aus (v2.1: mit Prerequisite-Check + Skip-Support)."""
        
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"STARTE PIPELINE-RUN v2.1")
        print(f"{'='*70}")
        print(f"Konfiguration: {config}")
        print(f"Analysis Mode: {config.analysis_mode}")
        print(f"{'='*70}\n")
        
        # ══════════════════════════════════════════════════════════════
        # PREREQUISITE-CHECK (NEU v2.1)
        # ══════════════════════════════════════════════════════════════
        
        prereq_ok, prereq_error = self.validate_prerequisites(config)
        
        if not prereq_ok:
            duration = time.time() - start_time
            
            print(f"\n{'='*70}")
            print(f"[SKIP] VP {config.vp_code} UEBERSPRUNGEN")
            print(f"{'='*70}")
            print(f"Grund: {prereq_error}")
            print(f"{'='*70}\n")
            
            return RunResult(
                success=False,
                run_id='skipped',
                run_folder=None,
                vp_code=config.vp_code,
                video_fps=config.video_fps,
                calib_mode=config.calib_mode,
                analysis_mode=config.analysis_mode,
                duration_seconds=duration,
                completed_steps=[],
                skipped=True,
                skip_reason=prereq_error
            )
        
        # Setup
        run_id, run_folder = self._create_run_folder(config)
        
        print(f"[INFO] Run-Ordner: {run_folder}")
        print(f"       ID: {run_id}\n")
        
        self._save_run_config(run_folder, config)
        
        output_dir = run_folder
        scanner = CrossRunCacheScanner(self.vp_data.vp_folder)
        
        # FIX v2.2: Aktuellen Run-Ordner fuer Cache-Suche ausschliessen
        current_run_folder = run_folder
      
        # Cache-Entscheidungen (v2.1)
        if config.cache_decisions is None:
            # Automatische Cache-Entscheidungen (Standard: alles aus Cache wenn moeglich)
            config.cache_decisions = create_default_cache_decisions(config, scanner)
            print(f"[CACHE] Automatische Cache-Entscheidungen:")
            print(f"        Aus Cache: {config.cache_decisions.n_from_cache} Steps")
            print(f"        Neu: {config.cache_decisions.n_fresh} Steps\n")

        completed_steps = []
        failed_step = None
        error_message = None
        
        # PKL-Pfade (werden in Kalibrierungs-Schritten gesetzt)
        calib_pkl_mediapipe = None
        calib_pkl_ptgaze = None
        
        try:
            # =========================================================
            # SCHRITT 1: debug_0 (Phase Detection)
            # =========================================================
            
            print(f"\n{'---'*23}")
            print("[1/10] debug_0_phase_detection.py")
            print(f"{'---'*23}")
            
            # Pruefe Cache-Entscheidung (v2.1)
            if config.cache_decisions and config.cache_decisions.use_cache_for('debug_0'):
                source_run = config.cache_decisions.decisions['debug_0'].source_path
                if source_run and scanner.copy_step_outputs(source_run, output_dir, 'debug_0'):
                    print(f"      [OK] debug_0 aus Cache (User-Entscheidung)")
                else:
                    # Fallback: Neu berechnen
                    success, msg = self._run_debug_0(config, output_dir)
                    if not success:
                        failed_step = 'debug_0'
                        error_message = msg
                        raise RuntimeError(f"debug_0 failed: {msg}")
                    print(f"      [OK] debug_0 erfolgreich (Cache-Kopie fehlgeschlagen)")
            else:
                # Kein Cache oder User will neu berechnen
                matching_run = scanner.find_compatible_run_for_step(config, 'debug_0')
                
                # Auto-Cache nur wenn keine explizite Entscheidung
                if config.cache_decisions is None and matching_run is not None:
                    if scanner.copy_step_outputs(matching_run, output_dir, 'debug_0'):
                        print(f"      [OK] debug_0 aus Cache (auto)")
                    else:
                        success, msg = self._run_debug_0(config, output_dir)
                        if not success:
                            failed_step = 'debug_0'
                            error_message = msg
                            raise RuntimeError(f"debug_0 failed: {msg}")
                        print(f"      [OK] debug_0 erfolgreich")
                else:
                    success, msg = self._run_debug_0(config, output_dir)
                    if not success:
                        failed_step = 'debug_0'
                        error_message = msg
                        raise RuntimeError(f"debug_0 failed: {msg}")
                    print(f"      [OK] debug_0 erfolgreich")
            
            completed_steps.append('debug_0')
            
            # =========================================================
            # SCHRITT 2: debug_1 MediaPipe (Pupillen-Detektion)
            # =========================================================
            
            print(f"\n{'---'*23}")
            print("[2/10] debug_1_video_analysis.py (MediaPipe)")
            print(f"{'---'*23}")
            
            matching_run = scanner.find_matching_run(config, 'debug_1_mediapipe', current_run_folder)
            
            if matching_run is not None and scanner.copy_step_outputs(matching_run, output_dir, 'debug_1_mediapipe'):
                print(f"      [OK] debug_1_mediapipe aus Cache")
            else:
                success, msg = self._run_debug_1_mediapipe(config, output_dir)
                if not success:
                    failed_step = 'debug_1_mediapipe'
                    error_message = msg
                    raise RuntimeError(f"debug_1_mediapipe failed: {msg}")
                print(f"      [OK] debug_1_mediapipe erfolgreich")
            
            completed_steps.append('debug_1_mediapipe')
            
            # =========================================================
            # SCHRITT 3: debug_1 ptgaze (Gaze-Detektion)
            # =========================================================
            
            if config.methods in ['ptgaze', 'both']:
                print(f"\n{'---'*23}")
                print("[3/10] debug_1_ptgaze.py (ptgaze)")
                print(f"{'---'*23}")
                
                matching_run = scanner.find_matching_run(config, 'debug_1_ptgaze', current_run_folder)
            
                if matching_run is not None and scanner.copy_step_outputs(matching_run, output_dir, 'debug_1_ptgaze'):
                    print(f"      [OK] debug_1_ptgaze aus Cache")
                else:
                    success, msg = self._run_debug_1_ptgaze(config, output_dir)
                    if not success:
                        failed_step = 'debug_1_ptgaze'
                        error_message = msg
                        raise RuntimeError(f"debug_1_ptgaze failed: {msg}")
                    print(f"      [OK] debug_1_ptgaze erfolgreich")
                
                completed_steps.append('debug_1_ptgaze')

            else:
                print(f"\n[SKIP] debug_1_ptgaze (methods={config.methods})")

            # =========================================================
            # SCHRITT 4: offline_calibration MediaPipe (PKL erstellen)
            # =========================================================
            
            if config.create_calibration:
                print(f"\n{'---'*23}")
                print("[4/10] offline_calibration.py (MediaPipe)")
                print(f"{'---'*23}")
                
                matching_run = scanner.find_matching_run(config, 'offline_calibration', current_run_folder)
                
                if matching_run is not None and scanner.copy_step_outputs(matching_run, output_dir, 'offline_calibration'):
                    pkl_files = list(output_dir.glob('calibration_*.pkl'))
                    pkl_files = [f for f in pkl_files if 'ptgaze' not in f.name]
                    if pkl_files:
                        calib_pkl_mediapipe = pkl_files[0]
                        print(f"      [OK] offline_calibration aus Cache: {calib_pkl_mediapipe.name}")
                    else:
                        success, msg, calib_pkl_mediapipe = self._run_offline_calibration_mediapipe(config, output_dir)
                        if not success:
                            failed_step = 'offline_calibration'
                            error_message = msg
                            raise RuntimeError(f"offline_calibration failed: {msg}")
                        print(f"      [OK] offline_calibration erfolgreich")
                else:
                    success, msg, calib_pkl_mediapipe = self._run_offline_calibration_mediapipe(config, output_dir)
                    if not success:
                        failed_step = 'offline_calibration'
                        error_message = msg
                        raise RuntimeError(f"offline_calibration failed: {msg}")
                    print(f"      [OK] offline_calibration erfolgreich")
                
                completed_steps.append('offline_calibration')
            else:
                print(f"\n[SKIP] offline_calibration.py (nutze Custom PKL)")
                calib_pkl_mediapipe = Path(config.calibration_pkl_path)
            
            # =========================================================
            # SCHRITT 5: offline_calibration_ptgaze (PKL erstellen)
            # =========================================================
            
            if config.create_calibration and config.methods in ['ptgaze', 'both']:
                print(f"\n{'---'*23}")
                print("[5/10] offline_calibration_ptgaze.py (ptgaze)")
                print(f"{'---'*23}")
                
                matching_run = scanner.find_matching_run(config, 'offline_calibration_ptgaze', current_run_folder)
                
                if matching_run is not None and scanner.copy_step_outputs(matching_run, output_dir, 'offline_calibration_ptgaze'):
                    pkl_files = list(output_dir.glob('calibration_ptgaze_*.pkl'))
                    if pkl_files:
                        calib_pkl_ptgaze = pkl_files[0]
                        print(f"      [OK] offline_calibration_ptgaze aus Cache: {calib_pkl_ptgaze.name}")
                    else:
                        success, msg, calib_pkl_ptgaze = self._run_offline_calibration_ptgaze(config, output_dir)
                        if not success:
                            failed_step = 'offline_calibration_ptgaze'
                            error_message = msg
                            raise RuntimeError(f"offline_calibration_ptgaze failed: {msg}")
                        print(f"      [OK] offline_calibration_ptgaze erfolgreich")
                else:
                    success, msg, calib_pkl_ptgaze = self._run_offline_calibration_ptgaze(config, output_dir)
                    if not success:
                        failed_step = 'offline_calibration_ptgaze'
                        error_message = msg
                        raise RuntimeError(f"offline_calibration_ptgaze failed: {msg}")
                    print(f"      [OK] offline_calibration_ptgaze erfolgreich")
                
                completed_steps.append('offline_calibration_ptgaze')
            else:
                print(f"\n[SKIP] offline_calibration_ptgaze.py (nutze Custom PKL)")
                calib_pkl_ptgaze = Path(config.calibration_ptgaze_pkl_path) if config.calibration_ptgaze_pkl_path else None
            
            # =========================================================
            # SCHRITT 6: debug_3 (EyeLink Import) - Nur Modus 2
            # =========================================================
            
            if config.analysis_mode == 2:
                print(f"\n{'---'*23}")
                print("[6/10] debug_3_eyetracker_load.py")
                print(f"{'---'*23}")
                
                matching_run = scanner.find_matching_run(config, 'debug_3', current_run_folder)
                
                if matching_run is not None and scanner.copy_step_outputs(matching_run, output_dir, 'debug_3'):
                    print(f"      [OK] debug_3 aus Cache")
                else:
                    success, msg = self._run_debug_3(config, output_dir)
                    if not success:
                        failed_step = 'debug_3'
                        error_message = msg
                        raise RuntimeError(f"debug_3 failed: {msg}")
                    print(f"      [OK] debug_3 erfolgreich")
                
                completed_steps.append('debug_3')
            else:
                print(f"\n[SKIP] debug_3 (Modus 1 = kein EyeLink)")
            
            # =========================================================
            # SCHRITT 7: debug_4 (Synchronisation) - Nur Modus 2
            # =========================================================
            
            if config.analysis_mode == 2:
                print(f"\n{'---'*23}")
                print("[7/10] debug_4_synchronization.py")
                print(f"{'---'*23}")
                
                matching_run = scanner.find_matching_run(config, 'debug_4', current_run_folder)
                
                if matching_run is not None and scanner.copy_step_outputs(matching_run, output_dir, 'debug_4'):
                    print(f"      [OK] debug_4 aus Cache")
                else:
                    success, msg = self._run_debug_4(config, output_dir)
                    if not success:
                        failed_step = 'debug_4'
                        error_message = msg
                        raise RuntimeError(f"debug_4 failed: {msg}")
                    print(f"      [OK] debug_4 erfolgreich")
                
                completed_steps.append('debug_4')
            else:
                print(f"\n[SKIP] debug_4 (Modus 1 = keine Synchronisation)")
            
            # =========================================================
            # SCHRITT 8: debug_5 MediaPipe (Kalibrierung anwenden)
            # =========================================================
            
            print(f"\n{'---'*23}")
            print("[8/10] debug_5_partial_calibration.py (MediaPipe)")
            print(f"{'---'*23}")
            
            matching_run = scanner.find_matching_run(config, 'debug_5_mediapipe', current_run_folder)
            
            if matching_run is not None and scanner.copy_step_outputs(matching_run, output_dir, 'debug_5_mediapipe'):
                print(f"      [OK] debug_5_mediapipe aus Cache")
            else:
                success, msg = self._run_debug_5_mediapipe(config, output_dir, calib_pkl_mediapipe)
                if not success:
                    failed_step = 'debug_5_mediapipe'
                    error_message = msg
                    raise RuntimeError(f"debug_5_mediapipe failed: {msg}")
                print(f"      [OK] debug_5_mediapipe erfolgreich")
            
            completed_steps.append('debug_5_mediapipe')
            
            # =========================================================
            # SCHRITT 9: debug_5 ptgaze (Kalibrierung anwenden)
            # =========================================================
            
            if calib_pkl_ptgaze is not None and config.methods in ['ptgaze', 'both']:
                print(f"\n{'---'*23}")
                print("[9/10] debug_5_ptgaze.py (ptgaze)")

                print(f"{'---'*23}")
                
                matching_run = scanner.find_matching_run(config, 'debug_5_ptgaze', current_run_folder)
                
                if matching_run is not None and scanner.copy_step_outputs(matching_run, output_dir, 'debug_5_ptgaze'):
                    print(f"      [OK] debug_5_ptgaze aus Cache")
                else:
                    success, msg = self._run_debug_5_ptgaze(config, output_dir, calib_pkl_ptgaze)
                    if not success:
                        failed_step = 'debug_5_ptgaze'
                        error_message = msg
                        raise RuntimeError(f"debug_5_ptgaze failed: {msg}")
                    print(f"      [OK] debug_5_ptgaze erfolgreich")
                
                completed_steps.append('debug_5_ptgaze')
            else:
                print(f"\n[SKIP] debug_5_ptgaze (kein ptgaze PKL)")
            
            # =========================================================
            # SCHRITT 10: debug_6 (3-Wege-Vergleich)
            # =========================================================
            
            print(f"\n{'---'*23}")
            print("[10/10] debug_6_extended_comparison.py")
            print(f"{'---'*23}")
            
            matching_run = scanner.find_matching_run(config, 'debug_6', current_run_folder)
            
            if matching_run is not None and scanner.copy_step_outputs(matching_run, output_dir, 'debug_6'):
                print(f"      [OK] debug_6 aus Cache")
            else:
                success, msg = self._run_debug_6(config, output_dir)
                if not success:
                    failed_step = 'debug_6'
                    error_message = msg
                    raise RuntimeError(f"debug_6 failed: {msg}")
                print(f"      [OK] debug_6 erfolgreich")
            
            completed_steps.append('debug_6')
                    
            # =========================================================
            # ERFOLG!
            # =========================================================
            
            duration = time.time() - start_time
            
            # Speichere Success-Marker (NEU v2.1)
            self._save_success_marker(run_folder, completed_steps, duration)
            
            print(f"\n{'='*70}")
            print(f"[SUCCESS] PIPELINE ERFOLGREICH ABGESCHLOSSEN")
            print(f"{'='*70}")
            print(f"Dauer: {duration/60:.1f} Minuten")
            print(f"Output: {run_folder}")
            print(f"Abgeschlossene Schritte: {len(completed_steps)}")
            print(f"{'='*70}\n")
            
            return RunResult(
                success=True,
                run_id=run_id,
                run_folder=run_folder,
                vp_code=config.vp_code,
                video_fps=config.video_fps,
                calib_mode=config.calib_mode,
                analysis_mode=config.analysis_mode,
                duration_seconds=duration,
                completed_steps=completed_steps,
                output_files={}
            )
        
        except Exception as e:
            duration = time.time() - start_time
            
            # Bestimme ob es ein kritischer Fehler oder ein ueberspringbarer Fehler ist
            error_str = str(e)
            is_skippable = any(skip_keyword in error_str.lower() for skip_keyword in [
                'video', 'nicht gefunden', 'not found', 'file not found',
                'no such file', 'datei nicht', 'asc', 'pkl'
            ])
            
            if is_skippable:
                print(f"\n{'='*70}")
                print(f"[SKIP] VP {config.vp_code} UEBERSPRUNGEN (Fehler)")
                print(f"{'='*70}")
                print(f"Fehler in: {failed_step}")
                print(f"Message: {error_message or error_str}")
                print(f"Abgeschlossene Schritte: {', '.join(completed_steps) if completed_steps else 'Keine'}")
                print(f"{'='*70}\n")
            else:
                print(f"\n{'='*70}")
                print(f"[ERROR] PIPELINE FEHLGESCHLAGEN")
                print(f"{'='*70}")
                print(f"Fehler in: {failed_step}")
                print(f"Message: {error_message or error_str}")
                print(f"Abgeschlossene Schritte: {', '.join(completed_steps) if completed_steps else 'Keine'}")
                print(f"{'='*70}\n")
            
            # Speichere Error-Log (nur wenn run_folder existiert)
            if 'run_folder' in locals() and run_folder is not None:
                self._save_error_log(run_folder, failed_step, error_message, str(e))
            
            return RunResult(
                success=False,
                run_id=run_id if 'run_id' in locals() else 'unknown',
                run_folder=run_folder if 'run_folder' in locals() else None,
                vp_code=config.vp_code,
                video_fps=config.video_fps,
                calib_mode=config.calib_mode,
                analysis_mode=config.analysis_mode,
                duration_seconds=duration,
                completed_steps=completed_steps,
                failed_step=failed_step,
                error_message=error_message or str(e),
                skipped=is_skippable,
                skip_reason=error_message or error_str if is_skippable else None
            )

    # ═════════════════════════════════════════════════════════════════
    # PIPELINE-SCHRITTE (Wrapper fuer Skripte)
    # ═════════════════════════════════════════════════════════════════
    
    def _create_base_env(self, config: RunConfiguration, output_dir: Path) -> dict:
        """Erstellt Basis-Environment fuer alle Skripte (v2.2: inkl. calib_mode)."""
        env = os.environ.copy()
        env['PIPELINE_OUTPUT_BASE_DIR'] = str(output_dir)
        env['PIPELINE_ANALYSIS_MODE'] = str(config.analysis_mode)
        env['PIPELINE_CALIB_MODE'] = config.calib_mode  # NEU v2.2
        env['PIPELINE_METHODS'] = config.methods  # NEU v2.2
        return env
    
    def _run_debug_0(self, config: RunConfiguration, output_dir: Path) -> tuple:
        """Fuehrt debug_0_phase_detection.py aus."""
        
        script = self.debug_folder / 'debug_0_phase_detection.py'
        video_path = self.vp_data.videos.get(config.video_fps)
        
        if video_path is None:
            return False, f"Video {config.video_fps} nicht gefunden"
        
        env = self._create_base_env(config, output_dir)
        env['PIPELINE_MAIN_VIDEO_PATH'] = str(video_path)
        env['PIPELINE_EXPERIMENT_SYNC_JSON_PATH'] = str(self.vp_data.experiment_log)
        env['PIPELINE_CALIBRATION_TIMING_JSON_DIR'] = str(self.vp_data.vp_folder / 'Calibration')
        
        print(f"      >> Fuehre debug_0 aus...")
        
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=self.scripts_folder,
                env=env,
                timeout=600
            )
            
            return (True, "OK") if result.returncode == 0 else (False, f"Exit {result.returncode}")
        except subprocess.TimeoutExpired:
            return False, "Timeout (600s)"
        except Exception as e:
            return False, str(e)
        
    def _run_debug_1_mediapipe(self, config: RunConfiguration, output_dir: Path) -> tuple:
        """Fuehrt debug_1_video_analysis.py (MediaPipe) aus."""
        
        script = self.debug_folder / 'debug_1_video_analysis.py'
        video_path = self.vp_data.videos.get(config.video_fps)
        
        if video_path is None:
            return False, f"Video {config.video_fps} nicht gefunden"
        
        env = self._create_base_env(config, output_dir)
        env['PIPELINE_MAIN_VIDEO_PATH'] = str(video_path)
        
        print(f"      >> Fuehre debug_1_mediapipe aus...")
        
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=self.scripts_folder,
                env=env,
                timeout=1800  # 30 min fuer Video-Analyse
            )
            
            return (True, "OK") if result.returncode == 0 else (False, f"Exit {result.returncode}")
        except subprocess.TimeoutExpired:
            return False, "Timeout (1800s)"
        except Exception as e:
            return False, str(e)
    
    def _run_debug_1_ptgaze(self, config: RunConfiguration, output_dir: Path) -> tuple:
        """Fuehrt debug_1_ptgaze.py aus."""
        
        script = self.debug_folder / 'debug_1_ptgaze.py'
        video_path = self.vp_data.videos.get(config.video_fps)
        
        if video_path is None:
            return False, f"Video {config.video_fps} nicht gefunden"
        
        env = self._create_base_env(config, output_dir)
        env['PIPELINE_MAIN_VIDEO_PATH'] = str(video_path)
        
        print(f"      >> Fuehre debug_1_ptgaze aus...")
        
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=self.scripts_folder,
                env=env,
                timeout=2400  # 40 min (ptgaze ist langsamer)
            )
            
            return (True, "OK") if result.returncode == 0 else (False, f"Exit {result.returncode}")
        except subprocess.TimeoutExpired:
            return False, "Timeout (2400s)"
        except Exception as e:
            return False, str(e)
    
    def _run_offline_calibration_mediapipe(self, config: RunConfiguration, output_dir: Path) -> tuple:
        """Fuehrt offline_calibration.py (MediaPipe) aus."""
        
        script = self.calibration_folder / 'offline_calibration.py'
        video_path = self.vp_data.videos.get(config.video_fps)
        
        if video_path is None:
            return False, f"Video nicht gefunden", None
        
        env = self._create_base_env(config, output_dir)
        env['PIPELINE_MAIN_VIDEO_PATH'] = str(video_path)
        
        print(f"      >> Fuehre offline_calibration aus...")
        
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=self.scripts_folder,
                env=env,
                timeout=600
            )
            
            if result.returncode == 0:
                pkl_files = list(output_dir.glob('calibration_*.pkl'))
                pkl_files = [f for f in pkl_files if 'ptgaze' not in f.name]
                return (True, "OK", pkl_files[0]) if pkl_files else (False, "Keine PKL erstellt", None)
            return False, f"Exit {result.returncode}", None
        except subprocess.TimeoutExpired:
            return False, "Timeout (600s)", None
        except Exception as e:
            return False, str(e), None
    
    def _run_offline_calibration_ptgaze(self, config: RunConfiguration, output_dir: Path) -> tuple:
        """Fuehrt offline_calibration_ptgaze.py aus."""
        
        script = self.calibration_folder / 'offline_calibration_ptgaze.py'
        video_path = self.vp_data.videos.get(config.video_fps)
        
        if video_path is None:
            return False, f"Video nicht gefunden", None
        
        env = self._create_base_env(config, output_dir)
        env['PIPELINE_MAIN_VIDEO_PATH'] = str(video_path)
        
        print(f"      >> Fuehre offline_calibration_ptgaze aus...")
        
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=self.scripts_folder,
                env=env,
                timeout=600
            )
            
            if result.returncode == 0:
                pkl_files = list(output_dir.glob('calibration_ptgaze_*.pkl'))
                return (True, "OK", pkl_files[0]) if pkl_files else (False, "Keine ptgaze PKL erstellt", None)
            return False, f"Exit {result.returncode}", None
        except subprocess.TimeoutExpired:
            return False, "Timeout (600s)", None
        except Exception as e:
            return False, str(e), None

    def _run_debug_3(self, config: RunConfiguration, output_dir: Path) -> tuple:
        """Fuehrt debug_3_eyetracker_load.py aus."""
        
        script = self.debug_folder / 'debug_3_eyetracker_load.py'
        
        if self.vp_data.asc_file is None:
            return False, "ASC-Datei nicht gefunden"
        
        env = self._create_base_env(config, output_dir)
        env['PIPELINE_EYETRACKER_FILE_PATH'] = str(self.vp_data.asc_file)
        
        print(f"      >> Fuehre debug_3 aus...")
        
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=self.scripts_folder,
                env=env,
                timeout=300
            )
            
            return (True, "OK") if result.returncode == 0 else (False, f"Exit {result.returncode}")
        except subprocess.TimeoutExpired:
            return False, "Timeout (300s)"
        except Exception as e:
            return False, str(e)
    
    def _run_debug_4(self, config: RunConfiguration, output_dir: Path) -> tuple:
        """Fuehrt debug_4_synchronization.py aus (v2.0: beide Pipelines!)."""
        
        script = self.debug_folder / 'debug_4_synchronization.py'
        
        env = self._create_base_env(config, output_dir)
        
        print(f"      >> Fuehre debug_4 aus (MediaPipe + ptgaze)...")
        
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=self.scripts_folder,
                env=env,
                timeout=120
            )
            
            return (True, "OK") if result.returncode == 0 else (False, f"Exit {result.returncode}")
        except subprocess.TimeoutExpired:
            return False, "Timeout (120s)"
        except Exception as e:
            return False, str(e)
    
    def _run_debug_5_mediapipe(self, config: RunConfiguration, output_dir: Path, calib_pkl: Path) -> tuple:
        """Fuehrt debug_5_partial_calibration.py (MediaPipe) aus."""
        
        script = self.debug_folder / 'debug_5_partial_calibration.py'
        
        env = self._create_base_env(config, output_dir)
        env['PIPELINE_CALIBRATION_PKL_PATH'] = str(calib_pkl)
        
        print(f"      >> Fuehre debug_5_mediapipe aus...")
        
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=self.scripts_folder,
                env=env,
                timeout=120
            )
            
            return (True, "OK") if result.returncode == 0 else (False, f"Exit {result.returncode}")
        except subprocess.TimeoutExpired:
            return False, "Timeout (120s)"
        except Exception as e:
            return False, str(e)
    
    def _run_debug_5_ptgaze(self, config: RunConfiguration, output_dir: Path, calib_pkl: Path) -> tuple:
        """Fuehrt debug_5_ptgaze.py aus."""
        
        script = self.debug_folder / 'debug_5_ptgaze.py'
        
        env = self._create_base_env(config, output_dir)
        env['PIPELINE_CALIBRATION_PKL_PATH'] = str(calib_pkl)
        
        print(f"      >> Fuehre debug_5_ptgaze aus...")
        
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=self.scripts_folder,
                env=env,
                timeout=120
            )
            
            return (True, "OK") if result.returncode == 0 else (False, f"Exit {result.returncode}")
        except subprocess.TimeoutExpired:
            return False, "Timeout (120s)"
        except Exception as e:
            return False, str(e)
    
    def _run_debug_6(self, config: RunConfiguration, output_dir: Path) -> tuple:
        """Fuehrt debug_6_extended_comparison.py aus."""
        
        script = self.debug_folder / 'debug_6_extended_comparison.py'
        
        env = self._create_base_env(config, output_dir)
        
        print(f"      >> Fuehre debug_6 aus...")
        
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=self.scripts_folder,
                env=env,
                timeout=300
            )
            
            return (True, "OK") if result.returncode == 0 else (False, f"Exit {result.returncode}")
        except subprocess.TimeoutExpired:
            return False, "Timeout (300s)"
        except Exception as e:
            return False, str(e)
    
    # ═════════════════════════════════════════════════════════════════
    # HELPER-METHODEN
    # ═════════════════════════════════════════════════════════════════
    
    def _create_run_folder(self, config: RunConfiguration) -> tuple:
        """Erstellt Run-Ordner mit Timestamp."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"Run_{config.video_fps}_{config.calib_mode}_mode{config.analysis_mode}_{timestamp}"
        
        analysis_folder = self.vp_data.vp_folder / 'Analyse'
        analysis_folder.mkdir(exist_ok=True)
        
        run_folder = analysis_folder / run_id
        run_folder.mkdir(exist_ok=True)
        
        return run_id, run_folder
    
    def _save_run_config(self, run_folder: Path, config: RunConfiguration):
        """Speichert Config als JSON (v2.1: inkl. methods)."""
        
        config_dict = {
            'vp_code': config.vp_code,
            'video_fps': config.video_fps,
            'calib_mode': config.calib_mode,
            'analysis_mode': config.analysis_mode,
            'methods': config.methods,  # NEU v2.1
            'create_calibration': config.create_calibration,
            'calibration_pkl_path': str(config.calibration_pkl_path) if config.calibration_pkl_path else None,
            'calibration_ptgaze_pkl_path': str(config.calibration_ptgaze_pkl_path) if config.calibration_ptgaze_pkl_path else None,
            'timestamp': datetime.now().isoformat(),
            'pipeline_version': '2.1'
        }
        
        with open(run_folder / 'config_used.json', 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def _run_step_with_cache(self, step: str, config: RunConfiguration,
                             output_dir: Path, scanner: CrossRunCacheScanner,
                             run_func, *run_args) -> Tuple[bool, str]:
        """
        Fuehrt einen Pipeline-Step mit Cache-Logik aus.
        
        Args:
            step: Step-Name (z.B. 'debug_0')
            config: Run-Konfiguration
            output_dir: Output-Verzeichnis
            scanner: Cache-Scanner
            run_func: Funktion zum Ausfuehren des Steps
            *run_args: Argumente fuer run_func
        
        Returns:
            (success, message)
        """
        # Pruefe Cache-Entscheidung
        use_cache = False
        source_run = None
        
        if config.cache_decisions and config.cache_decisions.use_cache_for(step):
            use_cache = True
            decision = config.cache_decisions.decisions[step]
            source_run = decision.source_path
        
        if use_cache and source_run:
            # Versuche aus Cache zu kopieren
            if scanner.copy_step_outputs(source_run, output_dir, step):
                return True, "aus Cache (User-Entscheidung)"
            else:
                print(f"      [!] Cache-Kopie fehlgeschlagen, berechne neu...")
        
        # Kein Cache oder Cache-Kopie fehlgeschlagen -> Neu berechnen
        result = run_func(*run_args)
        
        # run_func gibt entweder (success, msg) oder (success, msg, extra) zurueck
        if len(result) == 2:
            success, msg = result
            return success, msg if not success else "erfolgreich"
        else:
            success, msg, extra = result
            return success, msg if not success else "erfolgreich"

    def _save_error_log(self, run_folder: Path, failed_step: str, 
                       error_message: str, traceback: str):
        """Speichert Error-Log."""
        
        error_dict = {
            'failed_step': failed_step,
            'error_message': error_message,
            'traceback': traceback,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(run_folder / 'error_log.json', 'w', encoding='utf-8') as f:
            json.dump(error_dict, f, indent=2, ensure_ascii=False)
    
    def _save_success_marker(self, run_folder: Path, completed_steps: List[str],
                            duration_seconds: float):
        """
        Speichert Success-Marker (NEU v2.1).
        
        Wird genutzt um erfolgreiche Runs zu identifizieren.
        """
        success_dict = {
            'success': True,
            'completed_steps': completed_steps,
            'n_steps': len(completed_steps),
            'duration_seconds': duration_seconds,
            'duration_minutes': duration_seconds / 60,
            'completed_at': datetime.now().isoformat()
        }
        
        with open(run_folder / 'run_success.json', 'w', encoding='utf-8') as f:
            json.dump(success_dict, f, indent=2, ensure_ascii=False)

# =================================================================================
# MAIN (TEST)
# =================================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ANALYSIS PIPELINE v2.0")
    print("="*70)
    print("\n[OK] analysis_pipeline.py geladen")
    print("     -> Nutze master_cli.py fuer Ausfuehrung")
    print("\nNeue Features in v2.0:")
    print("  - ptgaze-Pipeline (debug_1_ptgaze, offline_calibration_ptgaze, debug_5_ptgaze)")
    print("  - ANALYSIS_MODE Environment-Variable")
    print("  - Erweiterte Cache-Unterstuetzung fuer ptgaze")
    print("  - 10 Pipeline-Schritte (statt 7)")
    print("\n" + "="*70)
