"""
=================================================================================
STATISTICAL_CLI.PY v1.0 - Interaktives Terminal für statistische Analyse
=================================================================================

Features:
---------
- Zeigt Datenverfügbarkeit
- Führt durch Konfigurationsoptionen
- Ermöglicht selektive Datengenerierung
- Preview-Modus für vorläufige Analysen

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
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from statistic.data_availability_scanner import (
    DataAvailabilityScanner,
    DataAvailabilityReport,
    STATISTICAL_CALIB_CONFIGS,
    FRAMERATES
)
from statistic.statistical_pipeline import (
    StatisticalPipeline,
    AnalysisPlan,
    PipelineResult
)

# =================================================================================
# CLI HELPER FUNCTIONS
# =================================================================================

def print_header(text: str, width: int = 70):
    """Gibt formatierten Header aus."""
    print(f"\n{'='*width}")
    print(f"{text.center(width)}")
    print(f"{'='*width}\n")


def print_divider(char: str = '-', width: int = 70):
    """Gibt Trennlinie aus."""
    print(f"{char*width}")


def ask_choice(prompt: str, options: List[str], allow_back: bool = True,
               allow_multiple: bool = False) -> Optional[List[str]]:
    """
    Fragt User nach Auswahl.
    
    Args:
        prompt: Frage-Text
        options: Liste von Optionen
        allow_back: Erlaubt [B]ack Option
        allow_multiple: Erlaubt Mehrfachauswahl
    
    Returns:
        Liste der gewählten Optionen (oder None für Back)
    """
    print(f"\n{prompt}")
    
    for i, option in enumerate(options, 1):
        print(f"  [{i}] {option}")
    
    if allow_multiple:
        print(f"  [A] Alle auswählen")
    if allow_back:
        print(f"  [B] Zurück")
    print(f"  [Q] Beenden")
    
    while True:
        if allow_multiple:
            choice = input("\nAuswahl (Komma-getrennt für mehrere): ").strip().upper()
        else:
            choice = input("\nAuswahl: ").strip().upper()
        
        if choice == 'Q':
            print("\nAuf Wiedersehen!")
            sys.exit(0)
        
        if allow_back and choice == 'B':
            return None
        
        if allow_multiple and choice == 'A':
            return options.copy()
        
        # Parse Auswahl
        try:
            if allow_multiple and ',' in choice:
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
                selected = [options[i] for i in indices if 0 <= i < len(options)]
                if selected:
                    return selected
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return [options[idx]]
        except (ValueError, IndexError):
            pass
        
        print("Ungültige Eingabe!")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Fragt User nach Ja/Nein."""
    suffix = "[J/n]" if default else "[j/N]"
    
    while True:
        choice = input(f"{prompt} {suffix}: ").strip().upper()
        
        if choice == '':
            return default
        if choice in ['J', 'Y', 'YES', 'JA']:
            return True
        if choice in ['N', 'NO', 'NEIN']:
            return False
        
        print("Bitte J oder N eingeben!")


# =================================================================================
# STATISTICAL CLI
# =================================================================================

class StatisticalCLI:
    """
    Interaktives Terminal für statistische Analyse.
    """
    
    def __init__(self, base_folder: Path, scripts_folder: Path):
        self.base_folder = Path(base_folder)
        self.scripts_folder = Path(scripts_folder)
        
        self.pipeline = StatisticalPipeline(base_folder, scripts_folder)
        self.scanner = DataAvailabilityScanner(base_folder)
    
    def run(self):
        """Haupt-Loop des CLI."""
        
        print_header("STATISTIKMODUS v1.0")
        
        while True:
            choice = self._show_main_menu()
            
            if choice is None:
                return  # Zurück zum Hauptmenü
            
            if choice == 'overview':
                self._show_data_overview()
            elif choice == 'analyze':
                self._run_analysis_workflow()
            elif choice == 'generate':
                self._run_data_generation()
            elif choice == 'preview':
                self._run_preview_analysis()
    
    def _show_main_menu(self) -> Optional[str]:
        """Zeigt Hauptmenü."""
        
        # Schneller Scan für Übersicht
        report = self.scanner.scan_all()
        
        print_divider()
        print("DATENVERFÜGBARKEIT (Schnellübersicht)")
        print_divider()
        print(f"  VPs gesamt:        {report.n_vps_total}")
        print(f"  Mit 25Hz:          {report.n_vps_with_25hz}")
        print(f"  Mit 60Hz:          {report.n_vps_with_60hz}")
        print(f"  Mit beiden:        {report.n_vps_with_both}")
        
        # Zeige Machbarkeit
        feasibility = report.get_analysis_feasibility()
        
        print(f"\n  Forschungsfragen:")
        for ff_key, ff_info in feasibility.items():
            status = "[OK]" if ff_info['feasible'] else "[!!]"
            print(f"    {status} {ff_key}: {ff_info['n_vps']} VPs")
        
        print_divider()
        
        options = [
            "Detaillierte Datenübersicht",
            "Analyse starten",
            "Fehlende Daten generieren",
            "Preview-Analyse (nur verfügbare Daten)"
        ]
        
        choice = ask_choice("Was möchtest du tun?", options)
        
        if choice is None:
            return None
        
        if "Übersicht" in choice[0]:
            return 'overview'
        elif "Analyse starten" in choice[0]:
            return 'analyze'
        elif "generieren" in choice[0]:
            return 'generate'
        elif "Preview" in choice[0]:
            return 'preview'
        
        return None
    
    def _show_data_overview(self):
        """Zeigt detaillierte Datenübersicht."""
        
        print_header("DETAILLIERTE DATENÜBERSICHT")
        
        report = self.scanner.scan_all()
        report.print_summary()
        
        # Zeige Details pro Konfiguration
        print("\n[DETAILS PRO KONFIGURATION]")
        
        for k_label, k_info in STATISTICAL_CALIB_CONFIGS.items():
            calib_mode = k_info['calib_mode']
            print(f"\n  {k_label} ({calib_mode}):")
            print(f"    {k_info['description']}")
            
            for fps in FRAMERATES:
                vps = report.get_vps_for_config(fps, calib_mode)
                missing = report.get_vps_missing_config(fps, calib_mode)
                print(f"    {fps}: {len(vps)} komplett, {len(missing)} fehlend")
        
        input("\nDrücke ENTER um fortzufahren...")
    
    def _run_analysis_workflow(self):
        """Führt durch den Analyse-Workflow."""
        
        print_header("ANALYSE KONFIGURIEREN")
        
        # Schritt 1: Framerates
        print_divider()
        print("SCHRITT 1: Framerates")
        print_divider()
        
        fps_options = ["Nur 25Hz", "Nur 60Hz", "Beide (25Hz + 60Hz)"]
        fps_choice = ask_choice("Welche Framerates analysieren?", fps_options)
        
        if fps_choice is None:
            return
        
        if "25Hz" in fps_choice[0] and "60Hz" not in fps_choice[0]:
            framerates = ['25hz']
        elif "60Hz" in fps_choice[0] and "25Hz" not in fps_choice[0]:
            framerates = ['60hz']
        else:
            framerates = ['25hz', '60hz']
        
        # Schritt 2: Kalibrierungen
        print_divider()
        print("SCHRITT 2: Kalibrierungskonfigurationen")
        print_divider()
        
        print("\nVerfügbare Konfigurationen:")
        for k_label, k_info in STATISTICAL_CALIB_CONFIGS.items():
            print(f"  {k_label}: {k_info['description']}")
        
        calib_options = list(STATISTICAL_CALIB_CONFIGS.keys())
        calib_choice = ask_choice(
            "Welche Kalibrierungen analysieren?",
            calib_options,
            allow_multiple=True
        )
        
        if calib_choice is None:
            return
        
        calib_configs = calib_choice
        
        # Schritt 3: Plan erstellen
        print_divider()
        print("SCHRITT 3: Analyse-Plan")
        print_divider()
        
        plan = self.pipeline.create_analysis_plan(
            framerates=framerates,
            calib_configs=calib_configs,
            methods=['mediapipe', 'ptgaze']
        )
        
        # Prüfe auf fehlende Daten
        total_missing = sum(len(v) for v in plan.missing_data.values())
        
        if total_missing > 0:
            print(f"\n[!] {total_missing} VP-Config-Kombinationen fehlen")
            print("\nOptionen:")
            print("  [1] Trotzdem mit verfügbaren Daten fortfahren")
            print("  [2] Fehlende Daten zuerst generieren")
            print("  [B] Abbrechen")
            
            sub_choice = input("\nAuswahl: ").strip().upper()
            
            if sub_choice == 'B':
                return
            elif sub_choice == '2':
                self._generate_from_plan(plan)
                # Re-scan nach Generierung
                plan = self.pipeline.create_analysis_plan(
                    framerates=framerates,
                    calib_configs=calib_configs
                )
        
        # Schritt 4: Ausführung
        if not ask_yes_no("\nAnalysedatensatz jetzt erstellen?"):
            return
        
        result = self.pipeline.prepare_analysis_dataset(plan)
        
        if result.success:
            print("\n[OK] Analyse abgeschlossen!")
            print(f"\nOutput-Dateien:")
            print(f"  Frame-Level: {result.frame_level_csv}")
            print(f"  Trial-Level: {result.trial_level_csv}")
            print(f"  Lag-Report:  {result.lag_report_json}")
            print(f"\nNächster Schritt:")
            print(f"  Öffne die R-Skripte in statistics/r_scripts/")
            print(f"  und führe sie mit dem erstellten Datensatz aus.")
        else:
            print("\n[!!] Analyse fehlgeschlagen!")
            for error in result.errors:
                print(f"  - {error}")
        
        input("\nDrücke ENTER um fortzufahren...")
    
    def _run_data_generation(self):
        """Workflow für selektive Datengenerierung."""
        
        print_header("FEHLENDE DATEN GENERIEREN")
        
        report = self.scanner.scan_all()
        
        # Zeige fehlende Daten pro FF
        print("[FEHLENDE DATEN NACH FORSCHUNGSFRAGE]")
        
        options = []
        option_data = {}
        
        # FF1-FF3: Basis-Analyse
        for fps in FRAMERATES:
            calib_mode = STATISTICAL_CALIB_CONFIGS['K1']['calib_mode']
            missing = report.get_vps_missing_config(fps, calib_mode)
            
            if missing:
                key = f"FF1-FF3 ({fps}, K1)"
                options.append(f"{key}: {len(missing)} VPs fehlen")
                option_data[key] = {
                    'fps': fps,
                    'calib_mode': calib_mode,
                    'vps': missing
                }
        
        # FF4: Alle Kalibrierungen
        for k_label in ['K2', 'K3', 'K4', 'K5']:
            if k_label not in STATISTICAL_CALIB_CONFIGS:
                continue
            
            calib_mode = STATISTICAL_CALIB_CONFIGS[k_label]['calib_mode']
            
            for fps in FRAMERATES:
                missing = report.get_vps_missing_config(fps, calib_mode)
                
                if missing:
                    key = f"FF4 ({fps}, {k_label})"
                    options.append(f"{key}: {len(missing)} VPs fehlen")
                    option_data[key] = {
                        'fps': fps,
                        'calib_mode': calib_mode,
                        'vps': missing
                    }
        
        if not options:
            print("\n[OK] Keine fehlenden Daten!")
            input("\nDrücke ENTER um fortzufahren...")
            return
        
        choice = ask_choice(
            "Welche Daten generieren?",
            options,
            allow_multiple=True
        )
        
        if choice is None:
            return
        
        # Sammle zu generierende Configs
        to_generate = []
        
        for selected in choice:
            # Finde Key
            for key, data in option_data.items():
                if key in selected:
                    for vp_code in data['vps']:
                        to_generate.append({
                            'vp_code': vp_code,
                            'video_fps': data['fps'],
                            'calib_mode': data['calib_mode']
                        })
                    break
        
        print(f"\n[INFO] {len(to_generate)} Runs werden generiert...")
        
        if not ask_yes_no("Fortfahren?"):
            return
        
        # Erstelle Plan und generiere
        plan = self.pipeline.create_analysis_plan()
        
        # Überschreibe missing_data mit Auswahl
        plan.missing_data = {}
        for item in to_generate:
            key = f"{item['video_fps']}_{item['calib_mode']}"
            if key not in plan.missing_data:
                plan.missing_data[key] = []
            if item['vp_code'] not in plan.missing_data[key]:
                plan.missing_data[key].append(item['vp_code'])
        
        results = self.pipeline.generate_missing_data(plan)
        
        # Zusammenfassung
        n_success = sum(1 for r in results.values() if r.success)
        n_failed = sum(1 for r in results.values() if not r.success)
        
        print(f"\n[ZUSAMMENFASSUNG]")
        print(f"  Erfolgreich: {n_success}")
        print(f"  Fehlgeschlagen: {n_failed}")
        
        input("\nDrücke ENTER um fortzufahren...")
    
    def _run_preview_analysis(self):
        """Preview-Analyse mit nur verfügbaren Daten."""
        
        print_header("PREVIEW-ANALYSE")
        
        print("[INFO] Preview-Modus: Nur verfügbare Daten werden verwendet.")
        print("       Fehlende Daten werden ignoriert.\n")
        
        report = self.scanner.scan_all()
        
        # Finde beste verfügbare Konfiguration
        best_config = None
        best_n_vps = 0
        
        for k_label in STATISTICAL_CALIB_CONFIGS.keys():
            calib_mode = STATISTICAL_CALIB_CONFIGS[k_label]['calib_mode']
            
            for fps in FRAMERATES:
                vps = report.get_vps_for_config(fps, calib_mode)
                
                if len(vps) > best_n_vps:
                    best_n_vps = len(vps)
                    best_config = {
                        'fps': fps,
                        'k_label': k_label,
                        'calib_mode': calib_mode,
                        'vps': vps
                    }
        
        if best_config is None or best_n_vps == 0:
            print("[!!] Keine Daten verfügbar!")
            input("\nDrücke ENTER um fortzufahren...")
            return
        
        print(f"Beste verfügbare Konfiguration:")
        print(f"  Framerate: {best_config['fps']}")
        print(f"  Kalibrierung: {best_config['k_label']}")
        print(f"  VPs: {best_n_vps}")
        
        if not ask_yes_no("\nPreview-Analyse mit dieser Konfiguration starten?"):
            return
        
        plan = self.pipeline.create_analysis_plan(
            framerates=[best_config['fps']],
            calib_configs=[best_config['k_label']]
        )
        
        result = self.pipeline.prepare_analysis_dataset(plan)
        
        if result.success:
            print(f"\n[OK] Preview erstellt!")
            print(f"     {result.n_vps_processed} VPs, {result.n_trials_total} Trials")
            print(f"\n     ACHTUNG: Dies ist eine vorläufige Analyse!")
            print(f"     Für die finale Analyse alle fehlenden Daten generieren.")
        
        input("\nDrücke ENTER um fortzufahren...")
    
    def _generate_from_plan(self, plan: AnalysisPlan):
        """Generiert fehlende Daten basierend auf Plan."""
        
        total_missing = sum(len(v) for v in plan.missing_data.values())
        
        print(f"\n[GENERIERUNG] {total_missing} VP-Config-Kombinationen")
        
        # Zeige Details
        for config_key, vps in plan.missing_data.items():
            if vps:
                print(f"  {config_key}: {len(vps)} VPs")
                print(f"    {', '.join(vps[:5])}{'...' if len(vps) > 5 else ''}")
        
        if not ask_yes_no("\nGenerierung starten?"):
            return
        
        results = self.pipeline.generate_missing_data(plan)
        
        n_success = sum(1 for r in results.values() if r.success)
        n_failed = sum(1 for r in results.values() if not r.success)
        
        print(f"\n[ERGEBNIS]")
        print(f"  Erfolgreich: {n_success}")
        print(f"  Fehlgeschlagen: {n_failed}")


# =================================================================================
# MAIN
# =================================================================================

def run_statistical_cli(base_folder: Path, scripts_folder: Path):
    """Entry Point für Statistical CLI."""
    cli = StatisticalCLI(base_folder, scripts_folder)
    cli.run()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("STATISTICAL CLI v1.0 - TEST")
    print("="*70)
    
    # Test-Pfade
    base_folder = Path(r"C:\Users\imanu\Documents\Imanuel\Studium\Psy\Master\Pupillendetektion\Ergebnisse")
    scripts_folder = Path(r"C:\Users\imanu\Documents\Imanuel\Studium\Psy\Master\Pupillendetektion\Skripte\Debug\v3.0.9")
    
    if base_folder.exists():
        run_statistical_cli(base_folder, scripts_folder)
    else:
        print(f"\n[!] Test-Ordner nicht gefunden: {base_folder}")
