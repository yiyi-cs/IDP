"""
=================================================================================
MASTER CLI v2.2 - Command-Line Interface fuer Master-Pipeline
=================================================================================

Abfrage-Reihenfolge:
-------------------------
1. Video-Auswahl (60Hz/25Hz)
2. Kalibrierungs-Modi
3. PKL-Quelle
4. Analyse-Modus (1/2)
5. Methoden-Auswahl (MediaPipe/ptgaze/beide)
6. VP-Auswahl (einzelne/mehrere/alle/ohne passende Config)
7. Cache-Check pro VP (interaktiv)
8. Bestaetigung
9. Ausfuehrung

Version: 2.2
Datum: 2026-01
=================================================================================
"""

# =================================================================================
# PATH SETUP (fuer manuelle Ausfuehrung + Package-Import)
# =================================================================================
import sys
from pathlib import Path

# Fuege Projekt-Root zu sys.path hinzu
_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# =================================================================================
# IMPORTS
# =================================================================================
import os
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.vp_data_manager import RunConfig, CacheOverview
    from utils.analysis_pipeline import CacheDecisions, StepCacheDecision, CrossRunCacheScanner
import time
from datetime import datetime

# Eigene Module (aus utils/)
from utils.vp_data_manager import (
    VPDataManager, 
    VPData, 
    ValidationResult,
    RunConfig,
    CacheFile,
    CacheOverview,
    RunInfo,
    CACHE_CRITERIA,
    STEP_OUTPUT_FILES
)
from utils.analysis_pipeline import (
    AnalysisPipeline, 
    RunConfiguration, 
    RunResult,
    CacheDecisions,
    StepCacheDecision,
    CrossRunCacheScanner,
    create_default_cache_decisions,
    create_no_cache_decisions
)
from utils.calibration_mode_manager import CalibrationModes, list_available_modes


# =================================================================================
# KONFIGURATION
# =================================================================================

# Pfade (ANPASSEN FÜR DEINE UMGEBUNG!)
BASE_FOLDER = Path(r"/Users/yiyi_mac/IDP_Code/Ergebnisse") # Ergebnisse
SCRIPTS_FOLDER = Path(r"/Users/yiyi_mac/IDP_Code/Skripte") # Skripte

# =================================================================================
# HELPER-FUNKTIONEN
# =================================================================================

MASTER_DEFAULTS_FOLDER = BASE_FOLDER / 'master_defaults'
MASTER_SYNC_LOG_PATH = MASTER_DEFAULTS_FOLDER / 'master_experiment_sync_log.json'

def print_header(text: str, width: int = 70):
    """Gibt formatierten Header aus"""
    print(f"\n{'='*width}")
    print(f"{text.center(width)}")
    print(f"{'='*width}\n")


def print_divider(char: str = '─', width: int = 70):
    """Gibt Trennlinie aus"""
    print(f"{char*width}")


def ask_choice(prompt: str, options: List[str], allow_back: bool = True) -> Optional[str]:
    """
    Fragt User nach Auswahl.
    
    Args:
        prompt: Frage-Text
        options: Liste von Optionen (Zahlen werden automatisch hinzugefügt)
        allow_back: Erlaubt [B]ack Option
    
    Returns:
        Gewählte Option (oder None für Back)
    """
    
    print(f"\n{prompt}")
    
    for i, option in enumerate(options, 1):
        print(f"  [{i}] {option}")
    
    if allow_back:
        print(f"  [B] Zurück")
    
    print(f"  [Q] Beenden")
    
    while True:
        choice = input("\nAuswahl: ").strip().upper()
        
        if choice == 'Q':
            print("\n👋 Auf Wiedersehen!")
            sys.exit(0)
        
        if allow_back and choice == 'B':
            return None
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        
        print(" Ungültige Eingabe!")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Fragt User nach Ja/Nein"""
    
    suffix = "[J/n]" if default else "[j/N]"
    
    while True:
        choice = input(f"{prompt} {suffix}: ").strip().upper()
        
        if choice == '':
            return default
        
        if choice in ['J', 'Y', 'YES', 'JA']:
            return True
        
        if choice in ['N', 'NO', 'NEIN']:
            return False
        
        print(" Bitte J oder N eingeben!")


def ask_text(prompt: str, default: Optional[str] = None) -> str:
    """Fragt User nach Text-Eingabe"""
    
    if default:
        prompt += f" [{default}]"
    
    while True:
        text = input(f"{prompt}: ").strip()
        
        if text == '' and default:
            return default
        
        if text != '':
            return text
        
        print(" Eingabe erforderlich!")


# =================================================================================
# MASTER CLI
# =================================================================================

class MasterCLI:
    """
    Haupt-CLI für Master-Pipeline.
    
    Features:
    ---------
    - Interaktive Menüs
    - Analysemodus (VP-Analysen)
    - Vergleichsmodus (LMM-Statistik)
    - Progress-Tracking
    - Fehlerbehandlung
    """
    
    def __init__(self, base_folder: Path, scripts_folder: Path):
        """
        Args:
            base_folder: Basis-Ordner (Ergebnisse/)
            scripts_folder: Ordner mit debug_*.py Skripten
        """
        self.base_folder = Path(base_folder)
        self.scripts_folder = Path(scripts_folder)
        
        # Manager initialisieren
        self.vp_manager = VPDataManager(base_folder)
        
        print_header("MASTER PIPELINE MANAGER v1.0")
        
        print(f" Basis-Ordner: {base_folder}")
        print(f" Skript-Ordner: {scripts_folder}")
    
    # ═════════════════════════════════════════════════════════════════
    # HAUPT-LOOP
    # ═════════════════════════════════════════════════════════════════
    
    def run(self):
        """Haupt-Loop"""
        
        while True:
            mode = self._ask_mode()
            
            if mode == 'analysis':
                self.run_analysis_mode()
            elif mode == 'comparison':
                self.run_comparison_mode()
    
    def _ask_mode(self) -> str:
        """Fragt nach Modus"""
        
        print_header("HAUPTMENÜ")
        
        options = [
            "Analysemodus (VP-Daten analysieren)",
            "Statistikmodus (Analysedatensatz + R-Vorbereitung)"
        ]
        
        choice = ask_choice("Bitte wähle einen Modus:", options, allow_back=False)
        
        return 'analysis' if choice == options[0] else 'comparison'
    
    # ═════════════════════════════════════════════════════════════════
    # ANALYSEMODUS
    # ═════════════════════════════════════════════════════════════════
    
    def run_analysis_mode(self):
        """
        Analysemodus - Fuehrt VP-Analysen durch (v2.2: neue Reihenfolge!)
        
        Neue Reihenfolge:
        1. Config zuerst (Video, Calib, Mode, Methods)
        2. VP-Auswahl basierend auf Config
        3. Cache-Dialog pro VP
        4. Ausfuehrung
        """
        
        print_header("ANALYSEMODUS v2.2")
        
        # ══════════════════════════════════════════════════════════════
        # SCHRITT 1-5: KONFIGURATION ZUERST!
        # ══════════════════════════════════════════════════════════════
        
        print("\n[INFO] Zuerst Konfiguration festlegen, dann VP-Auswahl\n")
        
        # Schritt 1: Video-Auswahl
        video_options = self._ask_video_selection()
        if video_options is None:
            return
        
        # Schritt 2: Kalibrierungs-Modi
        calib_modes = self._ask_calibration_modes()
        if calib_modes is None:
            return
        
        # Schritt 3: PKL-Quelle
        create_calibration = self._ask_calibration_source()
        
        # Schritt 4: Analyse-Modus
        analysis_mode = self._ask_analysis_mode()
        
        # Schritt 5: Methoden-Auswahl
        methods = self._ask_method_selection()
        
        # Erstelle RunConfig fuer VP-Filterung (nutze erste Optionen)
        # Bei mehreren video_options/calib_modes wird pro Kombination geprueft
        base_config = RunConfig(
            video_fps=video_options[0],
            calib_mode=calib_modes[0],
            analysis_mode=analysis_mode,
            methods=methods
        )
        
        # ══════════════════════════════════════════════════════════════
        # SCHRITT 6: VP-AUSWAHL (BASIEREND AUF CONFIG!)
        # ══════════════════════════════════════════════════════════════
        
        vp_codes = self._ask_vp_selection_with_config(base_config, video_options, calib_modes)
        if vp_codes is None:
            return
        
        # ══════════════════════════════════════════════════════════════
        # SCHRITT 7: CACHE-DIALOG PRO VP
        # ══════════════════════════════════════════════════════════════
        
        cache_decisions_per_vp = self._ask_cache_decisions(
            vp_codes, video_options, calib_modes, analysis_mode, methods
        )
        
        # ══════════════════════════════════════════════════════════════
        # SCHRITT 8: ZUSAMMENFASSUNG & BESTAETIGUNG
        # ══════════════════════════════════════════════════════════════
        
        if not self._confirm_analysis_v2(
            vp_codes, video_options, calib_modes, create_calibration, 
            analysis_mode, methods, cache_decisions_per_vp
        ):
            print("\n[!] Analyse abgebrochen")
            return
        
        # ══════════════════════════════════════════════════════════════
        # SCHRITT 9: AUSFUEHRUNG
        # ══════════════════════════════════════════════════════════════
        
        self._execute_analyses_v2(
            vp_codes, video_options, calib_modes, create_calibration,
            analysis_mode, methods, cache_decisions_per_vp
        )

    def _ask_vp_selection(self) -> Optional[List[str]]:
        """Fragt nach VP-Auswahl"""
        
        print_divider()
        print("VP-AUSWAHL")
        print_divider()
        
        options = [
            "Einzelne VP analysieren (VP-Code eingeben)",
            "Alle neuen VPs (ohne Analyse-Ordner)"
        ]
        
        choice = ask_choice("Welche VPs analysieren?", options)
        
        if choice is None:
            return None
        
        if choice == options[0]:
            # Einzelne VP
            all_vps = self.vp_manager.scan_all_vps()
            
            print(f"\nVerfügbare VPs: {', '.join(all_vps)}")
            
            vp_code = ask_text("VP-Code eingeben (z.B. bjs4)")
            
            if vp_code not in all_vps:
                print(f"\n VP '{vp_code}' nicht gefunden!")
                return None
            
            # Validiere VP
            vp_data = self.vp_manager.load_vp_data(vp_code)
            validation = self.vp_manager.validate_vp_data(vp_data)
            
            self.vp_manager.print_vp_summary(vp_data)
            print(f"\n{validation}")
            
            if not validation.is_valid:
                if not ask_yes_no("VP hat Fehler. Trotzdem fortfahren?", default=False):
                    return None
            
            return [vp_code]
        
        else:
            # Alle neuen VPs
            new_vps = self.vp_manager.scan_new_vps()
            
            if not new_vps:
                print("\n Keine neuen VPs gefunden (alle bereits analysiert)")
                return None
            
            print(f"\n Neue VPs: {', '.join(new_vps)} ({len(new_vps)} gesamt)")
            
            if not ask_yes_no(f"Alle {len(new_vps)} VPs analysieren?"):
                return None
            
            return new_vps
    
    def _ask_vp_selection_with_config(self, base_config: 'RunConfig',
                                      video_options: List[str],
                                      calib_modes: List[str]) -> Optional[List[str]]:
        """
        VP-Auswahl BASIEREND auf bereits gewaehlter Config (v2.2).
        
        Zeigt:
        - VPs ohne passende Analyse (fuer diese Config!)
        - VPs mit fehlgeschlagenen Runs
        - Option fuer manuelle Mehrfach-Eingabe
        """
        
        print_divider()
        print("VP-AUSWAHL")
        print_divider()
        
        print(f"\nAktuelle Config: {base_config}")
        
        # Sammle Statistik
        all_vps = self.vp_manager.scan_all_vps()
        
        # VPs ohne passende erfolgreiche Analyse
        vps_without_config = []
        vps_with_failed = []
        vps_completed = []
        
        for vp_code in all_vps:
            try:
                status = self.vp_manager.get_vp_analysis_status(vp_code, base_config)
                
                if status['status'] == 'completed':
                    vps_completed.append(vp_code)
                elif status['status'] == 'failed':
                    vps_with_failed.append(vp_code)
                elif status['status'] in ['not_started', 'no_video']:
                    if status.get('has_video', True):  # Nur wenn Video vorhanden
                        vps_without_config.append(vp_code)
            except Exception:
                pass
        
        # Zeige Statistik
        print(f"\n[STATISTIK]")
        print(f"  Gesamt: {len(all_vps)} VPs")
        print(f"  Bereits analysiert (diese Config): {len(vps_completed)}")
        print(f"  Fehlgeschlagen: {len(vps_with_failed)}")
        print(f"  Noch nicht analysiert: {len(vps_without_config)}")
        
        # Optionen
        options = [
            "Einzelne VP eingeben",
            "Mehrere VPs eingeben (Komma-getrennt)",
            "Alle VPs",
        ]
        
        if vps_without_config:
            options.append(f"VPs ohne passende Analyse ({len(vps_without_config)})")
        
        if vps_with_failed:
            options.append(f"VPs mit fehlgeschlagenen Runs ({len(vps_with_failed)})")
        
        choice = ask_choice("Welche VPs analysieren?", options)
        
        if choice is None:
            return None
        
        # ════════════════════════════════════════════════════════════
        # Option 1: Einzelne VP
        # ════════════════════════════════════════════════════════════
        if choice == options[0]:
            print(f"\nVerfuegbare VPs: {', '.join(all_vps)}")
            
            vp_code = ask_text("VP-Code eingeben (z.B. ldj9)")
            
            if vp_code not in all_vps:
                print(f"\n[!] VP '{vp_code}' nicht gefunden!")
                return None
            
            # Zeige Status
            status = self.vp_manager.get_vp_analysis_status(vp_code, base_config)
            print(f"\nStatus: {status['status']} - {status['message']}")
            
            # Validiere VP
            vp_data = self.vp_manager.load_vp_data(vp_code)
            validation = self.vp_manager.validate_vp_data(vp_data)
            
            self.vp_manager.print_vp_summary(vp_data)
            
            if not validation.is_valid:
                if not ask_yes_no("VP hat Fehler. Trotzdem fortfahren?", default=False):
                    return None
            
            return [vp_code]
        
        # ════════════════════════════════════════════════════════════
        # Option 2: Mehrere VPs (NEU v2.2!)
        # ════════════════════════════════════════════════════════════
        elif choice == options[1]:
            print(f"\nVerfuegbare VPs: {', '.join(all_vps)}")
            print("\nFormat: VP-Codes mit Komma und Leerzeichen trennen")
            print("Beispiel: ldj9, abc1, xyz3")
            
            vp_input = ask_text("VP-Codes eingeben")
            
            # Parse Input (Komma + optionales Leerzeichen)
            input_codes = [vp.strip() for vp in vp_input.replace(' ', '').split(',')]
            input_codes = [vp for vp in input_codes if vp]  # Leere entfernen
            
            # Validiere
            valid_vps = []
            invalid_vps = []
            
            for vp_code in input_codes:
                if vp_code in all_vps:
                    valid_vps.append(vp_code)
                else:
                    invalid_vps.append(vp_code)
            
            if invalid_vps:
                print(f"\n[!] Unbekannte VP-Codes: {', '.join(invalid_vps)}")
            
            if not valid_vps:
                print(f"\n[!] Keine gueltigen VPs!")
                return None
            
            print(f"\n[OK] {len(valid_vps)} gueltige VPs: {', '.join(valid_vps)}")
            
            # Zeige Status pro VP
            print(f"\n[STATUS]")
            for vp_code in valid_vps:
                status = self.vp_manager.get_vp_analysis_status(vp_code, base_config)
                status_icon = {
                    'completed': '[OK]',
                    'failed': '[X]',
                    'not_started': '[NEW]',
                    'no_video': '[!]'
                }.get(status['status'], '[?]')
                print(f"  {status_icon} {vp_code}: {status['message'][:50]}")
            
            if not ask_yes_no(f"\nMit {len(valid_vps)} VPs fortfahren?"):
                return None
            
            return valid_vps
        
        # ════════════════════════════════════════════════════════════
        # Option 3: Alle VPs
        # ════════════════════════════════════════════════════════════
        elif choice == options[2]:
            print(f"\n[INFO] Alle {len(all_vps)} VPs werden analysiert")
            
            if not ask_yes_no(f"Alle {len(all_vps)} VPs analysieren?"):
                return None
            
            return all_vps
        
        # ════════════════════════════════════════════════════════════
        # Option 4: VPs ohne passende Analyse
        # ════════════════════════════════════════════════════════════
        elif "ohne passende Analyse" in choice:
            if not vps_without_config:
                print("\n[INFO] Alle VPs haben bereits eine passende Analyse!")
                return None
            
            print(f"\n[INFO] {len(vps_without_config)} VPs ohne passende Analyse:")
            print(f"       {', '.join(vps_without_config)}")
            
            if not ask_yes_no(f"Diese {len(vps_without_config)} VPs analysieren?"):
                return None
            
            return vps_without_config
        
        # ════════════════════════════════════════════════════════════
        # Option 5: VPs mit fehlgeschlagenen Runs
        # ════════════════════════════════════════════════════════════
        elif "fehlgeschlagenen Runs" in choice:
            if not vps_with_failed:
                print("\n[INFO] Keine VPs mit fehlgeschlagenen Runs!")
                return None
            
            print(f"\n[INFO] {len(vps_with_failed)} VPs mit fehlgeschlagenen Runs:")
            for vp_code in vps_with_failed:
                status = self.vp_manager.get_vp_analysis_status(vp_code, base_config)
                print(f"  [X] {vp_code}: {status.get('message', 'Fehler')[:50]}")
            
            if not ask_yes_no(f"Diese {len(vps_with_failed)} VPs erneut analysieren?"):
                return None
            
            return vps_with_failed
        
        return None

    def _ask_video_selection(self) -> Optional[List[str]]:
        """Fragt nach Video-Auswahl"""
        
        print_divider()
        print("VIDEO-AUSWAHL")
        print_divider()
        
        options = [
            "Nur 60 Hz",
            "Nur 25 Hz",
            "Beide (separate Runs + Vergleich)"
        ]
        
        choice = ask_choice("Welche Videos analysieren?", options)
        
        if choice is None:
            return None
        
        if choice == options[0]:
            return ['60hz']
        elif choice == options[1]:
            return ['25hz']
        else:
            return ['60hz', '25hz']
    
    def _ask_calibration_modes(self) -> Optional[List[str]]:
        """Fragt nach Kalibrierungs-Modi (v2.2: erweiterte Optionen)"""
        
        print_divider()
        print("KALIBRIERUNGS-MODUS")
        print_divider()
        
        print("\nVerfuegbare Modi:")
        print("")
        print("  [Vollstaendig]")
        print("    FullCalib     : Alle Phasen (beg+mid+end) + Fixationen")
        print("")
        print("  [Ohne Mid-Kalibrierung]")
        print("    BegEnd        : beg + end (ohne mid, ohne Fixationen)")
        print("    BegEndFix     : beg + end + Fixationen")
        print("")
        print("  [Nur Anfang]")
        print("    OnlyBeg       : Nur beg (alle 20 Punkte)")
        print("    BegFirst10    : Nur erste 10 Punkte von beg")
        print("    BegSecond10   : Nur zweite 10 Punkte von beg")
        print("    OnlyBegFix    : Nur beg + Fixationen") 
        print("")
        print("  [Kombinationen]")
        print("    BegFirst10End : Erste 10 von beg + end")
        print("    BegSecond10End: Zweite 10 von beg + end")
        print("    BegFirst10EndFix: Erste 10 von beg + end + Fixationen")
        print("")
        print("  [Nur Ende]")
        print("    OnlyEnd       : Nur end-Kalibrierung")
        
        options = [
            "FullCalib (Standard - alle Daten)",
            "BegEnd (ohne Mid, ohne Fixationen)",
            "OnlyBeg (nur Anfangs-Kalibrierung)",
            "BegFirst10EndFix (Erste 10 von beg + end + Fixationen)",
            "Mehrere Modi auswaehlen",
            "Custom (interaktiv konfigurieren)"
        ]
        
        choice = ask_choice("Welcher Modus?", options)
        
        if choice is None:
            return None
        
        if choice == options[0]:
            return ['FullCalib']
        elif choice == options[1]:
            return ['BegEnd']
        elif choice == options[2]:
            return ['OnlyBeg']
        elif choice == options[3]:
            return ['BegFirst10EndFix']
        elif choice == options[4]:
            # Mehrere Modi
            print("\nVerfuegbare Modi (Komma-getrennt eingeben):")
            print("  FullCalib, BegEnd, BegEndFix, OnlyBeg, OnlyBegFix,")
            print("  BegFirst10, BegSecond10, BegFirst10End, BegSecond10End, OnlyEnd")
            
            modes_input = ask_text("Modi eingeben (z.B. FullCalib, BegEnd)")
            modes = [m.strip() for m in modes_input.split(',')]
            
            # Validiere
            valid_modes = ['FullCalib', 'BegEnd', 'BegEndFix', 'OnlyBeg', 'OnlyBegFix',
                          'BegFirst10', 'BegSecond10', 'BegFirst10End', 'BegSecond10End', 'OnlyEnd', 'BegFirst10EndFix']
            
            invalid = [m for m in modes if m not in valid_modes]
            if invalid:
                print(f"[!] Ungueltige Modi: {', '.join(invalid)}")
                return None
            
            return modes
        elif choice == options[5]:
            # Custom
            return self._ask_custom_calibration_mode()
        
        return ['FullCalib']
    
    def _ask_custom_calibration_mode(self) -> Optional[List[str]]:
        """Interaktive Custom-Konfiguration fuer Kalibrierung."""
        
        print_divider()
        print("CUSTOM KALIBRIERUNGS-KONFIGURATION")
        print_divider()
        
        # Phasen auswaehlen
        print("\n[1] Welche Phasen verwenden?")
        use_beg = ask_yes_no("  BEG-Kalibrierung verwenden?", default=True)
        use_mid = ask_yes_no("  MID-Kalibrierung verwenden?", default=False)
        use_end = ask_yes_no("  END-Kalibrierung verwenden?", default=True)
        
        phases = []
        if use_beg:
            phases.append('beg')
        if use_mid:
            phases.append('mid')
        if use_end:
            phases.append('end')
        
        if not phases:
            print("[!] Mindestens eine Phase muss ausgewaehlt werden!")
            return None
        
        # BEG Range
        beg_range = (1, 20)
        if use_beg:
            print("\n[2] BEG-Kalibrierung hat 20 Punkte. Welche verwenden?")
            beg_options = [
                "Alle 20 Punkte (1-20)",
                "Nur erste 10 (1-10)",
                "Nur zweite 10 (11-20)",
                "Custom Range"
            ]
            beg_choice = ask_choice("BEG Punkte?", beg_options)
            
            if beg_choice == beg_options[0]:
                beg_range = (1, 20)
            elif beg_choice == beg_options[1]:
                beg_range = (1, 10)
            elif beg_choice == beg_options[2]:
                beg_range = (11, 20)
            else:
                start = int(ask_text("Start-Punkt (1-20)") or "1")
                end = int(ask_text("End-Punkt (1-20)") or "20")
                beg_range = (max(1, min(20, start)), max(1, min(20, end)))
        
        # Fixationen
        print("\n[3] Fixationskreuz-Phasen als zusaetzliche Kalibrierpunkte?")
        use_fix = ask_yes_no("  Fixationen verwenden?", default=False)
        
        # Zusammenfassung
        print("\n[ZUSAMMENFASSUNG]")
        print(f"  Phasen: {', '.join(phases)}")
        if use_beg:
            print(f"  BEG Range: {beg_range[0]}-{beg_range[1]}")
        print(f"  Fixationen: {'Ja' if use_fix else 'Nein'}")
        
        if not ask_yes_no("\nMit dieser Konfiguration fortfahren?"):
            return None
        
        # Speichere Custom-Config in Environment (wird von offline_calibration gelesen)
        os.environ['PIPELINE_CALIB_PHASES'] = ','.join(phases)
        os.environ['PIPELINE_CALIB_BEG_RANGE'] = f"{beg_range[0]},{beg_range[1]}"
        os.environ['PIPELINE_CALIB_USE_FIXATIONS'] = 'true' if use_fix else 'false'
        
        return ['Custom']
    
    def _ask_calibration_source(self) -> bool:
        """Fragt nach PKL-Quelle"""
        
        print_divider()
        print("KALIBRIERUNG")
        print_divider()
        
        options = [
            "Automatisch erstellen (offline_calibration.py)",
            "Eigene PKL-Datei angeben (noch nicht unterstützt)"
        ]
        
        choice = ask_choice("PKL-Datei:", options)
        
        if choice is None or choice == options[1]:
            print("\n Eigene PKL noch nicht implementiert")
            return True  # Fallback auf Auto
        
        return True  # Automatisch erstellen
    
    def _ask_analysis_mode(self) -> int:
        """Fragt nach Analysis Mode (1=Standalone, 2=Comparison)"""
        
        print_divider()
        print("ANALYSE-MODUS")
        print_divider()
        
        print("\nModus-Uebersicht:")
        print("  Modus 1: STANDALONE WEBCAM")
        print("           - Nur Webcam-Video (keine EyeLink-Daten)")
        print("           - Skripte: debug_0 >> debug_1 >> debug_5 >> debug_6")
        print("")
        print("  Modus 2: COMPARISON (WEBCAM + EYELINK)")
        print("           - Webcam + EyeLink parallel")
        print("           - Skripte: debug_0 >> debug_1 >> debug_3 >> debug_4 >> debug_5 >> debug_6")
        print("           - 3-Wege-Vergleich: MediaPipe vs. ptgaze vs. EyeLink")
        
        options = [
            "Modus 1: Standalone Webcam (ohne EyeLink)",
            "Modus 2: Comparison (mit EyeLink) [EMPFOHLEN]"
        ]
        
        choice = ask_choice("Welcher Analyse-Modus?", options)
        
        if choice is None:
            return 2  # Default bei Abbruch
        
        if choice == options[0]:
            return 1
        else:
            return 2
        
    def _ask_method_selection(self) -> str:
        """Fragt nach Detektions-Methode (MediaPipe, ptgaze, beide)"""
        
        print_divider()
        print("METHODEN-AUSWAHL")
        print_divider()
        
        print("\nVerfuegbare Methoden:")
        print("  MediaPipe: Face Mesh + Iris Tracking")
        print("")
        print("  ptgaze:    ETH-XGaze CNN Model")
        print("")
        print("  Beide:     MediaPipe + ptgaze parallel")
        
        options = [
            "Nur MediaPipe",
            "Nur ptgaze",
            "Beide Methoden parallel"
        ]
        
        choice = ask_choice("Welche Methode(n)?", options)
        
        if choice is None:
            return 'both'  # Default bei Abbruch
        if choice == options[0]:
            return 'mediapipe'
        elif choice == options[1]:
            return 'ptgaze'
        else:
            return 'both'
    
    def _ask_cache_decisions(self, vp_codes: List[str], video_options: List[str],
                            calib_modes: List[str], analysis_mode: int,
                            methods: str) -> Dict[str, Dict[str, 'CacheDecisions']]:
        """
        Interaktiver Cache-Dialog pro VP (NEU v2.2).
        
        Zeigt fuer jede VP und Config-Kombination:
        - Welche Cache-Daten verfuegbar sind
        - Welche kompatibel sind
        - Laesst User entscheiden
        
        Returns:
            Dict[vp_code, Dict[config_key, CacheDecisions]]
        """
        
        print_header("CACHE-KONFIGURATION")
        
        all_cache_decisions = {}
        
        # Frage ob globale oder individuelle Entscheidung
        print("Cache-Optionen:")
        print("  [A] Automatisch (alle kompatiblen Daten aus Cache)")
        print("  [N] Keine Cache-Nutzung (alles neu berechnen)")
        print("  [I] Individuell pro VP entscheiden")
        
        global_choice = input("\nCache-Strategie [A/N/I]: ").strip().upper()
        
        if global_choice == 'N':
            # Kein Cache fuer alle
            print("\n[INFO] Cache deaktiviert - alle Daten werden neu berechnet")
            
            for vp_code in vp_codes:
                all_cache_decisions[vp_code] = {}
                
                for video_fps in video_options:
                    for calib_mode in calib_modes:
                        config_key = f"{video_fps}_{calib_mode}"
                        
                        temp_config = RunConfiguration(
                            vp_code=vp_code,
                            video_fps=video_fps,
                            calib_mode=calib_mode,
                            create_calibration=True,
                            analysis_mode=analysis_mode,
                            methods=methods
                        )
                        
                        all_cache_decisions[vp_code][config_key] = create_no_cache_decisions(temp_config)
            
            return all_cache_decisions
        
        elif global_choice == 'A' or global_choice == '':
            # Automatisch fuer alle
            print("\n[INFO] Automatischer Cache - alle kompatiblen Daten werden wiederverwendet")
            
            for vp_code in vp_codes:
                all_cache_decisions[vp_code] = {}
                
                try:
                    vp_data = self.vp_manager.load_vp_data(vp_code)
                except:
                    continue
                
                scanner = CrossRunCacheScanner(vp_data.vp_folder)
                
                for video_fps in video_options:
                    for calib_mode in calib_modes:
                        config_key = f"{video_fps}_{calib_mode}"
                        
                        temp_config = RunConfiguration(
                            vp_code=vp_code,
                            video_fps=video_fps,
                            calib_mode=calib_mode,
                            create_calibration=True,
                            analysis_mode=analysis_mode,
                            methods=methods
                        )
                        
                        cache_decisions = create_default_cache_decisions(temp_config, scanner)
                        all_cache_decisions[vp_code][config_key] = cache_decisions
                        
                        if cache_decisions.n_from_cache > 0:
                            print(f"  {vp_code} ({config_key}): {cache_decisions.n_from_cache} Steps aus Cache")
            
            return all_cache_decisions
        
        else:
            # Individuell pro VP
            print("\n[INFO] Individuelle Cache-Entscheidung pro VP")
            
            for i, vp_code in enumerate(vp_codes, 1):
                print(f"\n{'='*60}")
                print(f"VP {i}/{len(vp_codes)}: {vp_code}")
                print(f"{'='*60}")
                
                all_cache_decisions[vp_code] = {}
                
                try:
                    vp_data = self.vp_manager.load_vp_data(vp_code)
                except Exception as e:
                    print(f"[!] Fehler beim Laden: {e}")
                    continue
                
                for video_fps in video_options:
                    for calib_mode in calib_modes:
                        config_key = f"{video_fps}_{calib_mode}"
                        
                        target_config = RunConfig(
                            video_fps=video_fps,
                            calib_mode=calib_mode,
                            analysis_mode=analysis_mode,
                            methods=methods
                        )
                        
                        temp_run_config = RunConfiguration(
                            vp_code=vp_code,
                            video_fps=video_fps,
                            calib_mode=calib_mode,
                            create_calibration=True,
                            analysis_mode=analysis_mode,
                            methods=methods
                        )
                        
                        # Hole Cache-Uebersicht
                        cache_overview = self.vp_manager.get_cache_overview(vp_data, target_config)
                        
                        print(f"\n[{config_key}]")
                        
                        if not cache_overview.compatible_files:
                            print("  Keine kompatiblen Cache-Daten vorhanden")
                            all_cache_decisions[vp_code][config_key] = create_no_cache_decisions(temp_run_config)
                            continue
                        
                        # Zeige verfuegbare Cache-Daten
                        print("  Verfuegbare Cache-Daten:")
                        
                        step_order = [
                            'debug_0', 'debug_1_mediapipe', 'debug_1_ptgaze',
                            'offline_calibration', 'offline_calibration_ptgaze',
                            'debug_3', 'debug_4',
                            'debug_5_mediapipe', 'debug_5_ptgaze', 'debug_6'
                        ]
                        
                        for step in step_order:
                            if step in cache_overview.compatible_files:
                                cf = cache_overview.compatible_files[step]
                                print(f"    [OK] {step}: {cf.source_run}")
                            elif step in cache_overview.missing_steps:
                                print(f"    [ ] {step}: (neu berechnen)")
                        
                        # Frage nach Entscheidung
                        print(f"\n  Optionen:")
                        print(f"    [A] Alle kompatiblen aus Cache")
                        print(f"    [N] Alles neu berechnen")
                        print(f"    [D] Details (einzeln auswaehlen)")
                        
                        sub_choice = input(f"  Auswahl [{config_key}]: ").strip().upper()
                        
                        scanner = CrossRunCacheScanner(vp_data.vp_folder)
                        
                        if sub_choice == 'N':
                            all_cache_decisions[vp_code][config_key] = create_no_cache_decisions(temp_run_config)
                        elif sub_choice == 'D':
                            # Detail-Auswahl
                            cache_decisions = self._ask_detailed_cache_decisions(
                                vp_code, temp_run_config, cache_overview, scanner
                            )
                            all_cache_decisions[vp_code][config_key] = cache_decisions
                        else:
                            # Default: Automatisch
                            all_cache_decisions[vp_code][config_key] = create_default_cache_decisions(
                                temp_run_config, scanner
                            )
            
            return all_cache_decisions
    
    def _ask_detailed_cache_decisions(self, vp_code: str, 
                                      config: 'RunConfiguration',
                                      cache_overview: 'CacheOverview',
                                      scanner: 'CrossRunCacheScanner') -> 'CacheDecisions':
        """
        Detail-Auswahl: User entscheidet pro Step.
        """
        
        decisions = {}
        
        step_order = [
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
        
        print(f"\n  Detail-Auswahl fuer {vp_code}:")
        print(f"  (Enter = Cache nutzen, N = Neu berechnen)")
        
        for step, description in step_order:
            if step in cache_overview.compatible_files:
                cf = cache_overview.compatible_files[step]
                choice = input(f"    {description} [{cf.source_run}] [Y/n]: ").strip().upper()
                
                if choice == 'N':
                    decisions[step] = StepCacheDecision(
                        step=step,
                        use_cache=False,
                        reason="User hat Neu-Berechnung gewaehlt"
                    )
                else:
                    decisions[step] = StepCacheDecision(
                        step=step,
                        use_cache=True,
                        source_run=cf.source_run,
                        source_path=cf.path.parent,
                        reason="User hat Cache gewaehlt"
                    )
            else:
                decisions[step] = StepCacheDecision(
                    step=step,
                    use_cache=False,
                    reason="Keine kompatiblen Daten"
                )
        
        return CacheDecisions(
            vp_code=vp_code,
            target_config=config.to_run_config(),
            decisions=decisions
        )

    def _confirm_analysis(self, vp_codes: List[str], video_options: List[str],
                         calib_modes: List[str], create_calibration: bool,
                         analysis_mode: int = 2, methods: str = 'both') -> bool:
        """Zeigt Zusammenfassung & fragt nach Bestaetigung (v2.1: inkl. Methoden)"""
        
        print_divider()
        print("ZUSAMMENFASSUNG")
        print_divider()
        
        mode_desc = "Standalone (ohne EyeLink)" if analysis_mode == 1 else "Comparison (mit EyeLink)"
        methods_desc = {
            'mediapipe': 'Nur MediaPipe',
            'ptgaze': 'Nur ptgaze',
            'both': 'MediaPipe + ptgaze'
        }.get(methods, 'both')
        
        print(f"\n Konfiguration:")
        print(f"  - VPs: {', '.join(vp_codes)} ({len(vp_codes)} gesamt)")
        print(f"  - Videos: {', '.join(video_options)}")
        print(f"  - Kalibrierungs-Modi: {', '.join(calib_modes)}")
        print(f"  - PKL: {'Automatisch' if create_calibration else 'Eigene'}")
        print(f"  - Analyse-Modus: {analysis_mode} ({mode_desc})")
        print(f"  - Methoden: {methods_desc}")
        
        # Berechne Anzahl Runs
        n_runs = len(vp_codes) * len(video_options) * len(calib_modes)
        
        print(f"\n Gesamt: {n_runs} Runs")
        print(f"⏱️ Geschätzte Dauer: ~{n_runs * 15} Minuten ({n_runs * 15 / 60:.1f} Stunden)")
        
        return ask_yes_no("\nAnalysen starten?")
    
    def _confirm_analysis_v2(self, vp_codes: List[str], video_options: List[str],
                            calib_modes: List[str], create_calibration: bool,
                            analysis_mode: int, methods: str,
                            cache_decisions: Dict) -> bool:
        """
        Zusammenfassung mit Cache-Info (v2.2).
        """
        
        print_divider()
        print("ZUSAMMENFASSUNG")
        print_divider()
        
        mode_desc = "Standalone (ohne EyeLink)" if analysis_mode == 1 else "Comparison (mit EyeLink)"
        methods_desc = {
            'mediapipe': 'Nur MediaPipe',
            'ptgaze': 'Nur ptgaze',
            'both': 'MediaPipe + ptgaze'
        }.get(methods, 'beide')
        
        print(f"\n[KONFIGURATION]")
        print(f"  VPs: {', '.join(vp_codes)} ({len(vp_codes)} gesamt)")
        print(f"  Videos: {', '.join(video_options)}")
        print(f"  Kalibrierungs-Modi: {', '.join(calib_modes)}")
        print(f"  PKL: {'Automatisch' if create_calibration else 'Eigene'}")
        print(f"  Analyse-Modus: {analysis_mode} ({mode_desc})")
        print(f"  Methoden: {methods_desc}")
        
        # Berechne Anzahl Runs
        n_runs = len(vp_codes) * len(video_options) * len(calib_modes)
        
        # Cache-Statistik
        total_from_cache = 0
        total_fresh = 0
        
        for vp_code, configs in cache_decisions.items():
            for config_key, decisions in configs.items():
                if hasattr(decisions, 'n_from_cache'):
                    total_from_cache += decisions.n_from_cache
                    total_fresh += decisions.n_fresh
        
        print(f"\n[CACHE]")
        print(f"  Steps aus Cache: {total_from_cache}")
        print(f"  Steps neu berechnen: {total_fresh}")
        
        print(f"\n[AUSFUEHRUNG]")
        print(f"  Gesamt: {n_runs} Runs")
        
        # Geschaetzte Zeit (reduziert wenn Cache)
        base_time_per_run = 15  # Minuten
        cache_factor = 0.5 if total_from_cache > 0 else 1.0
        estimated_time = n_runs * base_time_per_run * cache_factor
        
        print(f"  Geschaetzte Dauer: ~{estimated_time:.0f} Minuten ({estimated_time/60:.1f} Stunden)")
        
        return ask_yes_no("\nAnalysen starten?")

    def _execute_analyses(self, vp_codes: List[str], video_options: List[str],
                         calib_modes: List[str], create_calibration: bool,
                         analysis_mode: int = 2, methods: str = 'both'):
        """Fuehrt alle Analysen aus (v2.2: mit Fehlertoleranz + Zusammenfassung)"""
        
        print_header("AUSFUEHRUNG")
        
        # Ergebnis-Tracking (NEU v2.2)
        all_results = []
        successful_runs = []
        failed_runs = []
        skipped_runs = []
        
        total_runs = len(vp_codes) * len(video_options) * len(calib_modes)
        current_run = 0
        
        for vp_code in vp_codes:
            # Lade VP-Daten (v2.2 FIX: korrekte Methode load_vp_data)
            try:
                vp_data = self.vp_manager.load_vp_data(vp_code)
            except (ValueError, FileNotFoundError) as e:
                print(f"\n[SKIP] VP {vp_code}: {e}")
                skipped_runs.append({
                    'vp_code': vp_code,
                    'reason': str(e),
                    'video_fps': 'alle',
                    'calib_mode': 'alle'
                })
                current_run += len(video_options) * len(calib_modes)
                continue
            
            if vp_data is None:
                print(f"\n[SKIP] VP {vp_code}: Daten nicht gefunden")
                skipped_runs.append({
                    'vp_code': vp_code,
                    'reason': 'VP-Daten nicht gefunden',
                    'video_fps': 'alle',
                    'calib_mode': 'alle'
                })
                current_run += len(video_options) * len(calib_modes)
                continue
            
            # ══════════════════════════════════════════════════════════════
            # NEU v2.3: Setze Environment-Variablen für Master-JSON Support
            # ══════════════════════════════════════════════════════════════
            
            # VP-Code für Korrekturwert-Lookup in debug_0
            os.environ['PIPELINE_VP_CODE'] = vp_code
            
            # Master-JSON Pfad für Fallback wenn VP-spezifische JSON fehlt
            if MASTER_SYNC_LOG_PATH.exists():
                os.environ['PIPELINE_MASTER_SYNC_LOG_PATH'] = str(MASTER_SYNC_LOG_PATH)
                print(f"    [INFO] Master-JSON verfügbar: {MASTER_SYNC_LOG_PATH.name}")
                # Master-Calibration Ordner (selber Ordner wie Master-Sync-Log)
                os.environ['PIPELINE_MASTER_CALIBRATION_FOLDER'] = str(MASTER_DEFAULTS_FOLDER)
                print(f"    [INFO] Master-Calibration-Folder gesetzt: {MASTER_DEFAULTS_FOLDER}")               
            else:
                os.environ.pop('PIPELINE_MASTER_SYNC_LOG_PATH', None)
            
            pipeline = AnalysisPipeline(vp_data, self.scripts_folder)
            
            for video_fps in video_options:
                for calib_mode in calib_modes:
                    current_run += 1
                    
                    print(f"\n{'='*70}")
                    print(f"RUN {current_run}/{total_runs}")
                    print(f"{'='*70}")
                    
                    # Erstelle Config (v2.2: inkl. methods)
                    config = RunConfiguration(
                        vp_code=vp_code,
                        video_fps=video_fps,
                        calib_mode=calib_mode,
                        create_calibration=create_calibration,
                        analysis_mode=analysis_mode,
                        methods=methods
                    )
                    
                    # Fuehre Pipeline aus (Fehler werden intern behandelt!)
                    try:
                        result = pipeline.run_full_analysis(config)
                        all_results.append(result)
                        
                        if result.success:
                            successful_runs.append(result)
                        elif result.skipped:
                            skipped_runs.append({
                                'vp_code': vp_code,
                                'reason': result.skip_reason or result.error_message,
                                'video_fps': video_fps,
                                'calib_mode': calib_mode
                            })
                        else:
                            failed_runs.append(result)
                    
                    except Exception as e:
                        # Unerwarteter Fehler - trotzdem weitermachen!
                        print(f"\n[!] Unerwarteter Fehler bei VP {vp_code}: {e}")
                        print(f"    >> Fahre mit naechster Konfiguration fort...\n")
                        
                        skipped_runs.append({
                            'vp_code': vp_code,
                            'reason': f'Unerwarteter Fehler: {str(e)}',
                            'video_fps': video_fps,
                            'calib_mode': calib_mode
                        })
        
        # ══════════════════════════════════════════════════════════════
        # ZUSAMMENFASSUNG (NEU v2.2)
        # ══════════════════════════════════════════════════════════════
        
        print(f"\n\n{'='*70}")
        print("BATCH-ZUSAMMENFASSUNG")
        print(f"{'='*70}")
        
        print(f"\n[STATISTIK]")
        print(f"  Gesamt:       {total_runs} Runs")
        print(f"  Erfolgreich:  {len(successful_runs)} ({100*len(successful_runs)/max(1,total_runs):.1f}%)")
        print(f"  Fehlgeschlagen: {len(failed_runs)}")
        print(f"  Uebersprungen:  {len(skipped_runs)}")
        
        # Details zu erfolgreichen Runs
        if successful_runs:
            print(f"\n[ERFOLGREICH]")
            for result in successful_runs:
                print(f"  [OK] {result.vp_code} | {result.video_fps} | {result.calib_mode}")
                print(f"       Dauer: {result.duration_seconds/60:.1f} min | Schritte: {len(result.completed_steps)}")
        
        # Details zu fehlgeschlagenen Runs
        if failed_runs:
            print(f"\n[FEHLGESCHLAGEN]")
            for result in failed_runs:
                print(f"  [X] {result.vp_code} | {result.video_fps} | {result.calib_mode}")
                print(f"      Fehler in: {result.failed_step}")
                print(f"      Message: {result.error_message[:100] if result.error_message else 'Unbekannt'}...")
        
        # Details zu uebersprungenen Runs
        if skipped_runs:
            print(f"\n[UEBERSPRUNGEN]")
            for skip in skipped_runs:
                print(f"  [SKIP] {skip['vp_code']} | {skip['video_fps']} | {skip['calib_mode']}")
                print(f"         Grund: {skip['reason'][:80] if skip['reason'] else 'Unbekannt'}...")
        
        print(f"\n{'='*70}")
        
        # Warte auf Bestaetigung
        input("\nDruecke ENTER um fortzufahren...")
    
    def _execute_analyses_v2(self, vp_codes: List[str], video_options: List[str],
                            calib_modes: List[str], create_calibration: bool,
                            analysis_mode: int, methods: str,
                            cache_decisions_all: Dict[str, Dict[str, 'CacheDecisions']]):
        """
        Fuehrt Analysen mit vordefinierten Cache-Entscheidungen aus (v2.2).
        """
        global MASTER_SYNC_LOG_PATH 
       
        print_header("AUSFUEHRUNG")
        
        # Ergebnis-Tracking
        all_results = []
        successful_runs = []
        failed_runs = []
        skipped_runs = []
        
        total_runs = len(vp_codes) * len(video_options) * len(calib_modes)
        current_run = 0
        
        for vp_code in vp_codes:
            # Lade VP-Daten
            try:
                vp_data = self.vp_manager.load_vp_data(vp_code)
            except (ValueError, FileNotFoundError) as e:
                print(f"\n[SKIP] VP {vp_code}: {e}")
                skipped_runs.append({
                    'vp_code': vp_code,
                    'reason': str(e),
                    'video_fps': 'alle',
                    'calib_mode': 'alle'
                })
                current_run += len(video_options) * len(calib_modes)
                continue
            
            if vp_data is None:
                print(f"\n[SKIP] VP {vp_code}: Daten nicht gefunden")
                skipped_runs.append({
                    'vp_code': vp_code,
                    'reason': 'VP-Daten nicht gefunden',
                    'video_fps': 'alle',
                    'calib_mode': 'alle'
                })
                current_run += len(video_options) * len(calib_modes)
                continue
            
            # ══════════════════════════════════════════════════════════════
            # NEU v2.3: Setze Environment-Variablen für Master-JSON Support
            # ══════════════════════════════════════════════════════════════
            
            # VP-Code für Korrekturwert-Lookup in debug_0
            os.environ['PIPELINE_VP_CODE'] = vp_code
            
            # Master-JSON Pfad für Fallback wenn VP-spezifische
            if MASTER_SYNC_LOG_PATH.exists():
                os.environ['PIPELINE_MASTER_SYNC_LOG_PATH'] = str(MASTER_SYNC_LOG_PATH)
                print(f"    [INFO] Master-JSON verfügbar: {MASTER_SYNC_LOG_PATH.name}")
            else:
                # Entferne Variable falls sie von vorherigem Run existiert
                os.environ.pop('PIPELINE_MASTER_SYNC_LOG_PATH', None)
            
            pipeline = AnalysisPipeline(vp_data, self.scripts_folder)
            
            for video_fps in video_options:
                for calib_mode in calib_modes:
                    current_run += 1
                    config_key = f"{video_fps}_{calib_mode}"
                    
                    print(f"\n{'='*70}")
                    print(f"RUN {current_run}/{total_runs}: {vp_code} | {config_key}")
                    print(f"{'='*70}")
                    
                    # Hole Cache-Entscheidungen fuer diese Config
                    vp_cache = cache_decisions_all.get(vp_code, {})
                    cache_decisions = vp_cache.get(config_key, None)
                    
                    # Erstelle Config
                    config = RunConfiguration(
                        vp_code=vp_code,
                        video_fps=video_fps,
                        calib_mode=calib_mode,
                        create_calibration=create_calibration,
                        analysis_mode=analysis_mode,
                        methods=methods,
                        cache_decisions=cache_decisions
                    )
                    
                    # Zeige Cache-Status
                    if cache_decisions:
                        print(f"[CACHE] {cache_decisions.n_from_cache} Steps aus Cache, {cache_decisions.n_fresh} neu")
                    
                    # Fuehre Pipeline aus
                    try:
                        result = pipeline.run_full_analysis(config)
                        all_results.append(result)
                        
                        if result.success:
                            successful_runs.append(result)
                        elif result.skipped:
                            skipped_runs.append({
                                'vp_code': vp_code,
                                'reason': result.skip_reason or result.error_message,
                                'video_fps': video_fps,
                                'calib_mode': calib_mode
                            })
                        else:
                            failed_runs.append(result)
                    
                    except Exception as e:
                        print(f"\n[!] Unerwarteter Fehler bei VP {vp_code}: {e}")
                        print(f"    >> Fahre mit naechster Konfiguration fort...\n")
                        
                        skipped_runs.append({
                            'vp_code': vp_code,
                            'reason': f'Unerwarteter Fehler: {str(e)}',
                            'video_fps': video_fps,
                            'calib_mode': calib_mode
                        })
        
        # ══════════════════════════════════════════════════════════════
        # ZUSAMMENFASSUNG
        # ══════════════════════════════════════════════════════════════
        
        print(f"\n\n{'='*70}")
        print("BATCH-ZUSAMMENFASSUNG")
        print(f"{'='*70}")
        
        print(f"\n[STATISTIK]")
        print(f"  Gesamt:         {total_runs} Runs")
        print(f"  Erfolgreich:    {len(successful_runs)} ({100*len(successful_runs)/max(1,total_runs):.1f}%)")
        print(f"  Fehlgeschlagen: {len(failed_runs)}")
        print(f"  Uebersprungen:  {len(skipped_runs)}")
        
        if successful_runs:
            print(f"\n[ERFOLGREICH]")
            for result in successful_runs:
                print(f"  [OK] {result.vp_code} | {result.video_fps} | {result.calib_mode}")
                print(f"       Dauer: {result.duration_seconds/60:.1f} min | Steps: {len(result.completed_steps)}")
        
        if failed_runs:
            print(f"\n[FEHLGESCHLAGEN]")
            for result in failed_runs:
                print(f"  [X] {result.vp_code} | {result.video_fps} | {result.calib_mode}")
                print(f"      Fehler in: {result.failed_step}")
                error_msg = result.error_message[:80] if result.error_message else 'Unbekannt'
                print(f"      Message: {error_msg}...")
        
        if skipped_runs:
            print(f"\n[UEBERSPRUNGEN]")
            for skip in skipped_runs:
                print(f"  [SKIP] {skip['vp_code']} | {skip['video_fps']} | {skip['calib_mode']}")
                reason = skip['reason'][:60] if skip['reason'] else 'Unbekannt'
                print(f"         Grund: {reason}...")
        
        print(f"\n{'='*70}")
        
        input("\nDruecke ENTER um fortzufahren...")

    def _print_analysis_summary(self, results: List[RunResult], duration_total: float):
        """Gibt Zusammenfassung der Analysen aus"""
        
        print_header("ANALYSEN ABGESCHLOSSEN")
        
        n_success = sum(1 for r in results if r.success)
        n_failed = len(results) - n_success
        
        print(f" Ergebnisse:")
        print(f"   Erfolgreich: {n_success}/{len(results)}")
        print(f"   Fehlgeschlagen: {n_failed}/{len(results)}")
        print(f"  ⏱️ Gesamtdauer: {duration_total/60:.1f} Minuten")
        
        if n_success > 0:
            print(f"\n Output-Ordner:")
            for result in results:
                if result.success:
                    print(f"  • {result.vp_code} ({result.video_fps}): {result.run_folder}")
        
        if n_failed > 0:
            print(f"\n Fehlgeschlagene Runs:")
            for result in results:
                if not result.success:
                    print(f"  • {result.vp_code} ({result.video_fps}): {result.failed_step}")
    
    # ═════════════════════════════════════════════════════════════════
    # VERGLEICHSMODUS
    # ═════════════════════════════════════════════════════════════════
    
    def run_comparison_mode(self):
        """
        Statistikmodus - Ersetzt den alten Vergleichsmodus (v2.3)
        
        Ruft das separate statistical_cli Modul auf.
        """
        
        print_header("STATISTIKMODUS v1.0")
        
        try:
            from statistic.statistical_cli import run_statistical_cli
            run_statistical_cli(self.base_folder, self.scripts_folder)
        
        except ImportError as e:
            print(f"[!] Statistikmodul nicht verfügbar!")
            print(f"    Fehler: {e}")
            print(f"\n    Bitte stelle sicher, dass der Ordner 'statistics/' existiert")
            print(f"    mit folgenden Dateien:")
            print(f"      - __init__.py")
            print(f"      - data_availability_scanner.py")
            print(f"      - statistical_data_prep.py")
            print(f"      - statistical_pipeline.py")
            print(f"      - statistical_cli.py")
            
            input("\nDrücke ENTER um fortzufahren...")
        
        except Exception as e:
            print(f"[!] Fehler im Statistikmodus: {e}")
            import traceback
            traceback.print_exc()
            
            input("\nDrücke ENTER um fortzufahren...")
    
# =================================================================================
# MAIN
# =================================================================================

def main():
    """Entry Point"""
    
    try:
        cli = MasterCLI(BASE_FOLDER, SCRIPTS_FOLDER)
        cli.run()
    
    except KeyboardInterrupt:
        print(f"\n\n👋 Programm beendet durch User")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n Kritischer Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
