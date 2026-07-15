"""
DEBUG 3: Eyetracker-Daten laden v2.3
====================================
NEU in v2.3:
------------
 Unterstützt end_fixation Messages (FVE_Empra_v51.m)
 Unterstützt video_calibration_* Messages
 Dokumentiert Timeline (440Hz >> end_fixation >> start_trial)
 Erweiterte Metadaten für Synchronisation
 Rückwärtskompatibel (alte EDFs ohne neue Messages)

Timeline-Struktur (NEU):
------------------------
Block 1 (Fixation):
  START >> start_fixation_1 (440Hz) >> ... >> end_fixation_1 (880Hz) >> END
  
Block 2 (Trial):
  START >> start_trial_1 >> ... (7s Bild) >> end_trial_1 >> END
  
Timing:
  end_fixation = Audio-Ende (880Hz, ~1.3s nach Start)
  start_trial = ~100ms nach end_fixation
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

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import os
import subprocess
import json

from config import *

# Environment-Variable-Overrides (für master_cli.py)
import os

if 'PIPELINE_OUTPUT_BASE_DIR' in os.environ:
    OUTPUT_BASE_DIR = Path(os.environ['PIPELINE_OUTPUT_BASE_DIR'])

if 'PIPELINE_MAIN_VIDEO_PATH' in os.environ:
    VIDEO_PATH = Path(os.environ['PIPELINE_MAIN_VIDEO_PATH'])

if 'PIPELINE_EYETRACKER_FILE_PATH' in os.environ:
    EYETRACKER_FILE_PATH = Path(os.environ['PIPELINE_EYETRACKER_FILE_PATH'])

if 'PIPELINE_CALIBRATION_PKL_PATH' in os.environ:
    CALIBRATION_PKL_PATH = Path(os.environ['PIPELINE_CALIBRATION_PKL_PATH'])

# ==================== DATENSTRUKTUREN ====================

@dataclass
class BlockData:
    block_number: int
    trial_number: int        
    is_practice: bool        
    block_type: str
    start_time_ms: int
    end_time_ms: int
    duration_ms: int
    samples: pd.DataFrame
    fixations: pd.DataFrame
    saccades: pd.DataFrame
    blinks: pd.DataFrame
    messages: pd.DataFrame
    
    def to_dict(self):
        """Konvertiert zu Dict (für CSV/JSON-Export) v2.5"""
        
        #  Extrahiere relevante Messages
        msg_list = []
        if not self.messages.empty:
            for _, msg in self.messages.iterrows():
                msg_text = msg['text']
                if any(keyword in msg_text for keyword in ['start_fixation', 'end_fixation', 'start_trial', 'end_trial']):
                    msg_list.append({
                        'timestamp_ms': int(msg['timestamp']),
                        'text': msg_text
                    })
        
        return {
            'block_number': self.block_number,
            'trial_number': self.trial_number,  #  Nutze Property
            'is_practice': self.is_practice,    #  Nutze Property
            'block_type': self.block_type,      #  Nutze Property ('fixation' oder 'trial')
            'start_time_ms': self.start_time_ms,
            'end_time_ms': self.end_time_ms,
            'duration_ms': self.duration_ms,
            'n_samples': len(self.samples),
            'n_fixations': len(self.fixations),
            'n_saccades': len(self.saccades),
            'n_blinks': len(self.blinks),
            'n_messages': len(msg_list),
            'key_messages': json.dumps(msg_list) if msg_list else None
        }

# ==================== EYETRACKER-LOADER ====================

class EyetrackerLoader:
    def __init__(self, mode: int = ANALYSIS_MODE):
        self.mode = mode
        self.blocks = []
        self.video_calibrations = []  #  NEU: Tracke Video-Kalibrierungen
        
        if mode == 1:
            print(" WARNUNG: Modus 1 (Standalone) benötigt keinen Eyetracker!")
            print("   debug_3 sollte nur für Modus 2 und 3 verwendet werden.")
    
    def load_eyetracker_data(self, file_path: str) -> List[BlockData]:
        """Lädt Eyetracker-Daten (ASC oder EDF) v2.3"""
        
        print(f"\n{'='*70}")
        print("DEBUG 3: EYETRACKER-DATEN LADEN v2.3")
        print(f"{'='*70}\n")
        
        file_path = Path(file_path)
        
        # Prüfe Dateityp
        if file_path.suffix.lower() == '.edf':
            print(f"EDF-Datei erkannt: {file_path.name}")
            asc_path = self._convert_edf_to_asc(file_path)
            if asc_path is None:
                return []
        elif file_path.suffix.lower() == '.asc':
            print(f"ASC-Datei erkannt: {file_path.name}")
            asc_path = file_path
        else:
            print(f" FEHLER: Unbekanntes Dateiformat '{file_path.suffix}'")
            return []
        
        # Parse ASC
        self.blocks = self._parse_asc_message_based(asc_path)
        
        # Validierung
        self._validate_blocks()
        
        # Export
        self._export_blocks()
        
        return self.blocks
    
    def _convert_edf_to_asc(self, edf_path: Path) -> Optional[Path]:
        """Konvertiert EDF zu ASC via edf2asc Tool"""
        # [GLEICH WIE VORHER]
        print("\n🔄 Konvertiere EDF >> ASC...")
        
        asc_path = edf_path.with_suffix('.asc')
        
        try:
            result = subprocess.run(['edf2asc', '-h'], capture_output=True, text=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(" FEHLER: edf2asc nicht gefunden!")
            return None
        
        try:
            cmd = ['edf2asc', str(edf_path), str(asc_path), '-s', '-e', '-nst', '-miss', '9999', '-t']
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if asc_path.exists():
                print(f" Konvertierung erfolgreich: {asc_path.name}")
                return asc_path
            else:
                print(" ASC-Datei wurde nicht erstellt!")
                return None
        except subprocess.CalledProcessError as e:
            print(f" Konvertierung fehlgeschlagen: {e.stderr}")
            return None
    
    def _parse_asc_message_based(self, asc_path: Path) -> List[BlockData]:
        """Parst ASC mit MESSAGE-BASIERTER Block-Erkennung v2.5"""
        
        print(f"\n📖 Parse ASC (Message-basiert): {asc_path.name}...")
        
        # ═══════════════════════════════════════════════════════════════
        # SCHRITT 1: Parse ALLE Daten (Samples, Fixationen, Sakkaden, Messages)
        # ═══════════════════════════════════════════════════════════════
        
        all_samples = []
        all_fixations = []
        all_saccades = []
        all_blinks = []
        all_messages = []
        
        total_lines = 0
        with open(asc_path, 'r', encoding='latin-1') as f:
            for line_num, line in enumerate(f, 1):
                total_lines = line_num
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                # SAMPLES
                if parts[0].replace('.', '').replace('-', '').isdigit():
                    try:
                        all_samples.append({
                            'timestamp': int(float(parts[0])),
                            'x_pos': float(parts[1]) if parts[1] not in ['.', '9999'] else np.nan,
                            'y_pos': float(parts[2]) if parts[2] not in ['.', '9999'] else np.nan,
                            'pupil_size': float(parts[3]) if len(parts) > 3 and parts[3] not in ['.', '9999'] else np.nan
                        })
                    except:
                        pass
                
                # FIXATIONEN (EFIX)
                elif line.startswith('EFIX'):
                    try:
                        all_fixations.append({
                            'eye': parts[1],
                            'start_time': int(float(parts[2])),
                            'end_time': int(float(parts[3])),
                            'duration': int(float(parts[4])),
                            'x_pos': float(parts[5]) if parts[5] not in ['.', '9999'] else np.nan,
                            'y_pos': float(parts[6]) if parts[6] not in ['.', '9999'] else np.nan,
                            'pupil_size': float(parts[7]) if len(parts) > 7 and parts[7] not in ['.', '9999'] else np.nan
                        })
                    except:
                        pass
                
                # SAKKADEN (ESACC)
                elif line.startswith('ESACC'):
                    try:
                        all_saccades.append({
                            'eye': parts[1],
                            'start_time': int(float(parts[2])),
                            'end_time': int(float(parts[3])),
                            'duration': int(float(parts[4])),
                            'start_x': float(parts[5]) if parts[5] not in ['.', '9999'] else np.nan,
                            'start_y': float(parts[6]) if parts[6] not in ['.', '9999'] else np.nan,
                            'end_x': float(parts[7]) if parts[7] not in ['.', '9999'] else np.nan,
                            'end_y': float(parts[8]) if parts[8] not in ['.', '9999'] else np.nan,
                            'amplitude': float(parts[9]) if len(parts) > 9 and parts[9] not in ['.', '9999'] else np.nan,
                            'peak_velocity': float(parts[10]) if len(parts) > 10 and parts[10] not in ['.', '9999'] else np.nan
                        })
                    except:
                        pass
                
                # BLINKS
                elif line.startswith('EBLINK'):
                    try:
                        all_blinks.append({
                            'eye': parts[1],
                            'start_time': int(float(parts[2])),
                            'end_time': int(float(parts[3])),
                            'duration': int(float(parts[4]))
                        })
                    except:
                        pass
                
                # MESSAGES
                elif line.startswith('MSG'):
                    try:
                        timestamp = int(float(parts[1]))
                        text = ' '.join(parts[2:])
                        all_messages.append({'timestamp': timestamp, 'text': text})
                    except:
                        pass
                
                if line_num % 100000 == 0:
                    print(f"  {line_num:,} Zeilen...")
        
        print(f"  {total_lines:,} Zeilen gesamt")
        print(f"  {len(all_samples):,} Samples")
        print(f"  {len(all_fixations):,} Fixationen")
        print(f"  {len(all_saccades):,} Sakkaden")
        print(f"  {len(all_messages):,} Messages")
        
        # ═══════════════════════════════════════════════════════════════
        # SCHRITT 2: Extrahiere Trial-Grenzen aus Messages
        # ═══════════════════════════════════════════════════════════════
        
        import re
        
        print(f"\n📋 Extrahiere Trial-Grenzen aus Messages...")
        
        trials = {}
        
        for msg in all_messages:
            text = msg['text']
            ts = msg['timestamp']
            
            # ═══════════════════════════════════════════════════════════
            #  FIX: PRACTICE-TRIAL ZUERST (vor normalem Regex!)
            # ═══════════════════════════════════════════════════════════
            
            # start_fixation_practice
            if 'start_fixation_practice' in text:
                trial_num = 0  # Practice = Trial 0
                if trial_num not in trials:
                    trials[trial_num] = {'is_practice': True}
                trials[trial_num]['fixation_start'] = ts
                continue  # ← Überspringe normale Regex!
            
            # end_fixation_practice
            if 'end_fixation_practice' in text:
                trial_num = 0
                if trial_num not in trials:
                    trials[trial_num] = {'is_practice': True}
                trials[trial_num]['fixation_end'] = ts
                continue
            
            # start_trial_practice
            if 'start_trial_practice' in text:
                trial_num = 0
                if trial_num not in trials:
                    trials[trial_num] = {'is_practice': True}
                trials[trial_num]['trial_start'] = ts
                continue
            
            # end_trial_practice
            if 'end_trial_practice' in text:
                trial_num = 0
                if trial_num not in trials:
                    trials[trial_num] = {'is_practice': True}
                trials[trial_num]['trial_end'] = ts
                continue
            
            # ═══════════════════════════════════════════════════════════
            # Normale Trials (1-24)
            # ═══════════════════════════════════════════════════════════
            
            # start_fixation_X
            match = re.search(r'start_fixation_(\d+)', text)
            if match:
                trial_num = int(match.group(1))
                if trial_num not in trials:
                    trials[trial_num] = {'is_practice': False}
                trials[trial_num]['fixation_start'] = ts
            
            # end_fixation_X
            match = re.search(r'end_fixation_(\d+)', text)
            if match:
                trial_num = int(match.group(1))
                if trial_num not in trials:
                    trials[trial_num] = {'is_practice': False}
                trials[trial_num]['fixation_end'] = ts
            
            # start_trial_X
            match = re.search(r'start_trial_(\d+)', text)
            if match:
                trial_num = int(match.group(1))
                if trial_num not in trials:
                    trials[trial_num] = {'is_practice': False}
                trials[trial_num]['trial_start'] = ts
            
            # end_trial_X
            match = re.search(r'end_trial_(\d+)', text)
            if match:
                trial_num = int(match.group(1))
                if trial_num not in trials:
                    trials[trial_num] = {'is_practice': False}
                trials[trial_num]['trial_end'] = ts
        
        print(f"  {len(trials)} Trials gefunden: {sorted(trials.keys())}")
        
        # ═══════════════════════════════════════════════════════════════
        # SCHRITT 3: Erstelle Blöcke MIT Fixationen/Sakkaden-Zuordnung
        # ═══════════════════════════════════════════════════════════════
        
        print(f"\n🔨 Erstelle Blöcke (mit Fixationen/Sakkaden)...")
        
        blocks = []
        block_number = 1
        
        for trial_num in sorted(trials.keys()):
            t = trials[trial_num]
            
            #  KEINE PRACTICE-LOGIK! (Practice ist nicht im EDF)
            is_practice = t.get('is_practice', False)
            
            # ───────────────────────────────────────────────────────────
            # FIXATION-BLOCK
            # ───────────────────────────────────────────────────────────
            
            if 'fixation_start' in t and 'fixation_end' in t:
                start_ms = t['fixation_start']
                end_ms = t['fixation_end']
                
                # Filtere Daten für diesen Zeitbereich (zeitbasiert!)
                samples = [s for s in all_samples if start_ms <= s['timestamp'] <= end_ms]
                fixations = [f for f in all_fixations if start_ms <= f['start_time'] <= end_ms]
                saccades = [s for s in all_saccades if start_ms <= s['start_time'] <= end_ms]
                blinks = [b for b in all_blinks if start_ms <= b['start_time'] <= end_ms]
                messages = [m for m in all_messages if start_ms <= m['timestamp'] <= end_ms]
                
                blocks.append(BlockData(
                    block_number=block_number,
                    trial_number=trial_num,  #  trial_num = 1-24 (nicht 0!)
                    is_practice=is_practice,
                    block_type='fixation',
                    start_time_ms=start_ms,
                    end_time_ms=end_ms,
                    duration_ms=end_ms - start_ms,
                    samples=pd.DataFrame(samples),
                    fixations=pd.DataFrame(fixations),
                    saccades=pd.DataFrame(saccades),
                    blinks=pd.DataFrame(blinks),
                    messages=pd.DataFrame(messages)
                ))
                block_number += 1
            
            # ───────────────────────────────────────────────────────────
            # TRIAL-BLOCK
            # ───────────────────────────────────────────────────────────
            
            if 'trial_start' in t and 'trial_end' in t:
                start_ms = t['trial_start']
                end_ms = t['trial_end']
                
                samples = [s for s in all_samples if start_ms <= s['timestamp'] <= end_ms]
                fixations = [f for f in all_fixations if start_ms <= f['start_time'] <= end_ms]
                saccades = [s for s in all_saccades if start_ms <= s['start_time'] <= end_ms]
                blinks = [b for b in all_blinks if start_ms <= b['start_time'] <= end_ms]
                messages = [m for m in all_messages if start_ms <= m['timestamp'] <= end_ms]
                
                blocks.append(BlockData(
                    block_number=block_number,
                    trial_number=trial_num,
                    is_practice=is_practice,
                    block_type='trial',
                    start_time_ms=start_ms,
                    end_time_ms=end_ms,
                    duration_ms=end_ms - start_ms,
                    samples=pd.DataFrame(samples),
                    fixations=pd.DataFrame(fixations),
                    saccades=pd.DataFrame(saccades),
                    blinks=pd.DataFrame(blinks),
                    messages=pd.DataFrame(messages)
                ))
                block_number += 1
        
        print(f"  {len(blocks)} Blöcke erstellt")
        print(f"  {sum(1 for b in blocks if b.block_type == 'fixation')} Fixations-Blöcke")
        print(f"  {sum(1 for b in blocks if b.block_type == 'trial')} Trial-Blöcke")
        print(f"   Fixationen/Sakkaden zeitbasiert zugeordnet")
        
        return blocks

    def _validate_blocks(self):
        """Validiert die geladenen Blöcke v2.3"""
        
        print(f"\n{'='*70}")
        print("VALIDIERUNG & TIMELINE-ANALYSE")
        print(f"{'='*70}\n")
        
        if not self.blocks:
            print(" Keine Blöcke gefunden!")
            return
        
        fixation_blocks = [b for b in self.blocks if b.block_type == 'fixation']
        trial_blocks = [b for b in self.blocks if b.block_type == 'trial']
                
        print(f"Blöcke gesamt: {len(self.blocks)}")
        print(f"  Fixationskreuz: {len(fixation_blocks)}")
        print(f"  Trials: {len(trial_blocks)}")
        
        #  NEU: Prüfe Messages
        blocks_with_messages = [b for b in self.blocks if not b.messages.empty]
        print(f"  Mit Messages: {len(blocks_with_messages)}/{len(self.blocks)}")
        
        # Zeitbereich
        start_time = min(b.start_time_ms for b in self.blocks)
        end_time = max(b.end_time_ms for b in self.blocks)
        duration_s = (end_time - start_time) / 1000
        
        print(f"\nZeitbereich:")
        print(f"  {start_time:,} - {end_time:,} ms ({duration_s:.1f}s)")
        
        #  NEU: Timeline-Struktur analysieren (erste 3 Trials)
        print(f"\n{'='*70}")
        print("TIMELINE-STRUKTUR (Erste 3 Trials)")
        print(f"{'='*70}\n")
        
        for trial_num in range(1, min(4, len(trial_blocks)+1)):
            fix_blocks = [b for b in self.blocks if b.trial_number == trial_num and b.block_type == 'fixation']
            tr_blocks = [b for b in self.blocks if b.trial_number == trial_num and b.block_type == 'trial']

            if not fix_blocks or not tr_blocks:
                continue

            fixation_block = fix_blocks[0]
            trial_block = tr_blocks[0]
            
            print(f"Trial {trial_num}:")
            
            # Fixation-Phase
            if not fixation_block.messages.empty:
                fix_messages = fixation_block.messages[
                    fixation_block.messages['text'].str.contains('fixation|trial', case=False, na=False)
                ]
                
                for _, msg in fix_messages.iterrows():
                    msg_time_rel = msg['timestamp'] - fixation_block.start_time_ms
                    print(f"  [{msg_time_rel:4.0f}ms] {msg['text']}")
            
            print(f"  Fixation Block Ende: {fixation_block.end_time_ms} ms "
                  f"(Dauer: {fixation_block.duration_ms}ms)")
            
            # Trial-Phase
            if not trial_block.messages.empty:
                trial_messages = trial_block.messages[
                    trial_block.messages['text'].str.contains('trial', case=False, na=False)
                ]
                
                for _, msg in trial_messages.iterrows():
                    msg_time_rel = msg['timestamp'] - trial_block.start_time_ms
                    print(f"  [{msg_time_rel:4.0f}ms] {msg['text']}")
            
            print(f"  Trial Block Ende: {trial_block.end_time_ms} ms "
                  f"(Dauer: {trial_block.duration_ms}ms)")
            
            # Timing-Check
            gap_ms = trial_block.start_time_ms - fixation_block.end_time_ms
            print(f"  ⏱️ Gap: {gap_ms}ms (erwartet: ~0-10ms)\n")
    
    def _export_blocks(self):
        """Exportiert Blöcke als CSV v2.3"""
        
        print(f"\n{'='*70}")
        print("EXPORT")
        print(f"{'='*70}\n")
        
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        
        # CSV 1: Samples (OPTIONAL)
        if SAVE_EYETRACKER_SAMPLES:
            print(" Exportiere EyeLink-Samples...")
            
            all_samples = []
            for block in self.blocks:
                if not block.samples.empty:
                    samples = block.samples.copy()
                    samples['block_number'] = block.block_number
                    samples['block_type'] = block.block_type
                    samples.rename(columns={'timestamp': 'timestamp_ms'}, inplace=True)
                    all_samples.append(samples)
            
            if all_samples:
                combined_df = pd.concat(all_samples, ignore_index=True)
                output_path = os.path.join(OUTPUT_BASE_DIR, OUTPUT_FILES['debug_3_eyetracker'])
                combined_df.to_csv(output_path, index=False)
                
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f" Samples: {output_path} ({file_size_mb:.1f} MB)")
        
        # CSV 2: Block-Übersicht (PRIMÄR)
        block_info = [b.to_dict() for b in self.blocks]
        blocks_df = pd.DataFrame(block_info)
        
        output_path = os.path.join(OUTPUT_BASE_DIR, OUTPUT_FILES['debug_3_blocks'])
        blocks_df.to_csv(output_path, index=False)
        print(f"\n Block-Übersicht: {output_path}")
        
        #  NEU: CSV 3: Video-Kalibrierungen (Optional)
        if self.video_calibrations:
            vcal_df = pd.DataFrame(self.video_calibrations)
            vcal_path = os.path.join(OUTPUT_BASE_DIR, "debug_3_video_calibrations.csv")
            vcal_df.to_csv(vcal_path, index=False)
            print(f" Video-Kalibrierungen: {vcal_path}")
        
        # CSV 3: Fixationen (für Fixationszeit-Analyse in debug_6)
        if SAVE_EYELINK_FIXATIONS:
            print(" Exportiere EyeLink-Fixationen...")
            
            all_fixations = []
            for block in self.blocks:
                if not block.fixations.empty:
                    fixations = block.fixations.copy()
                    fixations['block_number'] = block.block_number
                    fixations['block_type'] = block.block_type
                    fixations['trial_number'] = block.trial_number
                    fixations['is_practice'] = block.is_practice
                    all_fixations.append(fixations)
            
            if all_fixations:
                fixations_df = pd.concat(all_fixations, ignore_index=True)
                output_path = os.path.join(OUTPUT_BASE_DIR, "debug_3_fixations.csv")
                fixations_df.to_csv(output_path, index=False)
                
                total_fixation_time_s = fixations_df['duration'].sum() / 1000
                print(f" Fixationen: {output_path}")
                print(f"   • {len(fixations_df)} Fixationen")
                print(f"   • Gesamtdauer: {total_fixation_time_s:.1f}s")

        # JSON: Metadaten
        start_time_ms = min(b.start_time_ms for b in self.blocks) if self.blocks else 0
        end_time_ms = max(b.end_time_ms for b in self.blocks) if self.blocks else 0
        recording_duration_ms = end_time_ms - start_time_ms
        total_samples = sum(len(b.samples) for b in self.blocks)
        sample_rate_hz = (total_samples / (recording_duration_ms / 1000)) if recording_duration_ms > 0 else 0
        
        tracked_eye = TRACKED_EYE
        if tracked_eye is None:
            for block in self.blocks:
                if not block.fixations.empty:
                    tracked_eye = block.fixations.iloc[0]['eye']
                    break
        
        metadata = {
            'n_blocks': len(self.blocks),
            'n_fixation_blocks': sum(1 for b in self.blocks if b.block_type == 'fixation'),
            'n_trial_blocks': sum(1 for b in self.blocks if b.block_type == 'trial'),
            'start_time_ms': start_time_ms,
            'end_time_ms': end_time_ms,
            'recording_duration_ms': recording_duration_ms,
            'total_samples': total_samples,
            'sample_rate_hz': round(sample_rate_hz, 1),
            'tracked_eye': tracked_eye if tracked_eye else 'unknown',
            'mode': self.mode,
            'first_trial_is_practice': FIRST_TRIAL_IS_PRACTICE,
            'n_practice_trials': 1 if FIRST_TRIAL_IS_PRACTICE else 0,
            'n_experimental_trials': N_TRIALS,
            'total_fixations': sum(len(b.fixations) for b in self.blocks),
            'total_saccades': sum(len(b.saccades) for b in self.blocks),
            'total_blinks': sum(len(b.blinks) for b in self.blocks),
            'n_video_calibrations': len(self.video_calibrations),  #  NEU
            'video_calibrations': self.video_calibrations if self.video_calibrations else [],  #  NEU
            'csv_version': '2.3',  #  NEU
            'supports_end_fixation_messages': True  #  NEU
        }
        
        output_path = os.path.join(OUTPUT_BASE_DIR, OUTPUT_FILES['debug_3_metadata'])
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f" Metadaten: {output_path}")

# ==================== MAIN ====================

if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("DEBUG 3: EYETRACKER-DATEN LADEN v2.3")
    print(f"{'='*70}")
    
    if ANALYSIS_MODE == 1:
        print("\n Modus 1 (Standalone) benötigt keinen Eyetracker!")
        print("   debug_3 sollte nur für Modus 2 und 3 verwendet werden.")
        print("\n💡 Pipeline für Modus 1:")
        print("   debug_1 >> debug_2 >> debug_5 (Kalibrierung) >> debug_6 (Standalone-Analyse)")
        print("\nNächster Schritt: debug_5_partial_calibration.py")
        exit(0)
    
    print(f"\nKonfiguration:")
    print(f"  Modus: {ANALYSIS_MODE}")
    
    # Prüfe ob Datei existiert
    if not os.path.exists(EYETRACKER_FILE_PATH):
        print(f"\n FEHLER: Eyetracker-Datei nicht gefunden!")
        print(f"   Pfad: {EYETRACKER_FILE_PATH}")
        print(f"\n💡 Bitte in config.py anpassen:")
        print(f"   EYETRACKER_FILE_PATH = 'C:/Pfad/zu/datei.edf'")
        exit(1)
    
    print(f"  Datei: {Path(EYETRACKER_FILE_PATH).name}")
    
    # Lade Daten
    loader = EyetrackerLoader(mode=ANALYSIS_MODE)
    blocks = loader.load_eyetracker_data(EYETRACKER_FILE_PATH)
    
    if blocks:
        print(f"\n{'='*70}")
        print("ERFOLGREICH!")
        print(f"{'='*70}")
        print(f"\nOutput:")
        print(f"  • debug_3_blocks_overview.csv (PRIMÄR)")
        print(f"  • debug_3_metadata.json")
        
        if SAVE_EYETRACKER_SAMPLES:
            print(f"  • debug_3_eyetracker_data.csv")
        
        if SAVE_EYELINK_FIXATIONS:
            print(f"  • debug_3_fixations.csv")
        
        print(f"\n Wichtige Metadaten:")
        with open(os.path.join(OUTPUT_BASE_DIR, OUTPUT_FILES['debug_3_metadata']), 'r') as f:
            metadata = json.load(f)
        
        print(f"  Recording: {metadata['recording_duration_ms']/1000:.1f} s ({metadata['recording_duration_ms']/60000:.1f} min)")
        print(f"  Sample-Rate: {metadata['sample_rate_hz']} Hz")
        print(f"  Tracked Eye: {metadata['tracked_eye']}")
        print(f"  Blöcke: {metadata['n_blocks']} ({metadata['n_fixation_blocks']} Fixation + {metadata['n_trial_blocks']} Trial)")
        print(f"  Trials: {metadata['n_experimental_trials']} (+ {metadata['n_practice_trials']} Übung)")
        
        #  NEU: Video-Kalibrierungen (falls vorhanden)
        if loader.video_calibrations:
            print(f"\n  Video-Kalibrierungen erkannt: {len(loader.video_calibrations)}")
            for vcal in loader.video_calibrations:
                print(f"    • {vcal['label']}: {vcal['start_time_ms']:,} ms (Dauer: {vcal['duration_ms']/1000:.1f}s)")
        
        print(f"\n💡 Spalten in blocks_overview.csv:")
        blocks_df = pd.read_csv(os.path.join(OUTPUT_BASE_DIR, OUTPUT_FILES['debug_3_blocks']))
        for col in blocks_df.columns:
            print(f"  - {col}")
        
        print(f"\nNächster Schritt: debug_4_synchronization.py")
        
    else:
        print(f"\n{'='*70}")
        print("FEHLER: Keine Blöcke geladen!")
        print(f"{'='*70}")
        print(f"\n💡 Mögliche Ursachen:")
        print(f"  • EDF/ASC Datei ist leer oder korrupt")
        print(f"  • Keine START/END Blöcke gefunden")
        print(f"  • Falsches Dateiformat")
        print(f"\n Prüfe die ASC-Datei manuell:")
        print(f"   $ head -100 {EYETRACKER_FILE_PATH}")
        exit(1)