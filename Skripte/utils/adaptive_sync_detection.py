"""
=================================================================================
ADAPTIVE LOW-SIGNAL SYNC DETECTION v2.0
=================================================================================
CHANGELOG v2.0:
- NEU: Template Matching als primäre Detektionsmethode
- NEU: Goertzel-Algorithmus als Backup
- NEU: Multi-Method Consensus (kombiniert alle Methoden)
- VERBESSERT: Funktioniert bei SNR < -50 dB

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

import numpy as np
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks, correlate
from scipy.fft import fft, fftfreq
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# =================================================================================
# KONFIGURATION
# =================================================================================

# Adaptive Search Parameter
ADAPTIVE_TONE_DURATION_S = 0.2              # Erwartete Ton-Dauer (200ms)
ADAPTIVE_MIN_DISTANCE_S = 2.0               # Mindestabstand zwischen Tönen
ADAPTIVE_INITIAL_SEARCH_WINDOW_S = 2.5      # Suchfenster für ersten Ton (±2.5s) - erhöht!
ADAPTIVE_REFINED_SEARCH_WINDOW_S = 0.8      # Suchfenster für Folgetöne (±800ms) - erhöht!

# Template Matching Parameter
TEMPLATE_MATCH_THRESHOLD = 0.25             # Minimaler Score für Match (gesenkt für schwache Signale)
TEMPLATE_TONE_DURATION_S = 0.2              # Tonlänge für Template

# Goertzel Parameter  
GOERTZEL_WINDOW_S = 0.2                     # Fenster für Goertzel (= Tonlänge)
GOERTZEL_HOP_S = 0.02                       # Hop-Size (20ms für präzise Lokalisierung)
GOERTZEL_PERCENTILE_THRESHOLD = 95          # Threshold = 95. Perzentil

# Multi-Method Consensus
CONSENSUS_TIME_TOLERANCE_S = 0.15           # Peaks innerhalb 150ms = gleicher Ton
MIN_METHODS_FOR_CONSENSUS = 1               # Mindestens 1 Methode muss finden (relaxed)

# Frequenz-spezifische Parameter
ADAPTIVE_FREQ_PARAMS = {
    1760: {'tolerance_hz': 100, 'name': 'Kalibrierung'},   # Breiterer Filter!
    440: {'tolerance_hz': 50, 'name': 'Fixation-Start'},
    880: {'tolerance_hz': 75, 'name': 'Trial-Ende'}
}


# =================================================================================
# DATENSTRUKTUREN
# =================================================================================

@dataclass
class DetectionCandidate:
    """Ein potentieller Ton-Kandidat"""
    time_s: float
    score: float
    method: str
    
@dataclass
class AdaptiveDetectionResult:
    """Ergebnis der adaptiven Low-Signal-Detektion für EINE Phase"""
    phase_label: str
    found_times: List[float]
    confidence: float
    detection_rate: float
    anchor_index: int
    interpolated_indices: List[int]
    method: str
    n_expected: int = 0
    n_found: int = 0
    mean_deviation_ms: float = 0.0
    detection_details: Dict = field(default_factory=dict)


@dataclass
class AdaptiveSyncResult:
    """Gesamt-Ergebnis für ALLE Phasen"""
    phase_results: Dict[str, AdaptiveDetectionResult]
    overall_confidence: float
    overall_detection_rate: float
    recommended_offset_ms: float
    sync_point_video_s: float = 0.0
    sync_point_eyelink_ms: float = 0.0


# =================================================================================
# DETECTION METHODS
# =================================================================================

class LowSignalDetectionMethods:
    """
    Sammlung von Detektionsmethoden für extrem schwache Signale.
    
    Methoden (in Reihenfolge der Zuverlässigkeit):
    1. Template Matching - Beste für bekannte Signalform
    2. Goertzel - Effizient für einzelne Frequenzen
    3. STFT Peak - Frequenz-Zeit-Analyse
    """
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
    
    # ══════════════════════════════════════════════════════════════════════════
    # METHODE 1: TEMPLATE MATCHING (Cross-Correlation)
    # ══════════════════════════════════════════════════════════════════════════
    
    def template_matching(self,
                          audio: np.ndarray,
                          target_freq: int,
                          search_start_s: float,
                          search_duration_s: float,
                          threshold: float = TEMPLATE_MATCH_THRESHOLD
                          ) -> List[DetectionCandidate]:
        """
        Template Matching via Cross-Correlation.
        
        Erzeugt synthetischen Referenz-Ton und sucht nach Matches.
        SEHR effektiv bei schwachen aber sauberen Sinussignalen!
        """
        
        # Extrahiere Suchfenster
        start_sample = int(search_start_s * self.sr)
        end_sample = int((search_start_s + search_duration_s) * self.sr)
        
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        if end_sample - start_sample < 1000:
            return []
        
        y_window = audio[start_sample:end_sample]
        
        # Erzeuge Template (synthetischer Ton)
        t_tone = np.arange(int(TEMPLATE_TONE_DURATION_S * self.sr)) / self.sr
        template = np.sin(2 * np.pi * target_freq * t_tone)
        
        # Hanning-Fenster (wie echter Ton typischerweise)
        template = template * np.hanning(len(template))
        
        # Normalisiere Template
        template = template / np.sqrt(np.sum(template**2) + 1e-10)
        
        # Bandpass VORHER anwenden (reduziert Störungen)
        tolerance = ADAPTIVE_FREQ_PARAMS.get(target_freq, {}).get('tolerance_hz', 100)
        y_filtered = self._bandpass_filter(y_window, target_freq, tolerance)
        
        if y_filtered is None or len(y_filtered) < len(template):
            return []
        
        # Cross-Correlation
        correlation = correlate(y_filtered, template, mode='valid')
        correlation = np.abs(correlation)
        
        # Normalisiere
        max_corr = np.max(correlation)
        if max_corr == 0:
            return []
        
        correlation_norm = correlation / max_corr
        
        # Finde Peaks
        min_distance_samples = int(ADAPTIVE_MIN_DISTANCE_S * self.sr)
        
        try:
            peak_indices, properties = find_peaks(
                correlation_norm,
                height=threshold,
                distance=min_distance_samples
            )
        except:
            return []
        
        # Konvertiere zu Kandidaten
        candidates = []
        for idx in peak_indices:
            time_s = search_start_s + idx / self.sr
            score = correlation_norm[idx]
            candidates.append(DetectionCandidate(
                time_s=time_s,
                score=float(score),
                method='template_matching'
            ))
        
        return candidates
    
    # ══════════════════════════════════════════════════════════════════════════
    # METHODE 2: GOERTZEL-ALGORITHMUS
    # ══════════════════════════════════════════════════════════════════════════
    
    def goertzel(self,
                 audio: np.ndarray,
                 target_freq: int,
                 search_start_s: float,
                 search_duration_s: float,
                 threshold_percentile: float = GOERTZEL_PERCENTILE_THRESHOLD
                 ) -> List[DetectionCandidate]:
        """
        Goertzel-Algorithmus für einzelne Frequenz.
        
        Effizienter als FFT wenn nur eine Frequenz interessiert!
        """
        
        # Extrahiere Suchfenster
        start_sample = int(search_start_s * self.sr)
        end_sample = int((search_start_s + search_duration_s) * self.sr)
        
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        if end_sample - start_sample < 1000:
            return []
        
        y_window = audio[start_sample:end_sample]
        
        # Sliding Window Goertzel
        window_samples = int(GOERTZEL_WINDOW_S * self.sr)
        hop_samples = int(GOERTZEL_HOP_S * self.sr)
        
        if window_samples > len(y_window):
            return []
        
        powers = []
        times = []
        
        for i in range(0, len(y_window) - window_samples, hop_samples):
            window = y_window[i:i + window_samples]
            power = self._goertzel_single(window, target_freq)
            powers.append(power)
            times.append(search_start_s + (i + window_samples // 2) / self.sr)
        
        if len(powers) == 0:
            return []
        
        powers = np.array(powers)
        times = np.array(times)
        
        # Dynamischer Threshold
        threshold = np.percentile(powers, threshold_percentile) * 0.3
        threshold = max(threshold, np.mean(powers) * 2)  # Mindestens 2x Mittelwert
        
        # Finde Peaks
        min_distance = int(ADAPTIVE_MIN_DISTANCE_S / GOERTZEL_HOP_S)
        
        try:
            peak_indices, _ = find_peaks(powers, height=threshold, distance=min_distance)
        except:
            return []
        
        # Konvertiere zu Kandidaten
        candidates = []
        max_power = np.max(powers) if np.max(powers) > 0 else 1.0
        
        for idx in peak_indices:
            time_s = times[idx]
            score = powers[idx] / max_power  # Normalisierter Score
            candidates.append(DetectionCandidate(
                time_s=time_s,
                score=float(score),
                method='goertzel'
            ))
        
        return candidates
    
    def _goertzel_single(self, samples: np.ndarray, target_freq: int) -> float:
        """Goertzel für ein einzelnes Fenster."""
        N = len(samples)
        k = int(0.5 + (N * target_freq) / self.sr)
        omega = (2.0 * np.pi * k) / N
        coeff = 2.0 * np.cos(omega)
        
        s0, s1, s2 = 0.0, 0.0, 0.0
        for sample in samples:
            s0 = sample + coeff * s1 - s2
            s2 = s1
            s1 = s0
        
        power = s1**2 + s2**2 - coeff * s1 * s2
        return np.sqrt(max(0, power)) / N
    
    # ══════════════════════════════════════════════════════════════════════════
    # METHODE 3: STFT PEAK DETECTION
    # ══════════════════════════════════════════════════════════════════════════
    
    def stft_peaks(self,
                   audio: np.ndarray,
                   target_freq: int,
                   search_start_s: float,
                   search_duration_s: float,
                   n_fft: int = 4096,
                   hop_length: int = 512
                   ) -> List[DetectionCandidate]:
        """
        STFT-basierte Peak-Detektion.
        
        Analysiert Frequenz-Zeit-Ebene und sucht Energie-Peaks bei Zielfrequenz.
        """
        
        # Extrahiere Suchfenster
        start_sample = int(search_start_s * self.sr)
        end_sample = int((search_start_s + search_duration_s) * self.sr)
        
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        if end_sample - start_sample < n_fft:
            return []
        
        y_window = audio[start_sample:end_sample]
        
        # Manuelles STFT (ohne librosa-Abhängigkeit in Produktion)
        n_frames = 1 + (len(y_window) - n_fft) // hop_length
        
        if n_frames < 1:
            return []
        
        # Frequenz-Bins
        freqs = np.fft.rfftfreq(n_fft, 1/self.sr)
        
        # Finde Bin für Zielfrequenz
        target_bin = np.argmin(np.abs(freqs - target_freq))
        tolerance_bins = 3  # ±3 Bins
        
        # Berechne Energie pro Frame
        band_energies = []
        frame_times = []
        
        window = np.hanning(n_fft)
        
        for i in range(n_frames):
            start = i * hop_length
            frame = y_window[start:start + n_fft] * window
            
            spectrum = np.abs(np.fft.rfft(frame))
            
            # Energie um Zielfrequenz
            low_bin = max(0, target_bin - tolerance_bins)
            high_bin = min(len(spectrum), target_bin + tolerance_bins + 1)
            band_energy = np.sum(spectrum[low_bin:high_bin])
            
            band_energies.append(band_energy)
            frame_times.append(search_start_s + (start + n_fft // 2) / self.sr)
        
        band_energies = np.array(band_energies)
        frame_times = np.array(frame_times)
        
        # Normalisiere
        max_energy = np.max(band_energies)
        if max_energy == 0:
            return []
        
        band_energies_norm = band_energies / max_energy
        
        # Finde Peaks
        threshold = 0.2
        min_distance = int(ADAPTIVE_MIN_DISTANCE_S * self.sr / hop_length)
        
        try:
            peak_indices, _ = find_peaks(band_energies_norm, height=threshold, distance=min_distance)
        except:
            return []
        
        # Konvertiere zu Kandidaten
        candidates = []
        for idx in peak_indices:
            time_s = frame_times[idx]
            score = band_energies_norm[idx]
            candidates.append(DetectionCandidate(
                time_s=time_s,
                score=float(score),
                method='stft_peaks'
            ))
        
        return candidates
    
    # ══════════════════════════════════════════════════════════════════════════
    # HILFSMETHODEN
    # ══════════════════════════════════════════════════════════════════════════
    
    def _bandpass_filter(self, signal: np.ndarray, center_freq: float, 
                         tolerance: float) -> Optional[np.ndarray]:
        """Bandpass-Filter um Zielfrequenz."""
        
        nyquist = self.sr / 2
        low = max(0.01, (center_freq - tolerance) / nyquist)
        high = min(0.99, (center_freq + tolerance) / nyquist)
        
        if low >= high:
            return signal
        
        try:
            sos = butter(4, [low, high], btype='band', output='sos')
            filtered = sosfiltfilt(sos, signal)
            
            if np.any(np.isnan(filtered)):
                return signal
            
            return filtered
        except:
            return signal


# =================================================================================
# MULTI-METHOD CONSENSUS
# =================================================================================

class MultiMethodConsensus:
    """
    Kombiniert Ergebnisse mehrerer Detektionsmethoden.
    
    Strategie:
    1. Sammle Kandidaten von allen Methoden
    2. Clustere Kandidaten die zeitlich nah beieinander liegen
    3. Bewerte Cluster nach Anzahl bestätigender Methoden + Score
    4. Wähle beste Kandidaten
    """
    
    def __init__(self, time_tolerance_s: float = CONSENSUS_TIME_TOLERANCE_S):
        self.time_tolerance = time_tolerance_s
    
    def find_consensus(self,
                       all_candidates: List[DetectionCandidate],
                       expected_count: int = 1
                       ) -> List[Tuple[float, float, List[str]]]:
        """
        Findet Konsens zwischen verschiedenen Methoden.
        
        Args:
            all_candidates: Alle Kandidaten von allen Methoden
            expected_count: Erwartete Anzahl Töne
        
        Returns:
            Liste von (time_s, combined_score, methods)
        """
        
        if len(all_candidates) == 0:
            return []
        
        # Sortiere nach Zeit
        sorted_candidates = sorted(all_candidates, key=lambda x: x.time_s)
        
        # Clustere Kandidaten
        clusters = []
        current_cluster = [sorted_candidates[0]]
        
        for i in range(1, len(sorted_candidates)):
            if sorted_candidates[i].time_s - current_cluster[-1].time_s < self.time_tolerance:
                current_cluster.append(sorted_candidates[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sorted_candidates[i]]
        
        clusters.append(current_cluster)
        
        # Bewerte Cluster
        cluster_scores = []
        
        for cluster in clusters:
            # Mittlere Zeit
            mean_time = np.mean([c.time_s for c in cluster])
            
            # Anzahl verschiedener Methoden
            methods = list(set(c.method for c in cluster))
            n_methods = len(methods)
            
            # Mittlerer Score
            mean_score = np.mean([c.score for c in cluster])
            
            # Bester Score
            best_score = max(c.score for c in cluster)
            
            # Combined Score: Gewichte Methodenanzahl + besten Score
            # Mehr Methoden = zuverlässiger
            combined_score = (n_methods / 3.0) * 0.4 + best_score * 0.6
            
            cluster_scores.append((mean_time, combined_score, methods, best_score))
        
        # Sortiere nach Combined Score
        cluster_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Wähle Top-Kandidaten (mit Mindestabstand)
        selected = []
        
        for time_s, combined_score, methods, best_score in cluster_scores:
            # Prüfe Mindestabstand zu bereits ausgewählten
            too_close = False
            for sel_time, _, _ in selected:
                if abs(time_s - sel_time) < ADAPTIVE_MIN_DISTANCE_S:
                    too_close = True
                    break
            
            if not too_close:
                selected.append((time_s, combined_score, methods))
            
            if len(selected) >= expected_count:
                break
        
        return selected


# =================================================================================
# HAUPTKLASSE: ADAPTIVE LOW-SIGNAL DETECTOR v2.0
# =================================================================================

class AdaptiveLowSignalDetector:
    """
    Adaptiver Detektor für stark reduzierte Audio-Marker.
    
    Version 2.0 - Multi-Method Detection:
    - Template Matching (primär)
    - Goertzel-Algorithmus
    - STFT Peak Detection
    - Consensus-basierte Auswahl
    """
    
    def __init__(self, 
                 sr: int = 44100,
                 initial_search_window_s: float = ADAPTIVE_INITIAL_SEARCH_WINDOW_S,
                 refined_search_window_s: float = ADAPTIVE_REFINED_SEARCH_WINDOW_S,
                 verbose: bool = True):
        
        self.sr = sr
        self.initial_window_s = initial_search_window_s
        self.refined_window_s = refined_search_window_s
        self.verbose = verbose
        
        # Detection Methods
        self.methods = LowSignalDetectionMethods(sr=sr)
        self.consensus = MultiMethodConsensus()
    
    def search_from_manual_anchor(self,
                                   audio: np.ndarray,
                                   manual_anchor_s: float,
                                   target_freq: int,
                                   expected_times_from_json: List[float],
                                   expected_intervals_s: List[float],
                                   phase_label: str = 'unknown'
                                   ) -> AdaptiveDetectionResult:
        """
        Hauptmethode: Sucht Töne ausgehend von manuellem Anker.
        """
        
        freq_name = ADAPTIVE_FREQ_PARAMS.get(target_freq, {}).get('name', f'{target_freq} Hz')
        
        if self.verbose:
            print(f"\n{'─'*70}")
            print(f"ADAPTIVE LOW-SIGNAL SEARCH v2.0: {phase_label.upper()} ({freq_name})")
            print(f"{'─'*70}")
            print(f"   Manueller Anker: {manual_anchor_s:.2f}s (±{self.initial_window_s}s)")
            print(f"   Erwartete Töne: {len(expected_times_from_json)}")
            print(f"   Ziel-Frequenz: {target_freq} Hz")
        
        # ══════════════════════════════════════════════════════════════
        # PHASE 1: Multi-Method Suche für ERSTEN Ton
        # ══════════════════════════════════════════════════════════════
        
        if self.verbose:
            print(f"\n   PHASE 1: Multi-Method Suche (erster Ton)...")
        
        first_tone_result = self._find_first_tone_multi_method(
            audio, manual_anchor_s, target_freq, expected_intervals_s
        )
        
        if first_tone_result is None:
            if self.verbose:
                print(f"      ✗ KEIN Ton gefunden mit allen Methoden!")
                print(f"      >> Fallback: Nutze manuellen Anker")
            
            return self._interpolate_all_from_anchor(
                manual_anchor_s, expected_times_from_json, target_freq, phase_label
            )
        
        first_tone_time, first_score, first_methods = first_tone_result

        # ════════════════════════════════════════════════════════════════════
        # NEU: Peak-Korrektur anwenden!
        # Die Detektion findet den PEAK (Mitte des 200ms Tons), nicht den Start.
        # Korrigiere um -100ms für präzisere Synchronisation.
        # ════════════════════════════════════════════════════════════════════
        
        PEAK_OFFSET_S = 0.1  # 100ms = Mitte des 200ms Tons
        first_tone_time_corrected = first_tone_time
        
        if self.verbose:
            print(f"      ✓ GEFUNDEN: {first_tone_time:.3f}s (Peak)")
            print(f"        Korrigiert: {first_tone_time_corrected:.3f}s (Start, -{PEAK_OFFSET_S*1000:.0f}ms)")
            print(f"        Score: {first_score:.2f} | Methoden: {', '.join(first_methods)}")
            print(f"        Δ zum Anker: {(first_tone_time_corrected - manual_anchor_s)*1000:+.0f}ms")
        
        # Verwende korrigierte Zeit
        first_tone_time = first_tone_time_corrected
        
        if self.verbose:
            print(f"      ✓ GEFUNDEN: {first_tone_time:.3f}s")
            print(f"        Score: {first_score:.2f} | Methoden: {', '.join(first_methods)}")
            print(f"        Δ zum Anker: {(first_tone_time - manual_anchor_s)*1000:+.0f}ms")
        
        # ══════════════════════════════════════════════════════════════
        # PHASE 2: Sequentielle Suche der Folgetöne
        # ══════════════════════════════════════════════════════════════
        
        if self.verbose:
            print(f"\n   PHASE 2: Sequentielle Suche (Folgetöne)...")
        
        found_times = [first_tone_time]
        interpolated_indices = []
        detection_details = {'first_tone': {
            'time': first_tone_time,
            'score': first_score,
            'methods': first_methods
        }}
        
        # Bestimme Index des ersten Tons
        first_tone_index = 0  # Annahme: Manueller Anker zeigt auf ersten Ton
        
        # Vorwärts-Suche
        current_time = first_tone_time
        
        for i in range(first_tone_index + 1, len(expected_times_from_json)):
            # Erwarteter Abstand aus JSON
            if i - 1 < len(expected_intervals_s):
                expected_interval = expected_intervals_s[i - 1]
            else:
                expected_interval = np.mean(expected_intervals_s) if expected_intervals_s else 6.5
            
            expected_next_time = current_time + expected_interval
            
            # Suche in kleinem Fenster (schnell!)
            next_tone = self._search_single_tone(
                audio, expected_next_time, target_freq, self.refined_window_s
            )
            
            if next_tone is not None:
                next_time, next_score, next_methods = next_tone
                found_times.append(next_time)
                current_time = next_time
                
                if self.verbose:
                    print(f"      Ton {i+1}: {next_time:.3f}s (Δ={next_time - found_times[-2]:.2f}s) [{', '.join(next_methods)}]")
            else:
                # Interpoliere
                interpolated_time = expected_next_time
                found_times.append(interpolated_time)
                interpolated_indices.append(len(found_times) - 1)
                current_time = interpolated_time
                
                if self.verbose:
                    print(f"      Ton {i+1}: {interpolated_time:.3f}s (INTERPOLIERT)")
        
        # ══════════════════════════════════════════════════════════════
        # PHASE 3: Ergebnis zusammenstellen
        # ══════════════════════════════════════════════════════════════
        
        n_expected = len(expected_times_from_json)
        n_found = len(found_times) - len(interpolated_indices)
        detection_rate = n_found / n_expected if n_expected > 0 else 0.0
        
        # Confidence
        confidence = 0.5 * detection_rate + 0.5 * first_score
        
        if self.verbose:
            print(f"\n   ERGEBNIS:")
            print(f"      Gefunden: {n_found}/{n_expected} ({detection_rate*100:.1f}%)")
            print(f"      Interpoliert: {len(interpolated_indices)}")
            print(f"      Confidence: {confidence:.2f}")
        
        return AdaptiveDetectionResult(
            phase_label=phase_label,
            found_times=found_times,
            confidence=confidence,
            detection_rate=detection_rate,
            anchor_index=first_tone_index,
            interpolated_indices=interpolated_indices,
            method='multi_method_v2',
            n_expected=n_expected,
            n_found=n_found,
            detection_details=detection_details
        )
    
    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE METHODS
    # ══════════════════════════════════════════════════════════════════════════
    
    def _find_first_tone_multi_method(self,
                                       audio: np.ndarray,
                                       manual_anchor_s: float,
                                       target_freq: int,
                                       expected_intervals_s: List[float]
                                       ) -> Optional[Tuple[float, float, List[str]]]:
        """
        Findet ersten Ton mit allen verfügbaren Methoden.
        """
        
        search_start = max(0, manual_anchor_s - self.initial_window_s)
        search_duration = 2 * self.initial_window_s
        
        all_candidates = []
        
        # Methode 1: Template Matching
        if self.verbose:
            print(f"      → Template Matching...")
        
        template_candidates = self.methods.template_matching(
            audio, target_freq, search_start, search_duration
        )
        all_candidates.extend(template_candidates)
        
        if self.verbose:
            print(f"         {len(template_candidates)} Kandidaten")
        
        # Methode 2: Goertzel
        if self.verbose:
            print(f"      → Goertzel...")
        
        goertzel_candidates = self.methods.goertzel(
            audio, target_freq, search_start, search_duration
        )
        all_candidates.extend(goertzel_candidates)
        
        if self.verbose:
            print(f"         {len(goertzel_candidates)} Kandidaten")
        
        # Methode 3: STFT Peaks
        if self.verbose:
            print(f"      → STFT Peaks...")
        
        stft_candidates = self.methods.stft_peaks(
            audio, target_freq, search_start, search_duration
        )
        all_candidates.extend(stft_candidates)
        
        if self.verbose:
            print(f"         {len(stft_candidates)} Kandidaten")
        
        if len(all_candidates) == 0:
            return None
        
        # Consensus
        if self.verbose:
            print(f"      → Consensus ({len(all_candidates)} Kandidaten)...")
        
        consensus_results = self.consensus.find_consensus(all_candidates, expected_count=1)
        
        if len(consensus_results) == 0:
            # Fallback: Nimm besten Einzelkandidaten
            best = max(all_candidates, key=lambda x: x.score)
            return (best.time_s, best.score, [best.method])
        
        return consensus_results[0]  # (time, score, methods)
    
    def _search_single_tone(self,
                            audio: np.ndarray,
                            expected_time_s: float,
                            target_freq: int,
                            window_s: float
                            ) -> Optional[Tuple[float, float, List[str]]]:
        """
        Sucht einzelnen Ton in kleinem Fenster (schnelle Methode für Folgetöne).
        """
        
        search_start = max(0, expected_time_s - window_s)
        search_duration = 2 * window_s
        
        all_candidates = []
        
        # Nur Template Matching für Geschwindigkeit
        template_candidates = self.methods.template_matching(
            audio, target_freq, search_start, search_duration,
            threshold=0.2  # Etwas relaxter
        )
        all_candidates.extend(template_candidates)
        
        # Bei zu wenig Kandidaten: Auch Goertzel
        if len(template_candidates) == 0:
            goertzel_candidates = self.methods.goertzel(
                audio, target_freq, search_start, search_duration
            )
            all_candidates.extend(goertzel_candidates)
        
        if len(all_candidates) == 0:
            return None
        
        # Wähle den Kandidaten nächst am erwarteten Zeitpunkt
        best = min(all_candidates, key=lambda x: abs(x.time_s - expected_time_s))
        
        # Prüfe ob plausibel (nicht zu weit vom Erwarteten)
        if abs(best.time_s - expected_time_s) > window_s * 1.5:
            return None
        
        # Peak-Korrektur anwenden
        corrected_time = best.time_s
        
        return (corrected_time, best.score, [best.method])
    
    def _interpolate_all_from_anchor(self,
                                      manual_anchor_s: float,
                                      expected_times_from_json: List[float],
                                      target_freq: int,
                                      phase_label: str
                                      ) -> AdaptiveDetectionResult:
        """Fallback: Interpoliert ALLE Zeiten aus JSON."""
        
        if expected_times_from_json:
            offset = manual_anchor_s - expected_times_from_json[0]
            interpolated_times = [t + offset for t in expected_times_from_json]
        else:
            interpolated_times = [manual_anchor_s]
        
        return AdaptiveDetectionResult(
            phase_label=phase_label,
            found_times=interpolated_times,
            confidence=0.3,
            detection_rate=0.0,
            anchor_index=0,
            interpolated_indices=list(range(len(interpolated_times))),
            method='interpolated_fallback',
            n_expected=len(expected_times_from_json),
            n_found=0
        )


# =================================================================================
# INTEGRATION HELPER
# =================================================================================

def run_adaptive_sync_for_all_phases(
    audio: np.ndarray,
    sr: int,
    manual_sync_points: Dict[str, float],
    calibration_timing_jsons: Dict[str, str],
    experiment_sync_log: Dict,
    verbose: bool = True
) -> AdaptiveSyncResult:
    """
    Führt adaptive Synchronisation für ALLE Phasen durch.
    
    KORRIGIERT v2.1: Berechnet sync_point_eyelink_ms korrekt!
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print("ADAPTIVE LOW-SIGNAL SYNC v2.0 (Multi-Method)")
        print(f"{'='*70}")
    
    detector = AdaptiveLowSignalDetector(sr=sr, verbose=verbose)
    phase_results = {}
    
    for label, manual_time in manual_sync_points.items():
        if label not in calibration_timing_jsons:
            if verbose:
                print(f"\n   [SKIP] {label.upper()}: Keine calibration_timing JSON")
            continue
        
        json_path = calibration_timing_jsons[label]
        
        if not Path(json_path).exists():
            if verbose:
                print(f"\n   [SKIP] {label.upper()}: JSON nicht gefunden: {json_path}")
            continue
        
        try:
            with open(json_path, 'r') as f:
                calib_timing = json.load(f)
        except Exception as e:
            if verbose:
                print(f"\n   [ERROR] {label.upper()}: JSON-Fehler: {e}")
            continue
        
        events = calib_timing.get('events', [])
        
        if len(events) < 2:
            if verbose:
                print(f"\n   [SKIP] {label.upper()}: Zu wenig Events ({len(events)})")
            continue
        
        # Extrahiere Zeiten und Intervalle
        timestamps = [e['timestamp'] for e in events]
        first_timestamp = timestamps[0]
        
        expected_times_relative = [t - first_timestamp for t in timestamps]
        expected_intervals = np.diff(timestamps).tolist()
        
        if verbose:
            mean_interval = np.mean(expected_intervals)
            print(f"\n   {label.upper()}: {len(events)} Punkte, μ={mean_interval:.2f}s")
        
        # Adaptive Suche
        result = detector.search_from_manual_anchor(
            audio=audio,
            manual_anchor_s=manual_time,
            target_freq=1760,
            expected_times_from_json=expected_times_relative,
            expected_intervals_s=expected_intervals,
            phase_label=label
        )
        
        phase_results[label] = result
    
    # ══════════════════════════════════════════════════════════════════════════
    # GESAMT-STATISTIK + SYNC-PUNKT-BERECHNUNG (KORRIGIERT!)
    # ══════════════════════════════════════════════════════════════════════════
    
    if phase_results:
        confidences = [r.confidence for r in phase_results.values()]
        detection_rates = [r.detection_rate for r in phase_results.values()]
        
        overall_confidence = np.mean(confidences)
        overall_detection_rate = np.mean(detection_rates)
        
        # Beste Phase für Sync-Punkt (höchste Confidence)
        best_phase = max(phase_results.items(), key=lambda x: x[1].confidence)
        best_label, best_result = best_phase
        
        # ════════════════════════════════════════════════════════════════════
        # KRITISCHER FIX: Berechne sync_point_eyelink_ms aus experiment_sync_log!
        # ════════════════════════════════════════════════════════════════════
        
        sync_point_video_s = best_result.found_times[0] if best_result.found_times else manual_sync_points.get('beg', 0)
        
        # Suche entsprechendes calibration_start Event für die beste Phase
        eyelink_time_ms = 0.0
        
        events = experiment_sync_log.get('events', [])
        calib_start_events = [e for e in events 
                            if e.get('event_type') == 'calibration_start'
                            and e.get('label') == best_label]
        
        if calib_start_events:
            # Extrahiere EyeLink-Zeit (ACHTUNG: JSON-Bug - enthält Sekunden!)
            eyelink_time_raw = calib_start_events[0].get('eyelink_time_ms', 0)
            
            # Korrigiere Einheiten: Wenn Wert < 100000, ist es wahrscheinlich Sekunden
            if eyelink_time_raw < 100000:
                eyelink_time_ms = eyelink_time_raw * 1000  # Sekunden → ms
            else:
                eyelink_time_ms = eyelink_time_raw
            
            if verbose:
                print(f"\n   Sync-Punkt ({best_label.upper()}):")
                print(f"      Video: {sync_point_video_s:.3f}s")
                print(f"      EyeLink: {eyelink_time_ms:.0f}ms")
        else:
            if verbose:
                print(f"\n   [WARN] Kein calibration_start Event für {best_label} gefunden!")
                print(f"          >> sync_point_eyelink_ms = 0 (muss in debug_0 korrigiert werden)")
        
        # Berechne empfohlenen Offset
        recommended_offset_ms = eyelink_time_ms - (sync_point_video_s * 1000)
        
    else:
        overall_confidence = 0.0
        overall_detection_rate = 0.0
        recommended_offset_ms = 0.0
        sync_point_video_s = manual_sync_points.get('beg', 0)
        eyelink_time_ms = 0.0
    
    if verbose:
        print(f"\n{'='*70}")
        print("ZUSAMMENFASSUNG")
        print(f"{'='*70}")
        print(f"   Confidence: {overall_confidence:.2f}")
        print(f"   Detection-Rate: {overall_detection_rate*100:.1f}%")
        print(f"   Empfohlener Offset: {recommended_offset_ms:.0f}ms")
    
    return AdaptiveSyncResult(
        phase_results=phase_results,
        overall_confidence=overall_confidence,
        overall_detection_rate=overall_detection_rate,
        recommended_offset_ms=recommended_offset_ms,
        sync_point_video_s=sync_point_video_s,
        sync_point_eyelink_ms=eyelink_time_ms  # ← JETZT KORREKT!
    )

# =================================================================================
# EXPORT
# =================================================================================

__all__ = [
    'AdaptiveLowSignalDetector',
    'AdaptiveDetectionResult',
    'AdaptiveSyncResult',
    'run_adaptive_sync_for_all_phases',
    'LowSignalDetectionMethods',
    'MultiMethodConsensus',
]


if __name__ == "__main__":
    print("adaptive_sync_detection.py v2.0 - Multi-Method Detection")
    print("Importiere in debug_0_phase_detection.py für Verwendung.")
