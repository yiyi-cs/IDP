"""Known-onset tests and real production phase detector regression checks."""
import ast
import __future__
import json
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import unittest
from unittest.mock import Mock

import numpy as np
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

from shared.audio_sync import refine_880_centers, summarize_anchor_offsets


def production_symbols():
    # Isolate actual production functions from this legacy module's optional
    # librosa/CV imports and environment setup, not a reimplementation.
    path = Path(__file__).resolve().parents[1] / 'debug/debug_0_phase_detection.py'
    tree = ast.parse(path.read_text())
    names = {'FrequencyDetector', 'SyncResult', 'CompatMultiAnchorResult',
             'calculate_sync_offset', 'get_corrected_eyelink_time_for_peak',
             'get_eyelink_time_corrected', 'JSONFormat', 'PhaseReconstructor'}
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in names]
    namespace = dict(globals())
    # Config constants are the same ones imported by the production module.
    import config
    namespace.update({k: getattr(config, k) for k in dir(config) if k.isupper()})
    namespace.update(FREQ_CALIBRATION=1760, FREQ_TRIAL_START=440,
                     FREQ_TRIAL_END=880, PEAK_OFFSET_MS=100,
                     TONE_DURATION_MS=200, DEFAULT_LATENCY_MS=500.)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec', flags=__future__.annotations.compiler_flag), namespace)
    return namespace


class AudioSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.symbols = production_symbols()

    def test_880_center_is_stable_across_carrier_phase_noise_and_amplitude(self):
        sr = 16000
        t = np.arange(3 * sr) / sr
        for phase in np.linspace(0, 2 * np.pi, 9):
            for amplitude in (.3, 1.):
                rng = np.random.default_rng(71)
                a = rng.normal(0, .002, len(t))
                mask = (t >= 1.) & (t < 1.2)
                a[mask] += amplitude * np.sin(2*np.pi*880*t[mask] + phase)
                # Either old envelope maximum (near the start or end).
                for candidate in (1.01, 1.19):
                    centers = refine_880_centers(a, sr, [candidate])
                    self.assertEqual(len(centers), 1)
                    self.assertLess(abs(centers[0] - 1.1), .012)

    def test_silence_missing_matches_and_clipped_marker_are_not_detections(self):
        detector = self.symbols['FrequencyDetector']()
        self.assertEqual(detector.detect_frequency_peaks(
            np.zeros(32000), 16000, 880, expected_times=[1.]), [])
        self.assertEqual(detector._match_with_expected(np.array([2.]), [8.], 1.), [])
        self.assertEqual(refine_880_centers(np.zeros(32000), 16000, [1.]), [])
        t = np.arange(32000) / 16000
        clipped = np.sin(2*np.pi*880*t) * (t < .12)
        self.assertEqual(refine_880_centers(clipped, 16000, [.01]), [])

    def test_search_window_at_recording_start_uses_actual_local_time(self):
        cls = self.symbols['PhaseReconstructor']
        obj = cls.__new__(cls)
        obj.frequency_detector = Mock()
        obj.frequency_detector.detect_frequency_peaks.return_value = [.3]
        a = np.zeros(32000)
        for name in ('_search_440hz_around_time', '_search_880hz_around_time'):
            result = getattr(obj, name)(a, 16000, .3, window_s=5.)
            self.assertAlmostEqual(result, .3)
            kwargs = obj.frequency_detector.detect_frequency_peaks.call_args.kwargs
            self.assertEqual(kwargs['expected_times'], [.3])

    def test_old_sync_cache_and_changed_phases_are_rejected(self):
        from utils.analysis_pipeline import CrossRunCacheScanner
        from shared.audio_sync import SYNC_VERSION
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp)/'source', Path(tmp)/'target'
            source.mkdir()
            target.mkdir()
            old = {'sync_info': {'offset_ms': 1000.}, 'phases': {}}
            (source/'phases_detected.json').write_text(json.dumps(old))
            scanner = CrossRunCacheScanner(Path(tmp))
            self.assertFalse(scanner.copy_step_outputs(source, target, 'debug_0'))
            self.assertFalse((target/'phases_detected.json').exists())
            old['sync_info']['quality'] = {'version': SYNC_VERSION}
            (source/'phases_detected.json').write_text(json.dumps(old))
            self.assertTrue(scanner.copy_step_outputs(source, target, 'debug_0'))
            self.assertTrue(scanner._sync_cache_compatible(source, target, 'debug_1_ptgaze'))
            old['sync_info']['offset_ms'] = 1050.
            (target/'phases_detected.json').write_text(json.dumps(old))
            self.assertFalse(scanner._sync_cache_compatible(source, target, 'debug_1_ptgaze'))

    def test_family_disagreement_is_not_averaged_or_hidden(self):
        q = summarize_anchor_offsets([1000.,1002.,905.,907.], [440,440,880,880])
        self.assertEqual(q['selected_trial_offset_ms'], 1001.)
        self.assertEqual(q['frequency_families']['880']['median_offset_ms'], 906.)
        self.assertEqual(q['trial_frequency_difference_ms'], -95.)
        self.assertFalse(q['absolute_sync_verified'])
        self.assertFalse(q['drift_fitted'])

    def test_compatible_families_and_missing_trial_audio(self):
        q = summarize_anchor_offsets([1000.,1002.,995.,997.], [440,440,880,880])
        self.assertEqual(q['selected_trial_frequencies'], [440,880])
        self.assertEqual(q['selected_trial_offset_ms'], 998.5)
        q = summarize_anchor_offsets([500.], [1760])
        self.assertIsNone(q['selected_trial_offset_ms'])
        self.assertEqual(q['frequency_families']['440']['count'], 0)

    def test_production_offset_and_compat_export_preserve_observed_families(self):
        n = self.symbols
        anchors = []
        event = dict(audio_play_start_eyelink_ms=10., audio_play_actual_delay_ms=500.,
                     audio_duration_ms=2000)
        for freq, time in [(440,9.6),(880,11.495)]:
            anchors.append(dict(event=event, frequency=freq, video_time_s=time))
        result = n['calculate_sync_offset'](anchors, n['JSONFormat'].V2_EXTENDED, 500.)
        compat = n['CompatMultiAnchorResult'](result)
        self.assertAlmostEqual(compat.offset_ms, 1000.)
        self.assertAlmostEqual(compat.offset_880hz_ms, 905.)
        self.assertTrue(np.isnan(compat.r_squared))
        self.assertTrue(np.isnan(compat.drift_ms))
        self.assertEqual(compat.quality['selected_trial_frequencies'], [440])


if __name__ == '__main__':
    unittest.main()
