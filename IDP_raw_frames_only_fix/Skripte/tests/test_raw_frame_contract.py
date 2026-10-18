"""Ensure abandoned image resizing cannot enter production or its caches."""
import json
import tempfile
import unittest
from pathlib import Path

from config import PREPROCESS_MODE
from shared.shared_image_preprocessing import preprocess_frame
from utils.analysis_pipeline import CrossRunCacheScanner, RunConfiguration
from utils.vp_data_manager import VPDataManager
from blink.run_pipeline_extras import run_stage


class RawFrameTests(unittest.TestCase):
    def test_default_and_pixels_are_unchanged(self):
        self.assertEqual(PREPROCESS_MODE, 'none')
        import numpy as np
        frame = np.arange(36).reshape(3, 4, 3)
        self.assertIs(preprocess_frame(frame), frame)
        self.assertEqual(frame.shape, (3, 4, 3))

    def test_obsolete_modes_fail_before_processing(self):
        for mode in ['lanczos_2x', 'clahe', '']:
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, 'original frames'):
                    preprocess_frame(object(), mode)
                with self.assertRaises(ValueError):
                    RunConfiguration('beo7', '60hz', 'FullCalib', True, preprocess_mode=mode)
                with self.assertRaises(ValueError):
                    run_stage('/missing', '/missing', '/missing', 'beo7', preprocess_mode=mode)

    def test_historical_metadata_is_not_relabelled_as_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            manager = VPDataManager.__new__(VPDataManager)
            for data, expected in [({'preprocess_mode': 'lanczos_2x'}, 'lanczos_2x'),
                                   ({'preprocess_mode': 'none'}, 'none'), ({}, 'unknown')]:
                (run / 'config_used.json').write_text(json.dumps(data))
                self.assertEqual(manager.get_run_config(run).preprocess_mode, expected)

    def test_explicit_image_cache_cannot_bypass_raw_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            vp = Path(tmp)
            source, target = vp / 'source', vp / 'target'
            source.mkdir()
            target.mkdir()
            scanner = CrossRunCacheScanner(vp)
            filename = 'debug_5_ptgaze_calibrated.csv'
            (source / filename).write_text('frame\n1\n')
            (target / 'config_used.json').write_text(json.dumps({'preprocess_mode': 'none'}))
            for metadata in [{'preprocess_mode': 'lanczos_2x'}, {}]:
                (source / 'config_used.json').write_text(json.dumps(metadata))
                self.assertFalse(scanner.copy_step_outputs(source, target, 'debug_5_ptgaze'))
                self.assertFalse((target / filename).exists())
            (source / 'config_used.json').write_text(json.dumps({'preprocess_mode': 'none'}))
            self.assertTrue(scanner.copy_step_outputs(source, target, 'debug_5_ptgaze'))

    def test_eyelink_cache_is_independent_of_image_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            vp = Path(tmp)
            source, target = vp / 'source', vp / 'target'
            source.mkdir()
            target.mkdir()
            scanner = CrossRunCacheScanner(vp)
            scanner.output_mapping['debug_3'] = ['eyelink.csv']
            (source / 'eyelink.csv').write_text('timestamp\n1\n')
            (source / 'config_used.json').write_text(json.dumps({'preprocess_mode': 'lanczos_2x'}))
            self.assertTrue(scanner.copy_step_outputs(source, target, 'debug_3'))


if __name__ == '__main__':
    unittest.main()
