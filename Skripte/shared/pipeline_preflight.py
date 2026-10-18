"""Check decoder/model initialization before expensive pipeline stages."""
import argparse
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_runtime(video, methods='both'):
    if methods not in ('both', 'mediapipe', 'ptgaze'):
        raise ValueError(f'Unknown methods: {methods}')
    from shared.shared_image_preprocessing import validate_preprocess_mode
    from config import PREPROCESS_MODE
    validate_preprocess_mode(PREPROCESS_MODE)
    for executable in ['ffmpeg', 'ffprobe']:
        if shutil.which(executable) is None:
            raise RuntimeError(f'Required executable missing: {executable}')
    for module in ['cv2', 'mediapipe', 'sklearn', 'scipy', 'matplotlib', 'librosa',
                   'blink.generate_blink_diagnostics',
                   'fixation.generate_fixation_timeline_pdf']:
        importlib.import_module(module)
    import cv2
    import numpy as np
    from shared.shared_pupil_detection import RobustPupilDetector
    video = Path(video).resolve(strict=True)
    info = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'json', str(video)],
        check=True, capture_output=True, text=True, timeout=30).stdout)['streams'][0]
    width, height = int(info['width']), int(info['height'])
    encoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-map', '0:v:0',
        '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'],
        check=True, capture_output=True, timeout=30).stdout
    frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0 or frame.shape[:2] != (height, width):
        raise RuntimeError('Video decode dimensions disagree; check rotation metadata')
    detector = RobustPupilDetector()
    try:
        detector.extract_from_frame(frame.copy())
    finally:
        detector.close()
    if methods in ('ptgaze', 'both'):
        from debug.debug_1_ptgaze import PtgazeGazeAnalyzer
        analyzer = PtgazeGazeAnalyzer(video_width=width, video_height=height)
        try:
            analyzer.detector.extract_from_frame(frame.copy())
        finally:
            analyzer.close()
    print('[OK] Preflight: raw frame, decoder, requested models and PDF imports')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--methods', choices=['both', 'mediapipe', 'ptgaze'], default='both')
    args = parser.parse_args()
    check_runtime(args.video, args.methods)


if __name__ == '__main__':
    main()
