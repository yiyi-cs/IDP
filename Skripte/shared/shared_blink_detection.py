"""
shared_blink_detection.py

Shared EAR-based blink / eye-frame validity filter for MediaPipe and ptgaze pipelines.
The public interface is kept backward-compatible with the previous BlinkDetector.
"""

import sys
from pathlib import Path
from typing import Dict, Optional
from collections import deque

import numpy as np


# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

try:
    import config as _config
except ImportError:
    _config = None
    print("[shared_blink_detection] config.py not found, using defaults.")


def _cfg(name: str, default):
    return getattr(_config, name, default) if _config is not None else default


EAR_BLINK_THRESHOLD = _cfg("EAR_BLINK_THRESHOLD", 0.15)
EAR_CONSEC_FRAMES = _cfg("EAR_CONSEC_FRAMES", 2)
ENABLE_BLINK_DETECTION = _cfg("ENABLE_BLINK_DETECTION", True)

# Support both old/new config names where useful.
MAX_EAR_ASYMMETRY = _cfg("MAX_EAR_ASYMMETRY", _cfg("EAR_MAX_ASYMMETRY", 0.2))
TRANSITION_PADDING = _cfg("TRANSITION_PADDING", _cfg("BLINK_TRANSITION_PADDING", 2))

EAR_SMOOTHING_WINDOW = _cfg("EAR_SMOOTHING_WINDOW", 3)
EAR_USE_ADAPTIVE_THRESHOLD = _cfg("EAR_USE_ADAPTIVE_THRESHOLD", True)
EAR_ADAPTIVE_RATIO = _cfg("EAR_ADAPTIVE_RATIO", 0.65)
EAR_BASELINE_MIN_SAMPLES = _cfg("EAR_BASELINE_MIN_SAMPLES", 10)
EAR_BASELINE_WINDOW = _cfg("EAR_BASELINE_WINDOW", 90)
EAR_MIN_THRESHOLD = _cfg("EAR_MIN_THRESHOLD", 0.10)


# ---------------------------------------------------------------------
# Blink detector / eye-frame filter
# ---------------------------------------------------------------------

class BlinkDetector:
    """Backward-compatible EAR-based blink detector with eye-frame filtering."""

    RIGHT_EYE_EAR = [33, 159, 158, 133, 153, 145]
    LEFT_EYE_EAR = [362, 380, 374, 263, 386, 385]

    def __init__(
        self,
        threshold: float = None,
        consec_frames: int = None,
        max_ear_asymmetry: float = None,
        transition_padding: int = None,
        ear_smoothing_window: int = None,
        use_adaptive_threshold: bool = None,
        adaptive_ratio: float = None,
        baseline_min_samples: int = None,
        baseline_window: int = None,
        ear_min_threshold: float = None,
    ):
        self.threshold = threshold if threshold is not None else EAR_BLINK_THRESHOLD
        self.consec_frames = consec_frames if consec_frames is not None else EAR_CONSEC_FRAMES
        self.enabled = ENABLE_BLINK_DETECTION

        self.max_ear_asymmetry = (
            max_ear_asymmetry
            if max_ear_asymmetry is not None
            else MAX_EAR_ASYMMETRY
        )
        self.transition_padding = (
            transition_padding
            if transition_padding is not None
            else TRANSITION_PADDING
        )

        self.ear_smoothing_window = (
            ear_smoothing_window
            if ear_smoothing_window is not None
            else EAR_SMOOTHING_WINDOW
        )
        self.use_adaptive_threshold = (
            use_adaptive_threshold
            if use_adaptive_threshold is not None
            else EAR_USE_ADAPTIVE_THRESHOLD
        )
        self.adaptive_ratio = (
            adaptive_ratio
            if adaptive_ratio is not None
            else EAR_ADAPTIVE_RATIO
        )
        self.baseline_min_samples = (
            baseline_min_samples
            if baseline_min_samples is not None
            else EAR_BASELINE_MIN_SAMPLES
        )
        self.baseline_window = (
            baseline_window
            if baseline_window is not None
            else EAR_BASELINE_WINDOW
        )
        self.ear_min_threshold = (
            ear_min_threshold
            if ear_min_threshold is not None
            else EAR_MIN_THRESHOLD
        )

        # Blink state
        self.frame_counter = 0
        self.blink_counter = 0
        self.transition_counter = 0

        # Short-term smoothing buffers
        self.left_ear_buffer = deque(maxlen=self.ear_smoothing_window)
        self.right_ear_buffer = deque(maxlen=self.ear_smoothing_window)

        # Open-eye baseline buffers for adaptive thresholds
        self.left_open_ear_buffer = deque(maxlen=self.baseline_window)
        self.right_open_ear_buffer = deque(maxlen=self.baseline_window)

        print(
            f"   [OK] BlinkDetector initialized "
            f"(threshold={self.threshold}, "
            f"consec_frames={self.consec_frames}, "
            f"enabled={self.enabled}, "
            f"max_ear_asymmetry={self.max_ear_asymmetry}, "
            f"transition_padding={self.transition_padding}, "
            f"adaptive={self.use_adaptive_threshold})"
        )

    # -----------------------------------------------------------------
    # EAR calculation
    # -----------------------------------------------------------------

    def _calculate_ear_from_points(self, eye_points: np.ndarray) -> float:
        if eye_points is None or len(eye_points) != 6:
            return 0.0

        try:
            vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])
            vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])
            horizontal = np.linalg.norm(eye_points[0] - eye_points[3])

            if horizontal == 0:
                return 0.0

            return float((vertical_1 + vertical_2) / (2.0 * horizontal))

        except (IndexError, TypeError):
            return 0.0

    def calculate_ear_from_landmarks(
        self,
        face_landmarks,
        frame_width: int,
        frame_height: int,
        eye_indices: list,
    ) -> float:
        try:
            points = []
            for idx in eye_indices:
                landmark = face_landmarks.landmark[idx]
                points.append([
                    landmark.x * frame_width,
                    landmark.y * frame_height,
                ])

            return self._calculate_ear_from_points(np.array(points))

        except (IndexError, AttributeError, TypeError):
            return 0.0

    def calculate_ear_from_array(
        self,
        landmarks_array: np.ndarray,
        eye_indices: list,
    ) -> float:
        try:
            points = landmarks_array[eye_indices, :2]
            return self._calculate_ear_from_points(points)

        except (IndexError, TypeError):
            return 0.0

    # -----------------------------------------------------------------
    # Public detection methods
    # -----------------------------------------------------------------

    def detect_blink(
        self,
        face_landmarks,
        frame_width: int,
        frame_height: int,
    ) -> Dict:
        if not self.enabled:
            return self._create_empty_result(
                reason="blink_detection_disabled",
                valid_eye_frame=True,
            )

        left_ear = self.calculate_ear_from_landmarks(
            face_landmarks,
            frame_width,
            frame_height,
            self.LEFT_EYE_EAR,
        )
        right_ear = self.calculate_ear_from_landmarks(
            face_landmarks,
            frame_width,
            frame_height,
            self.RIGHT_EYE_EAR,
        )

        return self._process_ear_values(left_ear, right_ear)

    def detect_blink_from_array(self, landmarks_array: np.ndarray) -> Dict:
        if not self.enabled:
            return self._create_empty_result(
                reason="blink_detection_disabled",
                valid_eye_frame=True,
            )

        left_ear = self.calculate_ear_from_array(
            landmarks_array,
            self.LEFT_EYE_EAR,
        )
        right_ear = self.calculate_ear_from_array(
            landmarks_array,
            self.RIGHT_EYE_EAR,
        )

        return self._process_ear_values(left_ear, right_ear)

    def detect_blink_from_face(
        self,
        face,
        frame_height: int,
        frame_width: int,
    ) -> Dict:
        if not self.enabled:
            return self._create_empty_result(
                reason="blink_detection_disabled",
                valid_eye_frame=True,
            )

        if face is None or not hasattr(face, "landmarks"):
            return self._create_empty_result(reason="missing_face_landmarks")

        if face.landmarks is None or len(face.landmarks) == 0:
            return self._create_empty_result(reason="missing_face_landmarks")

        landmarks = face.landmarks[:, :2].copy()
        landmarks[:, 0] *= frame_width
        landmarks[:, 1] *= frame_height

        return self.detect_blink_from_array(landmarks)

    # -----------------------------------------------------------------
    # Core processing
    # -----------------------------------------------------------------

    def _process_ear_values(self, left_ear: float, right_ear: float) -> Dict:
        if left_ear is None or right_ear is None or left_ear <= 0.0 or right_ear <= 0.0:
            return self._create_empty_result(reason="invalid_ear")

        # 1. 分别对左右眼 EAR 做短期时间平滑
        self.left_ear_buffer.append(float(left_ear))
        self.right_ear_buffer.append(float(right_ear))

        left_ear_smooth = float(np.median(self.left_ear_buffer))
        right_ear_smooth = float(np.median(self.right_ear_buffer))

        # avg_ear 只用于兼容旧代码和日志记录，不作为主要判断依据
        avg_ear = (left_ear_smooth + right_ear_smooth) / 2.0
        ear_asymmetry = abs(left_ear_smooth - right_ear_smooth)

        # 2. 获取当前左右眼阈值
        # 如果已有足够的睁眼 baseline 样本，则使用自适应阈值；
        # 否则退回固定阈值 self.threshold。
        left_threshold, right_threshold = self._get_current_thresholds()

        # 3. 分别判断左右眼是否闭合
        # raw 判断用于捕捉 blink 开始时的快速下降；
        # smooth 判断用于避免单帧噪声造成误判。
        left_closed_raw = float(left_ear) < left_threshold
        right_closed_raw = float(right_ear) < right_threshold

        left_closed_smooth = left_ear_smooth < left_threshold
        right_closed_smooth = right_ear_smooth < right_threshold

        # 只要 raw 或 smooth 显示闭合，就先认为该眼当前不可靠
        left_closed = left_closed_raw or left_closed_smooth
        right_closed = right_closed_raw or right_closed_smooth

        both_eyes_closed = left_closed and right_closed
        one_eye_closed = left_closed != right_closed
        asymmetry_invalid = ear_asymmetry > self.max_ear_asymmetry

        # 为了兼容旧 pipeline：
        # eyes_closed 表示双眼都闭合，不表示单眼异常
        eyes_closed = both_eyes_closed

        # 4. Blink event 状态逻辑
        # is_blink=True 表示一次 blink 结束，用于计数；
        # 它不等价于“当前帧无效”。
        is_blink = False

        if both_eyes_closed:
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.consec_frames:
                self.blink_counter += 1
                is_blink = True
                self.transition_counter = self.transition_padding

            self.frame_counter = 0

        # 5. Blink 结束后的 transition padding
        # 这些帧通常已经重新睁眼，但 pupil/gaze 仍可能不稳定。
        is_transition_frame = False
        if not both_eyes_closed and self.transition_counter > 0:
            is_transition_frame = True
            self.transition_counter -= 1

        # 6. 当前帧是否可用于后续 gaze / pupil 分析
        if both_eyes_closed:
            valid_eye_frame = False
            invalid_reason = "blink_or_eyes_closed"
        elif one_eye_closed:
            valid_eye_frame = False
            invalid_reason = "suspected_blink_or_landmark_error"
        elif is_transition_frame:
            valid_eye_frame = False
            invalid_reason = "blink_transition"
        elif asymmetry_invalid:
            valid_eye_frame = False
            invalid_reason = "ear_asymmetry"
        else:
            valid_eye_frame = True
            invalid_reason = None

        is_suspected_blink = both_eyes_closed or one_eye_closed or is_transition_frame

        # 7. 只用真正有效、稳定、睁眼的帧更新 open-eye baseline
        if valid_eye_frame:
            self._update_open_ear_baseline(
                left_ear_smooth,
                right_ear_smooth,
                left_closed,
                right_closed,
            )

        return {
            # 原有 blink 字段，保留用于兼容旧 pipeline
            "is_blink": bool(is_blink),
            "eyes_closed": bool(eyes_closed),
            "left_ear": float(left_ear_smooth),
            "right_ear": float(right_ear_smooth),
            "avg_ear": float(avg_ear),
            "blink_count": self.blink_counter,
            "frames_below_threshold": self.frame_counter,

            # 新增 frame filter 字段
            "valid_eye_frame": bool(valid_eye_frame),
            "invalid_reason": invalid_reason,
            "is_suspected_blink": bool(is_suspected_blink),
            "is_transition_frame": bool(is_transition_frame),

            # Debug 字段：用于后续检查阈值、平滑和自适应 baseline
            "left_ear_raw": float(left_ear),
            "right_ear_raw": float(right_ear),
            "left_closed": bool(left_closed),
            "right_closed": bool(right_closed),
            "left_closed_raw": bool(left_closed_raw),
            "right_closed_raw": bool(right_closed_raw),
            "left_closed_smooth": bool(left_closed_smooth),
            "right_closed_smooth": bool(right_closed_smooth),
            "left_threshold": float(left_threshold),
            "right_threshold": float(right_threshold),
            "ear_asymmetry": float(ear_asymmetry),
            "baseline_samples": min(
                len(self.left_open_ear_buffer),
                len(self.right_open_ear_buffer),
            ),
        }

    # -----------------------------------------------------------------
    # Result helpers
    # -----------------------------------------------------------------

    def _create_empty_result(
        self,
        reason: str = "invalid_eye_data",
        valid_eye_frame: bool = False,
    ) -> Dict:
        invalid_reason = None if valid_eye_frame else reason

        return {
            # 原有 blink 字段，保留用于兼容旧 pipeline
            "is_blink": False,
            "eyes_closed": False,
            "left_ear": None,
            "right_ear": None,
            "avg_ear": None,
            "blink_count": self.blink_counter,
            "frames_below_threshold": 0,

            # 新增 frame filter 字段
            "valid_eye_frame": bool(valid_eye_frame),
            "invalid_reason": invalid_reason,
            "is_suspected_blink": False,
            "is_transition_frame": False,

            # Debug 字段，保持返回结构统一
            "left_ear_raw": None,
            "right_ear_raw": None,
            "left_closed": False,
            "right_closed": False,
            "left_closed_raw": False,
            "right_closed_raw": False,
            "left_closed_smooth": False,
            "right_closed_smooth": False,
            "left_threshold": float(self.threshold),
            "right_threshold": float(self.threshold),
            "ear_asymmetry": None,
            "baseline_samples": min(
                len(self.left_open_ear_buffer),
                len(self.right_open_ear_buffer),
            ),
        }

    # -----------------------------------------------------------------
    # Adaptive threshold helpers
    # -----------------------------------------------------------------

    def _get_current_thresholds(self):
        """
        返回当前左右眼 EAR 阈值。

        如果已经收集到足够的有效睁眼 EAR 样本，则使用自适应阈值。
        否则使用固定阈值 self.threshold。
        """
        use_adaptive = (
            self.use_adaptive_threshold
            and len(self.left_open_ear_buffer) >= self.baseline_min_samples
            and len(self.right_open_ear_buffer) >= self.baseline_min_samples
        )

        if not use_adaptive:
            return float(self.threshold), float(self.threshold)

        left_baseline = float(np.median(self.left_open_ear_buffer))
        right_baseline = float(np.median(self.right_open_ear_buffer))

        left_threshold = max(
            left_baseline * self.adaptive_ratio,
            self.ear_min_threshold,
        )
        right_threshold = max(
            right_baseline * self.adaptive_ratio,
            self.ear_min_threshold,
        )

        return float(left_threshold), float(right_threshold)

    def _update_open_ear_baseline(
        self,
        left_ear_smooth: float,
        right_ear_smooth: float,
        left_closed: bool,
        right_closed: bool,
    ) -> None:
        """
        更新睁眼状态下的 EAR baseline。

        只有看起来可靠、且双眼都处于睁开状态的帧才会被加入 baseline。
        blink、疑似 blink、EAR 无效或左右眼差异过大的帧都会被排除。
        """
        if left_ear_smooth is None or right_ear_smooth is None:
            return

        if left_ear_smooth <= 0.0 or right_ear_smooth <= 0.0:
            return

        if left_closed or right_closed:
            return

        ear_asymmetry = abs(left_ear_smooth - right_ear_smooth)
        if ear_asymmetry > self.max_ear_asymmetry:
            return

        self.left_open_ear_buffer.append(float(left_ear_smooth))
        self.right_open_ear_buffer.append(float(right_ear_smooth))

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def reset(self):
        self.frame_counter = 0
        self.blink_counter = 0
        self.transition_counter = 0

        self.left_ear_buffer.clear()
        self.right_ear_buffer.clear()
        self.left_open_ear_buffer.clear()
        self.right_open_ear_buffer.clear()

    def close(self):
        pass


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------

def create_blink_detector(
    threshold: float = None,
    consec_frames: int = None,
    max_ear_asymmetry: float = None,
    transition_padding: int = None,
    ear_smoothing_window: int = None,
    use_adaptive_threshold: bool = None,
    adaptive_ratio: float = None,
    baseline_min_samples: int = None,
    baseline_window: int = None,
    ear_min_threshold: float = None,
) -> Optional[BlinkDetector]:
    if not ENABLE_BLINK_DETECTION:
        print("   [INFO] Blink filter disabled.")
        return None

    return BlinkDetector(
        threshold=threshold,
        consec_frames=consec_frames,
        max_ear_asymmetry=max_ear_asymmetry,
        transition_padding=transition_padding,
        ear_smoothing_window=ear_smoothing_window,
        use_adaptive_threshold=use_adaptive_threshold,
        adaptive_ratio=adaptive_ratio,
        baseline_min_samples=baseline_min_samples,
        baseline_window=baseline_window,
        ear_min_threshold=ear_min_threshold,
    )