from collections import Counter
from shared_blink_detection import BlinkDetector


def make_detector():
    return BlinkDetector(
        threshold=0.16,
        consec_frames=2,
        max_ear_asymmetry=0.20,
        transition_padding=2,
        ear_smoothing_window=3,
        use_adaptive_threshold=True,
        adaptive_ratio=0.65,
        baseline_min_samples=10,
        baseline_window=90,
        ear_min_threshold=0.10,
    )


def fmt(value, ndigits=4):
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.{ndigits}f}"
    return str(value)


def build_continuous_sequence():
    """
    模拟一个连续视频过程。

    每个元素格式：
        (label, left_ear, right_ear)
    """

    sequence = []

    # 1. 正常睁眼 warm-up，用于建立 adaptive threshold baseline
    sequence += [("open_warmup", 0.30, 0.31)] * 12

    # 2. 第一次 blink：双眼闭合 3 帧
    sequence += [("blink_closed_1", 0.10, 0.11)] * 3

    # 3. blink 后重新睁眼，应该包含 transition padding
    sequence += [("open_after_blink_1", 0.30, 0.31)] * 6

    # 4. 正常稳定睁眼
    sequence += [("stable_open_1", 0.31, 0.30)] * 5

    # 5. 单眼异常：可能是单眼闭合、遮挡或 landmark error
    sequence += [("one_eye_closed", 0.30, 0.10)] * 3

    # 6. 恢复正常
    sequence += [("stable_open_2", 0.30, 0.31)] * 5

    # 7. 左右 EAR 差异过大：疑似 landmark 异常
    sequence += [("ear_asymmetry", 0.55, 0.30)] * 3

    # 8. 恢复正常
    sequence += [("stable_open_3", 0.30, 0.31)] * 5

    # 9. invalid EAR：模拟检测失败 / EAR 算不出来
    sequence += [
        ("invalid_ear_left_zero", 0.00, 0.31),
        ("invalid_ear_right_zero", 0.30, 0.00),
        ("invalid_ear_left_none", None, 0.31),
        ("invalid_ear_right_none", 0.30, None),
    ]

    # 10. 再次恢复正常
    sequence += [("stable_open_4", 0.30, 0.31)] * 5

    # 11. 第二次 blink
    sequence += [("blink_closed_2", 0.09, 0.10)] * 2
    sequence += [("open_after_blink_2", 0.30, 0.31)] * 6

    return sequence


def run_continuous_test():
    detector = make_detector()
    sequence = build_continuous_sequence()

    valid_count = 0
    invalid_count = 0
    reason_counter = Counter()
    label_counter = Counter()
    valid_by_label = Counter()
    invalid_by_label = Counter()

    print("\n=== Continuous blink filter test ===\n")

    header = (
        f"{'idx':>3s} | "
        f"{'label':22s} | "
        f"{'L_raw':>7s} | "
        f"{'R_raw':>7s} | "
        f"{'L_smooth':>8s} | "
        f"{'R_smooth':>8s} | "
        f"{'L_thr':>7s} | "
        f"{'R_thr':>7s} | "
        f"{'valid':>5s} | "
        f"{'eyes':>5s} | "
        f"{'blink':>5s} | "
        f"{'blink_n':>7s} | "
        f"{'base':>4s} | "
        f"reason"
    )
    print(header)
    print("-" * len(header))

    for idx, (label, left_ear, right_ear) in enumerate(sequence):
        result = detector._process_ear_values(left_ear, right_ear)

        is_valid = result["valid_eye_frame"]
        reason = result["invalid_reason"]

        if is_valid:
            valid_count += 1
            valid_by_label[label] += 1
        else:
            invalid_count += 1
            invalid_by_label[label] += 1
            reason_counter[reason] += 1

        label_counter[label] += 1

        print(
            f"{idx:3d} | "
            f"{label:22s} | "
            f"{fmt(left_ear):>7s} | "
            f"{fmt(right_ear):>7s} | "
            f"{fmt(result.get('left_ear')):>8s} | "
            f"{fmt(result.get('right_ear')):>8s} | "
            f"{fmt(result.get('left_threshold')):>7s} | "
            f"{fmt(result.get('right_threshold')):>7s} | "
            f"{str(is_valid):>5s} | "
            f"{str(result.get('eyes_closed')):>5s} | "
            f"{str(result.get('is_blink')):>5s} | "
            f"{result.get('blink_count'):7d} | "
            f"{result.get('baseline_samples'):4d} | "
            f"{reason}"
        )

    print("\n=== Summary ===")
    print(f"total frames:   {len(sequence)}")
    print(f"valid frames:   {valid_count}")
    print(f"invalid frames: {invalid_count}")
    print(f"blink events:   {detector.blink_counter}")
    print(f"baseline left:  {len(detector.left_open_ear_buffer)}")
    print(f"baseline right: {len(detector.right_open_ear_buffer)}")

    print("\n=== Invalid reasons ===")
    if reason_counter:
        for reason, count in reason_counter.items():
            print(f"{reason:35s}: {count}")
    else:
        print("No invalid frames.")

    print("\n=== Per-label valid / invalid counts ===")
    for label in label_counter:
        total = label_counter[label]
        valid = valid_by_label[label]
        invalid = invalid_by_label[label]
        print(
            f"{label:22s}: "
            f"total={total:2d}, valid={valid:2d}, invalid={invalid:2d}"
        )


if __name__ == "__main__":
    run_continuous_test()