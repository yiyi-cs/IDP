from shared_blink_detection import BlinkDetector


def print_result(i, label, r):
    print(
        f"{label:22s} | "
        f"{i:02d} | "
        f"valid={str(r.get('valid_eye_frame')):5s} | "
        f"eyes_closed={str(r.get('eyes_closed')):5s} | "
        f"is_blink={str(r.get('is_blink')):5s} | "
        f"blink_count={r.get('blink_count')} | "
        f"left={r.get('left_ear')} | "
        f"right={r.get('right_ear')} | "
        f"lt={r.get('left_threshold')} | "
        f"rt={r.get('right_threshold')} | "
        f"baseline={r.get('baseline_samples')} | "
        f"reason={r.get('invalid_reason')}"
    )


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


def warm_up_open_eyes(detector, n=12):
    print("\n=== 1. Open-eye warm-up / adaptive threshold ===")
    for i in range(n):
        r = detector._process_ear_values(0.30, 0.31)
        print_result(i, "open_warmup", r)


def test_blink_sequence(detector):
    print("\n=== 2. Blink / closed eyes ===")
    closed_sequence = [
        (0.10, 0.11),
        (0.09, 0.10),
        (0.10, 0.11),
    ]

    for i, (left, right) in enumerate(closed_sequence):
        r = detector._process_ear_values(left, right)
        print_result(i, "blink_closed", r)

    print("\n=== 3. Open again / blink transition ===")
    open_again_sequence = [
        (0.30, 0.31),
        (0.30, 0.31),
        (0.30, 0.31),
        (0.30, 0.31),
        (0.30, 0.31),
    ]

    for i, (left, right) in enumerate(open_again_sequence):
        r = detector._process_ear_values(left, right)
        print_result(i, "open_again", r)


def test_one_eye_closed():
    print("\n=== 4. One-eye closed / suspected blink or landmark error ===")
    detector = make_detector()
    warm_up_open_eyes(detector, n=12)

    sequence = [
        (0.30, 0.10),
        (0.30, 0.10),
        (0.30, 0.10),
    ]

    for i, (left, right) in enumerate(sequence):
        r = detector._process_ear_values(left, right)
        print_result(i, "one_eye_closed", r)


def test_ear_asymmetry():
    print("\n=== 5. EAR asymmetry / suspicious landmark frame ===")
    detector = make_detector()
    warm_up_open_eyes(detector, n=12)

    sequence = [
        (0.55, 0.30),
        (0.55, 0.30),
        (0.55, 0.30),
    ]

    for i, (left, right) in enumerate(sequence):
        r = detector._process_ear_values(left, right)
        print_result(i, "ear_asymmetry", r)


def test_invalid_ear():
    print("\n=== 6. Invalid EAR ===")
    detector = make_detector()

    sequence = [
        (0.0, 0.31),
        (0.30, 0.0),
        (None, 0.31),
        (0.30, None),
    ]

    for i, (left, right) in enumerate(sequence):
        r = detector._process_ear_values(left, right)
        print_result(i, "invalid_ear", r)


def test_reset():
    print("\n=== 7. Reset behavior ===")
    detector = make_detector()

    warm_up_open_eyes(detector, n=12)

    print("\nBefore reset:")
    print("blink_counter:", detector.blink_counter)
    print("left_open_baseline_samples:", len(detector.left_open_ear_buffer))
    print("right_open_baseline_samples:", len(detector.right_open_ear_buffer))

    detector.reset()

    print("\nAfter reset:")
    print("blink_counter:", detector.blink_counter)
    print("frame_counter:", detector.frame_counter)
    print("transition_counter:", detector.transition_counter)
    print("left_open_baseline_samples:", len(detector.left_open_ear_buffer))
    print("right_open_baseline_samples:", len(detector.right_open_ear_buffer))

    r = detector._process_ear_values(0.30, 0.31)
    print_result(0, "after_reset_open", r)


if __name__ == "__main__":
    detector = make_detector()

    warm_up_open_eyes(detector, n=12)
    test_blink_sequence(detector)

    test_one_eye_closed()
    test_ear_asymmetry()
    test_invalid_ear()
    test_reset()