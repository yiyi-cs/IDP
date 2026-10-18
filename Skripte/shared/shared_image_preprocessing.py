"""Production input contract: preserve original frame size and pixel values."""

SUPPORTED_PREPROCESS_MODES = {"none"}


def validate_preprocess_mode(mode="none"):
    if mode != "none":
        raise ValueError(
            f"Image preprocessing {mode!r} is disabled. "
            "The production pipeline requires original frames (mode='none')."
        )
    return mode


def preprocess_frame(frame, mode="none"):
    validate_preprocess_mode(mode)
    return frame
