import cv2


SUPPORTED_PREPROCESS_MODES = {
    "none",
    "lanczos_2x",
}


def preprocess_frame(frame, mode="none"):
    """
    Apply image preprocessing before MediaPipe / PTGaze inference.

    Parameters
    ----------
    frame : np.ndarray
        Input frame in BGR format.
    mode : str
        Preprocessing mode.

    Returns
    -------
    np.ndarray
        Preprocessed frame.
    """
    if mode not in SUPPORTED_PREPROCESS_MODES:
        raise ValueError(f"Unsupported preprocess mode: {mode}")

    if mode == "none":
        return frame

    if mode == "lanczos_2x":
        return cv2.resize(
            frame,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_LANCZOS4,
        )