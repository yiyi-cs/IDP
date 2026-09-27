import cv2

from shared.shared_image_preprocessing import preprocess_frame
from shared.shared_gaze_detection_ptgaze import PtgazeGazeDetector


VIDEO_PATH = "/Volumes/Empra9/Videos_60/beo7.mp4"





cap = cv2.VideoCapture(VIDEO_PATH)

raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

processed_width = raw_width * 2
processed_height = raw_height * 2

print(f"Raw size: {raw_width}x{raw_height}")
print(f"PTGaze input size: {processed_width}x{processed_height}")

detector = PtgazeGazeDetector(
    video_width=processed_width,
    video_height=processed_height,
    device="cpu",
    model="eth-xgaze",
)

n_frames = 10
n_detected = 0

for i in range(n_frames):
    ret, frame = cap.read()
    if not ret:
        break

    frame = preprocess_frame(frame, "lanczos_2x")

    result = detector.extract_from_frame(frame)

    if result["detected"]:
        n_detected += 1

        print(
            f"Frame {i}: "
            f"pitch={result['gaze_pitch_deg']:.2f}, "
            f"yaw={result['gaze_yaw_deg']:.2f}, "
            f"EAR={result.get('avg_ear')}"
        )
    else:
        print(f"Frame {i}: no detection")

cap.release()
detector.close()

print(f"\nDetection: {n_detected}/{n_frames}")