"""Quick check of the production raw-frame contract (no models required)."""
from shared.shared_image_preprocessing import preprocess_frame


def main():
    original_frame = object()
    assert preprocess_frame(original_frame, "none") is original_frame
    print("OK: original frame returned unchanged")


if __name__ == "__main__":
    main()
