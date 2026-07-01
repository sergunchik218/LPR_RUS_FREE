import argparse
import time
import cv2
from detect import PlateRecognizer


def main():
    parser = argparse.ArgumentParser(description="Распознавание номеров с RTSP-камеры")
    parser.add_argument("--rtsp", type=str, required=True, help="RTSP URL камеры")
    parser.add_argument("--detector", type=str, default="nomera.pt")
    parser.add_argument("--ocr", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--interval", type=float, default=1.0, help="Интервал детекции (сек)")
    args = parser.parse_args()

    recognizer = PlateRecognizer(
        detector_path=args.detector,
        ocr_path=args.ocr,
        device="cpu",
    )

    cap = cv2.VideoCapture(args.rtsp)
    if not cap.isOpened():
        print(f"Error: cannot connect to {args.rtsp}")
        return

    print(f"Connected to {args.rtsp}")
    print("Press 'q' to quit")

    last_time = 0
    plates = []

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            continue

        display = frame.copy()
        now = time.time()

        if now - last_time > args.interval:
            plates = recognizer.detect_and_recognize(frame)
            last_time = now
            for p in plates:
                print(f"Plate: {p['text']}")

        display = recognizer.draw_results(display, plates)

        cv2.imshow("RTSP Camera", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
