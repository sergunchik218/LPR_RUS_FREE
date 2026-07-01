import argparse
import time
import cv2
from detect import PlateRecognizer


def main():
    parser = argparse.ArgumentParser(description="Распознавание номеров на видео")
    parser.add_argument("video", type=str, help="Путь к видеофайлу")
    parser.add_argument("--ocr", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--interval", type=float, default=1.0, help="Интервал детекции (сек)")
    parser.add_argument("--output", type=str, default="", help="Сохранить результат в файл")
    args = parser.parse_args()

    recognizer = PlateRecognizer(
        car_model="yolo11n.pt",
        plate_model="plate_detect_rus.pt",
        ocr_path=args.ocr,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: cannot open {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {args.video}")
    print(f"FPS: {fps:.1f}, Frames: {total_frames}, Duration: {total_frames/fps:.1f}s")
    print("Press 'q' to quit\n")

    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    last_time = 0
    plates = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display = frame.copy()
        now = time.time()

        if now - last_time > args.interval:
            plates = recognizer.detect_and_recognize(frame)
            last_time = now
            for p in plates:
                print(f"[{frame_count}/{total_frames}] Plate: {p['text']} (conf: {p['conf']:.3f})")

        display = recognizer.draw_results(display, plates)

        cv2.imshow("Video", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if out:
            out.write(display)

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Processed {frame_count} frames.")


if __name__ == "__main__":
    main()
