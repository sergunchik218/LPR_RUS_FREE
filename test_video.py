import argparse
import time
import cv2
from detect import PlateRecognizer


def main():
    parser = argparse.ArgumentParser(description="Распознавание номеров на видео")
    parser.add_argument("video", type=str, nargs="?", help="Путь к видеофайлу")
    parser.add_argument("--ocr", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--interval", type=float, default=1.0, help="Интервал детекции (сек)")
    parser.add_argument("--output", type=str, default="", help="Сохранить результат в файл")
    args = parser.parse_args()

    video_path = args.video
    if not video_path:
        video_path = input("Enter video path: ").strip().strip('"')

    recognizer = PlateRecognizer(
        car_model="yolo11n.pt",
        plate_model="plate_detect_rus.pt",
        ocr_path=args.ocr,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path}")
    print(f"FPS: {fps:.1f}, Frames: {total_frames}, Duration: {total_frames/fps:.1f}s")
    print("SPACE = pause/resume  |  S = step frame  |  Q = quit\n")

    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    last_time = 0
    plates = []
    frame_count = 0
    paused = False

    while True:
        if not paused:
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
            if out:
                out.write(display)
        else:
            display = recognizer.draw_results(frame.copy(), plates)

        cv2.imshow("Video", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
            state = "PAUSED" if paused else "RESUMED"
            print(f"[{state}]")
        elif key == ord("s") and paused:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                plates = recognizer.detect_and_recognize(frame)
                for p in plates:
                    print(f"[{frame_count}/{total_frames}] Plate: {p['text']} (conf: {p['conf']:.3f})")
                if out:
                    out.write(recognizer.draw_results(frame.copy(), plates))

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Processed {frame_count} frames.")


if __name__ == "__main__":
    main()
