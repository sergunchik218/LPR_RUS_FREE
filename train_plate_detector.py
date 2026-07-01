import sys
import time
from pathlib import Path
from ultralytics import YOLO


def main():
    data_yaml = Path(__file__).parent / "plate_detection_data" / "data.yaml"
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    print("=" * 60)
    print("TRAINING RUSSIAN PLATE DETECTOR")
    print(f"Device: {device}")
    print(f"Dataset: {data_yaml}")
    print("=" * 60)

    model = YOLO("yolo11n.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        workers=0,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        close_mosaic=10,
        amp=device == "cuda",
        project="runs",
        name="plate_detector_rus",
        exist_ok=True,
        verbose=True,
        plots=True,
        save=True,
    )

    best = Path("runs/plate_detector_rus/weights/best.pt")
    if best.exists():
        import shutil
        dst = Path(__file__).parent / "plate_detect_rus.pt"
        shutil.copy(best, dst)
        print(f"\n{'=' * 60}")
        print(f"TRAINING COMPLETE!")
        print(f"Model saved to: {dst}")
        print(f"Size: {dst.stat().st_size / 1e6:.1f} MB")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
