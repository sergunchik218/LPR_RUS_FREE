import argparse
import sys
import cv2
from detect import PlateRecognizer


def main():
    parser = argparse.ArgumentParser(description="License plate detection + recognition")
    parser.add_argument("image", type=str, nargs="?", help="Path to image")
    parser.add_argument("--plate-model", type=str, default="plate_detect_rus.pt")
    parser.add_argument("--car-model", type=str, default="yolo11n.pt")
    parser.add_argument("--ocr", type=str, default="checkpoints/best_model.pt")
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        image_path = input("Enter image path: ").strip().strip('"')

    recognizer = PlateRecognizer(
        car_model=args.car_model,
        plate_model=args.plate_model,
        ocr_path=args.ocr,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: cannot read {image_path}")
        sys.exit(1)

    print(f"Processing: {image_path}")
    plates = recognizer.detect_and_recognize(image)

    if not plates:
        print("No plates found")
        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    print(f"\nFound {len(plates)} plate(s):")
    for i, p in enumerate(plates):
        print(f"  [{i+1}] {p['text']}  (conf: {p['conf']:.3f})")

    result = recognizer.draw_results(image, plates)

    cv2.imshow("Result - press any key to close", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    out_path = image_path.rsplit(".", 1)[0] + "_result.jpg"
    cv2.imwrite(out_path, result)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
