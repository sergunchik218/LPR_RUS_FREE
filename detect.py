import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

from model import CRNN
from dataset import ALPHABET


class PlateRecognizer:
    def __init__(self, car_model="yolov8n.pt", plate_model="plate_detect.pt",
                 ocr_path="checkpoints/best_model.pt", device="cpu"):
        self.device = torch.device(device)

        self.car_detector = YOLO(car_model)
        self.plate_detector = YOLO(plate_model)

        ckpt = torch.load(ocr_path, map_location=self.device, weights_only=False)
        num_classes = ckpt["model_state_dict"]["fc.bias"].shape[0]
        self.ocr = CRNN(num_classes=num_classes).to(self.device)
        self.ocr.load_state_dict(ckpt["model_state_dict"])
        self.ocr.eval()
        self.blank_idx = num_classes - 1

    def detect_and_recognize(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        if image is None:
            return []

        h, w = image.shape[:2]

        car_results = self.car_detector(image, conf=0.35, classes=[2, 5, 7], verbose=False)

        plates = []
        for r in car_results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes.xyxy.cpu().numpy():
                cx1, cy1, cx2, cy2 = map(int, box)
                cx1, cy1 = max(0, cx1), max(0, cy1)
                cx2, cy2 = min(w, cx2), min(h, cy2)
                car_crop = image[cy1:cy2, cx1:cx2]
                if car_crop.size == 0:
                    continue

                plate_results = self.plate_detector(car_crop, conf=0.35, iou=0.45, verbose=False)
                for pr in plate_results:
                    if pr.boxes is None or len(pr.boxes) == 0:
                        continue
                    for pbox in pr.boxes.xyxy.cpu().numpy():
                        px1, py1, px2, py2 = map(int, pbox)
                        px1, py1 = max(0, px1), max(0, py1)
                        px2, py2 = min(car_crop.shape[1], px2), min(car_crop.shape[0], py2)
                        if px2 - px1 < 20 or py2 - py1 < 8:
                            continue

                        expand_x = int((px2 - px1) * 0.15)
                        ocr_x1 = max(0, px1 - expand_x)
                        ocr_x2 = min(car_crop.shape[1], px2 + expand_x)
                        plate_crop = car_crop[py1:py2, ocr_x1:ocr_x2]
                        text, conf = self._recognize_plate(plate_crop)
                        if not text or conf < 0.75:
                            continue

                        gx1 = cx1 + px1
                        gy1 = cy1 + py1
                        gx2 = cx1 + px2
                        gy2 = cy1 + py2

                        plates.append({
                            "bbox": (gx1, gy1, gx2, gy2),
                            "text": text,
                            "conf": conf,
                        })

        return plates

    def _recognize_plate(self, plate_crop):
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        pil_img = Image.fromarray(gray)

        w, h = pil_img.size
        new_w = max(int(32 * w / h), 4)
        pil_img = pil_img.resize((new_w, 32), Image.BICUBIC)

        img_tensor = torch.tensor(list(pil_img.getdata()), dtype=torch.float32)
        img_tensor = img_tensor.view(1, 1, 32, new_w) / 255.0
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            log_probs = self.ocr(img_tensor)
            decoded = self.ocr.decode_ctc(log_probs, self.blank_idx)
            if decoded and decoded[0][0]:
                chars, conf = decoded[0]
                text = "".join(ALPHABET[c] for c in chars)
                return text, conf
        return "", 0.0

    def draw_results(self, image, plates):
        for plate in plates:
            x1, y1, x2, y2 = plate["bbox"]
            text = plate["text"]
            conf = plate.get("conf", 0.0)
            label = f"{text} ({conf:.2f})" if conf > 0 else text
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return image
