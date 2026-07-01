import json
import os
from pathlib import Path
import random

DATASET = Path(r"D:\nomera\autoriaNumberplateDataset-2023-03-06\autoriaNumberplateDataset-2023-03-06")
OUT = Path(r"D:\nomera\plate_detection_data")


def convert_split(split):
    via_path = DATASET / split / "via_region_data.json"
    img_dir = DATASET / split
    out_img = OUT / "images" / split
    out_lbl = OUT / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    with open(via_path) as f:
        data = json.load(f)

    meta = data["_via_img_metadata"]
    count = 0

    for img_id, v in meta.items():
        plates = [r for r in v["regions"] if r["region_attributes"].get("class") == "numberplate"]
        if not plates:
            continue

        src_path = img_dir / v["filename"]
        if not src_path.exists():
            continue

        dst_name = f"{split}_{count:05d}"
        dst_img = out_img / f"{dst_name}.jpg"
        dst_lbl = out_lbl / f"{dst_name}.txt"

        import shutil
        shutil.copy2(src_path, dst_img)

        from PIL import Image
        img = Image.open(dst_img)
        iw, ih = img.size

        with open(dst_lbl, "w") as lf:
            for r in plates:
                shape = r["shape_attributes"]
                if shape["name"] == "polygon":
                    xs = shape["all_points_x"]
                    ys = shape["all_points_y"]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                else:
                    x1, y1 = shape["x"], shape["y"]
                    x2, y2 = x1 + shape["width"], y1 + shape["height"]

                xc = ((x1 + x2) / 2) / iw
                yc = ((y1 + y2) / 2) / ih
                w = (x2 - x1) / iw
                h = (y2 - y1) / ih

                lf.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        count += 1
        if count % 200 == 0:
            print(f"  {split}: {count} images converted...")

    print(f"  {split}: {count} images total")


def main():
    print("Converting train...")
    convert_split("train")
    print("Converting val...")
    convert_split("val")

    data_yaml = OUT / "data.yaml"
    with open(data_yaml, "w") as f:
        f.write(f"path: {OUT}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("\n")
        f.write("nc: 1\n")
        f.write("names: ['numberplate']\n")

    print(f"\nDone! YAML config: {data_yaml}")


if __name__ == "__main__":
    main()
