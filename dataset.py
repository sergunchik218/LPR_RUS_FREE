import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image, ImageEnhance
import random

ALPHABET = "ABEKMHOPCTYX0123456789"


class PlateDataset(Dataset):
    def __init__(self, root: str, split: str = "train", augment: bool = False, img_height: int = 32):
        self.root = Path(root) / split
        self.img_dir = self.root / "img"
        self.ann_dir = self.root / "ann"
        self.img_height = img_height
        self.augment = augment

        self.samples = []
        for ann_path in sorted(self.ann_dir.glob("*.json")):
            img_path = self.img_dir / (ann_path.stem + ".png")
            if img_path.exists():
                self.samples.append((str(img_path), str(ann_path)))

        self.char_to_idx = {c: i for i, c in enumerate(ALPHABET)}
        self.blank_idx = len(ALPHABET)
        self.num_classes = len(ALPHABET) + 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]

        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        text = ann["description"].upper()

        image = Image.open(img_path).convert("L")

        if self.augment:
            image = self._augment(image)

        w, h = image.size
        new_w = max(int(self.img_height * w / h), 4)
        image = image.resize((new_w, self.img_height), Image.BICUBIC)

        img_tensor = torch.tensor(list(image.getdata()), dtype=torch.float32)
        img_tensor = img_tensor.view(1, self.img_height, new_w) / 255.0

        label = torch.tensor([self.char_to_idx.get(c, self.blank_idx) for c in text], dtype=torch.long)

        return img_tensor, label, len(text)

    def _augment(self, image):
        if random.random() < 0.3:
            image = image.resize((
                max(int(image.width * random.uniform(0.9, 1.1)), 4),
                max(int(image.height * random.uniform(0.9, 1.1)), 4),
            ), Image.BICUBIC)

        if random.random() < 0.2:
            angle = random.uniform(-5, 5)
            image = image.rotate(angle, expand=False, fillcolor=255)

        if random.random() < 0.3:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.7, 1.3))

        if random.random() < 0.3:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.7, 1.3))

        return image


def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    max_w = max(img.shape[2] for img in images)
    max_len = max(len(lbl) for lbl in labels)

    padded_images = torch.zeros(len(images), 1, images[0].shape[1], max_w)
    padded_labels = torch.full((len(images), max_len), 0, dtype=torch.long)

    for i, img in enumerate(images):
        padded_images[i, :, :, :img.shape[2]] = img

    for i, lbl in enumerate(labels):
        padded_labels[i, :len(lbl)] = lbl

    label_lengths = torch.tensor(list(lengths), dtype=torch.long)
    input_lengths = torch.tensor([max_w // 4 for _ in images], dtype=torch.long)

    return padded_images, padded_labels, input_lengths, label_lengths
