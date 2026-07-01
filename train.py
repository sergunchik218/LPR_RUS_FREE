import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time

from dataset import PlateDataset, collate_fn, ALPHABET
from model import CRNN


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = Path(__file__).parent / "autoriaNumberplateOcrRu-2021-09-01"

    train_dataset = PlateDataset(str(data_root), "train", augment=True)
    val_dataset = PlateDataset(str(data_root), "val", augment=False)

    num_classes = train_dataset.num_classes

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        num_workers=0, pin_memory=False, collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=collate_fn
    )

    model = CRNN(num_classes=num_classes, hidden_size=128).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    ctc_loss = nn.CTCLoss(blank=train_dataset.blank_idx, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_acc = 0.0
    save_dir = Path(__file__).parent / "checkpoints"
    save_dir.mkdir(exist_ok=True)

    for epoch in range(1, 31):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, labels, input_lengths, label_lengths) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            log_probs = model(images)  # (T, B, C)

            loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx > 0 and batch_idx % 200 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        val_acc = evaluate(model, val_loader, device, train_dataset.blank_idx, ALPHABET)

        print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
            }, save_dir / "best_model.pt")
            print(f"  -> Saved best model (acc={best_acc:.4f})")

    print(f"\nTraining done. Best val accuracy: {best_acc:.4f}")


def evaluate(model, loader, device, blank_idx, alphabet):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, input_lengths, label_lengths in loader:
            images = images.to(device)
            log_probs = model(images)

            decoded = model.decode_ctc(log_probs, blank_idx)

            for i, (chars, _) in enumerate(decoded):
                pred_text = "".join(alphabet[c] for c in chars)
                true_len = label_lengths[i].item()
                true_text = "".join(alphabet[c] for c in labels[i][:true_len].tolist())
                if pred_text == true_text:
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    train()
