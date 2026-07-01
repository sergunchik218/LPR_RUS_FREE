import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import PlateDataset, collate_fn, ALPHABET
from model import CRNN


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = Path(__file__).parent / "autoriaNumberplateOcrRu-2021-09-01"

    train_dataset = PlateDataset(str(data_root), "train", augment=True, balance=True)
    val_dataset = PlateDataset(str(data_root), "val", augment=False)
    print(f"Train samples: {len(train_dataset)} (balanced), Val samples: {len(val_dataset)}")

    num_classes = train_dataset.num_classes
    blank_idx = train_dataset.blank_idx

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

    save_dir = Path(__file__).parent / "checkpoints"
    backup_path = save_dir / "best_model_backup.pt"
    start_epoch = 0
    best_acc = 0.0

    if backup_path.exists():
        ckpt = torch.load(backup_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 8)
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.4f}")
    else:
        raise FileNotFoundError("backup checkpoint not found")

    ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    for epoch in range(1, 16):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, labels, input_lengths, label_lengths) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            log_probs = model(images)
            loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx > 0 and batch_idx % 200 == 0:
                print(f"  Epoch {start_epoch+epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        val_acc, acc_8, acc_9 = evaluate_detailed(model, val_loader, device, blank_idx, ALPHABET)

        print(f"Epoch {start_epoch+epoch:2d} | Loss: {avg_loss:.4f} | Acc: {val_acc:.4f} (8ch: {acc_8:.4f}, 9ch: {acc_9:.4f})")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": start_epoch + epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
            }, save_dir / "best_model.pt")
            print(f"  -> Saved best model (acc={best_acc:.4f})")

    print(f"\nTraining done. Best val accuracy: {best_acc:.4f}")


def evaluate_detailed(model, loader, device, blank_idx, alphabet):
    model.eval()
    total = 0
    correct = 0
    correct_8 = 0
    correct_9 = 0
    total_8 = 0
    total_9 = 0

    with torch.no_grad():
        for images, labels, input_lengths, label_lengths in loader:
            images = images.to(device)
            log_probs = model(images)
            decoded = model.decode_ctc(log_probs, blank_idx)

            for i, (chars, _) in enumerate(decoded):
                pred_text = "".join(alphabet[c] for c in chars)
                true_len = label_lengths[i].item()
                true_text = "".join(alphabet[c] for c in labels[i][:true_len].tolist())
                ok = pred_text == true_text
                if ok:
                    correct += 1
                total += 1
                if true_len <= 8:
                    total_8 += 1
                    if ok:
                        correct_8 += 1
                else:
                    total_9 += 1
                    if ok:
                        correct_9 += 1

    return (
        correct / total if total > 0 else 0.0,
        correct_8 / total_8 if total_8 > 0 else 0.0,
        correct_9 / total_9 if total_9 > 0 else 0.0,
    )


if __name__ == "__main__":
    train()
