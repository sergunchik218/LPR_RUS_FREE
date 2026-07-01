import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    def __init__(self, num_classes: int = 23, hidden_size: int = 128):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, None)),
        )

        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (B, 1, H, W)
        conv = self.cnn(x)          # (B, 256, 1, W')
        conv = conv.squeeze(2)      # (B, 256, W')
        conv = conv.permute(2, 0, 1)  # (W', B, 256) for LSTM

        rnn_out, _ = self.rnn(conv)  # (W', B, 2*hidden)
        out = self.fc(rnn_out)       # (W', B, num_classes)
        return F.log_softmax(out, dim=2)

    def decode_ctc(self, log_probs, blank_idx=22):
        # log_probs: (T, B, C)
        log_probs = log_probs.cpu()
        probs = torch.exp(log_probs)
        max_probs, max_indices = probs.max(dim=2)
        max_indices = max_indices.transpose(0, 1)  # (B, T)

        results = []
        for b, indices in enumerate(max_indices):
            chars = []
            confs = []
            prev = blank_idx
            for t, idx in enumerate(indices.tolist()):
                if idx != blank_idx and idx != prev:
                    chars.append(idx)
                    confs.append(max_probs[t, b].item())
                prev = idx

            avg_conf = sum(confs) / len(confs) if confs else 0.0
            results.append((chars, avg_conf))
        return results
