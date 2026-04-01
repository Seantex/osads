"""Binary audio classifier: one model per insect type.

Instead of one model trying to distinguish all insect types,
we train a separate binary classifier per type:
  - mosquito_detector: mosquito vs. everything else
  - gnat_detector: gnat vs. everything else
  - fly_detector: fly vs. everything else

This is simpler, more accurate, and maps to the mode-switch UI.
Each model only needs to answer: "Is this MY insect? yes/no"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)


class BinaryInsectCNN(nn.Module):
    """Binary classifier: target insect vs. not-target.

    Simpler than multi-class → higher accuracy per target.
    """

    def __init__(self, n_mels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 2)  # Binary: [not_target, target]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


# Insect profiles for audio generation
INSECT_PROFILES = {
    "mosquito": {
        "freq_mean": 500, "freq_std": 60,
        "harmonics": [1.0, 0.35, 0.15, 0.05],
        "mod_freq": (12, 20),
    },
    "gnat": {
        "freq_mean": 700, "freq_std": 40,
        "harmonics": [1.0, 0.2, 0.05],
        "mod_freq": (15, 30),
    },
    "fly": {
        "freq_mean": 200, "freq_std": 25,
        "harmonics": [1.0, 0.4, 0.2, 0.1, 0.05],
        "mod_freq": (8, 15),
    },
}


class BinaryAudioDataset(Dataset):
    """Dataset for binary classification: target insect vs. everything else.

    Positive samples: target insect audio
    Negative samples: other insects + background noise
    """

    def __init__(
        self,
        target_insect: str,
        num_positive: int = 3000,
        num_negative: int = 3000,
        sample_rate: int = 44100,
        duration: float = 1.0,
        augment: bool = True,
    ) -> None:
        self.target = target_insect
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.total = num_positive + num_negative
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.augment = augment

        from osads.detection.acoustic import MelSpectrogramExtractor
        self.mel_extractor = MelSpectrogramExtractor(
            sample_rate=sample_rate, duration=duration,
        )

        # Non-target insects
        self.other_insects = [k for k in INSECT_PROFILES if k != target_insect]

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if idx < self.num_positive:
            # Positive: target insect
            audio = self._generate_insect(self.target)
            label = 1
        else:
            # Negative: other insect or background
            if np.random.random() < 0.5 and self.other_insects:
                other = np.random.choice(self.other_insects)
                audio = self._generate_insect(other)
            else:
                audio = self._generate_background()
            label = 0

        if self.augment:
            audio = self._augment(audio)

        mel = self.mel_extractor.extract(audio)
        return torch.from_numpy(mel).unsqueeze(0), label

    def _generate_insect(self, insect_type: str) -> np.ndarray:
        """Generate realistic insect audio."""
        t = np.linspace(0, self.duration, self.n_samples, endpoint=False)
        profile = INSECT_PROFILES[insect_type]

        base_freq = np.random.normal(profile["freq_mean"], profile["freq_std"])
        base_freq = max(50, base_freq)

        signal = np.zeros(self.n_samples, dtype=np.float64)
        for i, amp in enumerate(profile["harmonics"]):
            freq = base_freq * (i + 1)
            phase = np.random.uniform(0, 2 * np.pi)
            wobble = 1 + 0.01 * np.sin(2 * np.pi * np.random.uniform(1, 5) * t)
            signal += amp * np.sin(2 * np.pi * freq * wobble * t + phase)

        mod_min, mod_max = profile["mod_freq"]
        mod_freq = np.random.uniform(mod_min, mod_max)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
        signal *= envelope * np.random.uniform(0.1, 0.5)
        signal += np.random.normal(0, np.random.uniform(0.01, 0.05), self.n_samples)

        return signal.astype(np.float32)

    def _generate_background(self) -> np.ndarray:
        """Generate varied background noise including confusing frequencies."""
        noise_type = np.random.choice([
            "white", "pink", "hum", "silence", "high_tone", "random_tone",
            "modulated_hum", "modulated_high",
        ], p=[0.15, 0.10, 0.15, 0.10, 0.15, 0.10, 0.125, 0.125])
        t = np.linspace(0, self.duration, self.n_samples, endpoint=False)

        if noise_type == "white":
            return np.random.normal(0, np.random.uniform(0.01, 0.05),
                                    self.n_samples).astype(np.float32)
        elif noise_type == "pink":
            white = np.random.normal(0, 0.03, self.n_samples)
            return (np.cumsum(white) * 0.002).astype(np.float32)
        elif noise_type == "hum":
            # Plain electric hum — no insect-like modulation
            freq = np.random.choice([50.0, 60.0, 100.0, 150.0, 200.0])
            amp = np.random.uniform(0.05, 0.35)
            sig = amp * np.sin(2 * np.pi * freq * t)
            sig += np.random.normal(0, 0.01, self.n_samples)
            return sig.astype(np.float32)
        elif noise_type == "modulated_hum":
            # Hard negative: electric hum WITH insect-like AM — must NOT trigger
            freq = np.random.choice([50.0, 60.0, 100.0])
            mod = np.random.uniform(8, 35)   # overlaps insect mod_freq ranges
            amp = np.random.uniform(0.1, 0.35)
            sig = amp * np.sin(2 * np.pi * freq * t)
            sig *= 0.5 + 0.5 * np.sin(2 * np.pi * mod * t)
            sig += np.random.normal(0, 0.01, self.n_samples)
            return sig.astype(np.float32)
        elif noise_type == "high_tone":
            # High-frequency non-insect sounds (1-8 kHz), plain
            freq = np.random.uniform(1000, 8000)
            sig = np.random.uniform(0.05, 0.3) * np.sin(2 * np.pi * freq * t)
            sig += np.random.normal(0, 0.02, self.n_samples)
            return sig.astype(np.float32)
        elif noise_type == "modulated_high":
            # Hard negative: high-frequency tone WITH insect-like AM — must NOT trigger
            freq = np.random.uniform(1000, 6000)
            mod = np.random.uniform(8, 35)
            amp = np.random.uniform(0.1, 0.3)
            sig = amp * np.sin(2 * np.pi * freq * t)
            sig *= 0.5 + 0.5 * np.sin(2 * np.pi * mod * t)
            sig += np.random.normal(0, 0.015, self.n_samples)
            return sig.astype(np.float32)
        elif noise_type == "random_tone":
            # Random tones across full range, plain
            freq = np.random.uniform(50, 5000)
            sig = np.random.uniform(0.05, 0.2) * np.sin(2 * np.pi * freq * t)
            sig += np.random.normal(0, 0.02, self.n_samples)
            return sig.astype(np.float32)
        else:
            return np.random.normal(0, 0.005, self.n_samples).astype(np.float32)

    def _augment(self, audio: np.ndarray) -> np.ndarray:
        if np.random.random() < 0.5:
            audio = audio + np.random.normal(0, np.random.uniform(0.005, 0.03),
                                              len(audio)).astype(np.float32)
        if np.random.random() < 0.5:
            audio = audio * np.random.uniform(0.5, 2.0)
        if np.random.random() < 0.3:
            audio = np.roll(audio, np.random.randint(-len(audio) // 10, len(audio) // 10))
        return np.clip(audio, -1.0, 1.0)


class BinaryAudioTrainer:
    """Train a binary classifier for one insect type."""

    def __init__(self, target_insect: str, device: str = "auto") -> None:
        self.target = target_insect
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = BinaryInsectCNN().to(self.device)
        logger.info(f"[{target_insect}] Using device: {self.device}")

    def train(
        self,
        num_samples: int = 3000,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 0.001,
    ) -> dict[str, Any]:
        dataset = BinaryAudioDataset(
            target_insect=self.target,
            num_positive=num_samples,
            num_negative=num_samples,
            augment=True,
        )

        val_size = int(len(dataset) * 0.2)
        train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0

        for epoch in range(epochs):
            self.model.train()
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(bx), by)
                loss.backward()
                optimizer.step()

            # Validate
            self.model.eval()
            correct = total = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    pred = self.model(bx).argmax(1)
                    correct += (pred == by).sum().item()
                    total += by.size(0)

            acc = correct / total
            best_acc = max(best_acc, acc)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"[{self.target}] Epoch {epoch+1}/{epochs} | Val Acc: {acc:.3f}")

        return {"target": self.target, "best_val_acc": best_acc}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": self.model.state_dict(), "target": self.target}, path)
        logger.info(f"[{self.target}] Model saved to {path}")

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])

    def predict(self, mel_spectrogram: np.ndarray) -> tuple[bool, float]:
        """Predict if target insect is present.

        Returns:
            (is_detected, confidence)
        """
        self.model.eval()
        x = torch.from_numpy(mel_spectrogram).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.model(x), dim=1).cpu().numpy()[0]
        return bool(probs[1] > 0.5), float(probs[1])
