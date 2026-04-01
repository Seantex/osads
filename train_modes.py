#!/usr/bin/env python3
"""Train individual binary classifiers per insect mode.

Usage:
    python train_modes.py                      # Train all 3 modes
    python train_modes.py --mode mosquito      # Train only mosquito mode
    python train_modes.py --mode gnat          # Train only gnat mode
    python train_modes.py --mode fly           # Train only fly mode
    python train_modes.py --test               # Train + run tests

Each mode gets its own model file:
    models/mosquito_detector.pt
    models/gnat_detector.pt
    models/fly_detector.pt
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

sys.path.insert(0, "src")

from osads.training.binary_audio_model import (
    BinaryAudioTrainer, INSECT_PROFILES,
)
from osads.detection.acoustic import MelSpectrogramExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODES = ["mosquito", "gnat", "fly"]


def train_mode(mode: str, args: argparse.Namespace) -> BinaryAudioTrainer:
    logger.info(f"\n{'='*60}")
    logger.info(f"Training MODE: {mode.upper()}")
    logger.info(f"{'='*60}")

    trainer = BinaryAudioTrainer(target_insect=mode)
    result = trainer.train(
        num_samples=args.samples,
        epochs=args.epochs,
        batch_size=32,
    )
    logger.info(f"[{mode}] Best accuracy: {result['best_val_acc']:.3f}")
    trainer.save(f"models/{mode}_detector.pt")
    return trainer


def test_mode(trainer: BinaryAudioTrainer, mode: str) -> None:
    mel_ext = MelSpectrogramExtractor()
    profile = INSECT_PROFILES[mode]

    logger.info(f"\n--- Testing {mode.upper()} detector ---")
    logger.info(f"{'Test':<30} {'Detected':<10} {'Conf':<10} {'OK?'}")

    sr, dur = 44100, 1.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    results = []

    def make_insect(freq: float, harmonics: list[float]) -> np.ndarray:
        sig = np.zeros(len(t), dtype=np.float64)
        for i, amp in enumerate(harmonics):
            sig += amp * np.sin(2 * np.pi * freq * (i + 1) * t)
        sig *= (0.5 + 0.5 * np.sin(2 * np.pi * 18 * t)) * 0.3
        sig += np.random.normal(0, 0.02, len(t))
        return sig.astype(np.float32)

    # Positive tests (should detect)
    for name, freq in [(f"{mode} peak", profile["freq_mean"]),
                        (f"{mode} low", profile["freq_mean"] - profile["freq_std"]),
                        (f"{mode} high", profile["freq_mean"] + profile["freq_std"])]:
        audio = make_insect(freq, profile["harmonics"])
        mel = mel_ext.extract(audio)
        detected, conf = trainer.predict(mel)
        ok = "OK" if detected else "MISS"
        results.append(detected)
        logger.info(f"{name:<30} {str(detected):<10} {conf:<10.3f} {ok}")

    # Negative tests (should NOT detect)
    for name, freq, harmonics in [
        ("background noise", 0, []),
        ("silence", -1, []),
        ("electric hum 50Hz", 50, [1.0]),
        ("non-insect 2kHz", 2000, [1.0]),
        ("non-insect 4kHz", 4000, [1.0]),
    ]:
        if freq == 0:
            audio = np.random.normal(0, 0.02, len(t)).astype(np.float32)
        elif freq < 0:
            audio = np.random.normal(0, 0.002, len(t)).astype(np.float32)
        else:
            audio = make_insect(freq, harmonics)
        mel = mel_ext.extract(audio)
        detected, conf = trainer.predict(mel)
        ok = "OK" if not detected else "FALSE+"
        results.append(not detected)
        logger.info(f"{name:<30} {str(detected):<10} {conf:<10.3f} {ok}")

    total_ok = sum(results)
    logger.info(f"\n[{mode}] Test score: {total_ok}/{len(results)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OSADS per-mode detectors")
    parser.add_argument("--mode", choices=MODES, help="Train only this mode")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    modes = [args.mode] if args.mode else MODES

    for mode in modes:
        trainer = train_mode(mode, args)
        if args.test:
            test_mode(trainer, mode)


if __name__ == "__main__":
    main()
