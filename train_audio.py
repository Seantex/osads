#!/usr/bin/env python3
"""Train the insect audio classifier.

Usage:
    python train_audio.py                    # Quick training (1000 samples/class)
    python train_audio.py --full             # Full training (10000 samples/class)
    python train_audio.py --epochs 200       # Custom epochs
    python train_audio.py --test             # Train + run test predictions

The model learns to distinguish:
  - Mosquito wing beat (~550 Hz)
  - Gnat/Mücke wing beat (~400 Hz)
  - Fly/Fliege wing beat (~190 Hz)
  - Background noise (no insect)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

# Add src to path
sys.path.insert(0, "src")

from osads.training.audio_model import AudioTrainer, CLASSES, InsectAudioCNN
from osads.detection.acoustic import FrequencyAnalyzer, MelSpectrogramExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def train(args: argparse.Namespace) -> None:
    """Run training pipeline."""
    logger.info("=" * 60)
    logger.info("OSADS Audio Classifier Training")
    logger.info("=" * 60)

    samples = args.samples
    logger.info(f"Samples per class: {samples}")
    logger.info(f"Total samples: {samples * len(CLASSES)}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Classes: {CLASSES}")
    logger.info("")

    # Train
    trainer = AudioTrainer(learning_rate=args.lr)
    t_start = time.time()

    results = trainer.train(
        num_samples_per_class=samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    duration = time.time() - t_start
    logger.info(f"\nTraining complete in {duration:.1f}s")
    logger.info(f"Best validation accuracy: {results['best_val_acc']:.3f}")

    # Save model
    model_path = "models/insect_audio_classifier.pt"
    trainer.save_model(model_path)

    # Test predictions
    if args.test:
        logger.info("\n" + "=" * 60)
        logger.info("Running test predictions...")
        logger.info("=" * 60)
        _run_test_predictions(trainer)


def _run_test_predictions(trainer: AudioTrainer) -> None:
    """Run test predictions on synthetic audio."""
    mel_extractor = MelSpectrogramExtractor()

    # Test cases with frequencies matching the training profiles
    test_cases = [
        ("Mosquito 500Hz", 500.0, "mosquito"),
        ("Mosquito 450Hz", 450.0, "mosquito"),
        ("Mosquito 550Hz", 550.0, "mosquito"),
        ("Gnat 700Hz", 700.0, "gnat"),
        ("Gnat 720Hz", 720.0, "gnat"),
        ("Fly 200Hz", 200.0, "fly"),
        ("Fly 180Hz", 180.0, "fly"),
        ("Background", 0.0, "background"),
        ("Non-insect 2kHz", 2000.0, "background"),
        ("Non-insect 5kHz", 5000.0, "background"),
    ]

    logger.info(f"\n{'Test Case':<25} {'Predicted':<12} {'Confidence':<12} {'Correct?'}")
    logger.info("-" * 65)

    correct = 0
    total = len(test_cases)

    # Harmonic profiles matching the training data
    harmonic_profiles = {
        "mosquito": [1.0, 0.35, 0.15, 0.05],
        "gnat": [1.0, 0.2, 0.05],
        "fly": [1.0, 0.4, 0.2, 0.1, 0.05],
    }

    for name, freq, expected in test_cases:
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        if freq == 0:
            audio = np.random.normal(0, 0.02, len(t)).astype(np.float32)
        elif freq > 1000:
            audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            audio += np.random.normal(0, 0.02, len(t)).astype(np.float32)
        else:
            # Generate with correct harmonic profile for expected type
            harmonics = harmonic_profiles.get(expected, [1.0, 0.3, 0.1])
            signal = np.zeros(len(t), dtype=np.float64)
            for i, amp in enumerate(harmonics):
                signal += amp * np.sin(2 * np.pi * freq * (i + 1) * t)
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 18 * t)
            signal *= envelope * 0.3
            signal += np.random.normal(0, 0.02, len(t))
            audio = signal.astype(np.float32)

        mel = mel_extractor.extract(audio)
        probs = trainer.predict(mel)
        predicted = max(probs, key=probs.get)
        conf = probs[predicted]
        is_correct = predicted == expected
        if is_correct:
            correct += 1

        mark = "OK" if is_correct else "MISS"
        logger.info(f"{name:<25} {predicted:<12} {conf:<12.3f} {mark}")

    logger.info(f"\nTest accuracy: {correct}/{total} ({correct/total*100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OSADS audio classifier")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Samples per class")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--full", action="store_true",
                        help="Full training (10000 samples/class, 200 epochs)")
    parser.add_argument("--test", action="store_true",
                        help="Run test predictions after training")
    args = parser.parse_args()

    if args.full:
        args.samples = 10000
        args.epochs = 200

    train(args)


if __name__ == "__main__":
    main()
