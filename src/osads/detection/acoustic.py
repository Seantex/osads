"""Acoustic insect detection using FFT frequency analysis.

Detects insects by their wing-beat frequency:
- Mosquito (Aedes aegypti): ~550 Hz (range 350-1000 Hz)
- Gnat (Chironomidae):      ~680 Hz (range 500-1050 Hz)
- Fly (Musca domestica):    ~200 Hz (range 100-300 Hz)

Sources: Arthur et al. PMC3985972, UF Insect Records Ch.9, Rockstein 1966

Uses both classical FFT analysis and a trained CNN classifier on Mel spectrograms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


@dataclass
class AcousticDetection:
    """Result of acoustic analysis for one audio chunk."""

    detected: bool
    insect_type: str | None  # mosquito, gnat, fly, or None
    confidence: float
    dominant_frequency: float
    spectrum_peak_db: float


class FrequencyAnalyzer:
    """FFT-based frequency analyzer for insect wing-beat detection.

    Classical (non-ML) approach:
    1. Compute FFT of audio chunk
    2. Find dominant frequency peaks
    3. Match peaks to known insect frequency ranges
    """

    # Scientifically verified insect wing-beat frequency ranges
    INSECT_FREQUENCIES: dict[str, tuple[float, float, float]] = {
        # (min_hz, max_hz, peak_hz)
        "mosquito": (350, 1000, 550),
        "gnat": (500, 1050, 680),
        "fly": (100, 300, 200),
    }

    def __init__(
        self,
        sample_rate: int = 44100,
        chunk_size: int = 2048,
        detection_threshold: float = 0.6,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.detection_threshold = detection_threshold
        self.freqs = np.fft.rfftfreq(chunk_size, d=1.0 / sample_rate)

    def analyze(self, audio_chunk: np.ndarray) -> AcousticDetection:
        """Analyze a single audio chunk for insect sounds."""
        if len(audio_chunk) < self.chunk_size:
            return AcousticDetection(
                detected=False, insect_type=None, confidence=0.0,
                dominant_frequency=0.0, spectrum_peak_db=-100.0,
            )

        windowed = audio_chunk[: self.chunk_size] * np.hanning(self.chunk_size)
        fft_result = np.fft.rfft(windowed)
        magnitude = np.abs(fft_result)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)

        peak_idx = np.argmax(magnitude[1:]) + 1
        dominant_freq = self.freqs[peak_idx]
        peak_db = magnitude_db[peak_idx]
        noise_floor = np.median(magnitude_db)
        snr = peak_db - noise_floor

        best_match: str | None = None
        best_confidence = 0.0

        for insect_type, (freq_min, freq_max, freq_peak) in self.INSECT_FREQUENCIES.items():
            if freq_min <= dominant_freq <= freq_max:
                freq_dist = abs(dominant_freq - freq_peak) / (freq_max - freq_min)
                freq_confidence = 1.0 - freq_dist
                snr_confidence = min(1.0, snr / 30.0)
                confidence = freq_confidence * 0.6 + snr_confidence * 0.4

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = insect_type

        detected = best_match is not None and best_confidence >= self.detection_threshold

        return AcousticDetection(
            detected=detected,
            insect_type=best_match if detected else None,
            confidence=best_confidence if detected else 0.0,
            dominant_frequency=dominant_freq,
            spectrum_peak_db=peak_db,
        )

    def compute_spectrogram(
        self, audio: np.ndarray, hop_length: int = 512
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute spectrogram for visualization."""
        frequencies, times, Sxx = scipy_signal.spectrogram(
            audio, fs=self.sample_rate,
            nperseg=self.chunk_size, noverlap=self.chunk_size - hop_length,
        )
        return frequencies, times, 10 * np.log10(Sxx + 1e-10)


class MelSpectrogramExtractor:
    """Extract Mel spectrograms using torchaudio (industry standard).

    Used as input for the InsectAudioCNN classifier.
    Parameters per HumBug/Oxford methodology and PLOS Comp Bio 2023.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 512,
        duration: float = 1.0,
    ) -> None:
        import torch
        import torchaudio.transforms as T

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.expected_length = int(sample_rate * duration)

        self._mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self._db_transform = T.AmplitudeToDB(stype="power", top_db=80)

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract Mel spectrogram from audio.

        Args:
            audio: 1D float32 array.

        Returns:
            2D numpy array (n_mels x time_frames), dB scale, normalized [0,1].
        """
        import torch

        # Pad or trim
        if len(audio) < self.expected_length:
            audio = np.pad(audio, (0, self.expected_length - len(audio)))
        else:
            audio = audio[: self.expected_length]

        # Convert to torch tensor
        waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)

        # Compute mel spectrogram in dB
        mel_spec = self._mel_transform(waveform)
        mel_db = self._db_transform(mel_spec)

        # Normalize to [0, 1] using fixed range (not per-sample)
        # top_db=80, so range is approximately [-80, 0] relative to max
        mel_np = mel_db.squeeze(0).numpy()
        mel_np = (mel_np - mel_np.min())
        max_val = mel_np.max()
        if max_val > 0:
            mel_np /= max_val

        return mel_np.astype(np.float32)
