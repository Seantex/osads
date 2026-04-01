"""Tests for detection modules."""

import numpy as np
import pytest

from osads.detection.acoustic import FrequencyAnalyzer, MelSpectrogramExtractor
from osads.detection.visual import Detection, MotionDetector, VisualDetectionPipeline
from osads.detection.fusion import SensorFusion, FusedDetection
from osads.detection.acoustic import AcousticDetection


class TestFrequencyAnalyzer:
    """Tests for FFT-based acoustic detection."""

    def setup_method(self) -> None:
        self.analyzer = FrequencyAnalyzer(
            sample_rate=44100,
            chunk_size=2048,
            detection_threshold=0.4,
        )

    def _generate_tone(self, freq: float, duration: float = 0.1, amplitude: float = 0.3) -> np.ndarray:
        """Generate a pure tone at given frequency."""
        sr = 44100
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        signal = amplitude * np.sin(2 * np.pi * freq * t)
        return signal.astype(np.float32)

    def test_detect_mosquito_frequency(self) -> None:
        """Mosquito wing beat ~550Hz should be detected."""
        audio = self._generate_tone(550.0)
        result = self.analyzer.analyze(audio)
        assert result.detected
        assert result.insect_type == "mosquito"
        assert 500 <= result.dominant_frequency <= 600

    def test_detect_gnat_frequency(self) -> None:
        """Gnat/Midge wing beat ~680Hz should be detected (Chironomidae 650-700Hz)."""
        audio = self._generate_tone(680.0)
        result = self.analyzer.analyze(audio)
        assert result.detected
        assert result.insect_type == "gnat"

    def test_detect_fly_frequency(self) -> None:
        """House fly wing beat ~200Hz should be detected (M. domestica 190-250Hz)."""
        audio = self._generate_tone(200.0)
        result = self.analyzer.analyze(audio)
        assert result.detected
        assert result.insect_type == "fly"

    def test_no_detection_on_silence(self) -> None:
        """Silence should not trigger detection."""
        audio = np.zeros(4096, dtype=np.float32)
        result = self.analyzer.analyze(audio)
        assert not result.detected

    def test_no_detection_on_noise(self) -> None:
        """White noise should not trigger detection."""
        audio = np.random.normal(0, 0.01, 4096).astype(np.float32)
        result = self.analyzer.analyze(audio)
        # May or may not detect, but confidence should be low
        if result.detected:
            assert result.confidence < 0.8

    def test_no_detection_on_out_of_range(self) -> None:
        """Frequencies outside insect range should not detect."""
        audio = self._generate_tone(5000.0)  # 5kHz - not an insect
        result = self.analyzer.analyze(audio)
        assert not result.detected

    def test_short_audio_returns_no_detection(self) -> None:
        """Audio shorter than chunk_size should return no detection."""
        audio = np.zeros(100, dtype=np.float32)
        result = self.analyzer.analyze(audio)
        assert not result.detected


class TestMelSpectrogram:
    """Tests for Mel spectrogram extraction."""

    def test_output_shape(self) -> None:
        extractor = MelSpectrogramExtractor(n_mels=64, duration=1.0)
        audio = np.random.randn(44100).astype(np.float32)
        mel = extractor.extract(audio)
        assert mel.shape[0] == 64  # n_mels
        assert mel.shape[1] > 0   # time frames
        assert mel.dtype == np.float32

    def test_normalized_range(self) -> None:
        extractor = MelSpectrogramExtractor(n_mels=64)
        audio = np.random.randn(44100).astype(np.float32) * 0.5
        mel = extractor.extract(audio)
        assert mel.min() >= 0.0
        assert mel.max() <= 1.0

    def test_handles_short_audio(self) -> None:
        extractor = MelSpectrogramExtractor(n_mels=64, duration=1.0)
        audio = np.random.randn(1000).astype(np.float32)  # Much shorter than 1s
        mel = extractor.extract(audio)
        assert mel.shape[0] == 64


class TestMotionDetector:
    """Tests for background subtraction motion detection."""

    def test_detects_moving_object(self) -> None:
        detector = MotionDetector(min_area=5, max_area=500)
        # Feed background frames first
        bg = np.zeros((100, 100, 3), dtype=np.uint8)
        for _ in range(30):
            detector.detect(bg)

        # Add a bright moving dot
        frame = bg.copy()
        cv2.circle(frame, (50, 50), 5, (255, 255, 255), -1)
        dets = detector.detect(frame)
        assert len(dets) > 0

    def test_no_detection_on_static(self) -> None:
        detector = MotionDetector()
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # After many identical frames, no motion
        for _ in range(50):
            dets = detector.detect(frame)
        assert len(dets) == 0


class TestSensorFusion:
    """Tests for audio + visual fusion."""

    def test_agreement_boosts_confidence(self) -> None:
        fusion = SensorFusion(visual_weight=0.6, acoustic_weight=0.4)
        vis = [Detection(x=100, y=100, w=10, h=10, confidence=0.8, class_name="mosquito")]
        aco = AcousticDetection(detected=True, insect_type="mosquito", confidence=0.9,
                                 dominant_frequency=550, spectrum_peak_db=-10)
        fused = fusion.fuse(vis, aco)
        assert len(fused) == 1
        # Agreement should give higher confidence than visual alone
        assert fused[0].fused_confidence > 0.6 * 0.8

    def test_no_input_returns_empty(self) -> None:
        fusion = SensorFusion()
        result = fusion.fuse([], None)
        assert len(result) == 0


# Need cv2 for MotionDetector test
import cv2
