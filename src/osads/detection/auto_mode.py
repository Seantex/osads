"""AUTO-Modus: Alle 3 Insekten-Classifier parallel, automatische Erkennung.

Strategie:
  - Jeder Frame wird durch alle 3 binären Audio-Classifier geschickt
  - FFT-Analyse prüft welches Frequenzband aktiv ist
  - Gewichtetes Voting (Audio-CNN 60% + FFT 40%) bestimmt Insektentyp
  - Exponentielles Glätten über Zeit verhindert schnelle Fehlwechsel
  - Mindest-Konfidenz-Schwelle verhindert Fehlklassifikation bei Stille

Ergebnis: AutoDetection mit erkanntem Typ + Konfidenz je Klasse
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

INSECT_TYPES = ("mosquito", "gnat", "fly")

# Emojis für HUD-Anzeige
INSECT_EMOJI = {
    "mosquito": "🦟",
    "gnat":     "🪲",
    "fly":      "🪰",
    "none":     "—",
}


@dataclass
class AutoDetection:
    """Ergebnis einer AUTO-Modus-Analyse für einen Frame."""

    detected_type: str          # "mosquito" | "gnat" | "fly" | "none"
    confidence: float           # Gesamt-Konfidenz des erkannten Typs
    scores: dict[str, float]    # Rohwerte je Klasse
    smoothed: dict[str, float]  # Zeitlich geglättete Scores
    source: str                 # "audio_cnn" | "fft" | "combined" | "none"


class AutoDetector:
    """Kombiniert alle 3 Binary-Classifier + FFT für automatische Insektenerkennung.

    Läuft jeden Frame durch:
      1. Audio-Akkumulierung bis mind. MIN_AUDIO_SEC Sekunden gesammelt sind
         (Classifier wurden auf 1s-Samples trainiert; kurze Chunks würden
         mit Nullen aufgefüllt → falsche Spektrogramme)
      2. Alle 3 Binary-Audio-CNNs
      3. FFT-Band-Energie-Analyse (jeder Frame, kein Puffer nötig)
      4. Gewichtetes Voting → Gewinner-Klasse
      5. Exponentielles Glätten (EMA) über Zeit
    """

    MIN_AUDIO_SEC = 0.5   # Mindest-Pufferlänge für CNN-Inferenz

    def __init__(
        self,
        audio_classifiers: dict,           # {mode: BinaryAudioTrainer}
        mel_extractor,                     # MelSpectrogramExtractor
        frequency_analyzer,                # FrequencyAnalyzer
        sample_rate: int = 44100,
        cnn_weight: float = 0.65,
        fft_weight: float = 0.35,
        smoothing_alpha: float = 0.20,     # EMA-Glättungsfaktor (0=träge, 1=sofort)
        min_confidence: float = 0.40,      # Mindest-Score zum Auslösen
        min_confirm_frames: int = 4,       # Frames mit gleichem Typ vor Bestätigung
    ) -> None:
        self.classifiers = audio_classifiers
        self.mel_ext = mel_extractor
        self.fft = frequency_analyzer
        self.sample_rate = sample_rate
        self.cnn_weight = cnn_weight
        self.fft_weight = fft_weight
        self.alpha = smoothing_alpha
        self.min_confidence = min_confidence
        self.min_confirm_frames = min_confirm_frames

        # Audio-Akkumulierungspuffer für CNN-Inferenz
        self._min_samples = int(sample_rate * self.MIN_AUDIO_SEC)
        self._audio_buf: list[np.ndarray] = []
        self._buf_len = 0                  # Anzahl gespeicherter Samples

        # Letzte CNN-Scores (werden bei jedem Puffer-Flush aktualisiert)
        self._last_cnn_scores: dict[str, float] = {t: 0.0 for t in INSECT_TYPES}

        # Zeitlich geglättete Scores (EMA)
        self._smoothed: dict[str, float] = {t: 0.0 for t in INSECT_TYPES}
        # Bestätigungs-Zähler
        self._candidate: str = "none"
        self._candidate_frames: int = 0
        self._confirmed: str = "none"

    def analyze(self, audio: np.ndarray) -> AutoDetection:
        """Analysiert einen Audio-Chunk und gibt die Auto-Erkennung zurück.

        Args:
            audio: 1D float32 numpy Array (beliebige Länge, typisch 46ms-Chunks).

        Returns:
            AutoDetection mit erkanntem Typ und Konfidenzwerten.
        """
        cnn_scores: dict[str, float] = dict(self._last_cnn_scores)  # letzter bekannter Wert
        fft_scores: dict[str, float] = {t: 0.0 for t in INSECT_TYPES}
        active_sources: list[str] = []

        # ── 1. Audio puffern, CNN nur wenn genug Daten vorhanden ────────
        self._audio_buf.append(audio)
        self._buf_len += len(audio)

        if self._buf_len >= self._min_samples and self.mel_ext is not None and self.classifiers:
            # Puffer leeren, 1 Sekunde ausschneiden
            full = np.concatenate(self._audio_buf)
            self._audio_buf.clear()
            self._buf_len = 0
            try:
                mel = self.mel_ext.extract(full)  # trimmt/padded intern auf 1s
                for insect_type in INSECT_TYPES:
                    clf = self.classifiers.get(insect_type)
                    if clf is not None:
                        _, conf = clf.predict(mel)
                        cnn_scores[insect_type] = float(conf)
                self._last_cnn_scores = dict(cnn_scores)
                active_sources.append("audio_cnn")
            except Exception as e:
                logger.debug(f"CNN inference error: {e}")

        # ── 2. FFT-Band-Energie ──────────────────────────────────────────
        if self.fft is not None:
            try:
                fft_result = self.fft.analyze(audio)
                if fft_result.detected and fft_result.insect_type in fft_scores:
                    fft_scores[fft_result.insect_type] = fft_result.confidence
                active_sources.append("fft")
            except Exception as e:
                logger.debug(f"FFT analysis error: {e}")

        # ── 3. Gewichtetes Voting ────────────────────────────────────────
        combined: dict[str, float] = {}
        if active_sources:
            w_cnn = self.cnn_weight if "audio_cnn" in active_sources else 0.0
            w_fft = self.fft_weight if "fft" in active_sources else 0.0
            total_w = w_cnn + w_fft
            if total_w > 0:
                for t in INSECT_TYPES:
                    combined[t] = (
                        cnn_scores[t] * w_cnn + fft_scores[t] * w_fft
                    ) / total_w
            else:
                combined = {t: 0.0 for t in INSECT_TYPES}
        else:
            combined = {t: 0.0 for t in INSECT_TYPES}

        # ── 4. Exponentielles Glätten (EMA) ─────────────────────────────
        for t in INSECT_TYPES:
            self._smoothed[t] = (
                self.alpha * combined[t] + (1.0 - self.alpha) * self._smoothed[t]
            )

        # ── 5. Gewinner bestimmen ────────────────────────────────────────
        best_type = max(self._smoothed, key=self._smoothed.__getitem__)
        best_score = self._smoothed[best_type]

        # Soft-Mehrheit: Gewinner braucht mind. 1.5× den Zweitplatzierten
        scores_sorted = sorted(self._smoothed.values(), reverse=True)
        runner_up = scores_sorted[1] if len(scores_sorted) > 1 else 0.0
        majority = best_score >= runner_up * 1.5 if runner_up > 0 else True

        # Mindest-Konfidenz + Mehrheitscheck
        if best_score >= self.min_confidence and majority:
            candidate = best_type
        else:
            candidate = "none"

        # ── 6. Bestätigungs-Hysterese ────────────────────────────────────
        if candidate == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate = candidate
            self._candidate_frames = 1

        if self._candidate_frames >= self.min_confirm_frames:
            self._confirmed = self._candidate

        source = (
            "combined" if len(active_sources) > 1
            else active_sources[0] if active_sources
            else "none"
        )

        return AutoDetection(
            detected_type=self._confirmed,
            confidence=best_score if self._confirmed != "none" else 0.0,
            scores=combined,
            smoothed=dict(self._smoothed),
            source=source,
        )

    def reset(self) -> None:
        """Setzt den internen Zustand zurück (z.B. bei Modewechsel)."""
        self._audio_buf.clear()
        self._buf_len = 0
        self._last_cnn_scores = {t: 0.0 for t in INSECT_TYPES}
        self._smoothed = {t: 0.0 for t in INSECT_TYPES}
        self._candidate = "none"
        self._candidate_frames = 0
        self._confirmed = "none"
