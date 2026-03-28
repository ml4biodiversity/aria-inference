"""BirdNET runtime using ``birdnetlib``.

Unlike the BirdNET-only package (which calls ``birdnet-analyzer`` as a
subprocess and parses CSVs), the full ARIA package needs raw per-segment
probability vectors to feed into the fusion model.  ``birdnetlib``
provides this via its ``Recording`` / ``Analyzer`` API with custom
TFLite classifiers.

Temperature scaling is applied to the multi-label sigmoid probabilities
to reduce confidence saturation.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer


class BirdNETRuntime:
    """Wrapper around birdnetlib for per-segment probability extraction."""

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        min_conf: float = 0.01,
        temperature: float = 1.8,
    ):
        """
        Args:
            model_path: Path to custom ``.tflite`` classifier.
            labels_path: Path to the species labels file.
            min_conf: Minimum confidence for birdnetlib detections.
            temperature: Temperature scaling (T > 1 reduces saturation).
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"BirdNET model not found: {model_path}")
        if not Path(labels_path).exists():
            raise FileNotFoundError(f"BirdNET labels not found: {labels_path}")

        self.min_conf = min_conf
        self.temperature = temperature

        # Load species list
        with open(labels_path, "r", encoding="utf-8") as f:
            self.species = [line.strip() for line in f]
        self.n_species = len(self.species)
        self.species_to_idx = {sp: i for i, sp in enumerate(self.species)}

        # Initialize birdnetlib analyzer with custom classifier
        self.analyzer = Analyzer(
            classifier_model_path=str(model_path),
            classifier_labels_path=str(labels_path),
        )

    # ── temperature scaling ──────────────────────────────────────────

    def apply_temperature_scaling(self, probs: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to multi-label sigmoid probabilities.

        Does **not** normalize to sum-to-1 (BirdNET is multi-label).
        """
        if self.temperature == 1.0:
            return probs
        clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = np.log(clipped / (1 - clipped))
        scaled = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled))

    # ── per-file analysis ────────────────────────────────────────────

    def analyze_file(self, audio_path: str) -> Optional[Recording]:
        """Run birdnetlib analysis on an audio file.

        Returns the ``Recording`` object (with ``.detections``), or
        ``None`` if analysis fails.
        """
        try:
            recording = Recording(
                analyzer=self.analyzer,
                path=str(audio_path),
                min_conf=self.min_conf,
            )
            recording.analyze()
            return recording
        except Exception:
            return None

    # ── probability vector construction ──────────────────────────────

    def build_probability_vector(
        self,
        recording: Optional[Recording],
        start_time: float,
        end_time: float,
    ) -> np.ndarray:
        """Build an n_species-dim probability vector for one time segment.

        Overlapping BirdNET detections are merged by taking the max
        confidence per species.  Temperature scaling is applied.

        Args:
            recording: birdnetlib ``Recording`` with detections, or None.
            start_time: Segment start in seconds.
            end_time: Segment end in seconds.

        Returns:
            ``(n_species,)`` float32 array of scaled probabilities.
        """
        probs = np.zeros(self.n_species, dtype=np.float32)

        if recording is None or not hasattr(recording, "detections"):
            return self.apply_temperature_scaling(probs)

        for det in recording.detections:
            det_start = det.get("start_time", 0)
            det_end = det.get("end_time", 0)

            # Check temporal overlap with this segment
            if det_start < end_time and det_end > start_time:
                scientific = det.get("scientific_name", "").replace("_", " ").strip()
                common = det.get("common_name", "").replace("_", " ").strip()

                # Try multiple name formats to find a match
                candidates = [
                    f"{scientific}_{common}",
                    f"{det.get('scientific_name', '')}_{det.get('common_name', '')}",
                    scientific,
                    common,
                ]
                for name in candidates:
                    if name in self.species_to_idx:
                        idx = self.species_to_idx[name]
                        conf = det.get("confidence", 0.0)
                        probs[idx] = max(probs[idx], conf)
                        break

        return self.apply_temperature_scaling(probs)
