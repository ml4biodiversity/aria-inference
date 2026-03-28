"""Audio loading and segmentation for dual-model inference.

Handles loading audio at the correct sample rates for both BirdNET
(48 kHz, used by birdnetlib internally) and PERCH v2 (32 kHz), plus
fixed-length segmentation with optional overlap.
"""

from __future__ import annotations

import numpy as np
import librosa


PERCH_SR = 32_000  # PERCH v2 expected sample rate


def load_audio_for_perch(audio_path: str, sr: int = PERCH_SR) -> np.ndarray:
    """Load an audio file resampled to PERCH's sample rate (32 kHz).

    Args:
        audio_path: Path to audio file.
        sr: Target sample rate.

    Returns:
        1-D float32 array of audio samples.
    """
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    return audio.astype(np.float32)


def compute_segments(
    duration: float,
    segment_length: float = 3.0,
    overlap: float = 0.0,
) -> list[tuple[float, float]]:
    """Compute (start, end) pairs for fixed-length segmentation.

    Args:
        duration: Total audio duration in seconds.
        segment_length: Length of each segment in seconds.
        overlap: Overlap between consecutive segments in seconds.

    Returns:
        List of ``(start_time, end_time)`` tuples.
    """
    step = segment_length - overlap if overlap > 0 else segment_length
    segments: list[tuple[float, float]] = []
    start = 0.0
    while start + segment_length <= duration + 0.1:
        segments.append((start, start + segment_length))
        start += step
    if not segments:
        segments = [(0.0, min(segment_length, duration))]
    return segments


def extract_perch_segment(
    audio: np.ndarray,
    start_time: float,
    end_time: float,
    sr: int = PERCH_SR,
    segment_length: float = 3.0,
) -> np.ndarray:
    """Extract a fixed-length segment from PERCH-rate audio.

    Pads with zeros if the segment is shorter than expected.

    Args:
        audio: Full audio array at *sr* Hz.
        start_time: Segment start in seconds.
        end_time: Segment end in seconds.
        sr: Sample rate of *audio*.
        segment_length: Expected segment length in seconds.

    Returns:
        1-D float32 array of exactly ``int(segment_length * sr)`` samples.
    """
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    expected = int(segment_length * sr)

    segment = audio[start_sample:end_sample]
    if len(segment) < expected:
        segment = np.pad(segment, (0, expected - len(segment)), mode="constant")
    else:
        segment = segment[:expected]
    return segment.astype(np.float32)
