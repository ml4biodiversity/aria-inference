"""Tests for aria-inference package (no model files required)."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from aria_inference.audio import compute_segments, extract_perch_segment, PERCH_SR
from aria_inference.species_filter import (
    _parse_species_line,
    filter_predictions,
    list_aviaries,
    load_allowed_species,
    load_aviary_species,
)
from aria_inference.fusion_model import DualEmbeddingClassifier


# ── audio segmentation ───────────────────────────────────────────────

def test_compute_segments_basic():
    segs = compute_segments(duration=9.0, segment_length=3.0, overlap=0.0)
    assert segs == [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0)]


def test_compute_segments_with_overlap():
    segs = compute_segments(duration=6.0, segment_length=3.0, overlap=1.5)
    assert len(segs) >= 3


def test_compute_segments_short_audio():
    segs = compute_segments(duration=1.0, segment_length=3.0)
    assert len(segs) == 1
    assert segs[0] == (0.0, 1.0)


def test_extract_perch_segment_padding():
    audio = np.ones(PERCH_SR * 2, dtype=np.float32)  # 2 seconds
    seg = extract_perch_segment(audio, 0.0, 3.0, sr=PERCH_SR, segment_length=3.0)
    assert len(seg) == PERCH_SR * 3
    # First 2 seconds are 1.0, last second is 0.0 (padded)
    assert seg[0] == 1.0
    assert seg[-1] == 0.0


# ── species filter ───────────────────────────────────────────────────

def test_parse_birdnet_label():
    common, sci = _parse_species_line("Phoenicopterus roseus_Greater Flamingo")
    assert common == "Greater Flamingo"
    assert sci == "Phoenicopterus roseus"


def test_parse_bare_name():
    common, sci = _parse_species_line("Greater Flamingo")
    assert common == "Greater Flamingo"
    assert sci == ""


def test_load_allowed_species_none():
    assert load_allowed_species(None) is None


def test_load_allowed_species_file(tmp_path: Path):
    p = tmp_path / "sp.txt"
    p.write_text("Phoenicopterus roseus_Greater Flamingo\nHadada Ibis\n")
    allowed = load_allowed_species(p)
    assert allowed == {"Greater Flamingo", "Hadada Ibis"}


def _write_aviary_config(tmp_path: Path) -> Path:
    cfg = {
        "wild_birds": ["House Sparrow"],
        "aviaries": {
            "aviary_1": {"species": ["Greater Flamingo", "Hadada Ibis"]},
            "aviary_2": {"species": ["Red-billed Quelea"]},
        },
    }
    p = tmp_path / "aviary_config.json"
    p.write_text(json.dumps(cfg))
    return p


def test_load_aviary_species(tmp_path: Path):
    cfg = _write_aviary_config(tmp_path)
    allowed = load_aviary_species(cfg, "aviary_1")
    assert allowed == {"Greater Flamingo", "Hadada Ibis", "House Sparrow"}


def test_load_aviary_unknown(tmp_path: Path):
    cfg = _write_aviary_config(tmp_path)
    with pytest.raises(KeyError):
        load_aviary_species(cfg, "aviary_99")


def test_list_aviaries(tmp_path: Path):
    cfg = _write_aviary_config(tmp_path)
    assert list_aviaries(cfg) == ["aviary_1", "aviary_2"]


def test_filter_predictions():
    rows = [
        {"species": "Greater Flamingo", "confidence": 0.9},
        {"species": "House Sparrow", "confidence": 0.7},
    ]
    out = filter_predictions(rows, {"Greater Flamingo"})
    assert len(out) == 1
    assert out[0]["species"] == "Greater Flamingo"


def test_filter_predictions_none():
    rows = [{"species": "A", "confidence": 0.5}]
    assert filter_predictions(rows, None) == rows


# ── fusion model (CPU, no weights) ───────────────────────────────────

def test_fusion_model_forward():
    model = DualEmbeddingClassifier(
        n_species=92, birdnet_dim=87, perch_dim=87,
        hidden_dims=[256, 128, 64],
    )
    model.eval()

    import torch
    x = torch.randn(1, 174)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 92)


def test_fusion_model_dimensions():
    model = DualEmbeddingClassifier(
        n_species=10, birdnet_dim=5, perch_dim=5,
        hidden_dims=[32, 16],
    )
    model.eval()

    import torch
    x = torch.randn(4, 10)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (4, 10)


# ── CLI ──────────────────────────────────────────────────────────────

def test_cli_help():
    from click.testing import CliRunner
    from aria_inference.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "ARIA" in result.output


def test_cli_detect_help():
    from click.testing import CliRunner
    from aria_inference.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["detect", "--help"])
    assert result.exit_code == 0
    assert "--aviary-config" in result.output
    assert "--mode" in result.output
    assert "--tier1-threshold" in result.output


def test_cli_download_help():
    from click.testing import CliRunner
    from aria_inference.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["download-models", "--help"])
    assert result.exit_code == 0


def test_cli_list_aviaries_help():
    from click.testing import CliRunner
    from aria_inference.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["list-aviaries", "--help"])
    assert result.exit_code == 0
