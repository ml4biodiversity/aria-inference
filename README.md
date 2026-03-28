# aria-inference

Full ARIA inference package: **BirdNET + PERCH v2 + Fusion** for bird species detection in zoo aviaries.

This is the complete baseline system for the [BioDCASE 2026 Challenge](https://biodcase.github.io/). For BirdNET-only inference, see [aria-inference-birdnet](https://pypi.org/project/aria-inference-birdnet/).

## Installation

Requires **Python 3.11 or 3.12** and a working installation of FFmpeg.

```bash
pip install aria-inference
```

This automatically installs BirdNET (via [birdnetlib](https://pypi.org/project/birdnetlib/)), [PERCH v2](https://github.com/google-research/perch-hoplite) (via perch-hoplite), TensorFlow, and PyTorch.

**Note**: PERCH v2 auto-downloads its model from Kaggle on first use (~500 MB). This happens once and is cached locally.

## Quick start

### 1. Download model files

```bash
aria-inference download-models --dir ./models
```

This fetches from GitHub Releases:
- `ZooCustom_v1.tflite` – custom BirdNET classifier
- `ZooCustom_v1_Labels.txt` – species labels
- `fusion_model_perchv2_best.pth` – trained fusion model
- `architecture_info_public.json` – model architecture config
- `perch_v2_zoo_species_mapping.json` – BirdNET-to-PERCH species mapping

### 2. Run detection on an aviary

```bash
aria-inference detect \
  --input aviary_1/ \
  --output predictions_aviary_1.csv \
  --model-dir ./models \
  --aviary-config aviary_config.json \
  --aviary aviary_1
```

### 3. List available aviaries

```bash
aria-inference list-aviaries --aviary-config aviary_config.json
```

### 4. Python API

```python
from pathlib import Path
from aria_inference import run_full_inference

run_full_inference(
    input_path=Path("aviary_1/"),
    output_csv=Path("predictions_aviary_1.csv"),
    model_dir=Path("models/"),
    aviary_config=Path("aviary_config.json"),
    aviary_id="aviary_1",
)
```

## How it works

ARIA uses a three-model ensemble for robust species detection:

```
Audio (3-second segments)
  │
  ├─ BirdNET (custom TFLite) → 87-dim probability vector
  │
  ├─ PERCH v2 (foundation model) → 87-dim constrained logits
  │
  └─ Fusion model (PyTorch) → combines BirdNET + PERCH → predictions
      │
      └─ Ensemble voting merges all three → final species + confidence
```

Two inference modes are available:

- **Ensemble voting** (default, `--mode voting`): all three models run on every segment; predictions are merged using weighted voting with a consensus boost when 2+ models agree.
- **Tier-based fallback** (`--mode tiers`): BirdNET runs first; if confidence < threshold, fusion runs; if fusion is also low, PERCH provides a fallback.

## CLI reference

### `detect`

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Audio file or directory |
| `--output` | *(required)* | Output CSV path |
| `--model-dir` | *(required)* | Directory with all model files |
| `--aviary-config` | `None` | Path to `aviary_config.json` |
| `--aviary` | `None` | Aviary ID (e.g. `aviary_1`) |
| `--allowed-species-file` | `None` | Flat species whitelist (alternative) |
| `--temperature` | `1.8` | BirdNET confidence scaling |
| `--tier1-threshold` | `0.85` | BirdNET-only acceptance threshold |
| `--min-confidence` | `0.30` | Global minimum confidence |
| `--perch-min-confidence` | `0.10` | PERCH fallback minimum |
| `--top-k` | `3` | Max species per segment |
| `--segment-length` | `3.0` | Segment length in seconds |
| `--overlap` | `0.0` | Segment overlap in seconds |
| `--mode` | `voting` | `voting` or `tiers` |

### `download-models`

| Flag | Default | Description |
|------|---------|-------------|
| `--dir` | *(required)* | Local directory for model files |

### `list-aviaries`

| Flag | Default | Description |
|------|---------|-------------|
| `--aviary-config` | *(required)* | Path to `aviary_config.json` |

## Output format

```csv
File name,Start (s),End (s),Species,Confidence,Method
recording_001.wav,0.00,3.00,Greater Flamingo,0.8734,Ensemble Voting
recording_001.wav,3.00,6.00,Hadada Ibis,0.7521,Ensemble Voting
```

## Notes

- This package is **inference-only**. It does not include training, data fetching, or segmentation.
- Model binaries are hosted as GitHub Release assets to keep the pip package small.
- PERCH v2 auto-downloads from Kaggle on first use and is cached locally.
- GPU is optional but recommended for faster PERCH inference.
- Python 3.10 is not supported due to dependency constraints.

## License

Apache-2.0
