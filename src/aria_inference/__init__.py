"""ARIA full inference package: BirdNET + PERCH v2 + Fusion.

Quick start::

    from aria_inference import run_full_inference
    from pathlib import Path

    run_full_inference(
        input_path=Path("aviary_1/"),
        output_csv=Path("predictions.csv"),
        model_dir=Path("models/"),
        aviary_config=Path("aviary_config.json"),
        aviary_id="aviary_1",
    )
"""

__version__ = "0.1.2"

from .ensemble import run_full_inference
from .model_store import ensure_model_assets

__all__ = ["run_full_inference", "ensure_model_assets"]
