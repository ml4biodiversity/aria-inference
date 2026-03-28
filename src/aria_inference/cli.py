"""Command-line interface for ``aria-inference``.

Usage::

    aria-inference download-models --dir ./models
    aria-inference detect \\
        --input aviary_1/ \\
        --output predictions.csv \\
        --model-dir ./models \\
        --aviary-config aviary_config.json \\
        --aviary aviary_1
"""

from __future__ import annotations

from pathlib import Path

import click


@click.group()
@click.version_option(package_name="aria-inference")
def main():
    """ARIA full inference: BirdNET + PERCH + Fusion."""


# ── download-models ──────────────────────────────────────────────────

@main.command("download-models")
@click.option(
    "--dir", "model_dir",
    type=click.Path(path_type=Path), required=True,
    help="Local directory to store model files.",
)
def download_models(model_dir: Path):
    """Download all model assets from GitHub Releases."""
    from .model_store import ensure_model_assets
    ensure_model_assets(model_dir, download=True)
    click.echo(f"ARIA model assets ready in {model_dir}")
    click.echo(
        "Note: PERCH v2 will auto-download from Kaggle on first inference."
    )


# ── list-aviaries ────────────────────────────────────────────────────

@main.command("list-aviaries")
@click.option(
    "--aviary-config",
    type=click.Path(exists=True, path_type=Path), required=True,
    help="Path to aviary_config.json.",
)
def list_aviaries_cmd(aviary_config: Path):
    """List available aviary IDs from the config."""
    import json
    from .species_filter import list_aviaries, load_aviary_species

    ids = list_aviaries(aviary_config)
    with open(aviary_config, "r") as f:
        n_wild = len(json.load(f).get("wild_birds", []))

    click.echo(f"Available aviaries ({len(ids)}):\n")
    for aid in ids:
        species = load_aviary_species(aviary_config, aid)
        n_aviary = len(species) - n_wild
        click.echo(f"  {aid}  ({n_aviary} aviary species + {n_wild} wild birds)")


# ── detect ───────────────────────────────────────────────────────────

@main.command("detect")
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path), required=True,
              help="Audio file or directory.")
@click.option("--output", "output_csv", type=click.Path(path_type=Path), required=True,
              help="Output CSV path.")
@click.option("--model-dir", type=click.Path(exists=True, path_type=Path), required=True,
              help="Directory with all model files.")
@click.option("--aviary-config", type=click.Path(exists=True, path_type=Path), default=None,
              help="Path to aviary_config.json.")
@click.option("--aviary", "aviary_id", type=str, default=None,
              help="Aviary ID (e.g. aviary_1). Requires --aviary-config.")
@click.option("--allowed-species-file", type=click.Path(exists=True, path_type=Path), default=None,
              help="Flat species whitelist (alternative to aviary config).")
@click.option("--temperature", type=float, default=1.8, show_default=True,
              help="BirdNET confidence scaling (T>1 reduces saturation).")
@click.option("--tier1-threshold", type=float, default=0.85, show_default=True,
              help="Confidence for BirdNET-only acceptance (tier mode).")
@click.option("--min-confidence", type=float, default=0.30, show_default=True,
              help="Global minimum confidence.")
@click.option("--perch-min-confidence", type=float, default=0.10, show_default=True,
              help="PERCH fallback minimum confidence.")
@click.option("--top-k", type=int, default=3, show_default=True,
              help="Max species per segment.")
@click.option("--segment-length", type=float, default=3.0, show_default=True,
              help="Audio segment length in seconds.")
@click.option("--overlap", type=float, default=0.0, show_default=True,
              help="Segment overlap in seconds.")
@click.option("--mode", type=click.Choice(["voting", "tiers"]), default="voting", show_default=True,
              help="Inference mode: ensemble voting or tier-based fallback.")
def detect(
    input_path: Path,
    output_csv: Path,
    model_dir: Path,
    aviary_config: Path | None,
    aviary_id: str | None,
    allowed_species_file: Path | None,
    temperature: float,
    tier1_threshold: float,
    min_confidence: float,
    perch_min_confidence: float,
    top_k: int,
    segment_length: float,
    overlap: float,
    mode: str,
):
    """Run full ARIA inference (BirdNET + PERCH + Fusion)."""
    from .ensemble import run_full_inference

    run_full_inference(
        input_path=input_path,
        output_csv=output_csv,
        model_dir=model_dir,
        aviary_config=aviary_config,
        aviary_id=aviary_id,
        allowed_species_file=allowed_species_file,
        temperature=temperature,
        tier1_threshold=tier1_threshold,
        min_confidence=min_confidence,
        perch_min_confidence=perch_min_confidence,
        top_k=top_k,
        segment_length=segment_length,
        overlap=overlap,
        use_ensemble_voting=(mode == "voting"),
    )
