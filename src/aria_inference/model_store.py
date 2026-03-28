"""Verify (and optionally download) ARIA runtime model assets.

Model binaries are hosted as GitHub Release attachments.  The
``download-models`` CLI command calls :func:`ensure_model_assets`
to fetch any missing files.

PERCH v2 is **not** included here — it auto-downloads via
``perch-hoplite`` on first use.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

_RELEASE_BASE = (
    "https://github.com/ml4biodiversity/aria-inference"
    "/releases/download"
)
_RELEASE_TAG = "v0.1.0"

# Required assets — inference fails without these.
_ASSETS: dict[str, str] = {
    "ZooCustom_v1.tflite": f"{_RELEASE_BASE}/{_RELEASE_TAG}/ZooCustom_v1.tflite",
    "ZooCustom_v1_Labels.txt": f"{_RELEASE_BASE}/{_RELEASE_TAG}/ZooCustom_v1_Labels.txt",
    "fusion_model_perchv2_best.pth": f"{_RELEASE_BASE}/{_RELEASE_TAG}/fusion_model_perchv2_best.pth",
    "architecture_info_public.json": f"{_RELEASE_BASE}/{_RELEASE_TAG}/architecture_info_public.json",
    "perch_v2_zoo_species_mapping.json": f"{_RELEASE_BASE}/{_RELEASE_TAG}/perch_v2_zoo_species_mapping.json",
}


def ensure_model_assets(
    model_dir: Path,
    *,
    download: bool = True,
) -> None:
    """Ensure all required model files exist in *model_dir*.

    When *download* is ``True`` (the default), missing files are fetched
    from the GitHub Release.  Otherwise a :class:`FileNotFoundError` is
    raised listing the missing files.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    for filename, url in _ASSETS.items():
        dest = model_dir / filename
        if dest.exists():
            continue
        if download:
            print(f"Downloading {filename} ...")
            try:
                urllib.request.urlretrieve(url, dest)
                print(f"  → {dest}")
            except urllib.error.HTTPError as e:
                print(f"  ✗ Failed to download {filename}: {e}")
                missing.append(filename)
        else:
            missing.append(filename)

    if missing:
        raise FileNotFoundError(
            f"Missing ARIA runtime assets in {model_dir}: "
            + ", ".join(missing)
        )
