"""Species filtering for ARIA inference.

Two filtering modes are supported:

1. **Aviary config** (recommended for BioDCASE): an ``aviary_config.json``
   maps each aviary ID to the species that actually live there.  Wild
   birds common to all aviaries are listed once and merged automatically.
   This is the anonymized replacement for the internal zoo configuration.

2. **Flat allowed-species file**: a newline-delimited text file listing
   every species to keep.  Supports both BirdNET label format
   (``Scientific_Common``) and bare common names.

In both cases, filtering happens against BirdNET's ``Common name``
CSV column.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .utils import find_col_index


# ── line parser ──────────────────────────────────────────────────────

def _parse_species_line(line: str) -> tuple[str, str]:
    """Parse a single line from an allowed-species file.

    Handles both ``Scientific name_Common name`` (BirdNET label format)
    and bare ``Common name``.

    Returns:
        ``(common_name, scientific_name)`` where *scientific_name* may
        be an empty string.
    """
    line = line.strip()
    if "_" in line:
        scientific, common = line.split("_", 1)
        return common.strip(), scientific.strip()
    return line, ""


# ── aviary config ────────────────────────────────────────────────────

def load_aviary_species(
    config_path: Path,
    aviary_id: str,
) -> set[str]:
    """Load allowed species for a single aviary from the config JSON.

    The returned set is the **union** of the aviary's own species and
    the ``wild_birds`` list (species that may appear in any aviary).

    Args:
        config_path: Path to ``aviary_config.json``.
        aviary_id: Aviary identifier, e.g. ``"aviary_1"``.

    Returns:
        Set of common-name strings.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
        KeyError: If *aviary_id* is not found in the config.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    wild_birds = set(config.get("wild_birds", []))

    aviaries = config.get("aviaries", {})
    if aviary_id not in aviaries:
        available = ", ".join(sorted(aviaries.keys()))
        raise KeyError(
            f"Aviary '{aviary_id}' not found in {config_path}. "
            f"Available: {available}"
        )

    aviary_species = set(aviaries[aviary_id].get("species", []))
    combined = aviary_species | wild_birds

    print(
        f"[ARIA] Aviary '{aviary_id}': "
        f"{len(aviary_species)} aviary species + "
        f"{len(wild_birds)} wild birds = "
        f"{len(combined)} total allowed"
    )
    return combined


def list_aviaries(config_path: Path) -> list[str]:
    """Return sorted list of aviary IDs from the config."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return sorted(config.get("aviaries", {}).keys())


# ── flat allowed-species file ────────────────────────────────────────

def load_allowed_species(path: Path | None) -> set[str] | None:
    """Load a newline-delimited species list.

    Each non-blank line is parsed with :func:`_parse_species_line`.
    The returned set contains **common names** for filtering.

    Args:
        path: Path to the allowed-species file.

    Returns:
        A set of common-name strings, or ``None`` when *path* is
        ``None`` (meaning "allow everything").
    """
    if path is None:
        return None
    common_names: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            common, _ = _parse_species_line(line)
            if common:
                common_names.add(common)
    return common_names


# ── shared filtering logic ───────────────────────────────────────────

def filter_predictions(
    rows: list[dict],
    allowed: set[str] | None,
) -> list[dict]:
    """Keep only predictions whose species is in *allowed*.

    When *allowed* is ``None`` every row passes through unchanged.
    """
    if allowed is None:
        return rows
    return [r for r in rows if r["species"] in allowed]


def filter_csv_in_place(
    csv_path: Path,
    allowed_species: set[str] | None,
) -> None:
    """Re-write a BirdNET results CSV keeping only allowed species.

    Args:
        csv_path: Path to a ``.BirdNET.results.csv`` file.
        allowed_species: Set of common names to keep.  If ``None`` the
            file is left untouched.
    """
    if allowed_species is None:
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if len(rows) <= 1:
        return

    header = rows[0]
    com_idx = find_col_index(header, ["Common name", "Common_name", "Common Name"])
    if com_idx is None:
        return

    kept = [r for r in rows[1:] if r[com_idx].strip() in allowed_species]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(kept)
