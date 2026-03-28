"""Shared utility functions for CSV column lookup and label formatting."""


def find_col_index(header: list[str], candidates: list[str]) -> int | None:
    """Return first matching column index from candidate names (case-insensitive).

    Args:
        header: CSV header row.
        candidates: Column name candidates in priority order.

    Returns:
        Column index or ``None`` if no candidate matches.
    """
    low = [h.lower() for h in header]
    for cand in candidates:
        if cand.lower() in low:
            return low.index(cand.lower())
    return None


def class_key(scientific: str, common: str) -> str:
    """Build a ``Scientific_Common`` label key matching the BirdNET training convention."""
    s = (scientific or "").strip()
    c = (common or "").strip()
    return f"{s}_{c}" if s and c else f"{s}{c}"
