"""PERCH v2 runtime for embedding and logit extraction.

Uses ``perch-hoplite`` to load the pre-trained PERCH v2 model (which
auto-downloads from Kaggle on first use).  Supports constrained logit
extraction: selecting only the PERCH output indices that correspond to
the zoo species, producing an 87-dim (or similar) vector that aligns
with the BirdNET species list for fusion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


class PERCHRuntime:
    """PERCH v2 model wrapper for embedding and logit extraction."""

    def __init__(
        self,
        species_mapping_path: str,
        birdnet_species: list[str],
        use_constrained: bool = True,
    ):
        """
        Args:
            species_mapping_path: Path to ``perch_v2_zoo_species_mapping.json``
                mapping BirdNET label strings to PERCH global indices.
            birdnet_species: Ordered list of BirdNET species labels (from
                the labels file).  Used to align PERCH indices with
                BirdNET's species order.
            use_constrained: If ``True``, return constrained logits
                (one per BirdNET species).  If ``False``, return the
                full ~15 000-dim logit vector.
        """
        from perch_hoplite.zoo import model_configs

        self.use_constrained = use_constrained

        # Load PERCH v2 model (auto-downloads on first use)
        print("[ARIA] Loading PERCH v2 model ...")
        self.model = model_configs.load_model_by_name("perch_v2")
        print("[ARIA] PERCH v2 loaded")

        # Build constrained species index mapping
        self.species_indices: Optional[list[int]] = None
        if use_constrained and Path(species_mapping_path).exists():
            with open(species_mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)

            self.species_indices = []
            unmapped: list[str] = []
            for bn_species in birdnet_species:
                idx = self._find_perch_index(bn_species, mapping)
                if idx is not None:
                    self.species_indices.append(idx)
                else:
                    unmapped.append(bn_species)
                    # Fallback: use first mapping value
                    self.species_indices.append(
                        int(next(iter(mapping.values())))
                    )

            n_mapped = len(birdnet_species) - len(unmapped)
            print(
                f"[ARIA] PERCH species mapping: "
                f"{n_mapped}/{len(birdnet_species)} mapped"
            )
            if unmapped:
                shown = unmapped[:5]
                extra = f" ... and {len(unmapped)-5} more" if len(unmapped) > 5 else ""
                print(f"[ARIA] Unmapped: {shown}{extra}")
        elif use_constrained:
            print(
                f"[ARIA] Warning: species mapping not found at "
                f"{species_mapping_path}, using unconstrained PERCH"
            )
            self.use_constrained = False

    @staticmethod
    def _find_perch_index(
        bn_species: str,
        mapping: dict[str, int],
    ) -> Optional[int]:
        """Find the PERCH index for a BirdNET species label.

        Tries exact match, then normalized match, then substring match.
        """
        # Exact match
        if bn_species in mapping:
            return int(mapping[bn_species])

        # Normalized match
        clean = bn_species.lower().replace("_", " ").strip()
        for key, idx in mapping.items():
            if key.lower().replace("_", " ").strip() == clean:
                return int(idx)

        # Substring match
        for key, idx in mapping.items():
            key_clean = key.lower().replace("_", " ").strip()
            if clean in key_clean or key_clean in clean:
                return int(idx)

        return None

    def extract_embedding(
        self, audio_segment: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract PERCH logits and raw embeddings from a 32 kHz segment.

        Args:
            audio_segment: 1-D float32 array at 32 kHz.

        Returns:
            ``(logits, raw_embedding)`` where:
            - *logits* is ``(n_constrained,)`` if constrained, else
              ``(~15000,)``
            - *raw_embedding* is always ``(1536,)``
        """
        outputs = self.model.embed(audio_segment[np.newaxis, :])

        # Raw 1536-dim embeddings
        raw = np.array(outputs.embeddings)
        if raw.ndim == 3:
            raw = raw.mean(axis=1)
        if raw.ndim == 2:
            raw = raw[0]
        raw = raw.astype(np.float32)

        # Logits from classification head
        logits = None
        if hasattr(outputs, "logits"):
            logits_raw = outputs.logits
            if isinstance(logits_raw, dict):
                logits_raw = logits_raw.get("label", list(logits_raw.values())[0])
            logits = np.array(logits_raw)
            if logits.ndim == 2:
                logits = logits[0]

            # Constrain to zoo species
            if self.use_constrained and self.species_indices is not None:
                logits = logits[self.species_indices].astype(np.float32)
            else:
                logits = logits.astype(np.float32)
        else:
            # Fallback: zeros
            n = len(self.species_indices) if self.species_indices else 15_000
            logits = np.zeros(n, dtype=np.float32)

        return logits, raw
