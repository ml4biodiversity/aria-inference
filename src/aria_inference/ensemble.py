"""Three-tier ensemble inference with hybrid voting.

Combines BirdNET, PERCH v2, and a trained fusion model for bird
species detection.  Two inference modes:

1. **Ensemble voting** (default): all three models run on every
   segment; predictions are merged using weighted voting with a
   consensus boost when multiple models agree.
2. **Tier-based fallback**: BirdNET runs first; if its top
   confidence is below the tier-1 threshold, fusion runs; if
   fusion is also low, PERCH provides a fallback.

Species filtering is controlled by an anonymized aviary config
or a flat allowed-species file — no zoo-specific logic.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .audio import (
    PERCH_SR,
    compute_segments,
    extract_perch_segment,
    load_audio_for_perch,
)
from .birdnet_runtime import BirdNETRuntime
from .fusion_model import DualEmbeddingClassifier, load_fusion_model
from .perch_runtime import PERCHRuntime
from .species_filter import load_allowed_species, load_aviary_species


# ── species-name helpers ─────────────────────────────────────────────

def _common_name(species: str) -> str:
    """Extract common name from ``Scientific_Common`` or bare name."""
    return species.split("_", 1)[1] if "_" in species else species


# ── ensemble voting ──────────────────────────────────────────────────

def _hybrid_ensemble_voting(
    birdnet_preds: list[tuple[str, float]],
    fusion_preds: list[tuple[str, float]],
    perch_preds: list[tuple[str, float]],
    *,
    weights: dict[str, float] | None = None,
    consensus_boost: float = 0.5,
    top_k: int = 3,
) -> list[tuple[str, float, dict]]:
    """Merge predictions from three models via weighted voting.

    Species names are normalized to common names before merging so that
    ``"Turdus merula_Common Blackbird"`` and ``"Common Blackbird"`` are
    treated as the same species.

    Returns:
        List of ``(species, max_confidence, metadata)`` sorted by
        voting score, at most *top_k* entries.
    """
    if weights is None:
        weights = {"BirdNET": 1.0, "Fusion": 0.8, "PERCH": 0.6}

    data: dict[str, dict] = defaultdict(lambda: {
        "score": 0.0,
        "models": [],
        "confs": {},
        "max_conf": 0.0,
        "source": None,
        "best_name": None,
    })

    for model_name, preds, w in [
        ("BirdNET", birdnet_preds, weights["BirdNET"]),
        ("Fusion", fusion_preds, weights["Fusion"]),
        ("PERCH", perch_preds, weights["PERCH"]),
    ]:
        for species, conf in preds:
            key = _common_name(species)
            d = data[key]
            d["score"] += conf * w
            d["models"].append(model_name)
            d["confs"][model_name] = conf
            if conf > d["max_conf"]:
                d["max_conf"] = conf
                d["source"] = model_name
                d["best_name"] = species

    # Consensus boost
    for d in data.values():
        if len(d["models"]) >= 2:
            d["score"] += consensus_boost

    ranked = sorted(data.items(), key=lambda x: x[1]["score"], reverse=True)
    selected = ranked[: top_k] if len(ranked) > top_k else ranked

    result: list[tuple[str, float, dict]] = []
    for norm, d in selected:
        meta = {
            "vote_count": len(d["models"]),
            "models": d["models"],
            "confidences": d["confs"],
            "source_of_max": d["source"],
            "voting_score": d["score"],
        }
        result.append((d["best_name"] or norm, d["max_conf"], meta))
    return result


# ── main inference class ─────────────────────────────────────────────

class ARIAInference:
    """Full three-tier ARIA inference engine."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        tier1_threshold: float = 0.85,
        min_confidence: float = 0.30,
        perch_min_confidence: float = 0.10,
        top_k: int = 3,
        temperature: float = 1.8,
        segment_length: float = 3.0,
        overlap: float = 0.0,
        use_ensemble_voting: bool = True,
        ensemble_weights: dict[str, float] | None = None,
        consensus_boost: float = 0.5,
    ):
        model_dir = Path(model_dir)

        self.tier1_threshold = tier1_threshold
        self.min_confidence = min_confidence
        self.perch_min_confidence = perch_min_confidence
        self.top_k = top_k
        self.segment_length = segment_length
        self.overlap = overlap
        self.use_ensemble_voting = use_ensemble_voting
        self.consensus_boost = consensus_boost
        self.ensemble_weights = ensemble_weights or {
            "BirdNET": 1.0, "Fusion": 0.8, "PERCH": 0.6,
        }

        # Allowed species (set later via set_allowed_species)
        self.allowed_species: set[str] | None = None
        self.allowed_indices: list[int] | None = None

        # ── load architecture config ─────────────────────────────────
        config_path = model_dir / "architecture_info_public.json"
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        arch = self.config["architecture"]
        self.birdnet_dim = arch["birdnet_dim"]
        self.perch_dim = arch["perch_dim"]
        self.n_species = self.config["n_species"]

        self.idx_to_species = {
            int(k): v
            for k, v in self.config["species_mapping"]["idx_to_species"].items()
        }
        self.species_to_idx = {v: k for k, v in self.idx_to_species.items()}

        print(f"[ARIA] Architecture: birdnet={self.birdnet_dim}, "
              f"perch={self.perch_dim}, species={self.n_species}")

        # ── BirdNET ──────────────────────────────────────────────────
        self.birdnet = BirdNETRuntime(
            model_path=str(model_dir / "ZooCustom_v1.tflite"),
            labels_path=str(model_dir / "ZooCustom_v1_Labels.txt"),
            min_conf=0.01,
            temperature=temperature,
        )

        # ── PERCH v2 ─────────────────────────────────────────────────
        self.perch = PERCHRuntime(
            species_mapping_path=str(
                model_dir / "perch_v2_zoo_species_mapping.json"
            ),
            birdnet_species=self.birdnet.species,
            use_constrained=True,
        )

        # ── Fusion model ─────────────────────────────────────────────
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.fusion_model, _ = load_fusion_model(
            weights_path=str(model_dir / "fusion_model_perchv2_best.pth"),
            config_path=str(config_path),
            device=self.device,
        )

        mode = "ENSEMBLE VOTING" if use_ensemble_voting else "TIER-BASED"
        print(f"[ARIA] Inference ready  (mode: {mode}, device: {self.device})")

    # ── species constraints ──────────────────────────────────────────

    def set_allowed_species(self, species: set[str]) -> None:
        """Set the allowed species set and precompute BirdNET indices."""
        self.allowed_species = species

        self.allowed_indices = []
        for idx, sp in enumerate(self.birdnet.species):
            if self._is_allowed(sp):
                self.allowed_indices.append(idx)

        print(
            f"[ARIA] Species constraints: {len(species)} allowed, "
            f"{len(self.allowed_indices)}/{self.birdnet.n_species} "
            f"BirdNET indices"
        )

    def _is_allowed(self, species: str) -> bool:
        if self.allowed_species is None:
            return True
        return _common_name(species) in self.allowed_species

    # ── per-model predictions ────────────────────────────────────────

    def _birdnet_predictions(
        self, probs: np.ndarray
    ) -> list[tuple[str, float]]:
        preds = []
        for idx, p in enumerate(probs):
            if p >= self.min_confidence and idx < len(self.birdnet.species):
                sp = self.birdnet.species[idx]
                if self._is_allowed(sp):
                    preds.append((sp, float(p)))
        preds.sort(key=lambda x: x[1], reverse=True)
        return preds

    def _fusion_predictions(
        self, combined: np.ndarray
    ) -> list[tuple[str, float]]:
        with torch.no_grad():
            t = torch.FloatTensor(combined).unsqueeze(0).to(self.device)
            logits = self.fusion_model(t)[0]
            probs = F.softmax(logits, dim=0).cpu().numpy()

        preds = []
        for idx in range(len(probs)):
            conf = float(probs[idx])
            if conf >= self.min_confidence and idx in self.idx_to_species:
                sp = self.idx_to_species[idx]
                if self._is_allowed(sp):
                    preds.append((sp, conf))
        preds.sort(key=lambda x: x[1], reverse=True)
        return preds

    def _perch_predictions(
        self,
        perch_logits: np.ndarray,
        use_threshold: bool = True,
    ) -> list[tuple[str, float]]:
        if self.allowed_indices is None or len(self.allowed_indices) == 0:
            return []
        if len(perch_logits) == 0:
            return []

        filtered = perch_logits[self.allowed_indices]

        # Softmax over allowed species
        exp_l = np.exp(filtered - np.max(filtered))
        probs = exp_l / np.sum(exp_l)

        preds = []
        threshold = self.perch_min_confidence if use_threshold else 0.0
        for i, prob in enumerate(probs):
            if prob >= threshold:
                bn_idx = self.allowed_indices[i]
                sp = self.birdnet.species[bn_idx]
                preds.append((sp, float(prob)))
        preds.sort(key=lambda x: x[1], reverse=True)
        return preds

    # ── segment prediction ───────────────────────────────────────────

    def predict_segment(self, seg: dict) -> dict:
        """Predict species for a single segment dict.

        *seg* must contain ``birdnet_probs``, ``perch_embedding``,
        ``combined``, ``start_time``, ``end_time``.
        """
        if self.use_ensemble_voting:
            return self._predict_voting(seg)
        return self._predict_tiers(seg)

    def _predict_voting(self, seg: dict) -> dict:
        bn = self._birdnet_predictions(seg["birdnet_probs"])
        fu = self._fusion_predictions(seg["combined"])
        pe = self._perch_predictions(seg["perch_embedding"])

        voted = _hybrid_ensemble_voting(
            bn, fu, pe,
            weights=self.ensemble_weights,
            consensus_boost=self.consensus_boost,
            top_k=self.top_k,
        )
        predictions = [(sp, c) for sp, c, _ in voted]
        return {
            "predictions": predictions,
            "method": "Ensemble Voting",
            "tier": "N/A",
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
        }

    def _predict_tiers(self, seg: dict) -> dict:
        # Tier 1: BirdNET
        t1 = self._birdnet_predictions(seg["birdnet_probs"])
        if t1 and t1[0][1] >= self.tier1_threshold:
            return self._seg_result(t1, 1, "BirdNET", seg)

        # Tier 2: Fusion
        t2 = self._fusion_predictions(seg["combined"])
        if t2 and t2[0][1] > self.min_confidence:
            return self._seg_result(t2, 2, "Fusion", seg)

        # Tier 3: PERCH
        t3 = self._perch_predictions(seg["perch_embedding"])
        if t3:
            return self._seg_result(t3, 3, "PERCH v2", seg)

        # Best available fallback
        candidates = [
            (t1, 1, "BirdNET"),
            (t2, 2, "Fusion"),
            (t3, 3, "PERCH v2"),
        ]
        candidates = [(p, t, m) for p, t, m in candidates if p]
        if candidates:
            best = max(candidates, key=lambda x: x[0][0][1])
            return self._seg_result(best[0], best[1], best[2], seg)

        return self._seg_result([], 0, "No prediction", seg)

    @staticmethod
    def _seg_result(preds, tier, method, seg):
        return {
            "predictions": preds,
            "tier": tier,
            "method": method,
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
        }

    # ── file-level prediction ────────────────────────────────────────

    def predict_file(self, audio_path: str) -> list[dict]:
        """Run inference on a single audio file.

        Segments the audio, runs BirdNET analysis, extracts PERCH
        embeddings, builds combined vectors, and predicts per segment.

        Returns:
            List of detection dicts (one per segment with predictions).
        """
        audio_path_str = str(audio_path)

        # Load audio for PERCH
        try:
            perch_audio = load_audio_for_perch(audio_path_str)
        except Exception as e:
            print(f"[ARIA] Error loading {audio_path}: {e}")
            return []

        duration = len(perch_audio) / PERCH_SR
        if duration < self.segment_length:
            min_samples = int(self.segment_length * PERCH_SR)
            perch_audio = np.pad(
                perch_audio, (0, min_samples - len(perch_audio)), mode="constant"
            )
            duration = self.segment_length

        segments = compute_segments(duration, self.segment_length, self.overlap)

        # Run BirdNET on the whole file
        recording = self.birdnet.analyze_file(audio_path_str)

        # Process each segment
        detections: list[dict] = []
        for seg_idx, (start, end) in enumerate(segments):
            # BirdNET probability vector
            bn_probs = self.birdnet.build_probability_vector(
                recording, start, end
            )

            # PERCH embedding + logits
            perch_seg = extract_perch_segment(
                perch_audio, start, end,
                sr=PERCH_SR, segment_length=self.segment_length,
            )
            try:
                perch_logits, perch_raw = self.perch.extract_embedding(perch_seg)
            except Exception:
                perch_logits = np.zeros(self.perch_dim, dtype=np.float32)

            # Combined vector for fusion
            combined = np.concatenate([bn_probs, perch_logits])

            seg_data = {
                "birdnet_probs": bn_probs,
                "perch_embedding": perch_logits,
                "combined": combined,
                "start_time": start,
                "end_time": end,
            }

            result = self.predict_segment(seg_data)
            result["segment_index"] = seg_idx

            if result["predictions"]:
                detections.append(result)

        return detections

    # ── directory processing ─────────────────────────────────────────

    def process_directory(
        self,
        directory: str | Path,
        audio_extensions: list[str] | None = None,
    ) -> list[dict]:
        """Process all audio files in a directory.

        Returns a flat list of file-result dicts.
        """
        if audio_extensions is None:
            audio_extensions = [".wav", ".mp3", ".flac", ".ogg"]

        directory = Path(directory)
        files: list[Path] = []
        for ext in audio_extensions:
            files.extend(directory.rglob(f"*{ext}"))
            files.extend(directory.rglob(f"*{ext.upper()}"))
        files = sorted(set(files))

        print(f"[ARIA] Processing {len(files)} files from {directory}")

        results: list[dict] = []
        for i, f in enumerate(files, 1):
            try:
                dets = self.predict_file(str(f))
                results.append({
                    "file_name": f.name,
                    "file_path": str(f),
                    "detections": dets,
                    "status": "success",
                })
                top = (
                    f"{dets[0]['predictions'][0][0]}({dets[0]['predictions'][0][1]:.0%})"
                    if dets and dets[0]["predictions"]
                    else "no predictions"
                )
                print(f"  [{i}/{len(files)}] {f.name:<50} {top}")
            except Exception as e:
                results.append({
                    "file_name": f.name,
                    "file_path": str(f),
                    "error": str(e),
                    "status": "failed",
                })
                print(f"  [{i}/{len(files)}] {f.name:<50} ERROR: {e}")

        return results

    # ── CSV output ───────────────────────────────────────────────────

    def save_results_csv(
        self,
        results: list[dict],
        output_path: str | Path,
    ) -> None:
        """Write results to a CSV file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        header = [
            "File name", "Start (s)", "End (s)",
            "Species", "Confidence", "Method",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)

            for r in results:
                if r["status"] != "success":
                    continue
                for det in r.get("detections", []):
                    for sp, conf in det.get("predictions", []):
                        w.writerow([
                            r["file_name"],
                            f"{det['start_time']:.2f}",
                            f"{det['end_time']:.2f}",
                            _common_name(sp),
                            f"{conf:.4f}",
                            det.get("method", ""),
                        ])

        print(f"[ARIA] Results saved → {output_path}")


# ── top-level convenience function ───────────────────────────────────

def run_full_inference(
    input_path: Path,
    output_csv: Path,
    model_dir: Path,
    *,
    aviary_config: Path | None = None,
    aviary_id: str | None = None,
    allowed_species_file: Path | None = None,
    temperature: float = 1.8,
    tier1_threshold: float = 0.85,
    min_confidence: float = 0.30,
    perch_min_confidence: float = 0.10,
    top_k: int = 3,
    segment_length: float = 3.0,
    overlap: float = 0.0,
    use_ensemble_voting: bool = True,
    ensemble_weights: dict[str, float] | None = None,
    consensus_boost: float = 0.5,
) -> Path:
    """Run the full ARIA inference pipeline.

    Args:
        input_path: Audio file or directory.
        output_csv: Output CSV path.
        model_dir: Directory with all model files.
        aviary_config: Path to ``aviary_config.json``.
        aviary_id: Aviary identifier (e.g. ``"aviary_1"``).
        allowed_species_file: Flat species whitelist (alternative).
        temperature: BirdNET confidence scaling.
        tier1_threshold: Confidence for BirdNET-only acceptance.
        min_confidence: Global minimum confidence.
        perch_min_confidence: PERCH fallback minimum confidence.
        top_k: Max species per segment.
        segment_length: Audio segment length in seconds.
        overlap: Segment overlap in seconds.
        use_ensemble_voting: Use voting (True) or tier fallback (False).
        ensemble_weights: Per-model voting weights.
        consensus_boost: Extra score for multi-model agreement.

    Returns:
        Path to the output CSV.
    """
    # Resolve species constraints
    allowed: set[str] | None = None
    if aviary_config is not None and aviary_id is not None:
        allowed = load_aviary_species(aviary_config, aviary_id)
    elif aviary_config is not None:
        raise ValueError(
            "--aviary-config provided but --aviary is missing"
        )
    elif allowed_species_file is not None:
        allowed = load_allowed_species(allowed_species_file)

    # Initialize engine
    engine = ARIAInference(
        model_dir=model_dir,
        tier1_threshold=tier1_threshold,
        min_confidence=min_confidence,
        perch_min_confidence=perch_min_confidence,
        top_k=top_k,
        temperature=temperature,
        segment_length=segment_length,
        overlap=overlap,
        use_ensemble_voting=use_ensemble_voting,
        ensemble_weights=ensemble_weights,
        consensus_boost=consensus_boost,
    )

    if allowed is not None:
        engine.set_allowed_species(allowed)

    # Run
    if input_path.is_dir():
        results = engine.process_directory(input_path)
    else:
        dets = engine.predict_file(str(input_path))
        results = [{
            "file_name": input_path.name,
            "file_path": str(input_path),
            "detections": dets,
            "status": "success",
        }]

    # Save
    engine.save_results_csv(results, output_csv)
    return output_csv
