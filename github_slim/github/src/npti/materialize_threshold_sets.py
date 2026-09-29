#!/usr/bin/env python3
"""Materialize compatible NPTI sets from current-bank dense discovery stats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def entries(delta: torch.Tensor, a95: torch.Tensor, threshold: float, positive: bool) -> list[dict[str, Any]]:
    indices = torch.nonzero(delta > threshold if positive else delta < -threshold, as_tuple=False).flatten()
    indices = indices[torch.argsort(delta[indices], descending=positive)] if indices.numel() else indices
    return [{"neuron_id": int(index), "delta": float(delta[index]), "a95": float(a95[index]), "rank": rank} for rank, index in enumerate(indices.tolist(), 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-dir", type=Path, default=Path("use/npti_set/current_bank_11class"))
    parser.add_argument("--output-dir", type=Path, default=Path("use/npti_set/current_bank_11class/threshold_sets"))
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25")
    args = parser.parse_args()
    dense_paths = sorted(args.dense_dir.glob("dense_neuron_stats_shard*_of_*.pt"))
    if not dense_paths:
        raise FileNotFoundError(f"no dense stats in {args.dense_dir}")
    dense: dict[str, Any] = {}
    for path in dense_paths:
        dense.update(torch.load(path, map_location="cpu", weights_only=False))
    for threshold in [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]:
        target = args.output_dir / f"threshold_{threshold:.2f}".replace(".", "p")
        output = {}
        for emotion, stats in dense.items():
            delta, a95 = stats["delta"].float(), stats["a95"].float()
            positive, negative, layer_statistics = {}, {}, {}
            for layer in range(delta.shape[0]):
                key = f"layer_{layer + 1}"
                positive[key] = entries(delta[layer], a95[layer], threshold, True)
                negative[key] = entries(delta[layer], a95[layer], threshold, False)
                layer_statistics[key] = {"positive_count": len(positive[key]), "negative_count": len(negative[key])}
            output[emotion] = {"sample_count": stats["sample_count"], "target_token_count": stats["target_token_count"], "neutral_token_count": stats["neutral_token_count"], "positive_neurons": positive, "negative_neurons": negative, "layer_statistics": layer_statistics}
        target.mkdir(parents=True, exist_ok=True)
        (target / "emotion_neuron_sets_11class.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (target / "metadata.json").write_text(json.dumps({"dense_dir": str(args.dense_dir), "threshold": threshold, "source": "current bank train_successful_steering records"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"materialized {target}", flush=True)


if __name__ == "__main__":
    main()
