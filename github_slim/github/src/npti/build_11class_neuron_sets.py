#!/usr/bin/env python3
"""Build NPTI FFN neuron sets from the current 11-class memory-bank pairs.

Only successful training records in the current bank are used.  Test memory
records are deliberately excluded from discovery.  For each clean pair, the
emotion_description and neutral_description are forwarded through Qwen's FFN;
delta is the difference between over-zero activation probabilities.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BANK = ROOT / "use/build/multilayer_11class_trainplus_test_memory/qwen_multilayer_memory_bank_regenerated_classified_6927.pt"
DEFAULT_OUTPUT = ROOT / "use/npti_set/current_bank_11class"
DEFAULT_MODEL = "./local/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--qwen-model-path", default=DEFAULT_MODEL)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--activation-max", type=float, default=3.0)
    p.add_argument("--activation-bin", type=float, default=0.01)
    p.add_argument("--gpu", default="0")
    p.add_argument("--emotion-shard-index", type=int, default=0)
    p.add_argument("--emotion-shard-count", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--merge-shards", action="store_true")
    p.add_argument("--save-dense-stats", action="store_true", help="Save all-neuron delta and a95 tensors for threshold sweeps.")
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class Stats:
    def __init__(self, layers: int, width: int, bins: int, histograms: bool, device: torch.device):
        self.over_zero = torch.zeros(layers, width, dtype=torch.int32, device=device)
        self.histograms = torch.zeros(layers, width, bins, dtype=torch.float32, device=device) if histograms else None
        self.token_count = 0


class Collector:
    def __init__(self, model: Any, activation_max: float, activation_bin: float):
        self.model = model
        self.device = next(model.parameters()).device
        self.layers = int(model.config.num_hidden_layers)
        self.width = int(model.config.intermediate_size)
        edges = torch.arange(0.0, activation_max + activation_bin, activation_bin)
        self.bin_edges_cpu = torch.cat([torch.tensor([-float("inf")]), edges])
        self.bin_edges_cpu[-1] = float("inf")
        self.bin_edges = self.bin_edges_cpu.to(self.device)
        self.active: Stats | None = None
        for index, layer in enumerate(model.model.layers):
            layer.mlp.forward = MethodType(self._make_forward(index), layer.mlp)

    def _make_forward(self, layer_index: int):
        owner = self

        def forward(module: Any, x: torch.Tensor):
            activation = module.act_fn(module.gate_proj(x))
            stats = owner.active
            if stats is not None:
                values = activation.detach().float().reshape(-1, activation.shape[-1])
                stats.over_zero[layer_index] += (values > 0).sum(dim=0).to(torch.int32)
                if stats.histograms is not None:
                    indices = (torch.bucketize(values, owner.bin_edges) - 1).transpose(0, 1)
                    increment = torch.ones_like(indices, dtype=stats.histograms.dtype)
                    stats.histograms[layer_index].scatter_add_(1, indices, increment)
                if layer_index == 0:
                    stats.token_count += int(values.shape[0])
            return module.down_proj(activation * module.up_proj(x))

        return forward

    def new_stats(self, histograms: bool) -> Stats:
        return Stats(self.layers, self.width, self.bin_edges.numel() - 1, histograms, self.device)

    @torch.inference_mode()
    def collect(self, tokenizer: Any, stats: Stats, text: str) -> None:
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        self.active = stats
        try:
            self.model(**encoded, use_cache=False)
        finally:
            self.active = None


def percentile(histograms: torch.Tensor, bins: torch.Tensor, q: float) -> torch.Tensor:
    cumulative = torch.cumsum(histograms.cpu(), dim=-1)
    thresholds = cumulative[..., -1].clamp_min(1) * q
    index = (cumulative >= thresholds.unsqueeze(-1)).float().argmax(dim=-1)
    values = bins.cpu()[index].float()
    return torch.nan_to_num(values, nan=0.0, neginf=0.0, posinf=float(bins.cpu()[-2]))


def entries(delta: torch.Tensor, a95: torch.Tensor, threshold: float, positive: bool) -> list[dict[str, Any]]:
    selected = torch.nonzero(delta > threshold if positive else delta < -threshold, as_tuple=False).flatten()
    if not selected.numel():
        return []
    ordered = selected[torch.argsort(delta[selected], descending=positive)]
    return [{"neuron_id": int(index), "delta": float(delta[index]), "a95": float(a95[index]), "rank": rank} for rank, index in enumerate(ordered.tolist(), 1)]


def load_training_records(bank_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    records = [row for row in bank["records"] if row.get("memory_source") == "train_successful_steering"]
    if not records:
        raise ValueError("current bank contains no train_successful_steering records")
    required = {"emotion_label", "emotion_description", "neutral_description"}
    bad = [row.get("memory_id") for row in records if not required.issubset(row) or not row["emotion_description"] or not row["neutral_description"]]
    if bad:
        raise ValueError(f"training records missing clean descriptions: {len(bad)}")
    labels = set(bank["manifest"]["labels"])
    if len(labels) != 11:
        raise ValueError(f"expected 11 bank labels, got {sorted(labels)}")
    if {row["emotion_label"] for row in records} != labels:
        raise ValueError("training labels do not match bank manifest labels")
    return bank, records


def merge(args: argparse.Namespace) -> None:
    shards = sorted(args.output_dir.glob("emotion_neuron_sets_shard*_of_*.json"))
    if not shards:
        raise FileNotFoundError(f"no neuron-set shards in {args.output_dir}")
    merged: dict[str, Any] = {}
    manifests = []
    for shard in shards:
        merged.update(json.loads(shard.read_text(encoding="utf-8")))
        manifest = shard.with_name(shard.name.replace("emotion_neuron_sets_", "manifest_"))
        if manifest.exists():
            manifests.append(json.loads(manifest.read_text(encoding="utf-8")))
    if len(merged) != 11:
        raise ValueError(f"expected 11 merged emotions, got {len(merged)}")
    write_json(args.output_dir / "emotion_neuron_sets_11class.json", merged)
    write_json(args.output_dir / "metadata.json", {"bank": str(args.bank), "model": args.qwen_model_path, "threshold": args.threshold, "activation_site": "FFN intermediate act_fn(gate_proj(x))", "score": "Pr(emotion activation > 0) - Pr(neutral activation > 0)", "selection": "delta > threshold / delta < -threshold", "source": "current bank train_successful_steering records only", "num_emotions": len(merged), "sample_counts": {emotion: value["sample_count"] for emotion, value in merged.items()}, "shards": len(shards)})


def main() -> None:
    setting = parse_args()
    if setting.merge_shards:
        merge(setting)
        print(f"merged: {setting.output_dir / 'emotion_neuron_sets_11class.json'}", flush=True)
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = setting.gpu
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if not 0 <= setting.emotion_shard_index < setting.emotion_shard_count:
        raise ValueError("invalid emotion shard")
    bank, records = load_training_records(setting.bank)
    labels = sorted(bank["manifest"]["labels"])
    labels = [label for index, label in enumerate(labels) if index % setting.emotion_shard_count == setting.emotion_shard_index]
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(setting.qwen_model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(setting.qwen_model_path, torch_dtype=torch.float16, trust_remote_code=True, local_files_only=True).to(device).eval()
    collector = Collector(model, setting.activation_max, setting.activation_bin)
    output: dict[str, Any] = {}
    dense_output: dict[str, Any] = {}
    for emotion in labels:
        group = [row for row in records if row["emotion_label"] == emotion]
        target = collector.new_stats(False)
        neutral = collector.new_stats(True)
        for index, row in enumerate(group, 1):
            collector.collect(tokenizer, target, str(row["emotion_description"]))
            collector.collect(tokenizer, neutral, str(row["neutral_description"]))
            if index % setting.progress_every == 0 or index == len(group):
                print(f"{emotion}: {index}/{len(group)}", flush=True)
        delta = target.over_zero.float().cpu() / target.token_count - neutral.over_zero.float().cpu() / neutral.token_count
        a95 = percentile(neutral.histograms, collector.bin_edges_cpu, 0.95)
        positive, negative, layer_statistics = {}, {}, {}
        for layer in range(collector.layers):
            pos = entries(delta[layer], a95[layer], setting.threshold, True)
            neg = entries(delta[layer], a95[layer], setting.threshold, False)
            key = f"layer_{layer + 1}"
            positive[key], negative[key] = pos, neg
            layer_statistics[key] = {"positive_count": len(pos), "negative_count": len(neg)}
        output[emotion] = {"sample_count": len(group), "target_token_count": target.token_count, "neutral_token_count": neutral.token_count, "positive_neurons": positive, "negative_neurons": negative, "layer_statistics": layer_statistics}
        if setting.save_dense_stats:
            dense_output[emotion] = {
                "delta": delta.float(),
                "a95": a95.float(),
                "sample_count": len(group),
                "target_token_count": target.token_count,
                "neutral_token_count": neutral.token_count,
            }
        del target, neutral
        torch.cuda.empty_cache()
    setting.output_dir.mkdir(parents=True, exist_ok=True)
    shard = setting.output_dir / f"emotion_neuron_sets_shard{setting.emotion_shard_index}_of_{setting.emotion_shard_count}.json"
    write_json(shard, output)
    if setting.save_dense_stats:
        torch.save(dense_output, setting.output_dir / f"dense_neuron_stats_shard{setting.emotion_shard_index}_of_{setting.emotion_shard_count}.pt")
    write_json(setting.output_dir / f"manifest_{setting.emotion_shard_index}_of_{setting.emotion_shard_count}.json", {"bank": str(setting.bank), "model": setting.qwen_model_path, "emotions": labels, "threshold": setting.threshold, "activation_site": "FFN intermediate act_fn(gate_proj(x))", "a95": "95th percentile of neutral_description activations", "source": "current bank train_successful_steering records only"})
    print(f"wrote {shard}", flush=True)


if __name__ == "__main__":
    main()
