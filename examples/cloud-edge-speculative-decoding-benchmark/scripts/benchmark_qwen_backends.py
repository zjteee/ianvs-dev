#!/usr/bin/env python
"""Benchmark edge-only and cloud-only baseline backends on full datasets."""

import csv
import gc
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

import torch


ROOT_DIR = Path(__file__).resolve().parents[3]
SPECDEC_DIR = ROOT_DIR / "examples" / "cloud-edge-speculative-decoding-benchmark" / "testalgorithms" / "speculative-decoding"
AR_DIR = SPECDEC_DIR / "algorithms" / "ar"
SEDNA_SRC = ROOT_DIR / "examples" / "resources" / "third_party" / "sedna-0.6.0.1-src"
for path in (str(AR_DIR), str(SPECDEC_DIR), str(ROOT_DIR), str(SEDNA_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from drafter import SpeculativeDraftModel
from verifier import SpeculativeVerifyModel


DATASETS = {
    "gsm8k": ROOT_DIR / "dataset" / "gsm8k" / "train_data" / "data.jsonl",
    "humaneval": ROOT_DIR / "dataset" / "humaneval" / "train_data" / "data.jsonl",
}
MODELS = {
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
}
BACKENDS = ["custom", "transformers", "vllm"]
MODES = ["edge-only", "cloud-only"]


def parse_filter_env(name, default_values):
    """Parse an optional comma-separated environment filter."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default_values)
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    return selected or list(default_values)


def load_dataset(name):
    """Load one benchmark dataset into normalized request items."""
    path = DATASETS[name]
    rows = []
    with path.open() as handle:
        for index, line in enumerate(handle):
            payload = json.loads(line)
            rows.append(
                {
                    "request_id": f"{name}-{index:03d}",
                    "query": payload.get("question", ""),
                    "gold": payload.get("answer", ""),
                    "task_name": name,
                }
            )
    return rows


def build_model(mode, model_name, backend):
    """Construct the runtime object for one baseline mode/backend pair."""
    prompt_tokens = int(os.environ.get("QWEN_PROMPT_TOKENS", "1024"))
    max_new_tokens = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "128"))
    common_kwargs = {
        "model": model_name,
        "generation_backend": backend,
        "draft_tokens_per_step": 8,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "stop_mode": "none",
        "device": "auto",
        "trust_remote_code": True,
        "enable_network_sleep": False,
        "sample_temperature": 0.0,
    }
    if mode == "edge-only":
        runtime = SpeculativeDraftModel(
            inference_mode="edge-only",
            **common_kwargs,
        )
        runtime.load()
        return runtime
    runtime = SpeculativeVerifyModel(**common_kwargs)
    return runtime


def run_mode_dataset(mode, backend, model_alias, model_name, dataset_name):
    """Run one full benchmark slice and aggregate response metrics."""
    runtime = build_model(mode, model_name, backend)
    rows = load_dataset(dataset_name)
    responses = []
    started = time.perf_counter()
    for row in rows:
        if mode == "edge-only":
            response = runtime.inference(data=row)
        else:
            response = runtime.inference(data=row)
        responses.append(response)
    wall_s = time.perf_counter() - started

    avg_ttft = sum(item["perf"]["time_to_first_token"] for item in responses) / len(responses)
    avg_itl = sum(item["perf"]["internal_token_latency"] for item in responses) / len(responses)
    avg_throughput = sum(item["perf"]["throughput"] for item in responses) / len(responses)
    avg_e2e = sum(item["simulation"]["end_to_end_latency"] for item in responses) / len(responses)
    total_completion_tokens = sum(item["usage"]["completion_tokens"] for item in responses)
    batch_throughput = total_completion_tokens / max(wall_s, 1e-6)

    runtime.cleanup()
    del runtime
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "dataset": dataset_name,
        "mode": mode,
        "backend": backend,
        "model_alias": model_alias,
        "model": model_name,
        "samples": len(responses),
        "avg_ttft_s": round(avg_ttft, 6),
        "avg_itl_s": round(avg_itl, 6),
        "avg_throughput_tok_s": round(avg_throughput, 6),
        "avg_e2e_s": round(avg_e2e, 6),
        "batch_wall_s": round(wall_s, 6),
        "batch_completion_tokens": int(total_completion_tokens),
        "batch_throughput_tok_s": round(batch_throughput, 6),
    }


def main():
    """Run the full baseline backend matrix sequentially."""
    output_root = Path(
        os.environ.get(
            "QWEN_BACKEND_BENCH_OUT",
            ROOT_DIR / ".codex_run_logs" / f"qwen_backend_bench_{time.strftime('%Y%m%d-%H%M%S')}",
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    datasets = parse_filter_env("QWEN_DATASET_FILTER", DATASETS.keys())
    modes = parse_filter_env("QWEN_MODE_FILTER", MODES)
    backends = parse_filter_env("QWEN_BACKEND_FILTER", BACKENDS)
    model_aliases = parse_filter_env("QWEN_MODEL_FILTER", MODELS.keys())
    max_samples = int(os.environ.get("QWEN_MAX_SAMPLES", "0") or "0")

    original_load_dataset = load_dataset

    def filtered_load_dataset(name):
        rows = original_load_dataset(name)
        if max_samples > 0:
            return rows[:max_samples]
        return rows

    globals()["load_dataset"] = filtered_load_dataset

    for dataset_name in datasets:
        for mode in modes:
            for model_alias in model_aliases:
                model_name = MODELS[model_alias]
                for backend in backends:
                    print(
                        f"=== RUN dataset={dataset_name} mode={mode} model={model_alias} backend={backend} ===",
                        flush=True,
                    )
                    result = run_mode_dataset(
                        mode=mode,
                        backend=backend,
                        model_alias=model_alias,
                        model_name=model_name,
                        dataset_name=dataset_name,
                    )
                    results.append(result)
                    print(json.dumps(result, ensure_ascii=False), flush=True)

    csv_path = output_root / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    json_path = output_root / "summary.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
