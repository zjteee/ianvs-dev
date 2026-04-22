#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IANVS_BIN="${IANVS_BIN:-/home/zhangjuntao/miniconda3/envs/ianvs-experiment/bin/ianvs}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangjuntao/miniconda3/envs/ianvs-experiment/bin/python}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_BASE_URL="${HUGGINGFACE_HUB_BASE_URL:-$HF_ENDPOINT}"

RUN_ROOT="${1:-$ROOT_DIR/.codex_run_logs/baseline_backend_suite_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RUN_ROOT"

declare -A DATASET_SAMPLE_SIZE=(
  [gsm8k]=8
  [humaneval]=8
)

SMALL_MODEL="Qwen/Qwen3-0.6B"
LARGE_MODEL="Qwen/Qwen3-8B"
BACKENDS=(custom transformers vllm)
MODES=(edge-only cloud-only)
DATASETS=(gsm8k humaneval)
MODELS=("$SMALL_MODEL" "$LARGE_MODEL")

for dataset in "${DATASETS[@]}"; do
  for mode in "${MODES[@]}"; do
    for model in "${MODELS[@]}"; do
      for backend in "${BACKENDS[@]}"; do
        run_name="${dataset}-${mode}-$(basename "${model}")-${backend}"
        run_dir="$RUN_ROOT/$run_name"
        mkdir -p "$run_dir"

        cp "$ROOT_DIR/examples/cloud-edge-speculative-decoding-benchmark/benchmarkingjob.yaml" "$run_dir/benchmarkingjob.yaml"
        cp "$ROOT_DIR/examples/cloud-edge-speculative-decoding-benchmark/testenv/testenv.yaml" "$run_dir/testenv.yaml"
        cp "$ROOT_DIR/examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/test_speculative_decoding.yaml" "$run_dir/test_speculative_decoding.yaml"
        cp "$ROOT_DIR/examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/profiles/base.yaml" "$run_dir/base.yaml"

        "$PYTHON_BIN" - <<'PY' "$run_dir" "$ROOT_DIR" "$dataset" "$mode" "$model" "$backend" "${DATASET_SAMPLE_SIZE[$dataset]}" "$SMALL_MODEL"
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
root = Path(sys.argv[2])
dataset = sys.argv[3]
mode = sys.argv[4]
active_model = sys.argv[5]
backend = sys.argv[6]
sample_size = int(sys.argv[7])
small_model = sys.argv[8]

bench_path = run_dir / "benchmarkingjob.yaml"
testenv_path = run_dir / "testenv.yaml"
algo_path = run_dir / "test_speculative_decoding.yaml"
profile_path = run_dir / "base.yaml"

bench_text = bench_path.read_text()
bench_text = bench_text.replace(
    "./workspace-cloud-edge-speculative-decoding-benchmark-gsm8k",
    str(run_dir / "workspace"),
)
bench_text = bench_text.replace(
    "./examples/cloud-edge-speculative-decoding-benchmark/testenv/testenv.yaml",
    str(testenv_path),
)
bench_text = bench_text.replace(
    "./examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/test_speculative_decoding.yaml",
    str(algo_path),
)
bench_path.write_text(bench_text)

testenv_text = testenv_path.read_text()
testenv_text = testenv_text.replace(
    "./dataset/humaneval/train_data/data.jsonl",
    f"./dataset/{dataset}/train_data/data.jsonl",
)
testenv_text = testenv_text.replace(
    "./dataset/humaneval/test_data/metadata.json",
    f"./dataset/{dataset}/test_data/metadata.json",
)
testenv_path.write_text(testenv_text)

profile_lines = []
for line in profile_path.read_text().splitlines():
    if line.startswith("generation_backend:"):
        profile_lines.append(f'generation_backend: "{backend}"')
    else:
        profile_lines.append(line)
profile_path.write_text("\n".join(profile_lines) + "\n")

lines = algo_path.read_text().splitlines()
patched = []
state = None
for line in lines:
    stripped = line.strip()
    if stripped == 'name: "SpeculativeDraftModel"':
        state = "drafter"
    elif stripped == 'name: "SpeculativeVerifyModel"':
        state = "verifier"
    if stripped == "- sample_size:":
        patched.append(line)
        continue
    if stripped == "- 50":
        patched.append(line.replace("- 50", f"- {sample_size}"))
        continue
    if stripped == '- "./examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/profiles/base.yaml"':
        patched.append(line.replace('./examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/profiles/base.yaml', str(profile_path)))
        continue
    if stripped == '- "collaboration"':
        patched.append(line.replace('"collaboration"', f'"{mode}"'))
        continue
    if stripped == '- "Qwen/Qwen2.5-0.5B-Instruct"':
        target = active_model if (mode == "edge-only" and state == "drafter") else small_model
        patched.append(line.replace('Qwen/Qwen2.5-0.5B-Instruct', target))
        continue
    if stripped == '- "Qwen/Qwen2.5-7B-Instruct"':
        target = active_model if (mode == "cloud-only" and state == "verifier") else small_model
        patched.append(line.replace('Qwen/Qwen2.5-7B-Instruct', target))
        continue
    patched.append(line)
algo_path.write_text("\n".join(patched) + "\n")
PY

        echo "=== RUN $run_name ==="
        (cd "$ROOT_DIR" && "$IANVS_BIN" -f "$run_dir/benchmarkingjob.yaml" > "$run_dir/run.log" 2>&1)
        tail -n 12 "$run_dir/run.log"
        echo
      done
    done
  done
done

echo "Run root: $RUN_ROOT"
