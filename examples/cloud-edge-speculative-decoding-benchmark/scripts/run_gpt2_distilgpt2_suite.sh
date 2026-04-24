#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/zhangjuntao/ianvs"
PY="/home/zhangjuntao/miniconda3/envs/ianvs-experiment/bin/python"
HF_CLI="/home/zhangjuntao/miniconda3/envs/ianvs-experiment/bin/huggingface-cli"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_ENDPOINT

BASE_DIR="${1:-$ROOT_DIR/.codex_run_logs/gpt2_distilgpt2_mirror_$(date +%Y%m%d-%H%M%S)}"
SAMPLE_SIZE="${SAMPLE_SIZE:-20}"
WARMUP_SAMPLES="${WARMUP_SAMPLES:-3}"
DRAFT_TOKENS_PER_STEP="${DRAFT_TOKENS_PER_STEP:-8}"
PROMPT_TOKENS="${PROMPT_TOKENS:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
RUNAWAY_GUARD_MAX_NEW_TOKENS="${RUNAWAY_GUARD_MAX_NEW_TOKENS:-1024}"
SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-0.0}"
DATASETS="${DATASETS:-gsm8k,humaneval}"
MODELS_DIR="$BASE_DIR/models"
EDGE_MODEL_DIR="$MODELS_DIR/distilgpt2"
CLOUD_MODEL_DIR="$MODELS_DIR/gpt2"

mkdir -p "$MODELS_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

download_model() {
  local repo_id="$1"
  local local_dir="$2"
  shift 2
  mkdir -p "$local_dir"
  log "Downloading $repo_id to $local_dir via $HF_ENDPOINT"
  "$HF_CLI" download "$repo_id" \
    --include "$@" \
    --local-dir "$local_dir" \
    --max-workers 8
}

prepare_models() {
  download_model gpt2 "$CLOUD_MODEL_DIR" \
    config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json model.safetensors
  download_model distilgpt2 "$EDGE_MODEL_DIR" \
    config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json model.safetensors
}

prepare_run_dir() {
  local run_dir="$1"
  local dataset_name="$2"
  local mode="$3"

  mkdir -p "$run_dir"
  cp "$ROOT_DIR/examples/cloud-edge-speculative-decoding-benchmark/benchmarkingjob.yaml" "$run_dir/benchmarkingjob.yaml"
  cp "$ROOT_DIR/examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/test_speculative_decoding.yaml" "$run_dir/test_speculative_decoding.yaml"
  cp "$ROOT_DIR/examples/cloud-edge-speculative-decoding-benchmark/testenv/testenv.yaml" "$run_dir/testenv.yaml"

  cat > "$run_dir/runtime.yaml" <<RUNTIME
device: auto
trust_remote_code: true
enable_network_sleep: false
network_rtt_ms: 0
network_jitter_ms: 0
network_uplink_ratio: 0.5
network_uplink_bandwidth_mbps: 0
network_downlink_bandwidth_mbps: 0
network_seed: 42
sample_output_log: specdec_sample_outputs.jsonl
draft_tokens_per_step: ${DRAFT_TOKENS_PER_STEP}
prompt_tokens: ${PROMPT_TOKENS}
max_new_tokens: ${MAX_NEW_TOKENS}
runaway_guard_max_new_tokens: ${RUNAWAY_GUARD_MAX_NEW_TOKENS}
stop_mode: none
sample_temperature: ${SAMPLE_TEMPERATURE}
RUNTIME

  RUN_DIR="$run_dir" \
  DATASET="$dataset_name" \
  MODE="$mode" \
  EDGE_MODEL="$EDGE_MODEL_DIR" \
  CLOUD_MODEL="$CLOUD_MODEL_DIR" \
  SAMPLE_SIZE="$SAMPLE_SIZE" \
  WARMUP_SAMPLES="$WARMUP_SAMPLES" \
  DRAFT_TOKENS_PER_STEP="$DRAFT_TOKENS_PER_STEP" \
  "$PY" - <<'PY'
from pathlib import Path
import os
import yaml

run_dir = Path(os.environ["RUN_DIR"])
dataset = os.environ["DATASET"]
mode = os.environ["MODE"]
edge_model = os.environ["EDGE_MODEL"]
cloud_model = os.environ["CLOUD_MODEL"]
sample_size = int(os.environ["SAMPLE_SIZE"])
warmup_samples = int(os.environ["WARMUP_SAMPLES"])

benchmark = yaml.safe_load((run_dir / "benchmarkingjob.yaml").read_text())
benchmark["benchmarkingjob"]["workspace"] = str((run_dir / "workspace").resolve())
benchmark["benchmarkingjob"]["testenv"] = str((run_dir / "testenv.yaml").resolve())
benchmark["benchmarkingjob"]["test_object"]["algorithms"][0]["url"] = str((run_dir / "test_speculative_decoding.yaml").resolve())
(run_dir / "benchmarkingjob.yaml").write_text(yaml.safe_dump(benchmark, sort_keys=False, allow_unicode=True))

testenv = yaml.safe_load((run_dir / "testenv.yaml").read_text())
testenv["testenv"]["dataset"]["train_data"] = f"./dataset/{dataset}/train_data/data.jsonl"
testenv["testenv"]["dataset"]["test_data_info"] = f"./dataset/{dataset}/test_data/metadata.json"
(run_dir / "testenv.yaml").write_text(yaml.safe_dump(testenv, sort_keys=False, allow_unicode=True))

alg = yaml.safe_load((run_dir / "test_speculative_decoding.yaml").read_text())
alg["algorithm"]["warmup_samples"] = warmup_samples
mods = alg["algorithm"]["modules"]
mods[0]["hyperparameters"][0]["sample_size"]["values"] = [sample_size]
mods[1]["hyperparameters"][0]["other_hyperparameters"]["values"] = [str((run_dir / "runtime.yaml").resolve())]
mods[1]["hyperparameters"][1]["inference_mode"]["values"] = [mode]
mods[1]["hyperparameters"][2]["draft_tokens_per_step"]["values"] = [int(os.environ.get("DRAFT_TOKENS_PER_STEP", "8"))]
mods[1]["hyperparameters"][3]["model"]["values"] = [edge_model]
mods[2]["hyperparameters"][0]["other_hyperparameters"]["values"] = [str((run_dir / "runtime.yaml").resolve())]
mods[2]["hyperparameters"][1]["model"]["values"] = [cloud_model]
(run_dir / "test_speculative_decoding.yaml").write_text(yaml.safe_dump(alg, sort_keys=False, allow_unicode=True))
PY
}

run_case() {
  local dataset_name="$1"
  local mode="$2"
  local run_dir="$BASE_DIR/${dataset_name}-${mode}"
  local summary_file="$BASE_DIR/${dataset_name}-${mode}.summary"

  log "Running dataset=$dataset_name mode=$mode"
  prepare_run_dir "$run_dir" "$dataset_name" "$mode"
  {
    echo "$run_dir"
    echo "RUN_DIR=$run_dir"
    HF_ENDPOINT="$HF_ENDPOINT" "$PY" "$ROOT_DIR/benchmarking.py" -f "$run_dir/benchmarkingjob.yaml"
    echo "EXIT_CODE=0"
    cat "$run_dir/workspace/benchmarkingjob/rank/selected_rank.csv"
  } > "$summary_file" 2>&1
}

print_summary() {
  printf '\n=== summaries ===\n'
  for f in "$BASE_DIR"/*.summary; do
    echo "--- $(basename "$f") ---"
    tail -n 20 "$f"
  done
}

cd "$ROOT_DIR"
log "Using HF_ENDPOINT=$HF_ENDPOINT"
prepare_models
for dataset in ${DATASETS//,/ }; do
  for mode in collaboration cloud-only edge-only; do
    run_case "$dataset" "$mode"
  done
done
print_summary
log "Finished. Results are under $BASE_DIR"
