#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert LLMSpeculativeSampling jsonl prompts into Ianvs question/answer jsonl format."
    )
    parser.add_argument("--src", required=True, help="Source jsonl file path.")
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Target dataset directory name under ianvs/dataset.",
    )
    parser.add_argument(
        "--output-root",
        default="/home/zhangjuntao/ianvs/dataset",
        help="Ianvs dataset root directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    src = Path(args.src)
    dataset_root = Path(args.output_root) / args.dataset_name
    train_dir = dataset_root / "train_data"
    test_dir = dataset_root / "test_data"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with src.open("r", encoding="utf-8") as file:
        for line in file:
            dialogs = json.loads(line)
            if not dialogs:
                continue
            prompt = str(dialogs[0].get("content", "")).strip()
            if not prompt:
                continue
            rows.append({"question": prompt, "answer": ""})

    with (train_dir / "data.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (test_dir / "data.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            test_row = {
                "query": row["question"],
                "response": row["answer"],
                "level_3_dim": "open-ended-generation",
                "level_4_dim": args.dataset_name,
            }
            file.write(json.dumps(test_row, ensure_ascii=False) + "\n")

    metadata = {
        "dataset": args.dataset_name,
        "description": f"Converted from {src} into Ianvs single-turn jsonl format.",
        "level_1_dim": "single-modal",
        "level_2_dim": "text",
        "level_3_dim": "task-category",
        "level_4_dim": "benchmark-name",
    }
    with (test_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"Converted {len(rows)} samples to {dataset_root}")


if __name__ == "__main__":
    main()
