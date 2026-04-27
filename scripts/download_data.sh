#!/usr/bin/env bash
# download_data.sh — Download and cache a HuggingFace dataset

set -euo pipefail

DATASET=${1:-"ag_news"}
OUTPUT_DIR=${2:-"data/raw"}

echo "Downloading dataset: $DATASET"
mkdir -p "$OUTPUT_DIR"

python - <<EOF
from datasets import load_dataset
import sys

dataset_name = "$DATASET"
output_dir = "$OUTPUT_DIR/${DATASET//\//_}"

print(f"Loading {dataset_name}...")
ds = load_dataset(dataset_name)
ds.save_to_disk(output_dir)
print(f"Saved to {output_dir}")
for split, data in ds.items():
    print(f"  {split}: {len(data):,} examples")
EOF

echo "Done. Dataset saved to $OUTPUT_DIR"
