#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${REPO_ROOT}/venv/bin/activate"
cd "${REPO_ROOT}"

# Example: Flux inference + metrics
python tools/infer.py infer \
    --backend flux \
    --lora-path "/path/to/flux_lora/step-3000.safetensors" \
    --metadata-path "/path/to/metadata.json" \
    --output-dir "/path/to/output_images" \
    --results-file "/path/to/results.json"
