#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${REPO_ROOT}/venv/bin/activate"
cd "${REPO_ROOT}"

# Example: InsightFace/CCIP evaluation (Flux preset)
python tools/eval.py similarity \
    --preset flux \
    --dataset mbst \
    --output-dir "/path/to/eval_results"

# Example: GPT evaluation (requires OPENAI_API_KEY)
# export OPENAI_API_KEY="your_key_here"
python tools/eval.py gpt \
    --source flux \
    --dataset mbst \
    --variant normal \
    --max-images 10
