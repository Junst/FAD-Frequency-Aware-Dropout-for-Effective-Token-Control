#!/bin/bash
# Qwen-Image LoRA Training with AdaptiveDropout
#
# This script demonstrates how to train Qwen-Image LoRA with AdaptiveDropout method.
# AdaptiveDropout includes three types of dropout:
#   - rank_dropout: Randomly drops entire rank dimensions (recommended: 0.1-0.3)
#   - module_dropout: Randomly skips entire LoRA modules (recommended: 0.05-0.1)
#   - dropout: Standard neuron dropout on activations (optional)
#
# Reference: AdaptiveDropout method proven effective on SDXL
#
# IMPORTANT - Offloading Notes:
#   - AdaptiveDropout LoRA is compatible with DiffSynth's offloading mechanism
#   - LoRA weights automatically move to the same device as input during forward pass
#   - When using --offload_models, the base model will offload but LoRA stays in memory
#   - For disk offloading, do NOT include the LoRA base model in --offload_models
#
# Usage: bash Qwen-Image-AdaptiveDropout.sh

# Model paths (update these to your local paths)
MODEL_ID="Qwen/Qwen-Image"

# Dataset configuration
DATASET_BASE_PATH="/path/to/your/dataset"
DATASET_METADATA_PATH="/path/to/your/metadata.json"

# Training configuration
OUTPUT_PATH="./outputs/qwen_image_lora_adaptive_dropout"
LORA_RANK=32
LEARNING_RATE=1e-4
NUM_EPOCHS=10

# AdaptiveDropout configuration
RANK_DROPOUT=0.2
MODULE_DROPOUT=0.1
DROPOUT=0.0

python ../train.py \
    --model_id_with_origin_paths "${MODEL_ID}:*.safetensors" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --dataset_metadata_path "${DATASET_METADATA_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --lora_base_model dit \
    --lora_target_modules "to_q,to_k,to_v,to_out,ff.0,ff.2" \
    --lora_rank ${LORA_RANK} \
    --learning_rate ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --use_gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --use_adaptive_dropout \
    --rank_dropout ${RANK_DROPOUT} \
    --module_dropout ${MODULE_DROPOUT} \
    --dropout ${DROPOUT} \
    --remove_prefix_in_ckpt "pipe.dit."

echo "Training completed! Output saved to ${OUTPUT_PATH}"

