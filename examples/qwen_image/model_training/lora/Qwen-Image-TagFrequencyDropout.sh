#!/bin/bash
# Qwen-Image LoRA Training with Tag Frequency Dropout
#
# This script demonstrates how to train Qwen-Image LoRA with Tag Frequency Dropout.
# Tag Frequency Dropout dynamically drops tags based on their frequency in the dataset:
#   - Frequent tags (e.g., "1boy", "realistic") get higher dropout rates
#   - Rare tags (e.g., unique descriptors) get lower dropout rates
#   - Trigger tokens are never dropped
#
# This helps the model:
#   - Learn rare/unique features better
#   - Reduce overfitting on common tags
#   - Improve generalization
#
# Usage: bash Qwen-Image-TagFrequencyDropout.sh

# Model paths (update these to your local paths)
MODEL_ID="Qwen/Qwen-Image"

# Dataset configuration
DATASET_BASE_PATH="/path/to/your/dataset"
DATASET_METADATA_PATH="/path/to/your/metadata.json"

# Training configuration
OUTPUT_PATH="./outputs/qwen_image_tag_freq_dropout"
LORA_RANK=32
LEARNING_RATE=1e-4
NUM_EPOCHS=2

# Tag Frequency Dropout configuration
TAG_DROPOUT_MIN_RATE=0.0    # Min dropout for rare tags
TAG_DROPOUT_MAX_RATE=0.5    # Max dropout for frequent tags
TRIGGER_TOKEN="your_trigger"  # Never dropped

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
    --use_tag_frequency_dropout \
    --tag_dropout_min_rate ${TAG_DROPOUT_MIN_RATE} \
    --tag_dropout_max_rate ${TAG_DROPOUT_MAX_RATE} \
    --tag_dropout_trigger_tokens "${TRIGGER_TOKEN}" \
    --caption_key "prompt" \
    --remove_prefix_in_ckpt "pipe.dit."

echo "Training completed! Output saved to ${OUTPUT_PATH}"

