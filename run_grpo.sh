#!/bin/bash

# Shell script to run GRPO trainer
# Usage: ./run_grpo.sh [SFT_MODEL_PATH] [REWARD_MODEL_PATH]
# Example: ./run_grpo.sh /path/to/sft_model /path/to/reward_model

# Set default paths if not provided as arguments
SFT_MODEL_PATH="${1:-/workspace/dgx_spark_summary_from_human_feedback/sft_output/checkpoint_809}"
REWARD_MODEL_PATH="${2:-/workspace/dgx_spark_summary_from_human_feedback/reward_output/checkpoint_basemodel_qwen_2.5_1.5b_final_step}"

# Check if paths are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Warning: Using default paths. Provide SFT_MODEL_PATH and REWARD_MODEL_PATH as arguments to override."
    echo "Usage: $0 <SFT_MODEL_PATH> <REWARD_MODEL_PATH>"
    echo ""
fi

# Run GRPO trainer with specified parameters
python3 -m dgx_spark_summary_from_human_feedback.grpo \
    --sft_model_path "$SFT_MODEL_PATH" \
    --reward_model_path "$REWARD_MODEL_PATH" \
    --batch_size 16 \
    --mini_batch_size 64 \
    --num_responses_per_group 8 \
    --output_dir grpo_output_full_run_1 \
    --val_dir grpo_val_output_full_run_1 \
    --output_checkpoint_prefix grpo_full_run_1 \
    --eval_dataset_size 16 \
    --wandb_project summary_from_human_feedback_grpo \
    --wandb_run_name grpo_full_run_1 \
    --update_per_rollout 4

