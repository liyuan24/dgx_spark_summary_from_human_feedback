import argparse
from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch
import torch.nn as nn
import os

from .reward import RewardModel, ScalarModelConfig


def load_reward_model_with_safetensors(model_path: str):
    """
    Load a reward model from a checkpoint path.

    Args:
        model_path: Path to the checkpoint directory containing config.json and model weights

    Returns:
        The loaded RewardModel
    """

    model_config = ScalarModelConfig(
        base_model="Qwen/Qwen2.5-1.5B",
        base_config=AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B"),
    )
    reward_model = RewardModel(model_config)
    # Now load the saved state dict manually to avoid meta tensor issues
    from safetensors.torch import load_file as safe_load_file
    import glob

    # Load the saved state dict - handle both single and sharded safetensors files
    state_dict = None

    # load the local saved hf model
    print(glob.glob(os.path.join(model_path, "model-*-of-*.safetensors")))
    sharded_files = sorted(
        glob.glob(os.path.join(model_path, "model-*-of-*.safetensors"))
    )
    print(f"Loading {len(sharded_files)} sharded safetensors files")
    state_dict = {}
    for shard_file in sharded_files:
        print(f"Loading {os.path.basename(shard_file)}")
        shard_dict = safe_load_file(shard_file)
        state_dict.update(shard_dict)

    # Load state dict into model
    reward_model.load_state_dict(state_dict, strict=False)

    # Move to device and dtype
    if torch.cuda.is_available():
        reward_model = reward_model.to("cuda")
    print(f"reward model dtype: {reward_model.dtype}")
    return reward_model


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def load_model(model_path: str):
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, device_map="auto", dtype=torch.bfloat16
    )
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--push_to_hf", action="store_true", default=False)
    parser.add_argument("--hf_repo_id", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    reward_model = load_model(args.model_path)
    tokenizer = load_tokenizer(args.model_path)
    if args.push_to_hf:
        if args.hf_repo_id is None:
            raise ValueError("hf_repo_id is required when push_to_hf is True")
        reward_model.push_to_hub(args.hf_repo_id)
        tokenizer.push_to_hub(args.hf_repo_id)
