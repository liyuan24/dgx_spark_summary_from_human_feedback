import argparse
from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch
import torch.nn as nn
import os

from .reward import RewardModel, ScalarModelConfig


def load_model(model_path: str):
    """
    Load a reward model from a checkpoint path.

    Args:
        model_path: Path to the checkpoint directory containing config.json and model weights

    Returns:
        The loaded RewardModel
    """

    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, device_map="auto", dtype=torch.bfloat16
    )
    return model


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
    model = load_model(args.model_path)
    tokenizer = load_tokenizer(args.model_path)
    if args.push_to_hf:
        if args.hf_repo_id is None:
            raise ValueError("hf_repo_id is required when push_to_hf is True")
        model.push_to_hub(args.hf_repo_id)
        tokenizer.push_to_hub(args.hf_repo_id)
