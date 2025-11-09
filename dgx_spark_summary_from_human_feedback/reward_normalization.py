import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
import tqdm
from transformers import AutoModel, AutoTokenizer
import torch

from .reward import get_reward


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--sft_dataset_path", type=str, required=True)
    parser.add_argument("--push_model_to_hf", action="store_true", default=False)
    parser.add_argument("--hf_model_repo_id", type=str, required=False, default=None)
    parser.add_argument("--push_dataset_to_hf", action="store_true", default=False)
    return parser.parse_args()


def load_model(model_path: str):
    """
    Load the reward model and tokenizer from Hugging Face.

    Args:
        model_path: Hugging Face model identifier

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model and tokenizer from {model_path}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, device_map="auto", dtype=torch.bfloat16
    )
    print(f"Model loaded successfully!")

    return model, tokenizer


if __name__ == "__main__":
    args = parse_args()
    reward_model, tokenizer = load_model(args.model_path)
    reward_model.eval()  # set model to evaluation mode
    # with no gradient graph and compiled model, we can achieve roughly 25% speedup in inference
    torch.set_grad_enabled(False)  # disable gradient computation to speed up inference
    reward_model = torch.compile(
        reward_model, mode="max-autotune"
    )  # use optimized kernels for inference
    sft_dataset = load_dataset(args.sft_dataset_path, split="train")
    sft_dataset_tensor = sft_dataset.with_format(
        "torch", columns=["query_and_response_tokens"]
    )
    sft_dataloader = DataLoader(sft_dataset_tensor, batch_size=8, shuffle=False)
    n = 0
    avg_reward = 0.0
    rewards = []
    for batch in tqdm.tqdm(sft_dataloader):
        query_and_response_tokens = batch["query_and_response_tokens"].to(
            reward_model.device, non_blocking=True
        )
        batch_rewards = get_reward(reward_model, tokenizer, query_and_response_tokens)
        rewards.extend(batch_rewards.tolist())
        current_sum_reward = batch_rewards.sum().item()
        total_sum_reward = avg_reward * n + current_sum_reward
        n += batch_rewards.shape[0]
        avg_reward = total_sum_reward / n
    print(f"Average reward: {avg_reward}")
    reward_model.config.bias_value = avg_reward
    if args.push_model_to_hf:
        if args.hf_model_repo_id is None:
            raise ValueError("hf_repo_id is required when push_to_hf is True")
        print(f"Pushing reward model to Hugging Face...")
        reward_model.push_to_hub(args.hf_model_repo_id)
        tokenizer.push_to_hub(args.hf_model_repo_id)
        print(f"Reward model pushed to Hugging Face successfully!")
    if args.push_dataset_to_hf:
        sft_dataset = sft_dataset.add_column("reward", rewards)
        print(f"Pushing dataset to Hugging Face...")
        sft_dataset.push_to_hub(f"{args.sft_dataset_path}_with_rewards", split="train")
        print(f"Dataset pushed to Hugging Face successfully!")
