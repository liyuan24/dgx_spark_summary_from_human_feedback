import argparse
from dataclasses import dataclass
import math
from typing import Optional
from datasets import Dataset, load_dataset
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PretrainedConfig,
)


@dataclass
class RewardConfig:
    # training configs
    num_train_epochs: int = 1
    batch_size: int = 8
    grad_clip: float = 1.0
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 10
    lr_decay_steps: int = 1000
    train_dataset_size: Optional[int] = None
    seed: int = 42
    dataset: str = "seangogo/processed_tldr_comparison_dataset_20251102_065554"

    # evaluation
    num_eval_epochs: int = 1
    eval_interval: int = 10
    eval_dataset_size: Optional[int] = None

    # wandb
    wandb_log: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # optimizer
    use_adamw_fused: bool = True
    gradient_accumulation_steps: int = 8
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.95
    adamw_weight_decay: float = 0.1
    use_eight_bit_optimizer: bool = False

    # saving
    output_dir: str = "reward_output"  # directory to save the reward model


class ScalarModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-0.5B",
        base_config: PretrainedConfig = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B"),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.base_config = base_config


class RewardModel(PreTrainedModel):
    config_class = ScalarModelConfig

    def __init__(self, config: ScalarModelConfig):
        super().__init__(config)
        self.config = config
        self.lm_backbone = AutoModel.from_pretrained(
            config.base_model, config=config.base_config, trust_remote_code=True
        )
        self.scalar_head = nn.Linear(config.base_config.hidden_size, 1)
        nn.init.normal_(
            self.scalar_head.weight,
            mean=0.0,
            std=1.0 / math.sqrt(config.base_config.hidden_size + 1),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = self.lm_backbone(input_ids)
        # [batch_size, sequence_length, 1]
        scalar_output = self.scalar_head(output.last_hidden_state)
        # [batch_size, sequence_length]
        return scalar_output.squeeze(-1)


def get_reward(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    chosen_and_rejected_query_responses: torch.Tensor,
) -> torch.Tensor:
    """
    Forward pass the chosen and rejected query responses to get the reward. The rewards are only extracted
    at eos tokens for each example in the batch. When no eos token is found, the reward is set to -1.

    Args:
        model: the model to use for reward calculation
        tokenizer: the tokenizer to use for reward calculation
        chosen_and_rejected_query_responses: [batch_size, sequence_length] tensor of chosen and rejected query responses

    Returns:
        the reward of shape [batch_size, ]
    """
    # [batch_size, sequence_length, 1]
    model_output = model(chosen_and_rejected_query_responses)
    # find the index before the first padding token
    eos_token_id = tokenizer.eos_token_id
    # [batch_size, 1]
    eos_mask = (
        (chosen_and_rejected_query_responses == eos_token_id)
        .float()
        .argmax(dim=1, keepdim=True)
    )
    # [batch_size, 1]
    has_eos = eos_mask.any(dim=1, keepdim=True)
    # [batch_size, 1]
    # get the reward at the eos token for each example
    reward = torch.gather(model_output, 1, eos_mask)
    # set the reward to -1 if no eos token is found, [batch_size, 1]
    reward = torch.where(has_eos, reward, torch.full_like(reward, -1))
    # [batch_size, ]
    return reward.squeeze(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training a reward model on a TLDR comparison dataset"
    )

    parser.add_argument(
        "--model_path", type=str, default="Qwen/Qwen2.5-0.5B", help="reward model path"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="seangogo/processed_tldr_comparison_dataset_20251102_065554",
        help="Comparison dataset to use for training and evaluation",
    )
    parser.add_argument(
        "--use_torch_compile",
        action="store_true",
        help="Use torch.compile() for faster training",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--train_dataset_size", type=int, default=None, help="Training dataset size"
    )
    parser.add_argument(
        "--eval_dataset_size", type=int, default=1000, help="Evaluation dataset size"
    )

    # Training configs
    parser.add_argument(
        "--output_dir",
        type=str,
        default="reward_output",
        help="Directory to save the reward model",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-6, help="Learning rate"
    )
    parser.add_argument(
        "--min_learning_rate", type=float, default=3e-7, help="Minimum learning rate"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=10, help="Number of warmup steps"
    )
    parser.add_argument(
        "--lr_decay_steps",
        type=int,
        default=800,
        help="Number of steps for learning rate decay",
    )

    # Evaluation configs
    parser.add_argument(
        "--num_eval_epochs", type=int, default=1, help="Number of evaluation epochs"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=10, help="Evaluate every N steps"
    )

    # Wandb configs
    parser.add_argument(
        "--wandb_log", action="store_true", default=True, help="Enable wandb logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="summary_from_human_feedback_sft",
        help="summary_from_human_feedback_sft",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default="4th run", help="run name"
    )

    # Optimizer configs
    parser.add_argument(
        "--use_adamw_fused",
        action="store_true",
        default=True,
        help="Use fused AdamW optimizer (default: True)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--adamw_beta1", type=float, default=0.9, help="AdamW beta1 parameter"
    )
    parser.add_argument(
        "--adamw_beta2", type=float, default=0.95, help="AdamW beta2 parameter"
    )
    parser.add_argument(
        "--adamw_weight_decay", type=float, default=0.1, help="AdamW weight decay"
    )
    parser.add_argument(
        "--use_eight_bit_optimizer",
        action="store_true",
        default=True,
        help="Use 8-bit optimizer (default: True)",
    )

    return parser.parse_args()


def process_dataset(dataset: Dataset, dataset_size: Optional[int] = None) -> Dataset:
    if dataset_size is not None:
        dataset = dataset.select(range(dataset_size))
    torch_dataset = dataset.with_format(
        type="torch",
        columns=[
            "query_and_chosen_response_tokens",
            "query_and_rejected_response_tokens",
        ],
    )
    return torch_dataset


if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path
    model_config = ScalarModelConfig(
        base_model=model_path,
        base_config=AutoConfig.from_pretrained(model_path),
    )
    reward_model = RewardModel(model_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    raw_train_dataset = load_dataset(args.dataset, split="train")
    raw_validation_dataset = load_dataset(args.dataset, split="validation")
    train_dataset = process_dataset(raw_train_dataset, args.train_dataset_size)
    validation_dataset = process_dataset(raw_validation_dataset, args.eval_dataset_size)
    example = train_dataset[0:2]
    query_chosen_response_tokens = example["query_and_chosen_response_tokens"]
    query_rejected_response_tokens = example["query_and_rejected_response_tokens"]
    input_tokens = torch.concat(
        [query_chosen_response_tokens, query_rejected_response_tokens], dim=0
    )
    rewards = get_reward(reward_model, tokenizer, input_tokens)
    print(f"rewards: {rewards}, shape: {rewards.shape}")
