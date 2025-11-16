import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import math
import os
import random
from typing import List, Optional, Tuple
from datasets import Dataset, load_dataset
from datasets.table import np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PretrainedConfig,
)
import bitsandbytes as bnb
from .reward import get_reward

"""
python3 -m dgx_spark_summary_from_human_feedback.grpo --sft_model_path /workspace/dgx_spark_summary_from_human_feedback/sft_output/checkpoint_809 --reward_model_path /workspace/dgx_spark_summary_from_human_feedback/reward_output/checkpoint_basemodel_qwen_2.5_1.5b_final_step
"""


@dataclass
class GrpoConfig:
    # training configs
    num_train_epochs: int = 1
    batch_size: int = (
        16  # number of prompts to sample rollout data, effective batch size is batch_size * num_responses_per_group
    )
    mini_batch_size: int = (
        64  # number of prompt + responses pair to update the policy model weights
    )
    micro_batch_size: int = (
        8  # number of prompt + responses pair to do gradient accumulation
    )
    grad_clip: float = 1.0
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    train_dataset_size: Optional[int] = None
    seed: int = 42
    dataset: str = "seangogo/processed_tldr_sft_dataset_20251029_045736"

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
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.95
    adamw_weight_decay: float = 0.1
    use_eight_bit_optimizer: bool = False

    # saving
    output_dir: str = None  # directory to save the reward model
    output_checkpoint_prefix: str = None
    val_dir: str = None
    save_interval: int = 50

    # GRPO hyperparameters
    update_per_rollout: int = 4  # number of updates per rollout data collection
    clip_ratio: float = 0.2  # grpo importance sampling ratio clip range
    clip_ratio_c: float = 3.0  # Dual Clip loss clip ratio
    kl_coeff: float = 0.05  # kl coefficient for grpo
    kl_penalty_mode: str = "k3"  # kl penalty mode
    response_length: int = 63  # number of tokens generated during rollout
    temperature: float = 0.7  # temperature for rollout generation
    num_responses_per_group: int = 3  # number of responses per group
    normalize_adv_by_std_of_group: bool = (
        True  # normalize the advantages by the standard deviation of the group
    )
    no_eos_penalty: float = -1.0  # penalty for the response when there is no EOS token
    loss_agg_mode: str = "seq-mean-token-mean"  # loss aggregation mode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training a DPO model on a TLDR comparison dataset"
    )
    parser.add_argument("--sft_model_path", type=str, required=True)
    parser.add_argument("--reward_model_path", type=str, required=True)

    parser.add_argument(
        "--dataset",
        type=str,
        default="seangogo/processed_tldr_comparison_dataset_20251102_065554",
        help="Comparison dataset to use for training and evaluation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--train_dataset_size", type=int, default=None, help="Training dataset size"
    )
    parser.add_argument(
        "--eval_dataset_size", type=int, default=10, help="Evaluation dataset size"
    )

    # Training configs
    parser.add_argument(
        "--num_train_epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size when sampling rollout prompts",
    )
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=64,
        help="Batch size when updating policy model",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=8,
        help="Batch size when doing gradient accumulation",
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")

    # Evaluation configs
    parser.add_argument(
        "--num_eval_epochs", type=int, default=1, help="Number of evaluation epochs"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=100, help="Evaluate every N steps"
    )

    # Wandb configs
    parser.add_argument(
        "--wandb_log", action="store_true", default=True, help="Enable wandb logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="summary_from_human_feedback_dpo",
        help="the wandb project name",
    )
    parser.add_argument("--wandb_run_name", type=str, default=None, help="run name")

    # Optimizer configs
    parser.add_argument(
        "--use_adamw_fused",
        action="store_true",
        default=True,
        help="Use fused AdamW optimizer (default: True)",
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
        default=False,
        help="Use 8-bit optimizer (default: True)",
    )

    # output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the reward model",
    )
    parser.add_argument(
        "--output_checkpoint_prefix",
        type=str,
        default=None,
        help="Prefix for the checkpoint name",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default="grpo_val_outputs",
        help="Directory to save the validation outputs",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=50,
        help="Save the model every N steps",
    )

    # GRPO hyperparameters
    parser.add_argument(
        "--update_per_rollout",
        type=int,
        default=4,
        help="Number of updates per rollout data collection",
    )
    parser.add_argument(
        "--clip_ratio",
        type=float,
        default=0.2,
        help="GRPO importance sampling ratio clip range",
    )
    parser.add_argument(
        "--clip_ratio_c",
        type=float,
        default=3.0,
        help="Dual Clip loss clip ratio",
    )
    parser.add_argument(
        "--kl_coeff", type=float, default=0.05, help="KL coefficient for GRPO"
    )
    parser.add_argument(
        "--kl_penalty_mode",
        type=str,
        default="k3",
        help="KL penalty mode",
    )
    parser.add_argument(
        "--num_responses_per_group",
        type=int,
        default=8,
        help="Number of responses per group",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for rollout generation",
    )
    parser.add_argument(
        "--response_length",
        type=int,
        default=63,
        help="Number of tokens generated during rollout",
    )
    parser.add_argument(
        "--normalize_adv_by_std_of_group",
        action="store_true",
        default=True,
        help="Normalize the advantages by the standard deviation of the group",
    )
    parser.add_argument(
        "--no_eos_penalty",
        type=float,
        default=-1.0,
        help="Penalty for the response when there is no EOS token",
    )
    parser.add_argument(
        "--loss_agg_mode",
        type=str,
        default="seq-mean-token-mean",
        help="Loss aggregation mode",
    )

    return parser.parse_args()


def masked_mean(
    values: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None
) -> torch.Tensor:
    """
    Sum the values along the specified dimension, ignoring the values where the mask is False.

    Args:
        values: shape: (batch_size, response_length)
        mask: shape: (batch_size, response_length)
        dim: the dimension to sum along
    Returns:
        The masked sum of the values.
    """
    return torch.sum(values * mask, dim=dim) / (torch.sum(mask, dim=dim) + 1e-7)


def disable_dropout(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0


def process_dataset(dataset: Dataset, dataset_size: Optional[int] = None) -> Dataset:
    if dataset_size is not None:
        dataset = dataset.select(range(dataset_size))
    torch_dataset = dataset.with_format(
        type="torch",
        columns=[
            "query_tokens",  # already left padded with pad_token_id
        ],
    )
    return torch_dataset


class GrpoTrainer:
    def __init__(
        self,
        policy_model: PreTrainedModel,
        ref_model: PreTrainedModel,
        reward_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: GrpoConfig,
    ):
        self.policy_model = policy_model
        self.ref_model = torch.compile(ref_model, mode="max-autotune")
        self.ref_model.eval()
        self.reward_model = torch.compile(reward_model, mode="max-autotune")
        self.reward_model.eval()
        self.tokenizer = tokenizer
        self.config = config
        print(f"policy model device: {self.policy_model.device}")
        print(f"ref model device: {self.ref_model.device}")
        print(f"reward model device: {self.reward_model.device}")
        disable_dropout(self.policy_model)
        disable_dropout(self.ref_model)
        disable_dropout(self.reward_model)
        raw_train_dataset = load_dataset(args.dataset, split="train")
        raw_validation_dataset = load_dataset(args.dataset, split="validation")
        train_dataset = process_dataset(raw_train_dataset, args.train_dataset_size)
        validation_dataset = process_dataset(
            raw_validation_dataset, args.eval_dataset_size
        )
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        self.validation_dataloader = DataLoader(
            validation_dataset, batch_size=args.batch_size
        )
        self.total_training_steps = len(self.train_dataloader) * config.num_train_epochs
        self.warmup_steps = int(self.total_training_steps * config.warmup_ratio)
        self.config = config
        self.optimizer = self.setup_optimizer(config)
        self.generation_config = GenerationConfig(
            max_new_tokens=config.response_length,
            temperature=config.temperature + 1e-7,
            top_k=0,
            top_p=1.0,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        print(f"pad token id: {self.tokenizer.pad_token_id}")
        print(f"eos token id: {self.tokenizer.eos_token_id}")
        # for reproducibility
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True

    def setup_optimizer(self, config):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.policy_model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwis    no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": config.adamw_weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        if config.use_eight_bit_optimizer:
            # fuse is not supported
            optimizer = bnb.optim.AdamW8bit(
                optim_groups,
                lr=config.learning_rate,
                betas=(config.adamw_beta1, config.adamw_beta2),
            )
        else:
            optimizer = torch.optim.AdamW(
                optim_groups,
                lr=config.learning_rate,
                betas=(config.adamw_beta1, config.adamw_beta2),
                fused=config.use_adamw_fused,
            )
        return optimizer

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, step):
        # 1) linear warmup for warmup_iters steps
        if step < self.warmup_steps:
            return self.config.learning_rate * step / self.warmup_steps
        # # 2) if it > lr_decay_steps, return min learning rate
        # if step > self.config.lr_decay_steps:
        #     return self.config.min_learning_rate
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (step - self.warmup_steps) / (
            self.total_training_steps - self.warmup_steps
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return coeff * self.config.learning_rate

    def find_first_true_index(
        self, bools: torch.Tensor, dtype: torch.dtype = torch.long
    ) -> torch.Tensor:
        """
        Find the first true index in the boolean tensor.
        Find the minimum index of the first true value in each row.
        If all values are false, return the sequence length.
        Args:
            bools: A boolean tensor of shape (batch_size, sequence_length)
        Returns:
            a tensor of shape (batch_size,)
        """
        row_length = bools.shape[1]
        row_length_tensor = row_length * (~bools).type(dtype)
        # shape: (1, row_length)
        index_tensor = torch.arange(
            row_length, dtype=dtype, device=bools.device
        ).unsqueeze(0)
        all_tensor = row_length_tensor + index_tensor
        # when there are no true values, the minimum value is the sequence length at index 0
        # when there are true values, the minimum value is the index of the first true value
        return torch.min(all_tensor, dim=1).values

    # take reference from verl
    # https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py#L265
    @torch.no_grad()
    def compute_advantages(
        self,
        token_level_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        normalize_adv_by_std_of_group: bool = True,
    ) -> torch.Tensor:
        """
        Compute the advantages from the token level rewards which has reward only at the EOS token.

        Args:
            token_level_rewards: shape: (batch_size, response_length)
            response_mask: shape: (batch_size, response_length)
            epsilon: a small value to avoid division by zero
            normalize_adv_by_std_of_group: whether to normalize the advantages by the standard deviation of rewards of the group of responses
        Returns:
            advantages: shape: (batch_size, response_length)
        """
        scores = token_level_rewards.sum(dim=-1)
        bs = scores.shape[0]
        assert (
            bs % self.config.num_responses_per_group == 0
        ), f"batch size {bs} must be divisible by num_responses_per_group {self.config.num_responses_per_group}"
        group_index = np.arange(bs) // self.config.num_responses_per_group
        group_scores = defaultdict(list)
        group_scores_mean = {}
        group_scores_std = {}
        for i in range(bs):
            group_scores[group_index[i]].append(scores[i])
        for group in group_scores:
            group_tensor = torch.stack(group_scores[group])
            group_scores_mean[group] = group_tensor.mean()
            # notice by default, PyTorch std() is divided by N-1 instead of N
            group_scores_std[group] = group_tensor.std()
        for i in range(bs):
            group = group_index[i]
            scores[i] = scores[i] - group_scores_mean[group]
            if normalize_adv_by_std_of_group:
                scores[i] = scores[i] / (group_scores_std[group] + 1e-7)
        # broadcast the advantages to the response length
        # the response tokens will have the same advantage as the EOS token
        return scores.unsqueeze(-1) * response_mask

    @torch.no_grad()
    def get_token_level_rewards(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        rollout_data: torch.Tensor,
        response_mask: torch.Tensor,
        context_length: int,
    ) -> torch.Tensor:
        """
        Get the rewards for the rollout data.
        The rollout data is the query token ids and the generated response tokens.
        The context length is the length of the query token ids.

        Args:
            model: the reward model
            tokenizer: the tokenizer
            rollout_data: shape: (batch_size, context_length + response_length)
            response_mask: shape: (batch_size, response_length)
            context_length: the length of the query token ids
        Returns:
            token_level_rewards: shape: (batch_size, response_length)
        """
        bs = rollout_data.shape[0]
        attention_mask = rollout_data.ne(tokenizer.pad_token_id)
        # shape: (batch_size, sequence_length)
        rewards = model(rollout_data, attention_mask=attention_mask)
        # shape: (batch_size, response_length)
        response_rewards = rewards[:, context_length:]
        token_level_rewards = torch.zeros_like(response_rewards)
        valid_lengths = response_mask.sum(dim=-1)
        token_level_rewards[torch.arange(bs), valid_lengths - 1] = response_rewards[
            torch.arange(bs), valid_lengths - 1
        ]
        is_eos = rollout_data.eq(tokenizer.eos_token_id)
        first_eos_index = self.find_first_true_index(is_eos) - context_length
        response_length = response_rewards.shape[1]
        no_eos_mask = first_eos_index == response_length
        # when there is no EOS token, the reward is -1.0 to penalize the response
        token_level_rewards[no_eos_mask, valid_lengths[no_eos_mask] - 1] = (
            self.config.no_eos_penalty
        )
        return token_level_rewards

    @torch.no_grad()
    def generate_rollout(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        query_token_ids: torch.Tensor,
        generation_config: GenerationConfig,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a rollout of responses.
        The query_token_ids are left padded with pad_token_id.
        The generated responses will be right padded with pad_token_id after the EOS token.
        """
        context_length = query_token_ids.shape[1]
        attention_mask = query_token_ids.ne(tokenizer.pad_token_id)
        model_output = model.generate(
            query_token_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )
        response_tokens = model_output.sequences[:, context_length:]
        response_mask = response_tokens.ne(tokenizer.pad_token_id)
        # the scores are already divided by the temperature
        scores = torch.stack(model_output.scores, dim=1)
        log_probs = scores.log_softmax(dim=-1)
        log_probs = log_probs.gather(
            dim=2, index=response_tokens.unsqueeze(-1)
        ).squeeze(-1)
        return (
            torch.concat([query_token_ids, response_tokens], dim=1),
            log_probs,
            response_mask,
        )

    def forward(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        temperature: float,
    ) -> torch.Tensor:
        """
        Get the logits of the model.

        Args:
            model: The model to forward pass.
            input_ids: The input ids to forward pass, shape: (batch_size, sequence_length)
            tokenizer: The tokenizer to use.
        Returns:
            The logprobs of the model.
        """
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        # shape: (batch_size, sequence_length-1, vocab_size)
        outputs = model(
            input_ids[:, :-1], attention_mask=attention_mask, return_dict=True
        )
        # shape: (batch_size, sequence_length-1, vocab_size)
        logits = outputs.logits
        logits = logits / temperature
        logprobs = logits.log_softmax(dim=-1)
        # shape: (batch_size, sequence_length-1)
        logprobs = torch.gather(
            logprobs, dim=2, index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        return logprobs[:, context_length - 1 :]

    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        loss_agg_mode: str = "token-mean",
    ) -> torch.Tensor:
        """
        Compute the PPO loss.

        Args:
            log_probs: shape: (batch_size, response_length)
            old_log_probs: shape: (batch_size, response_length)
            advantages: shape: (batch_size, response_length)
            response_mask: shape: (batch_size, response_length)
            loss_agg_mode: the mode to aggregate the losses
        Returns:
            The PPO loss.
        """
        log_ratio = log_probs - old_log_probs
        # clamp ratio to avoid numerical instability
        log_ratio = torch.clamp(log_ratio, min=-20, max=20)
        ratio = torch.exp(log_ratio)
        pg_losses1 = -advantages * ratio
        clip_ratio = self.config.clip_ratio
        # https://arxiv.org/pdf/1912.09729
        clip_ratio_c = self.config.clip_ratio_c
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        pg_losses2 = -advantages * clipped_ratio
        clipped_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
        pg_losses3 = -advantages * clip_ratio_c
        clipped_pg_losses2 = torch.minimum(clipped_pg_losses1, pg_losses3)
        pg_losses = torch.where(advantages < 0, clipped_pg_losses2, clipped_pg_losses1)
        loss = self.agg_losses(pg_losses, response_mask, loss_agg_mode)
        return loss

    def agg_losses(
        self, losses: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str
    ):
        """
        Aggregate token level losses

        Args:
            losses: shape: (batch_size, response_length)
            loss_mask: shape: (batch_size, response_length)
            loss_agg_mode: the mode to aggregate the losses
        Returns:
            The aggregated loss.
        """
        if loss_agg_mode == "token-mean":
            # every token contribute equally to the loss
            return (losses * loss_mask).sum() / loss_mask.sum()
        elif loss_agg_mode == "seq-mean-token-sum":
            # the sequence level loss is the sum of token losses
            seq_losses = (losses * loss_mask).sum(dim=-1)
            seq_mask = (loss_mask.sum(dim=-1) > 0).float()
            return (seq_losses * seq_mask).sum() / seq_mask.sum()
        elif loss_agg_mode == "seq-mean-token-mean":
            # the sequence level loss is the mean of token losses
            # GRPO paper uses this mode https://arxiv.org/pdf/2402.03300
            seq_losses = (losses * loss_mask).sum(dim=-1) / (
                loss_mask.sum(dim=-1) + 1e-7
            )
            seq_mask = (loss_mask.sum(dim=-1) > 0).float()
            return (seq_losses * seq_mask).sum() / seq_mask.sum()
        else:
            raise ValueError(f"Invalid loss aggregation mode: {loss_agg_mode}")

    def kl_penalty(
        self, log_probs: torch.Tensor, ref_log_probs: torch.Tensor, kl_penalty_mode: str
    ):
        """
        Compute the KL penalty.
        J. Schulman. Approximating kl divergence, 2020.
        URL: http://joschu.net/blog/kl-approx.html.

        Args:
            log_probs: shape: (batch_size, response_length)
            ref_log_probs: shape: (batch_size, response_length)
            kl_penalty_mode: the mode to compute the KL penalty
        Returns:
            The KL penalty per token. shape: (batch_size, response_length)
        """
        if kl_penalty_mode == "k1":
            return log_probs - ref_log_probs
        if kl_penalty_mode == "k2":
            return 0.5 * (log_probs - ref_log_probs).square()
        if kl_penalty_mode == "k3":
            kl = log_probs - ref_log_probs
            kl = torch.clamp(kl, min=-20, max=20)
            ratio = torch.exp(kl)
            kld = (ratio - kl - 1).contiguous()
            return torch.clamp(kld, min=-10, max=10)
        raise NotImplementedError(f"Invalid KL penalty mode: {kl_penalty_mode}")

    def get_grpo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        loss_agg_mode: str,
        kl_penalty_mode: str,
    ) -> torch.Tensor:
        """
        Get the GRPO loss.
        Args:
            log_probs: shape: (batch_size, response_length)
            old_log_probs: shape: (batch_size, response_length)
            ref_log_probs: shape: (batch_size, response_length)
            advantages: shape: (batch_size, response_length)
            loss_agg_mode: the mode to aggregate the losses
            kl_penalty_mode: the mode to compute the KL penalty
        Returns:
            The GRPO loss.
        """
        policy_loss = self.compute_policy_loss(
            log_probs, old_log_probs, advantages, response_mask, loss_agg_mode
        )
        # per token KL loss
        kld = self.kl_penalty(log_probs, ref_log_probs, kl_penalty_mode)
        kl_loss = self.agg_losses(kld, response_mask, loss_agg_mode)
        return policy_loss + self.config.kl_coeff * kl_loss

    def train(self):
        if self.config.wandb_log:
            import wandb

            assert self.config.wandb_project is not None, "Wandb project is required"
            assert self.config.wandb_run_name is not None, "Wandb run name is required"
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=asdict(self.config),
                sync_tensorboard=True,
            )
        global_step = 0
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=global_step,
            desc="Training Progress",
        )
        global_step = 1
        for epoch in range(self.config.num_train_epochs):
            for batch in self.train_dataloader:
                query_token_ids = batch["query_tokens"].to(policy_model.device)
                # repeat the query token ids to sample multiple responses for each prompt
                query_token_ids = torch.repeat_interleave(
                    query_token_ids, self.config.num_responses_per_group, dim=0
                )
                context_length = query_token_ids.shape[1]
                # collect rollout data
                # rollout_data: shape (batch_size, context_length + response_length)
                # old_policy_logprobs: shape (batch_size, response_length)
                rollout_data, old_log_probs, response_mask = self.generate_rollout(
                    self.policy_model,
                    self.tokenizer,
                    query_token_ids,
                    self.generation_config,
                )
                print(f"Epoch {epoch + 1}, Global Step: {global_step}, rollout data with shape {rollout_data.shape} finished")
                with torch.no_grad():
                    # shape: (batch_size, response_length)
                    ref_log_probs = self.forward(
                        self.ref_model,
                        rollout_data,
                        self.tokenizer,
                        context_length=context_length,
                        temperature=self.config.temperature + 1e-7,
                    )
                print(f"Epoch {epoch + 1}, Global Step: {global_step}, ref log probs with shape {ref_log_probs.shape} finished")
                # shape: (batch_size, response_length)
                token_level_rewards = self.get_token_level_rewards(
                    self.reward_model,
                    self.tokenizer,
                    rollout_data,
                    response_mask,
                    context_length,
                )
                print(f"Epoch {epoch + 1}, Global Step: {global_step}, token level rewards with shape {token_level_rewards.shape} finished")
                mean_rewards = token_level_rewards.sum(dim=-1).mean().item()
                print(f"Epoch {epoch + 1}, Global Step: {global_step}, mean rewards: {mean_rewards}")
                # shape: (batch_size, response_length)
                advantages = self.compute_advantages(
                    token_level_rewards,
                    response_mask,
                    self.config.normalize_adv_by_std_of_group,
                )
                all_batch_size = query_token_ids.shape[0]
                gradient_accumulations = (
                    self.config.mini_batch_size // self.config.micro_batch_size
                )
                is_last_step = global_step >= self.total_training_steps
                for i in tqdm(range(self.config.update_per_rollout), desc="Updating the policy model in mini-batches"):
                    # shuffle the rollout data at the start of each epoch
                    b_inds = np.random.permutation(all_batch_size)
                    for batch_start in range(0, all_batch_size, args.mini_batch_size):
                        batch_end = min(
                            batch_start + args.mini_batch_size, all_batch_size
                        )
                        mb_inds = b_inds[batch_start:batch_end]
                        mb_old_log_probs = old_log_probs[mb_inds]
                        mb_ref_log_probs = ref_log_probs[mb_inds]
                        mb_advantages = advantages[mb_inds]
                        mb_query_responses = rollout_data[mb_inds]
                        mb_response_mask = response_mask[mb_inds]
                        # zero the gradients for each mini-batch
                        self.optimizer.zero_grad(set_to_none=True)
                        step_loss = 0
                        for micro_batch_start in range(
                            0, mb_query_responses.shape[0], self.config.micro_batch_size
                        ):
                            micro_batch_end = min(
                                micro_batch_start + self.config.micro_batch_size,
                                mb_query_responses.shape[0],
                            )
                            cur_query_responses = mb_query_responses[
                                micro_batch_start:micro_batch_end
                            ]
                            cur_old_log_probs = mb_old_log_probs[
                                micro_batch_start:micro_batch_end
                            ]
                            cur_ref_log_probs = mb_ref_log_probs[
                                micro_batch_start:micro_batch_end
                            ]
                            cur_advantages = mb_advantages[
                                micro_batch_start:micro_batch_end
                            ]
                            cur_response_mask = mb_response_mask[
                                micro_batch_start:micro_batch_end
                            ]
                            cur_log_probs = self.forward(
                                self.policy_model,
                                cur_query_responses,
                                self.tokenizer,
                                context_length=context_length,
                                temperature=self.config.temperature,
                            )
                            pg_loss = self.get_grpo_loss(
                                cur_log_probs,
                                cur_old_log_probs,
                                cur_ref_log_probs,
                                cur_advantages,
                                cur_response_mask,
                                self.config.loss_agg_mode,
                                self.config.kl_penalty_mode,
                            )
                            loss_scale_factor = 1 / gradient_accumulations
                            loss = loss_scale_factor * pg_loss
                            step_loss += loss.detach().item()
                            loss.backward()
                    # after all the micro-batches are processed, clip the gradients
                    if self.config.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.policy_model.parameters(), self.config.grad_clip
                        )
                    lr = self.get_lr(global_step)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr
                    self.optimizer.step()
                    if self.config.wandb_log:
                        wandb.log(
                            {
                                "train_minibatch_loss": step_loss,
                                "train_batch_mean_reward": mean_rewards,
                                "learning_rate": lr,
                            }
                        )
                if is_last_step or global_step % self.config.eval_interval == 0:
                    val_avg_reward = self.run_evaluation(global_step)
                    print(
                        f"Epoch {epoch + 1}, Global Step: {global_step}, average val reward: {val_avg_reward:.4f}"
                    )
                    if self.config.wandb_log:
                        wandb.log(
                            {
                                "val_avg_reward": val_avg_reward,
                            }
                        )
                if is_last_step or global_step % self.config.save_interval == 0:
                    output_dir = os.path.join(
                        self.config.output_dir,
                        f"{self.config.output_checkpoint_prefix}_{global_step}",
                    )
                    self.save_checkpoint(
                        output_dir,
                        self.policy_model,
                        self.tokenizer,
                        self.optimizer,
                        global_step,
                        self.config,
                    )
                progress_bar.update(1)
                global_step += 1
            self.optimizer.zero_grad(set_to_none=True)
            progress_bar.close()

    def save_checkpoint(
        self,
        output_dir: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        optimizer: torch.optim.Optimizer,
        global_step: int,
        config: GrpoConfig,
    ):
        """
        Saves:
        - HF model weights + config (save_pretrained)
        - HF tokenizer (save_pretrained)
        - Training state (optimizer/scheduler/scaler/custom) to training_state.pt
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1) Save model + tokenizer in HF-native format
        model.save_pretrained(output_dir)  # saves pytorch_model.bin + config.json
        tokenizer.save_pretrained(
            output_dir
        )  # saves tokenizer.json, tokenizer_config.json, etc.

        # 2) Save training state as a small .pt next to them
        training_state = {
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "config": config,
        }
        torch.save(training_state, os.path.join(output_dir, "training_state.pt"))

    @torch.no_grad()
    def run_evaluation(self, global_step: int) -> float:
        print(f"Running evaluation at global step {global_step}")
        self.policy_model.eval()
        rewards = []
        for batch in tqdm(self.validation_dataloader, desc="Evaluating"):
            query_token_ids = batch["query_tokens"].to(self.policy_model.device)
            context_length = query_token_ids.shape[1]
            rollout_data, _, response_mask = self.generate_rollout(
                self.policy_model,
                self.tokenizer,
                query_token_ids,
                self.generation_config,
            )
            token_level_rewards = self.get_token_level_rewards(
                self.reward_model,
                self.tokenizer,
                rollout_data,
                response_mask,
                context_length,
            )
            token_level_rewards = token_level_rewards.sum(dim=-1)
            rewards.append(token_level_rewards)
            text_outputs = self.tokenizer.batch_decode(
                rollout_data, skip_special_tokens=True
            )
            # dump the text outputs to a file
            self.dump_val_outputs(text_outputs, token_level_rewards.tolist(), global_step)

        avg_reward = torch.stack(rewards).mean().item()
        self.policy_model.train()
        return avg_reward

    def dump_val_outputs(
        self, text_outputs: List[str], rewards: List[float], global_step: int
    ):
        os.makedirs(self.config.val_dir, exist_ok=True)
        with open(
            os.path.join(self.config.val_dir, f"val_outputs_step_{global_step}.txt"),
            "a",
        ) as f:
            for text_output, reward in zip(text_outputs, rewards):
                f.write(f"Reward: {reward:.4f}\n")
                f.write(text_output + "\n")
                f.write("=" * 100 + "\n")


if __name__ == "__main__":
    args = parse_args()
    config_dict = {
        "num_train_epochs": args.num_train_epochs,
        "dataset": args.dataset,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "mini_batch_size": args.mini_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "grad_clip": args.grad_clip,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "train_dataset_size": args.train_dataset_size,
        "num_eval_epochs": args.num_eval_epochs,
        "eval_interval": args.eval_interval,
        "eval_dataset_size": args.eval_dataset_size,
        "wandb_log": args.wandb_log,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
        "use_adamw_fused": args.use_adamw_fused,
        "adamw_beta1": args.adamw_beta1,
        "adamw_beta2": args.adamw_beta2,
        "adamw_weight_decay": args.adamw_weight_decay,
        "use_eight_bit_optimizer": args.use_eight_bit_optimizer,
        "output_dir": args.output_dir,
        "output_checkpoint_prefix": args.output_checkpoint_prefix,
        "val_dir": args.val_dir,
        "save_interval": args.save_interval,
        "update_per_rollout": args.update_per_rollout,
        "clip_ratio": args.clip_ratio,
        "clip_ratio_c": args.clip_ratio_c,
        "kl_coeff": args.kl_coeff,
        "kl_penalty_mode": args.kl_penalty_mode,
        "num_responses_per_group": args.num_responses_per_group,
        "temperature": args.temperature,
        "response_length": args.response_length,
        "normalize_adv_by_std_of_group": args.normalize_adv_by_std_of_group,
        "no_eos_penalty": args.no_eos_penalty,
        "loss_agg_mode": args.loss_agg_mode,
    }
    config = GrpoConfig(**config_dict)
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    dataset = load_dataset(args.dataset, split="train")
    dataset = process_dataset(dataset, dataset_size=args.train_dataset_size)
    trainer = GrpoTrainer(
        policy_model=policy_model,
        ref_model=reference_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        config=config,
    )
    trainer.train()
