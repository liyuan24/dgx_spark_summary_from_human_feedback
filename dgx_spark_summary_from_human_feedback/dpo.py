import argparse
from dataclasses import asdict, dataclass
import os
import random
from typing import Optional, Tuple

from datasets import Dataset, load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import numpy as np
import math
import bitsandbytes as bnb
from tqdm import tqdm
import torch.nn.functional as F


@dataclass
class DPOConfig:
    # training configs
    num_train_epochs: int = 1
    batch_size: int = 8
    grad_clip: float = 1.0
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
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
    output_dir: str = None  # directory to save the reward model
    output_checkpoint_prefix: str = None

    # DPO hyperparameters
    beta: float = 0.3
    label_smoothing: float = 0.1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training a DPO model on a TLDR comparison dataset"
    )
    parser.add_argument("--sft_model_path", type=str, required=True)

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
        "--num_train_epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
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
        "--eval_interval", type=int, default=10, help="Evaluate every N steps"
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
        "--gradient_accumulation_steps",
        type=int,
        default=16,
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

    # DPO hyperparameters
    parser.add_argument("--beta", type=float, default=0.3, help="Beta for DPO loss")
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="Label smoothing for DPO"
    )

    return parser.parse_args()


def forward(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    query_token_length_without_padding: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the logits of the chosen and rejected responses.
    Args:
        model: The model to forward pass.
        input_ids: The input ids to forward pass, shape: (batch_size, sequence_length)
        query_token_length_without_padding: The length of the query without padding, shape: (batch_size,)
        tokenizer: The tokenizer to use.
    Returns:
        The logits of the chosen and rejected responses.
    """
    # Create a new tensor from input_ids and mask out query tokens as pad_token_id
    # For each batch element, replace the first query_token_length_without_padding tokens with pad_token_id
    #
    # Example:
    #   sequence_length = 10
    #   query_token_length_without_padding = [3, 5, 2]  # batch_size=3
    #
    #   torch.arange(sequence_length).unsqueeze(0) creates:
    #     [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]  # shape: (1, 10)
    #
    #   query_token_length_without_padding.unsqueeze(1) creates:
    #     [[3],
    #      [5],
    #      [2]]  # shape: (3, 1)
    #
    #   The comparison < broadcasts to:
    #     [[True,  True,  True,  False, False, False, False, False, False, False],  # batch 0: first 3 are query
    #      [True,  True,  True,  True,  True,  False, False, False, False, False],  # batch 1: first 5 are query
    #      [True,  True,  False, False, False, False, False, False, False, False]]  # batch 2: first 2 are query
    #
    #   This mask identifies which positions should be replaced with pad_token_id

    sequence_length = input_ids.shape[1]
    query_mask = torch.arange(sequence_length, device=input_ids.device).unsqueeze(
        0
    ) < query_token_length_without_padding.unsqueeze(1)
    # Create new tensor with query tokens masked as pad_token_id
    response_input_ids = input_ids.clone()
    # shape: (batch_size, sequence_length)
    response_input_ids[query_mask] = tokenizer.pad_token_id

    attention_mask = input_ids != tokenizer.pad_token_id
    outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
    # shape: (batch_size, sequence_length, vocab_size), distribution for token t+1
    logits = outputs.logits[:, :-1, :]
    # shape: (batch_size, sequence_length-1)
    labels = response_input_ids[:, 1:]
    # shape: (batch_size, sequence_length-1)
    loss_mask = labels != tokenizer.pad_token_id
    # shape: (batch_size, sequence_length-1)
    per_token_logp = torch.gather(
        logits.log_softmax(dim=-1), dim=2, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    # shape: (batch_size,)
    all_logps = (per_token_logp * loss_mask).sum(dim=-1)
    chosen_logps = all_logps[: input_ids.shape[0] // 2]
    rejected_logps = all_logps[input_ids.shape[0] // 2 :]
    return chosen_logps, rejected_logps


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
            "query_token_length_without_padding",
            "query_and_chosen_response_tokens",
            "query_and_rejected_response_tokens",
        ],
    )
    return torch_dataset


class DPOTrainer:
    def __init__(
        self,
        policy_model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: DPOConfig,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.ref_model = torch.compile(self.ref_model, mode="max-autotune")
        self.tokenizer = tokenizer
        self.config = config
        disable_dropout(self.policy_model)
        disable_dropout(self.ref_model)
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
        total_updates = (
            len(self.train_dataloader) + config.gradient_accumulation_steps - 1
        ) // config.gradient_accumulation_steps
        self.total_updates = total_updates * config.num_train_epochs
        self.warmup_steps = int(total_updates * config.warmup_ratio)
        self.config = config
        self.optimizer = self.setup_optimizer(config)
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
            return self.config.learning_rate * (step + 1) / (self.warmup_steps + 1)
        # # 2) if it > lr_decay_steps, return min learning rate
        # if step > self.config.lr_decay_steps:
        #     return self.config.min_learning_rate
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (step - self.warmup_steps) / (
            self.total_updates - self.warmup_steps
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return coeff * self.config.learning_rate

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
        # Training loop over epochs
        global_step = 0
        best_eval_loss = float("inf")
        step_loss = 0
        step_accuracy = 0
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(self.config.num_train_epochs):
            for i, batch in enumerate(
                tqdm(
                    self.train_dataloader,
                    desc=f"Epoch {epoch + 1}/{self.config.num_train_epochs}",
                )
            ):
                query_chosen_response_tokens = batch["query_and_chosen_response_tokens"]
                query_rejected_response_tokens = batch[
                    "query_and_rejected_response_tokens"
                ]
                query_token_length_without_padding = batch[
                    "query_token_length_without_padding"
                ]
                input_tokens = torch.concat(
                    [query_chosen_response_tokens, query_rejected_response_tokens],
                    dim=0,
                )
                query_token_length_without_padding = torch.concat(
                    [
                        query_token_length_without_padding,
                        query_token_length_without_padding,
                    ],
                    dim=0,
                )
                input_tokens = input_tokens.to(
                    self.policy_model.device, non_blocking=True
                )
                query_token_length_without_padding = (
                    query_token_length_without_padding.to(
                        self.policy_model.device, non_blocking=True
                    )
                )
                with torch.no_grad():
                    ref_chosen_logps, ref_rejected_logps = forward(
                        self.ref_model,
                        input_tokens,
                        query_token_length_without_padding,
                        self.tokenizer,
                    )
                policy_chosen_logps, policy_rejected_logps = forward(
                    self.policy_model,
                    input_tokens,
                    query_token_length_without_padding,
                    self.tokenizer,
                )
                z = (policy_chosen_logps - ref_chosen_logps) - (
                    policy_rejected_logps - ref_rejected_logps
                )
                accuracy = z.gt(0).float().mean()
                accuracy = accuracy / self.config.gradient_accumulation_steps

                loss = (
                    -(1 - config.label_smoothing)
                    * F.logsigmoid(self.config.beta * z).mean()
                    - config.label_smoothing
                    * F.logsigmoid(-self.config.beta * z).mean()
                )
                loss = loss / self.config.gradient_accumulation_steps
                step_loss += loss.item()
                step_accuracy += accuracy.item()
                loss.backward()
                if (i + 1) % self.config.gradient_accumulation_steps == 0:
                    lr = self.get_lr(global_step)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr
                    if self.config.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.policy_model.parameters(), self.config.grad_clip
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.config.wandb_log:
                        wandb.log(
                            {
                                "train_loss": step_loss,
                                "train_accuracy": step_accuracy,
                                "learning_rate": lr,
                            }
                        )
                    if (global_step + 1) % self.config.eval_interval == 0:
                        eval_loss, eval_accuracy = self.run_evaluation()
                        print(
                            f"Epoch {epoch + 1}, Global Step: {global_step}, training loss: {step_loss:.4f}, evaluation loss: {eval_loss:.4f}, evaluation accuracy: {eval_accuracy:.4f}"
                        )
                        if self.config.wandb_log:
                            wandb.log(
                                {
                                    "eval_loss": eval_loss,
                                    "eval_accuracy": eval_accuracy,
                                }
                            )
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
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
                                best_eval_loss,
                                self.config,
                            )
                    step_loss = 0
                    step_accuracy = 0
                    global_step += 1
        # always save the final checkpoint
        output_dir = os.path.join(
            self.config.output_dir,
            f"{self.config.output_checkpoint_prefix}_final_step",
        )
        self.save_checkpoint(
            output_dir,
            self.policy_model,
            self.tokenizer,
            self.optimizer,
            global_step,
            best_eval_loss,
            self.config,
        )

    def save_checkpoint(
        self,
        output_dir,
        model,
        tokenizer,
        optimizer,
        global_step,
        best_eval_loss,
        config,
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
            "best_eval_loss": best_eval_loss,
            "config": config,
        }
        torch.save(training_state, os.path.join(output_dir, "training_state.pt"))

    @torch.no_grad()
    def run_evaluation(self):
        total_loss = 0
        step = 0
        self.policy_model.eval()
        step = 0
        total_loss = 0
        total_accuracy = 0
        for batch in tqdm(self.validation_dataloader, desc="Evaluating"):
            query_chosen_response_tokens = batch["query_and_chosen_response_tokens"]
            query_rejected_response_tokens = batch["query_and_rejected_response_tokens"]
            input_tokens = torch.concat(
                [query_chosen_response_tokens, query_rejected_response_tokens], dim=0
            )
            query_token_length_without_padding = batch[
                "query_token_length_without_padding"
            ]
            query_token_length_without_padding = torch.concat(
                [
                    query_token_length_without_padding,
                    query_token_length_without_padding,
                ],
                dim=0,
            )
            input_tokens = input_tokens.to(self.policy_model.device, non_blocking=True)
            query_token_length_without_padding = query_token_length_without_padding.to(
                self.policy_model.device, non_blocking=True
            )
            ref_chosen_logps, ref_rejected_logps = forward(
                self.ref_model,
                input_tokens,
                query_token_length_without_padding,
                self.tokenizer,
            )
            policy_chosen_logps, policy_rejected_logps = forward(
                self.policy_model,
                input_tokens,
                query_token_length_without_padding,
                self.tokenizer,
            )
            z = (policy_chosen_logps - ref_chosen_logps) - (
                policy_rejected_logps - ref_rejected_logps
            )
            accuracy = z.gt(0).float().mean()
            total_accuracy += accuracy.item()
            loss = (
                -(1 - config.label_smoothing)
                * F.logsigmoid(self.config.beta * z).mean()
                - config.label_smoothing * F.logsigmoid(-self.config.beta * z).mean()
            )
            total_loss += loss.item()
            step += 1
        average_loss = total_loss / step
        average_accuracy = total_accuracy / step
        self.policy_model.train()
        return average_loss, average_accuracy


if __name__ == "__main__":
    args = parse_args()
    sft_model_path = args.sft_model_path
    ref_model = AutoModelForCausalLM.from_pretrained(
        sft_model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    # initialize the policy model as the sft model
    policy_model = AutoModelForCausalLM.from_pretrained(
        sft_model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    config_dict = {
        "num_train_epochs": args.num_train_epochs,
        "dataset": args.dataset,
        "seed": args.seed,
        "batch_size": args.batch_size,
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
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "adamw_beta1": args.adamw_beta1,
        "adamw_beta2": args.adamw_beta2,
        "adamw_weight_decay": args.adamw_weight_decay,
        "use_eight_bit_optimizer": args.use_eight_bit_optimizer,
        "output_dir": args.output_dir,
        "output_checkpoint_prefix": args.output_checkpoint_prefix,
        "beta": args.beta,
        "label_smoothing": args.label_smoothing,
    }
    config = DPOConfig(**config_dict)
    trainer = DPOTrainer(policy_model, ref_model, tokenizer, config)
    trainer.train()
