import argparse
from dataclasses import dataclass
import math
import os
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import bitsandbytes as bnb
import torch.nn.functional as F


@dataclass
class SFTConfig:
    # training configs
    num_train_epochs: int = 1
    batch_size: int = 8
    grad_clip: float = 1.0
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 10
    lr_decay_steps: int = 1000
    train_dataset_size: int = 10
    seed: int = 42
    dataset: str = "seangogo/processed_tldr_sft_dataset_20251028_232434"
    input_field: str = "query_and_response_tokens"
    label_field: str = "query_and_response_labels"

    # evaluation
    num_eval_epochs: int = 1
    eval_interval: int = 10  # eval every N st
    eval_dataset_size: int = 10

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
    output_dir: str = "sft_output"  # directory to save the fine-tuned model


class SFTTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        config: SFTConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        train_dataset_size = (
            config.train_dataset_size
            if config.train_dataset_size
            else len(load_dataset(config.dataset, split="train"))
        )
        eval_dataset_size = (
            config.eval_dataset_size
            if config.eval_dataset_size
            else len(load_dataset(config.dataset, split="validation"))
        )
        self.train_dataset = (
            load_dataset(config.dataset, split="train")
            .select(range(train_dataset_size))
            .shuffle(seed=config.seed)
        )
        self.eval_dataset = (
            load_dataset(config.dataset, split="train")
            .select(range(eval_dataset_size))
            .shuffle(seed=config.seed)
        )
        self.config = config
        self.input_field = config.input_field
        self.label_field = config.label_field
        self.optimizer = self.setup_optimizer(config)

    def setup_optimizer(self, config):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in model.named_parameters()}
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
        if step < self.config.warmup_steps:
            return (
                self.config.learning_rate * (step + 1) / (self.config.warmup_steps + 1)
            )
        # 2) if it > lr_decay_steps, return min learning rate
        if step > self.config.lr_decay_steps:
            return self.config.min_learning_rate
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (step - self.config.warmup_steps) / (
            self.config.lr_decay_steps - self.config.warmup_steps
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.config.min_learning_rate + coeff * (
            self.config.learning_rate - self.config.min_learning_rate
        )

    def train(self):
        if self.config.wandb_log:
            import wandb

            assert self.config.wandb_project is not None, "Wandb project is required"
            assert self.config.wandb_run_name is not None, "Wandb run name is required"
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=vars(self.config),
            )
        # Training loop over epochs
        global_step = 0
        best_eval_loss = float("inf")
        for epoch in range(self.config.num_train_epochs):
            # Shuffle the dataset at the start of each epoch
            # Use epoch number as seed to get different shuffles each epoch
            self.train_dataset = self.train_dataset.shuffle(seed=self.config.seed)
            self.optimizer.zero_grad(set_to_none=True)
            print(f"\nEpoch {epoch + 1}/{self.config.num_train_epochs}")
            # Iterate over batches in the current epoch
            total_iterations = (
                len(self.train_dataset) + self.config.batch_size - 1
            ) // self.config.batch_size
            step_loss = 0
            for iteration in range(total_iterations):
                i = iteration * self.config.batch_size
                batch = self.train_dataset[i : i + self.config.batch_size]
                # input_tensor [batch_size, sequence_length]
                # label_tensor [batch_size, sequence_length]
                input_tensor, label_tensor = self.prepare_batch(batch)
                # [batch_size, sequence_length, vocab_size]
                model_output = self.model(input_tensor).logits
                micro_loss = F.cross_entropy(
                    model_output.view(-1, model_output.shape[-1]), label_tensor.view(-1)
                )
                loss = micro_loss / self.config.gradient_accumulation_steps
                step_loss += loss.item()
                loss.backward()
                if (iteration + 1) % self.config.gradient_accumulation_steps == 0:
                    lr = self.get_lr(global_step)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr
                    if self.config.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.config.wandb_log:
                        wandb.log(
                            {
                                "train_loss": step_loss,
                            }
                        )
                    if (global_step + 1) % self.config.eval_interval == 0:
                        eval_loss = self.run_evaluation()
                        print(
                            f"Epoch {epoch + 1}, Global Step: {global_step}, training loss: {step_loss:.4f}, evaluation loss: {eval_loss:.4f}"
                        )
                        if self.config.wandb_log:
                            wandb.log(
                                {
                                    "eval_loss": eval_loss,
                                }
                            )
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            output_dir = os.path.join(
                                self.config.output_dir, f"checkpoint_{global_step}"
                            )
                            self.save_checkpoint(
                                output_dir,
                                self.model,
                                self.tokenizer,
                                self.optimizer,
                                global_step,
                                best_eval_loss,
                                self.config,
                            )
                    step_loss = 0
                    global_step += 1

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
        self.model.eval()
        for i in range(0, len(self.eval_dataset), self.config.batch_size):
            batch = self.eval_dataset[i : i + self.config.batch_size]
            input_tensor, label_tensor = self.prepare_batch(batch)
            model_output = self.model(input_tensor).logits
            loss = F.cross_entropy(
                model_output.view(-1, model_output.shape[-1]), label_tensor.view(-1)
            )
            total_loss += loss.item()
            step += 1
        average_loss = total_loss / step
        self.model.train()
        return average_loss

    def prepare_batch(self, batch):
        input_tensor = torch.tensor(batch[self.input_field])[:, :-1].to(
            self.model.device
        )
        label_tensor = torch.tensor(batch[self.label_field])[:, 1:].to(
            self.model.device
        )
        return input_tensor, label_tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tuning a model on a TLDR dataset"
    )

    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Model to fine-tune"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="seangogo/processed_tldr_sft_dataset_20251029_045736",
        help="Dataset to use for training and evaluation",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
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
        "--eval_dataset_size", type=int, default=100, help="Evaluation dataset size"
    )
    parser.add_argument(
        "--input_field",
        type=str,
        default="query_and_response_tokens",
        help="Input field in dataset",
    )
    parser.add_argument(
        "--label_field",
        type=str,
        default="query_and_response_labels",
        help="Label field in dataset",
    )

    # Training configs
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sft_output",
        help="Directory to save the fine-tuned model",
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


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    print(f"\n{'='*70}")
    print(f"Fine tuning on TLDR dataset with model: {model_name}")
    print(f"{'='*70}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=args.dtype, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

    # Create config dict from args, mapping to SFTConfig fields
    config_dict = {
        "num_train_epochs": args.num_train_epochs,
        "dataset": args.dataset,
        "input_field": args.input_field,
        "label_field": args.label_field,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "grad_clip": args.grad_clip,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "warmup_steps": args.warmup_steps,
        "lr_decay_steps": args.lr_decay_steps,
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
    }
    config = SFTConfig(**config_dict)
    input_field = args.input_field
    label_field = args.label_field
    sft_trainer = SFTTrainer(model, tokenizer, config)
    sft_trainer.train()
