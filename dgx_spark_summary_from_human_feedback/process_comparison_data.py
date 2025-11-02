import copy
from datetime import datetime
import os
import json
from dataclasses import dataclass, field
from pprint import pformat
from pprint import pprint
from typing import Any, Dict, Optional, Tuple

import datasets
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset, DatasetDict
from huggingface_hub import HfApi, RepoCard
from tqdm import tqdm
import tyro

from .summary_from_human_feedback import SummarizeFromFeedback


@dataclass
class DatasetPreprocessingParams:
    max_rm_response_length: Optional[int] = None
    max_rm_query_response_length: Optional[int] = None
    query_length: Optional[int] = None
    query_format_str: Optional[str] = None
    query_truncation_field: Optional[str] = None
    query_truncation_text: Optional[str] = None
    query_padding_side: Optional[str] = None


@dataclass
class Args:
    source_dataset_name: str = "openai/summarize_from_feedback"
    model_name: str = "Qwen/Qwen2.5-0.5B"
    output_dataset_name: str = "processed_tldr_comparison_dataset"
    debug: bool = False
    padding_token: str = "[PAD]"
    dataset_preprocessing_params: DatasetPreprocessingParams = field(
        default_factory=lambda: DatasetPreprocessingParams(
            query_length=512,
            query_format_str="SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:",
            query_truncation_field="post",
            query_truncation_text="\n",
            query_padding_side="left",
            max_rm_response_length=133,
            max_rm_query_response_length=645,
        )
    )


def process_query(
    query_info: Dict[str, str],
    params: DatasetPreprocessingParams,
    tokenizer: AutoTokenizer,
) -> Dict[str, Any]:
    query_info_copy = query_info.copy()
    formated_query = params.query_format_str.format(**query_info_copy)
    query_tokens = tokenizer.encode(formated_query)
    while len(query_tokens) > params.query_length:
        # when ind = -1, just remove the last character
        ind = query_info_copy[params.query_truncation_field].rfind(
            params.query_truncation_text
        )
        query_info_copy[params.query_truncation_field] = query_info_copy[
            params.query_truncation_field
        ][:ind]
        formated_query = params.query_format_str.format(**query_info_copy)
        query_tokens = tokenizer.encode(formated_query)
    # pad query tokens to params.query_length
    query_token_length_without_padding = len(query_tokens)
    padding_length = params.query_length - query_token_length_without_padding
    padding_token_id = tokenizer.pad_token_id
    if params.query_padding_side == "left":
        query_tokens = [padding_token_id] * padding_length + query_tokens
    elif params.query_padding_side == "right":
        query_tokens = query_tokens + [padding_token_id] * padding_length
    else:
        assert False, f"Invalid padding side: {params.query_padding_side}"
    query = tokenizer.decode(query_tokens, skip_special_tokens=True).lstrip()
    return {
        "query": query,
        "query_tokens": query_tokens,
        "query_token_length_without_padding": query_token_length_without_padding,
    }


def upload_dataset(hf_entity: str, final_dataset: DatasetDict, args: Args):
    # Login to Hugging Face (you'll need to provide your token)
    print("\nTo upload to Hugging Face, you need to:")
    print("1. Get your Hugging Face token from https://huggingface.co/settings/tokens")
    print("2. Run: huggingface-cli login")
    print("3. Or set HF_TOKEN environment variable")

    # Check if user is logged in
    try:
        # Upload the dataset
        upload_dataset_name = (
            f"{args.output_dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        hf_path = f"{hf_entity}/{upload_dataset_name}"
        print(f"\nUploading dataset to: {hf_path}")
        final_dataset.push_to_hub(hf_path)
        print(
            f"Dataset uploaded successfully to: https://huggingface.co/datasets/{hf_path}"
        )
        final_dataset_card = RepoCard.load(hf_path, repo_type="dataset")
        final_dataset_card.text = f"""\
# TL;DR Comparison Dataset for OpenAI's [Summarize from Feedback](https://arxiv.org/abs/2009.01325) task

The dataset is generated from https://huggingface.co/datasets/openai/summarize_from_feedback.
Please refer to https://github.com/liyuan24/dgx_spark_summary_from_human_feedback/tree/main?tab=readme-ov-file#download-the-comparison-dataset about how to download the dataset.

This is a comparison dataset used for training a reward model. Each example contains a query (post) and two responses (chosen and rejected) where the chosen response is preferred by human feedback.

## Columns

These columns are added by this preprocessing script:

* **query**: length-limited query for summarization. The main text (title + subreddit + post) is preprocessed to have exactly 512 tokens; if the main text is too long, it truncates at the last `\n`. If it's too short, it pads the main text. Padding uses the `[PAD]` token.
* **query_tokens**: tokenized version of `query` as a list of token IDs (length: 512)
* **query_token_length_without_padding**: length of query tokens without padding (used to mask query tokens during loss calculation)

* **chosen_response**: the preferred response summary with a trailing <|endoftext|> token
* **chosen_response_tokens**: tokenized version of `chosen_response`, padded/truncated to `max_rm_response_length` tokens
* **rejected_response**: the less preferred response summary with a trailing <|endoftext|> token
* **rejected_response_tokens**: tokenized version of `rejected_response`, padded/truncated to `max_rm_response_length` tokens

* **query_and_chosen_response**: the concatenation of `query` and `chosen_response`
* **query_and_chosen_response_tokens**: tokenized version of `query_and_chosen_response`, padded/truncated to `max_rm_query_response_length` tokens
* **query_and_rejected_response**: the concatenation of `query` and `rejected_response`
* **query_and_rejected_response_tokens**: tokenized version of `query_and_rejected_response`, padded/truncated to `max_rm_query_response_length` tokens

## Dataset Configuration

```python
{pformat(vars(args))}
```
"""
        final_dataset_card.push_to_hub(hf_path, repo_type="dataset")

    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")
        print("Please login first using: huggingface-cli login")


def load_comparison_dataset() -> Tuple[Dataset, Dataset]:
    summary_from_human_feedback = SummarizeFromFeedback(config_name="comparisons")
    summary_from_human_feedback.download_and_prepare()
    train_dataset = summary_from_human_feedback.as_dataset(split=datasets.Split.TRAIN)
    validation_dataset = summary_from_human_feedback.as_dataset(
        split=datasets.Split.VALIDATION
    )
    cnn_batches = ["batch0_cnndm", "cnndm0", "cnndm2"]
    validation_dataset = validation_dataset.filter(
        lambda x: x["batch"] not in cnn_batches
    )
    return train_dataset, validation_dataset


def process_and_upload_dataset(args: Args):
    api = HfApi()
    hf_entity = api.whoami()["name"]
    print(f"Logged in as: {hf_entity}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    padding_token = args.padding_token
    tokenizer.add_special_tokens({"pad_token": padding_token})
    data_preprocessing_params = args.dataset_preprocessing_params

    def process_comparison_data(x: Dict[str, str]) -> Dict[str, Any]:
        processed_query = process_query(x["info"], data_preprocessing_params, tokenizer)
        query = processed_query["query"]
        query_tokens = processed_query["query_tokens"]
        query_token_length_without_padding = processed_query[
            "query_token_length_without_padding"
        ]
        y = {
            "query": query,
            "query_tokens": query_tokens,
        }
        # add a prefix whitespace and a suffix <|endoftext|> eos token to reference summary
        # add <|endoftext|> after summary also make it easier for label shifting
        choice = x["choice"]
        # there is already a leading whitespace in the response, so we don't need to add it again
        chosen_response = f"{x['summaries'][choice]['text']}<|endoftext|>"
        rejected_response = f"{x['summaries'][1-choice]['text']}<|endoftext|>"
        chosen_response_tokens = tokenizer.encode(
            chosen_response,
            padding="max_length",
            truncation=True,
            max_length=data_preprocessing_params.max_rm_response_length,
        )
        rejected_response_tokens = tokenizer.encode(
            rejected_response,
            padding="max_length",
            truncation=True,
            max_length=data_preprocessing_params.max_rm_response_length,
        )
        query_and_chosen_response = query.strip() + chosen_response
        query_and_chosen_response_tokens = tokenizer.encode(
            query_and_chosen_response,
            padding="max_length",
            truncation=True,
            max_length=data_preprocessing_params.max_rm_query_response_length,
        )
        query_and_rejected_response = query.strip() + rejected_response
        query_and_rejected_response_tokens = tokenizer.encode(
            query_and_rejected_response,
            padding="max_length",
            truncation=True,
            max_length=data_preprocessing_params.max_rm_query_response_length,
        )
        return {
            "query": query,
            "query_tokens": query_tokens,
            "query_token_length_without_padding": query_token_length_without_padding,
            "chosen_response": chosen_response,
            "chosen_response_tokens": chosen_response_tokens,
            "rejected_response": rejected_response,
            "rejected_response_tokens": rejected_response_tokens,
            "query_and_chosen_response": query_and_chosen_response,
            "query_and_chosen_response_tokens": query_and_chosen_response_tokens,
            "query_and_rejected_response": query_and_rejected_response,
            "query_and_rejected_response_tokens": query_and_rejected_response_tokens,
        }

    train_dataset, validation_dataset = load_comparison_dataset()
    processed_train_dataset = train_dataset.map(
        process_comparison_data,
        load_from_cache_file=False,
        num_proc=1 if args.debug else os.cpu_count(),
    )
    processed_validation_dataset = validation_dataset.map(
        process_comparison_data,
        load_from_cache_file=False,
        num_proc=1 if args.debug else os.cpu_count(),
    )
    processed_comparison_dataset = DatasetDict(
        {
            "train": processed_train_dataset,
            "validation": processed_validation_dataset,
        }
    )
    upload_dataset(hf_entity, processed_comparison_dataset, args)


if __name__ == "__main__":
    args = tyro.cli(Args)
    pprint(args)
    process_and_upload_dataset(args)
