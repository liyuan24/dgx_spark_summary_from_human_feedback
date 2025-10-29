import copy
from datetime import datetime
import os
import json
from dataclasses import dataclass, field
from pprint import pformat
from pprint import pprint
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer
from datasets import Dataset, load_dataset, DatasetDict
from huggingface_hub import HfApi, RepoCard
from tqdm import tqdm
import tyro

"""
python process_sft_dataset.py \
--source_dataset_name=vwxyzjn/summarize_from_feedback_tldr_3_filtered \
--model_name=Qwen/Qwen2.5-0.5B
"""


@dataclass
class DatasetPreprocessingParams:
    max_sft_response_length: Optional[int] = None
    max_sft_query_response_length: Optional[int] = None
    query_length: Optional[int] = None
    query_format_str: Optional[str] = None
    query_truncation_field: Optional[str] = None
    query_truncation_text: Optional[str] = None
    query_padding_side: Optional[str] = None


@dataclass
class Args:
    source_dataset_name: str = "vwxyzjn/summarize_from_feedback_tldr_3_filtered"
    model_name: str = "Qwen/Qwen2.5-0.5B"
    output_dataset_name: str = "processed_tldr_sft_dataset"
    debug: bool = False
    padding_token: str = "[PAD]"
    dataset_preprocessing_params: DatasetPreprocessingParams = field(
        default_factory=lambda: DatasetPreprocessingParams(
            query_length=512,
            query_format_str="SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:",
            query_truncation_field="post",
            query_truncation_text="\n",
            query_padding_side="left",
            max_sft_response_length=63,
            max_sft_query_response_length=575,
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
        print(f"Dataset uploaded successfully to: https://huggingface.co/datasets/{hf_path}")
        final_dataset_card = RepoCard.load(hf_path, repo_type="dataset")
        final_dataset_card.text = f"""\
# TL;DR SFT Dataset for OpenAI's [Summarize from Feedback](https://arxiv.org/abs/2009.01325) task

The dataset is generated from https://huggingface.co/datasets/vwxyzjn/summarize_from_feedback_tldr_3_filtered

These columns are taken directly from the aforementioned dataset:

* **id**: unique identifier for the post
* **subreddit**: subreddit the post was taken from
* **title**: title of the post
* **post**: body of the post
* **summary**: summary of the post

These columns are added by this preprocessing script:
* **query**: length-limited query for summarization: OAI pre-processes the main text (title + subreddit + post), ensuring it has only 512 tokens; if the main text is too long, then it tries to truncate at the last `\n`. If it's too short it pads the main text. Padding is `[PAD]` token.
* **query_token**: tokenized version of `query`
* **query_token_length_without_padding**: length of query tokens without padding
* **response**: response for the post: a prefix whitespace and a suffix <|endoftext|> eos token to reference summary
* **response_tokens**: tokenized version of `response`
* **response_token_length**: length of `response_tokens`
* **query_and_response**: the concatenation of `query` and `response`
* **query_and_response_tokens**: tokenized version of `query_and_response`, up to `max_sft_query_response_length` tokens
* **query_and_response_labels**: labels for the query and response: the query tokens are masked as -100 to not calculate loss on user query, the padding tokens are masked as -100 to not calculate loss on padding tokens


# Args

```python
{pformat(vars(args))}
```
"""
        final_dataset_card.push_to_hub(hf_path, repo_type="dataset")

    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")
        print("Please login first using: huggingface-cli login")


def process_and_upload_dataset(args: Args):
    api = HfApi()
    hf_entity = api.whoami()['name']
    print(f"Logged in as: {hf_entity}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    padding_token = args.padding_token
    tokenizer.add_special_tokens({"pad_token": padding_token})
    data_preprocessing_params = args.dataset_preprocessing_params

    def process_sft_data(x: Dict[str, str]) -> Dict[str, Any]:
        processed_query = process_query(x, data_preprocessing_params, tokenizer)
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
        response = f" {x['summary']}<|endoftext|>"
        response_tokens = tokenizer.encode(
            response,
            padding="max_length",
            truncation=True,
            max_length=data_preprocessing_params.max_sft_response_length,
        )
        response_token_length = len(tokenizer.encode(response))
        query_and_response = query.strip() + response
        query_and_response_tokens = tokenizer.encode(
            query_and_response,
            padding="max_length",
            truncation=True,
            max_length=data_preprocessing_params.max_sft_query_response_length,
        )
        query_and_response_labels = copy.deepcopy(query_and_response_tokens)
        # mask the query tokens in the query and response as -100 to not calculate loss on user query
        query_and_response_labels[:query_token_length_without_padding] = [
            -100
        ] * query_token_length_without_padding
        # mask the padding tokens label as -100 to not calculate loss on padding tokens
        padded_query_and_response_tokens = [
            token
            for token in query_and_response_tokens
            if token == tokenizer.pad_token_id
        ]
        query_and_response_labels[-len(padded_query_and_response_tokens) :] = [
            -100
        ] * len(padded_query_and_response_tokens)
        return {
            "query": query,
            "query_tokens": query_tokens,
            "query_token_length_without_padding": query_token_length_without_padding,
            "response": response,
            "response_tokens": response_tokens,
            "response_token_length": response_token_length,
            "query_and_response": query_and_response,
            "query_and_response_tokens": query_and_response_tokens,
            "query_and_response_labels": query_and_response_labels,
        }

    sft_source_dataset = load_dataset(args.source_dataset_name)
    sft_source_dataset = sft_source_dataset.map(
        process_sft_data,
        load_from_cache_file=False,
        num_proc=1 if args.debug else os.cpu_count(),
    )
    upload_dataset(hf_entity, sft_source_dataset, args)

if __name__ == "__main__":
    args = tyro.cli(Args)
    pprint(args)
    process_and_upload_dataset(args)