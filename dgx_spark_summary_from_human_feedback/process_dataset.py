import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer


@dataclass
class DatasetPreprocessingParams:
    max_sft_response_length: Optional[int] = None
    max_sft_query_response_length: Optional[int] = None
    query_length: Optional[int] = None
    query_format_str: Optional[str] = None
    query_truncation_field: Optional[str] = None
    query_truncation_text: Optional[str] = None
    query_padding_token: Optional[str] = None
    query_padding_side: Optional[str] = None


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
        ind = query_info[params.query_truncation_field].rfind(
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
    padding_token_id = tokenizer.encode(params.query_padding_token)[0]
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


def process_sft_data(
    x: Dict[str, str], params: DatasetPreprocessingParams, tokenizer: AutoTokenizer
) -> Dict[str, Any]:
    processed_query = process_query(x, params, tokenizer)
    query = processed_query["query"]
    query_tokens = processed_query["query_tokens"]
    query_token_length_without_padding = processed_query["query_token_length_without_padding"]
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
        max_length=params.max_sft_response_length,
    )
    response_token_length = len(tokenizer.encode(response))
    query_and_response = query.strip() + response
    query_and_response_tokens = tokenizer.encode(
        query_and_response,
        padding="max_length",
        truncation=True,
        max_length=params.max_sft_query_response_length,
    )
    query_and_response_labels = copy.deepcopy(query_and_response_tokens)
    # mask the query tokens in the query and response as -100 to not calculate loss on user query
    query_and_response_labels[:query_token_length_without_padding] = [-100] * query_token_length_without_padding
    # mask the padding tokens label as -100 to not calculate loss on padding tokens
    padded_query_and_response_tokens = [token for token in query_and_response_tokens if token == tokenizer.pad_token_id]
    query_and_response_labels[-len(padded_query_and_response_tokens):] = [-100] * len(padded_query_and_response_tokens)
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

if __name__ == "__main__":
    x = {
        "subreddit": "test_subreddit",
        "title": "Test Title",
        "post": "This is a test post with some content.\n\nMore content here.",
        "summary": "This is a test summary with some content.",
    }
    params = DatasetPreprocessingParams(
        query_length=512,
        query_format_str="SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:",
        query_truncation_field="post",
        query_truncation_text="\n",
        query_padding_token="[PAD]",
        query_padding_side="right",
        max_sft_response_length=63,
        max_sft_query_response_length=575,
    )
    model = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    print("get tokenizer, padding token id: ", tokenizer.pad_token_id)
    print("process query")
    print(process_query(x, params, tokenizer))
    print("process sft data")
    print(process_sft_data(x, params, tokenizer))