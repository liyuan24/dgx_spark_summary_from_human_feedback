# dgx_spark_summary_from_human_feedback

This repo contains the code for reproducing [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) in one DGX Spark. [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://arxiv.org/abs/2403.17031) has already done this in an 8-H100 cluster. We will refer to this work for many details.

# Steps

1. Preprocessing the data
2. Supervised Fine-tuning
3. Reward Model Training
4. Reinforcement learning from human feedback

# Preprocessing the TLDR dataset
I will first use [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) as the base model and its tokenizer to preprocess the TLDR dataset. Checking the padding token and eos token of the tokenizer,

```python
from transformers import AutoTokenizer
model = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model)
print(f"padding token id: {tokenizer.pad_token_id}, eos token id: {tokenizer.eos_token_id}")
```

```
padding token id: 151643, eos token id: 151643
padding token: <|endoftext|>, eos token: <|endoftext|>
```

## SFT Dataset
We use [summarize_from_feedback_tldr_3_filtered](https://huggingface.co/datasets/vwxyzjn/summarize_from_feedback_tldr_3_filtered) as the SFT dataset.

1. for each reference summary, add a prepending whitespace and a trailing <|endoftext|> token. The maximum length of tokens of the reference response after formatting is **63**
