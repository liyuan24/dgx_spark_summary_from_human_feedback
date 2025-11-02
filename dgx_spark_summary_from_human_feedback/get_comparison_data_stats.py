from datasets import load_dataset
import datasets
from tqdm import tqdm
from transformers import AutoTokenizer

from .summary_from_human_feedback import SummarizeFromFeedback


def get_reference_response_lengths(dataset, tokenizer):
    response_lengths = []
    for i in tqdm(range(len(dataset))):
        summary = dataset[i]["summaries"]
        choice = dataset[i]["choice"]
        # for comparison dataset, there is already a leading whitespace, so we don't need to add it again
        chosen = f"{summary[choice]['text']}<|endoftext|>"
        rejected = f"{summary[1-choice]['text']}<|endoftext|>"
        chosen_tokens = tokenizer.encode(chosen)
        rejected_tokens = tokenizer.encode(rejected)
        response_lengths.append(len(chosen_tokens))
        response_lengths.append(len(rejected_tokens))
    return response_lengths


if __name__ == "__main__":
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
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    train_response_lengths = get_reference_response_lengths(train_dataset, tokenizer)
    validation_response_lengths = get_reference_response_lengths(
        validation_dataset, tokenizer
    )
    print(f"max length of training data: {max(train_response_lengths)}")
    print(f"max length of validation data: {max(validation_response_lengths)}")
