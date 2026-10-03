from typing import Dict, List

from transformers import PreTrainedTokenizerBase


IGNORE_INDEX = -100


def tokenize_example(
    example: Dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
) -> Dict[str, List[int]]:
    """
    Tokenize one formatted instruction example.

    Prompt tokens are masked with -100 so they do not contribute
    to the language-model training loss.
    """

    prompt_ids = tokenizer(
        example["prompt"],
        add_special_tokens=False,
    )["input_ids"]

    full_ids = tokenizer(
        example["text"],
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    prompt_length = min(len(prompt_ids), len(full_ids))

    labels = (
        [IGNORE_INDEX] * prompt_length
        + full_ids[prompt_length:]
    )

    labels = labels[:max_length]

    attention_mask = [1] * len(full_ids)

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }