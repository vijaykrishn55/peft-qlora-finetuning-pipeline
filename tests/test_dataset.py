from transformers import AutoTokenizer

from src.data.formatter import format_chatml
from src.data.tokenizer import tokenize_example, IGNORE_INDEX


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def test_chatml_formatting():
    result = format_chatml(
        "What is Python?",
        "Python is a programming language.",
    )

    assert "<|im_start|>user" in result["text"]
    assert "<|im_start|>assistant" in result["text"]
    assert result["text"].endswith("<|im_end|>")


def test_prompt_tokens_are_masked():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    formatted = format_chatml(
        "What is Python?",
        "Python is a programming language.",
    )

    result = tokenize_example(
        formatted,
        tokenizer,
        max_length=512,
    )

    labels = result["labels"]

    assert IGNORE_INDEX in labels
    assert any(label != IGNORE_INDEX for label in labels)


def test_lengths_match():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    formatted = format_chatml(
        "What is Python?",
        "Python is a programming language.",
    )

    result = tokenize_example(
        formatted,
        tokenizer,
        max_length=512,
    )

    assert len(result["input_ids"]) == len(result["attention_mask"])
    assert len(result["input_ids"]) == len(result["labels"])