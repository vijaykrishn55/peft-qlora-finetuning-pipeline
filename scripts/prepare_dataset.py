import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from src.data.formatter import format_chatml
from src.data.tokenizer import tokenize_example


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = Path("data/raw/train.json")


def main() -> None:
    with DATA_PATH.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    formatted_data = [
        format_chatml(
            item["instruction"],
            item["response"],
        )
        for item in raw_data
    ]

    dataset = Dataset.from_list(formatted_data)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_dataset = dataset.map(
        lambda example: tokenize_example(
            example,
            tokenizer,
            max_length=512,
        ),
        remove_columns=dataset.column_names,
    )

    tokenized_dataset.save_to_disk(
        "data/processed/train"
    )

    print(tokenized_dataset)
    print(tokenized_dataset[0])


if __name__ == "__main__":
    main()

