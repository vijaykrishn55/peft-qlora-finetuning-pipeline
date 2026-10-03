import sys

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

from src.model.loader import load_quantized_model
from src.model.lora import add_lora_adapters, get_trainable_parameter_stats
from src.training.config import create_training_config
from src.training.trainer import (
    create_trainer,
    save_training_metrics,
)


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_PATH = "data/processed/train"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. "
            "Run the training script in a CUDA-enabled environment "
            "such as the Kaggle GPU notebook."
        )

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    dataset = load_from_disk(DATASET_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = load_quantized_model(MODEL_NAME)

    model = add_lora_adapters(model)

    stats = get_trainable_parameter_stats(model)

    print("\nTrainable parameter statistics:")
    print(
        f"Trainable parameters: "
        f"{stats['trainable_parameters']:,}"
    )
    print(
        f"Total parameters: "
        f"{stats['total_parameters']:,}"
    )
    print(
        f"Trainable percentage: "
        f"{stats['trainable_percentage']:.4f}%"
    )

    if stats["trainable_percentage"] >= 2.0:
        raise RuntimeError(
            "LoRA configuration exceeds the 2% trainable "
            "parameter requirement."
        )

    training_config = create_training_config()

    trainer = create_trainer(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        training_config=training_config,
    )

    print("\nStarting QLoRA training...")

    trainer.train()

    trainer.save_model("artifacts/adapter")
    save_training_metrics(trainer)

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nTraining failed: {exc}", file=sys.stderr)
        raise