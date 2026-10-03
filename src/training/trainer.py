from pathlib import Path

from datasets import Dataset
from peft import PeftModel
from transformers import PreTrainedTokenizerBase
from trl import SFTTrainer, SFTConfig


def create_trainer(
    model: PeftModel,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    training_config: SFTConfig,
) -> SFTTrainer:
    """
    Create the TRL SFT trainer for QLoRA training.
    """

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    return trainer


def save_training_metrics(
    trainer: SFTTrainer,
    output_path: str = "artifacts/training_metrics.json",
) -> None:
    """
    Save the final training metrics to JSON.
    """

    metrics = trainer.state.log_history

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    import json

    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)