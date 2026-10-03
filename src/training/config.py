import torch
from trl import SFTConfig


def create_training_config(
    output_dir: str = "artifacts/checkpoints",
) -> SFTConfig:
    """
    Create the training configuration for QLoRA fine-tuning.
    """

    cuda_available = torch.cuda.is_available()

    use_bf16 = (
        cuda_available
        and torch.cuda.is_bf16_supported()
    )

    use_fp16 = (
        cuda_available
        and not use_bf16
    )

    return SFTConfig(
        output_dir=output_dir,

        # Training
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,

        # Precision
        bf16=use_bf16,
        fp16=use_fp16,

        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },

        # Optimizer
        optim="paged_adamw_8bit",

        # Sequence handling
        max_length=512,
        packing=False,

        # Logging
        logging_strategy="steps",
        logging_steps=5,
        report_to="none",

        # Checkpoints
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,

        # Model behavior
        use_cache=False,

        # Reproducibility
        seed=42,
    )