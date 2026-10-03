import torch

from src.training.config import create_training_config


def test_training_configuration():
    config = create_training_config()

    assert config.per_device_train_batch_size == 1
    assert config.gradient_accumulation_steps == 8
    assert config.learning_rate == 2e-4
    assert config.num_train_epochs == 3


def test_gradient_checkpointing():
    config = create_training_config()

    assert config.gradient_checkpointing is True


def test_paged_optimizer():
    config = create_training_config()

    assert config.optim == "paged_adamw_8bit"


def test_precision_configuration():
    config = create_training_config()

    assert not (
        config.bf16
        and config.fp16
    )


def test_cpu_environment_does_not_enable_gpu_precision():
    config = create_training_config()

    if not torch.cuda.is_available():
        assert config.bf16 is False
        assert config.fp16 is False