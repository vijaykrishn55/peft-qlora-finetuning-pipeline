from src.model.lora import (
    TARGET_MODULES,
    create_lora_config,
)


def test_lora_configuration():
    config = create_lora_config()

    assert config.r == 16
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.05
    assert config.bias == "none"
    assert config.task_type == "CAUSAL_LM"


def test_target_modules():
    config = create_lora_config()

    for module in TARGET_MODULES:
        assert module in config.target_modules