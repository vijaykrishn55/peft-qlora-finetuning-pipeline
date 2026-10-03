from peft import LoraConfig, get_peft_model


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def create_lora_config() -> LoraConfig:
    """
    Create the LoRA configuration used for QLoRA training.
    """

    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )


def add_lora_adapters(model):
    """
    Attach LoRA adapters to the prepared quantized model.
    """

    lora_config = create_lora_config()

    model = get_peft_model(
        model,
        lora_config,
    )

    return model

def get_trainable_parameter_stats(model) -> dict:
    """
    Calculate trainable and total parameter statistics.
    """

    trainable_params = 0
    total_params = 0

    for parameter in model.parameters():
        parameter_count = parameter.numel()

        total_params += parameter_count

        if parameter.requires_grad:
            trainable_params += parameter_count

    percentage = (
        trainable_params / total_params * 100
        if total_params > 0
        else 0.0
    )

    return {
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "trainable_percentage": percentage,
    }