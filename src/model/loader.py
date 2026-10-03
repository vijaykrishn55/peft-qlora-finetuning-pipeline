from transformers import AutoModelForCausalLM

from peft import prepare_model_for_kbit_training

from src.model.quantization import create_quantization_config


def load_quantized_model(model_name: str):
    """
    Load a causal language model using 4-bit QLoRA quantization
    and prepare it for k-bit training.
    """

    quantization_config = create_quantization_config()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)

    return model