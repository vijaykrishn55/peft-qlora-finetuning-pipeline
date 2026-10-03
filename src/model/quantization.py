import torch
from transformers import BitsAndBytesConfig


def get_compute_dtype() -> torch.dtype:
    """
    Select the computation dtype for 4-bit quantized training.

    BF16 is preferred when the GPU supports it.
    Otherwise FP16 is used.
    """

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def create_quantization_config() -> BitsAndBytesConfig:
    """
    Create the QLoRA 4-bit quantization configuration.
    """

    compute_dtype = get_compute_dtype()

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )