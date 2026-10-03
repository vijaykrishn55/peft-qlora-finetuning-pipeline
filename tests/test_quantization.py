import torch

from src.model.quantization import (
    create_quantization_config,
    get_compute_dtype,
)


def test_compute_dtype_is_supported():
    dtype = get_compute_dtype()

    assert dtype in {
        torch.float16,
        torch.bfloat16,
    }


def test_nf4_quantization_configuration():
    config = create_quantization_config()

    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_use_double_quant is True


def test_compute_dtype_configuration():
    config = create_quantization_config()

    assert config.bnb_4bit_compute_dtype in {
        torch.float16,
        torch.bfloat16,
    }