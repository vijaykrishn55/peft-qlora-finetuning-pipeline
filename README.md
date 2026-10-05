
# PEFT & QLoRA Fine-Tuning Pipeline

A small production-style pipeline for fine-tuning a language model using  **PEFT and QLoRA** . The project covers instruction formatting, 4-bit NF4 quantization, LoRA adapters, training, adapter merging, and inference verification.

## What this project does

The pipeline:

1. Formats instruction-response data using ChatML.
2. Tokenizes the dataset and masks prompt tokens with `<span>-100</span>`.
3. Loads `<span>Qwen/Qwen2.5-0.5B-Instruct</span>` using 4-bit NF4 quantization.
4. Applies LoRA adapters to the attention and feed-forward projections.
5. Fine-tunes using gradient checkpointing and an 8-bit paged optimizer.
6. Saves the trained LoRA adapter.
7. Reloads and merges the adapter into the base model.
8. Saves and reloads the merged model.
9. Verifies inference before and after fine-tuning.

## Project Structure

```text
├── data/
│   └── raw/
│       └── train.json
│
├── src/
│   ├── data/
│   │   ├── formatter.py
│   │   └── tokenizer.py
│   │
│   ├── model/
│   │   ├── loader.py
│   │   ├── lora.py
│   │   └── quantization.py
│   │
│   └── training/
│       ├── config.py
│       └── trainer.py
│
├── scripts/
│   ├── prepare_dataset.py
│   └── train.py
│
├── tests/
│   ├── test_dataset.py
│   ├── test_lora.py
│   ├── test_quantization.py
│   └── test_training.py
│
└── artifacts/
    ├── adapter/
    ├── merged_model/
    └── training_metrics.json
```

## QLoRA Configuration

* Model: `<span>Qwen/Qwen2.5-0.5B-Instruct</span>`
* Quantization: 4-bit NF4
* Double quantization: enabled
* Compute dtype: FP16 on the training GPU
* LoRA rank: 16
* LoRA alpha: 32
* LoRA dropout: 0.05
* Optimizer: `<span>paged_adamw_8bit</span>`
* Gradient checkpointing: enabled
* Learning rate: `<span>2e-4</span>`
* Epochs: 3
* Batch size: 1
* Gradient accumulation: 8

The final LoRA configuration reported:

```text
Trainable parameters: 8,798,208
Total parameters: 502,830,976
Trainable percentage: 1.7497%
```

This satisfies the assessment requirement of keeping trainable parameters below 2%.

## Training Result

Training completed successfully on a Kaggle GPU environment.

```text
Global steps: 3
Training loss: 2.7022
Mean token accuracy: 0.6732
Epochs: 3
Training runtime: 4.48 seconds
```

The trained adapter was saved to:

```text
artifacts/adapter/
```

The merged model was saved to:

```text
artifacts/merged_model/
```

## Verification

The adapter was successfully:

* saved
* reloaded
* merged into the base model
* saved as a standalone model
* reloaded independently
* used for inference

The fine-tuned and merged models produced the same response for the verification prompt, confirming that the merged model retained the adapter behavior.

## Tests

```text
13 passed
14 warnings
```

The tests cover:

* ChatML formatting
* Prompt masking
* Tokenization lengths
* LoRA configuration
* Target modules
* NF4 quantization
* Double quantization
* Compute dtype
* Training configuration
* Gradient checkpointing
* Paged optimizer
* Precision configuration

The warnings came from a PyTorch deprecation warning and did not affect the tests.

## Limitations

The training dataset used for this assessment is intentionally small. Because of that, the fine-tuning run demonstrates the complete QLoRA workflow but is not intended to measure meaningful model improvement.

For a real fine-tuning task, the dataset should be substantially larger and evaluated with a dedicated validation set and task-specific metrics.

## Running Tests

```bash
python -m pytest -v
```

The actual QLoRA training requires a CUDA-enabled GPU. The training script intentionally stops with an error when CUDA is unavailable rather than attempting to run 4-bit GPU training on a CPU-only machine.
