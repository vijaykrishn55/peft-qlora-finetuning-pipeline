from typing import Dict


def format_chatml(instruction: str, response: str) -> Dict[str, str]:
    """
    Convert an instruction-response pair into ChatML-style text.

    Args:
        instruction: The user's question or instruction.
        response: The expected assistant answer.

    Returns:
        A dictionary containing the prompt and complete training text.
    """

    prompt = (
        "<|im_start|>user\n"
        f"{instruction}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    full_text = (
        prompt
        + f"{response}\n"
        + "<|im_end|>"
    )

    return {
        "prompt": prompt,
        "text": full_text,
    }