"""Forward pass router — dispatches to family-specific implementation."""
import numpy as np


def forward_pass(weights, token_ids, cfg: dict):
    """
    Route to the correct forward pass based on model family.
    Returns (logits, attention_patterns).
    """
    family = cfg.get("family", "gpt2")

    if family == "gpt2":
        from psychic.core.forward_gpt2 import forward_pass as _forward
    elif family == "qwen2":
        from psychic.core.forward_qwen2 import forward_pass as _forward
    else:
        raise ValueError(f"Unknown model family: {family}")

    return _forward(weights, token_ids, cfg)