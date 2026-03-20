"""
Per-prompt head classification.
"""
import numpy as np
from psychic.core.analysis import ANALYSES


HEAD_TYPES = ["prev-token", "first-token", "self-attn", "sharp", "diffuse"]


def classify_single(scores: dict) -> str:
    if scores["prev_tok"] > 0.4:
        return "prev-token"
    if scores["first_tok"] > 0.5:
        return "first-token"
    if scores["diagonal"] > 0.5:
        return "self-attn"
    if scores["max_attn"] > 0.6:
        return "sharp"
    return "diffuse"


def scores_for_pattern(pat: np.ndarray) -> dict:
    """
    Given a full [seq_len, seq_len] attention pattern,
    compute all analysis scores over the full matrix.
    """
    return {
        name: fn(pat)
        for name, fn in ANALYSES.items()
    }


def classify_all_prompts(patterns_per_prompt: list, n_layers: int, n_heads: int) -> dict:
    counts = {
        layer: {
            head: {t: 0 for t in HEAD_TYPES}
            for head in range(n_heads)
        }
        for layer in range(n_layers)
    }

    for patterns in patterns_per_prompt:
        for layer in range(n_layers):
            for head in range(n_heads):
                pat = patterns[layer][head]  # [seq_len, seq_len]
                scores = scores_for_pattern(pat)
                t = classify_single(scores)
                counts[layer][head][t] += 1

    return counts


def dominant_type(counts: dict) -> str:
    return max(counts, key=counts.get)