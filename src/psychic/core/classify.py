"""
Per-prompt head classification.

Instead of averaging scores across all prompts and positions,
classify each head per prompt, then aggregate into a distribution.
"""
import numpy as np
from psychic.core.analysis import ANALYSES


HEAD_TYPES = ["prev-token", "first-token", "self-attn", "sharp", "diffuse"]


def classify_single(scores: dict) -> str:
    """
    Classify a single (head, prompt) observation into one head type.
    scores is a dict of analysis_name -> float, averaged over positions
    for this one prompt.
    """
    if scores["prev_tok"] > 0.4:
        return "prev-token"
    if scores["first_tok"] > 0.5:
        return "first-token"
    if scores["diagonal"] > 0.5:
        return "self-attn"
    if scores["max_attn"] > 0.6:
        return "sharp"
    return "diffuse"


def scores_for_prompt(patterns_layer_head: np.ndarray) -> dict:
    """
    Given a single [seq_len, seq_len] attention pattern,
    compute all analysis scores averaged over token positions.
    """
    return {
        name: float(np.mean([fn(patterns_layer_head[pos:pos+1, :pos+1])
                              if patterns_layer_head[pos:pos+1, :pos+1].shape[1] > 0
                              else fn(patterns_layer_head)
                              for pos in range(patterns_layer_head.shape[0])]))
        for name, fn in ANALYSES.items()
    }


def classify_all_prompts(patterns_per_prompt: list, n_layers: int, n_heads: int) -> dict:
    """
    patterns_per_prompt: list of per-prompt attention patterns
      each entry is a list of [n_heads, seq_len, seq_len] per layer

    Returns: dict[layer][head] -> dict of type -> count
    """
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
                scores = {
                    name: float(np.mean([fn(pat[pos:pos+1])
                                         for pos in range(pat.shape[0])]))
                    for name, fn in ANALYSES.items()
                }
                t = classify_single(scores)
                counts[layer][head][t] += 1

    return counts


def dominant_type(counts: dict) -> str:
    return max(counts, key=counts.get)