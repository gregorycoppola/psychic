"""
Analysis functions over attention pattern matrices.

Each function takes a single attention pattern matrix of shape
[seq_len, seq_len] and returns a float score.

Register new analyses in the ANALYSES dict at the bottom.
The patterns command picks up all registered analyses automatically.
"""
import numpy as np


def avg_entropy(pattern: np.ndarray) -> float:
    """
    Average entropy of attention rows.
    Low = sharp (concentrated), high = diffuse (spread out).
    """
    entropies = []
    for row in pattern:
        p = row + 1e-10
        entropies.append(float(-np.sum(p * np.log(p))))
    return float(np.mean(entropies))


def avg_max_attn(pattern: np.ndarray) -> float:
    """
    Average max attention weight per row.
    High = sharp, low = diffuse.
    """
    return float(np.mean(pattern.max(axis=-1)))


def prev_token_score(pattern: np.ndarray) -> float:
    """
    Average attention weight on the previous token position.
    pattern[i, i-1] for i > 0.
    High = previous token head.
    """
    if pattern.shape[0] < 2:
        return 0.0
    scores = [pattern[i, i - 1] for i in range(1, pattern.shape[0])]
    return float(np.mean(scores))


def first_token_score(pattern: np.ndarray) -> float:
    """
    Average attention weight on token position 0.
    High = global context / BOS head.
    """
    return float(np.mean(pattern[:, 0]))


def diagonal_score(pattern: np.ndarray) -> float:
    """
    Average attention weight on the current token (self-attention diagonal).
    High = self-attending head.
    """
    n = pattern.shape[0]
    return float(np.mean([pattern[i, i] for i in range(n)]))


# Registry: name -> function
# Add new analyses here. The patterns command picks them up automatically.
ANALYSES = {
    "entropy": avg_entropy,
    "max_attn": avg_max_attn,
    "prev_tok": prev_token_score,
    "first_tok": first_token_score,
    "diagonal": diagonal_score,
}


def classify_head(scores: dict) -> str:
    """
    Given a dict of analysis scores for a head, return a type hint string.
    Extend this as we learn more about what the scores mean.
    """
    if scores["prev_tok"] > 0.4:
        return "prev-token"
    if scores["first_tok"] > 0.4:
        return "first-token"
    if scores["diagonal"] > 0.4:
        return "self-attn"
    if scores["max_attn"] > 0.6:
        return "sharp"
    if scores["entropy"] < 1.0:
        return "moderate"
    return "diffuse"