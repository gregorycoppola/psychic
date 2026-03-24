"""Llama 3 family forward pass. Covers Llama 3.2 1B and 3B."""
import numpy as np


def rms_norm(x, weight, eps=1e-5):
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return weight * (x / rms)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def silu(x):
    return x / (1 + np.exp(-x))


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope(q, k, seq_len, d_head, rope_theta=500000.0):
    positions = np.arange(seq_len)
    freqs = 1.0 / (rope_theta ** (np.arange(0, d_head, 2).astype(np.float32) / d_head))
    angles = np.outer(positions, freqs)
    cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)
    sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)
    cos = cos[np.newaxis, :, :]
    sin = sin[np.newaxis, :, :]
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


def forward_pass(weights, token_ids, cfg: dict):
    """
    Llama 3 family forward pass.
    Returns (logits, attention_patterns).
      logits:             [seq_len, vocab_size]
      attention_patterns: list of [n_heads, seq_len, seq_len] per layer
    """
    n_layers = cfg["n_layers"]
    n_heads = cfg["n_heads"]
    n_kv_heads = cfg.get("n_kv_heads", n_heads)
    d_model = cfg["d_model"]
    d_head = cfg["d_head"]
    rope_theta = cfg.get("rope_theta", 500000.0)
    n_groups = n_heads // n_kv_heads

    seq_len = len(token_ids)
    token_ids = np.array(token_ids)

    embed = weights["model.embed_tokens.weight"]
    x = embed[token_ids].copy().astype(np.float32)

    all_patterns = []

    for layer in range(n_layers):
        # pre-attention RMSNorm
        ln1_w = weights[f"model.layers.{layer}.input_layernorm.weight"]
        x_ln = rms_norm(x, ln1_w)

        # QKV projections — no bias in Llama 3
        wq = weights[f"model.layers.{layer}.self_attn.q_proj.weight"]
        wk = weights[f"model.layers.{layer}.self_attn.k_proj.weight"]
        wv = weights[f"model.layers.{layer}.self_attn.v_proj.weight"]

        Q = (x_ln @ wq.T).reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
        K = (x_ln @ wk.T).reshape(seq_len, n_kv_heads, d_head).transpose(1, 0, 2)
        V = (x_ln @ wv.T).reshape(seq_len, n_kv_heads, d_head).transpose(1, 0, 2)

        Q, K = apply_rope(Q, K, seq_len, d_head, rope_theta)

        if n_groups > 1:
            K = np.repeat(K, n_groups, axis=0)
            V = np.repeat(V, n_groups, axis=0)

        scale = np.sqrt(d_head)
        scores = Q @ K.transpose(0, 2, 1) / scale
        mask = np.triu(np.full((seq_len, seq_len), -1e10), k=1)
        scores = scores + mask
        patterns = softmax(scores, axis=-1)
        all_patterns.append(patterns)

        attn_out = patterns @ V
        attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, d_model)

        wo = weights[f"model.layers.{layer}.self_attn.o_proj.weight"]
        attn_out = attn_out @ wo.T
        x = x + attn_out

        # post-attention RMSNorm
        ln2_w = weights[f"model.layers.{layer}.post_attention_layernorm.weight"]
        x_ln2 = rms_norm(x, ln2_w)

        # SwiGLU FFN
        w_gate = weights[f"model.layers.{layer}.mlp.gate_proj.weight"]
        w_up   = weights[f"model.layers.{layer}.mlp.up_proj.weight"]
        w_down = weights[f"model.layers.{layer}.mlp.down_proj.weight"]

        gate = x_ln2 @ w_gate.T
        up   = x_ln2 @ w_up.T
        ffn_out = (silu(gate) * up) @ w_down.T
        x = x + ffn_out

    # final norm
    ln_f_w = weights["model.norm.weight"]
    x = rms_norm(x, ln_f_w)

    # lm_head — Llama 3.2 1B does NOT tie embeddings
    lm_head = weights["lm_head.weight"]
    logits = x @ lm_head.T

    return logits, all_patterns