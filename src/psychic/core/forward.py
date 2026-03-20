"""GPT-2 forward pass. Returns logits and attention patterns."""
import numpy as np


def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return weight * (x - mean) / np.sqrt(var + eps) + bias


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def forward_pass(weights, token_ids, n_layers=12, n_heads=12, d_model=768, d_head=64):
    """
    Full GPT-2 forward pass.
    Returns (logits, attention_patterns).
      logits:           [seq_len, vocab_size]
      attention_patterns: list of [n_heads, seq_len, seq_len] per layer
    """
    seq_len = len(token_ids)
    token_ids = np.array(token_ids)

    wte = weights["wte.weight"]
    wpe = weights["wpe.weight"]
    x = wte[token_ids] + wpe[:seq_len]

    all_patterns = []

    for layer in range(n_layers):
        ln1_w = weights[f"h.{layer}.ln_1.weight"]
        ln1_b = weights[f"h.{layer}.ln_1.bias"]
        x_ln = layer_norm(x, ln1_w, ln1_b)

        c_attn_w = weights[f"h.{layer}.attn.c_attn.weight"]
        c_attn_b = weights[f"h.{layer}.attn.c_attn.bias"]
        qkv = x_ln @ c_attn_w + c_attn_b

        Q = qkv[:, :d_model]
        K = qkv[:, d_model:2*d_model]
        V = qkv[:, 2*d_model:]

        Q = Q.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
        K = K.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
        V = V.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)

        scale = np.sqrt(d_head)
        scores = Q @ K.transpose(0, 2, 1) / scale
        mask = np.triu(np.full((seq_len, seq_len), -1e10), k=1)
        scores = scores + mask
        patterns = softmax(scores, axis=-1)
        all_patterns.append(patterns)

        attn_out = patterns @ V
        attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, d_model)

        c_proj_w = weights[f"h.{layer}.attn.c_proj.weight"]
        c_proj_b = weights[f"h.{layer}.attn.c_proj.bias"]
        attn_out = attn_out @ c_proj_w + c_proj_b
        x = x + attn_out

        ln2_w = weights[f"h.{layer}.ln_2.weight"]
        ln2_b = weights[f"h.{layer}.ln_2.bias"]
        x_ln2 = layer_norm(x, ln2_w, ln2_b)

        fc_w = weights[f"h.{layer}.mlp.c_fc.weight"]
        fc_b = weights[f"h.{layer}.mlp.c_fc.bias"]
        proj_w = weights[f"h.{layer}.mlp.c_proj.weight"]
        proj_b = weights[f"h.{layer}.mlp.c_proj.bias"]

        h = gelu(x_ln2 @ fc_w + fc_b)
        ffn_out = h @ proj_w + proj_b
        x = x + ffn_out

    ln_f_w = weights["ln_f.weight"]
    ln_f_b = weights["ln_f.bias"]
    x = layer_norm(x, ln_f_w, ln_f_b)
    logits = x @ weights["wte.weight"].T

    return logits, all_patterns