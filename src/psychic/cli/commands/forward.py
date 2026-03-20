"""Run a single forward pass and time it."""
import time
import json
import struct
import numpy as np
from pathlib import Path
from rich.console import Console

console = Console()

DEFAULT_CACHE = Path.home() / ".cache" / "psychic"


def load_safetensors(path: Path) -> dict:
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes)
        data_start = 8 + header_len
        f.seek(data_start)
        data = f.read()
    tensors = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        shape = meta["shape"]
        raw = data[start:end]
        if meta["dtype"] == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16)
            u32 = u16.astype(np.uint32) << 16
            arr = u32.view(np.float32).reshape(shape)
        elif meta["dtype"] == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
        else:
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        tensors[name] = arr
    return tensors


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


def forward_pass(weights, token_ids, n_heads=12, d_model=768, d_head=64):
    """
    Full GPT-2 forward pass. Returns (logits, attention_patterns).
    attention_patterns: list of [n_heads, seq_len, seq_len] per layer.
    """
    seq_len = len(token_ids)
    token_ids = np.array(token_ids)

    # embeddings
    wte = weights["wte.weight"]   # [vocab, d_model]
    wpe = weights["wpe.weight"]   # [1024, d_model]
    x = wte[token_ids] + wpe[:seq_len]  # [seq, d_model]

    all_patterns = []

    for layer in range(12):
        # layer norm 1
        ln1_w = weights[f"h.{layer}.ln_1.weight"]
        ln1_b = weights[f"h.{layer}.ln_1.bias"]
        x_ln = layer_norm(x, ln1_w, ln1_b)

        # attention
        c_attn_w = weights[f"h.{layer}.attn.c_attn.weight"]  # [768, 2304]
        c_attn_b = weights[f"h.{layer}.attn.c_attn.bias"]    # [2304]
        qkv = x_ln @ c_attn_w + c_attn_b                     # [seq, 2304]

        Q = qkv[:, :d_model]            # [seq, 768]
        K = qkv[:, d_model:2*d_model]   # [seq, 768]
        V = qkv[:, 2*d_model:]          # [seq, 768]

        # split into heads
        Q = Q.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)  # [heads, seq, d_head]
        K = K.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
        V = V.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)

        # attention scores
        scale = np.sqrt(d_head)
        scores = Q @ K.transpose(0, 2, 1) / scale  # [heads, seq, seq]

        # causal mask
        mask = np.triu(np.full((seq_len, seq_len), -1e10), k=1)
        scores = scores + mask

        patterns = softmax(scores, axis=-1)  # [heads, seq, seq]
        all_patterns.append(patterns)

        # weighted sum
        attn_out = patterns @ V  # [heads, seq, d_head]
        attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, d_model)

        # output projection
        c_proj_w = weights[f"h.{layer}.attn.c_proj.weight"]  # [768, 768]
        c_proj_b = weights[f"h.{layer}.attn.c_proj.bias"]
        attn_out = attn_out @ c_proj_w + c_proj_b

        x = x + attn_out

        # layer norm 2
        ln2_w = weights[f"h.{layer}.ln_2.weight"]
        ln2_b = weights[f"h.{layer}.ln_2.bias"]
        x_ln2 = layer_norm(x, ln2_w, ln2_b)

        # FFN
        fc_w = weights[f"h.{layer}.mlp.c_fc.weight"]    # [768, 3072]
        fc_b = weights[f"h.{layer}.mlp.c_fc.bias"]
        proj_w = weights[f"h.{layer}.mlp.c_proj.weight"] # [3072, 768]
        proj_b = weights[f"h.{layer}.mlp.c_proj.bias"]

        h = gelu(x_ln2 @ fc_w + fc_b)
        ffn_out = h @ proj_w + proj_b

        x = x + ffn_out

    # final layer norm
    ln_f_w = weights["ln_f.weight"]
    ln_f_b = weights["ln_f.bias"]
    x = layer_norm(x, ln_f_w, ln_f_b)

    # logits
    logits = x @ weights["wte.weight"].T  # [seq, vocab]

    return logits, all_patterns


def add_subparser(subparsers):
    p = subparsers.add_parser("forward", help="Run a single forward pass and time it")
    p.add_argument("model", nargs="?", default="gpt2", help="Model name")
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="Cache directory")
    p.add_argument("--seq-len", type=int, default=16, help="Sequence length to test")
    p.set_defaults(func=cmd_forward)


def cmd_forward(args):
    cache = Path(args.cache)
    path = cache / f"{args.model}.safetensors"

    if not path.exists():
        console.print(f"[red]✗[/red] Not found: {path}")
        raise SystemExit(1)

    console.print(f"Loading weights...")
    t0 = time.time()
    weights = load_safetensors(path)
    t_load = time.time() - t0
    console.print(f"  loaded in {t_load:.2f}s")

    # hardcoded token ids — "Hello, my name is" in GPT-2 tokens
    token_ids = [15496, 11, 616, 1438, 318] + [0] * (args.seq_len - 5)
    token_ids = token_ids[:args.seq_len]

    console.print(f"Running forward pass (seq_len={args.seq_len})...")
    t0 = time.time()
    logits, patterns = forward_pass(weights, token_ids)
    t_fwd = time.time() - t0

    console.print(f"  [green]done in {t_fwd:.2f}s[/green]")
    console.print(f"  logits shape: {logits.shape}")
    console.print(f"  layers: {len(patterns)}")
    console.print(f"  patterns[0] shape: {patterns[0].shape}")

    # top predicted next token
    next_token_logits = logits[-1]
    top5 = np.argsort(next_token_logits)[-5:][::-1]
    console.print(f"  top-5 next token ids: {top5.tolist()}")