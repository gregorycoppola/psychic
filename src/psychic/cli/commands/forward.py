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
    seq_len = len(token_ids)
    token_ids = np.array(token_ids)

    wte = weights["wte.weight"]
    wpe = weights["wpe.weight"]
    x = wte[token_ids] + wpe[:seq_len]

    all_patterns = []

    for layer in range(12):
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


def add_subparser(subparsers):
    p = subparsers.add_parser("forward", help="Run a single forward pass on real text")
    p.add_argument("model", nargs="?", default="gpt2", help="Model name")
    p.add_argument("--text", type=str, default="Hello, my name is Claude and I think therefore I am", help="Input text")
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="Cache directory")
    p.set_defaults(func=cmd_forward)


def cmd_forward(args):
    cache = Path(args.cache)
    path = cache / f"{args.model}.safetensors"
    vocab_path = cache / f"{args.model}_vocab.json"
    merges_path = cache / f"{args.model}_merges.txt"

    if not path.exists():
        console.print(f"[red]✗[/red] Weights not found. Run psychic download.")
        raise SystemExit(1)
    if not vocab_path.exists() or not merges_path.exists():
        console.print(f"[red]✗[/red] Vocab not found. Run psychic download-vocab.")
        raise SystemExit(1)

    from psychic.core.tokenizer import BPETokenizer

    console.print(f"Loading tokenizer...")
    tokenizer = BPETokenizer(vocab_path, merges_path)

    console.print(f"Loading weights...")
    t0 = time.time()
    weights = load_safetensors(path)
    t_load = time.time() - t0
    console.print(f"  loaded in {t_load:.2f}s")

    token_ids = tokenizer.encode(args.text)
    decoded = [tokenizer.decode([t]) for t in token_ids]
    console.print(f"Input: [cyan]{args.text}[/cyan]")
    console.print(f"Tokens ({len(token_ids)}): {decoded}")

    console.print(f"Running forward pass...")
    t0 = time.time()
    logits, patterns = forward_pass(weights, token_ids)
    t_fwd = time.time() - t0
    console.print(f"  [green]done in {t_fwd:.2f}s[/green]")

    # top 5 predicted next tokens
    next_logits = logits[-1]
    top5 = np.argsort(next_logits)[-5:][::-1]
    top5_tokens = [tokenizer.decode([i]) for i in top5]
    console.print(f"Top-5 next tokens: {top5_tokens}")