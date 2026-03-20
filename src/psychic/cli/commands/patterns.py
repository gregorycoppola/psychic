"""Analyze attention patterns across a set of prompts."""
import json
import struct
import time
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_CACHE = Path.home() / ".cache" / "psychic"

PROMPTS = [
    "The cat sat on the mat and looked at the door",
    "In 1969 Neil Armstrong walked on the moon for the first time",
    "The president signed the bill into law yesterday afternoon",
    "She opened the book and began to read the first chapter",
    "The quick brown fox jumps over the lazy dog near the river",
    "Python is a programming language that emphasizes code readability",
    "The stock market fell sharply on fears of rising interest rates",
    "He walked into the room and saw that everyone had already left",
    "The train arrived at the station exactly on time this morning",
    "Scientists discovered a new species of bird in the Amazon rainforest",
    "The recipe calls for two cups of flour and one egg",
    "She said that she would call him back later that evening",
    "The company announced record profits for the third quarter",
    "All men are mortal and Socrates is a man therefore Socrates is mortal",
    "The cat chased the mouse and the mouse ran under the table",
    "He put the key in the lock and turned it slowly",
    "The sun rises in the east and sets in the west every day",
    "They decided to take the longer route to avoid the traffic",
    "The doctor told the patient to rest and drink plenty of water",
    "Once upon a time there was a princess who lived in a tower",
]


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


def attention_entropy(pattern_row: np.ndarray) -> float:
    """Entropy of one row of an attention pattern. Low = sharp, high = diffuse."""
    p = pattern_row + 1e-10
    return float(-np.sum(p * np.log(p)))


def max_attention(pattern_row: np.ndarray) -> float:
    """Max attention weight in a row. High = sharp, low = diffuse."""
    return float(pattern_row.max())


def prev_token_score(pattern: np.ndarray) -> float:
    """
    How much does this head attend to the previous token?
    Average of pattern[i, i-1] for i > 0.
    High = previous token head.
    """
    if pattern.shape[0] < 2:
        return 0.0
    scores = [pattern[i, i-1] for i in range(1, pattern.shape[0])]
    return float(np.mean(scores))


def add_subparser(subparsers):
    p = subparsers.add_parser("patterns", help="Analyze attention patterns across prompts")
    p.add_argument("model", nargs="?", default="gpt2", help="Model name")
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="Cache directory")
    p.add_argument("--layer", type=int, default=None, help="Show only this layer")
    p.set_defaults(func=cmd_patterns)


def cmd_patterns(args):
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

    tokenizer = BPETokenizer(vocab_path, merges_path)

    console.print(f"Loading weights...")
    weights = load_safetensors(path)

    n_layers = 12
    n_heads = 12

    # accumulators: layer -> head -> list of scores
    entropy_acc = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
    max_attn_acc = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
    prev_tok_acc = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    console.print(f"Running {len(PROMPTS)} prompts...")
    t0 = time.time()

    for i, prompt in enumerate(PROMPTS):
        token_ids = tokenizer.encode(prompt)
        _, patterns = forward_pass(weights, token_ids)

        for layer in range(n_layers):
            for head in range(n_heads):
                pat = patterns[layer][head]  # [seq, seq]
                # entropy and max averaged over all token positions
                entropies = [attention_entropy(pat[pos]) for pos in range(pat.shape[0])]
                maxes = [max_attention(pat[pos]) for pos in range(pat.shape[0])]
                entropy_acc[layer][head].append(np.mean(entropies))
                max_attn_acc[layer][head].append(np.mean(maxes))
                prev_tok_acc[layer][head].append(prev_token_score(pat))

        console.print(f"  [{i+1}/{len(PROMPTS)}] {prompt[:50]}")

    t_total = time.time() - t0
    console.print(f"[green]done in {t_total:.1f}s[/green]\n")

    # display results
    layers = [args.layer] if args.layer is not None else range(n_layers)

    table = Table(title=f"Attention Pattern Analysis: {args.model}")
    table.add_column("Layer", style="cyan", justify="right")
    table.add_column("Head", style="cyan", justify="right")
    table.add_column("Avg entropy", style="yellow", justify="right")
    table.add_column("Avg max-attn", style="green", justify="right")
    table.add_column("Prev-tok score", style="magenta", justify="right")
    table.add_column("Type hint", style="white")

    for layer in layers:
        for head in range(n_heads):
            avg_ent = np.mean(entropy_acc[layer][head])
            avg_max = np.mean(max_attn_acc[layer][head])
            avg_prev = np.mean(prev_tok_acc[layer][head])

            if avg_prev > 0.4:
                hint = "[bold]prev-token[/bold]"
            elif avg_max > 0.5:
                hint = "[bold]sharp[/bold]"
            elif avg_ent < 1.0:
                hint = "moderate"
            else:
                hint = "diffuse"

            table.add_row(
                str(layer),
                str(head),
                f"{avg_ent:.3f}",
                f"{avg_max:.3f}",
                f"{avg_prev:.3f}",
                hint,
            )

    console.print(table)