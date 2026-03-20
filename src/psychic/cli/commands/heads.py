"""Extract and analyze attention heads from model weights."""
import json
import struct
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_CACHE = Path.home() / ".cache" / "psychic"

# GPT-2 small config
GPT2_CONFIG = {
    "n_layers": 12,
    "n_heads": 12,
    "d_model": 768,
    "d_head": 64,  # d_model / n_heads
}


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


def effective_rank(matrix: np.ndarray) -> float:
    """
    Compute the effective rank of a matrix via singular value entropy.
    A rank-1 matrix has effective rank ~1.
    A full-rank matrix has effective rank ~min(rows, cols).
    """
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    s = s[s > 1e-10]  # drop near-zero singular values
    s_norm = s / s.sum()
    entropy = -np.sum(s_norm * np.log(s_norm + 1e-10))
    return float(np.exp(entropy))


def top_singular_ratio(matrix: np.ndarray) -> float:
    """
    Ratio of largest singular value to sum of all singular values.
    Close to 1.0 = very rank-1-like (sharp, BP-style).
    Close to 0.0 = diffuse, spread across many dimensions.
    """
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    if s.sum() < 1e-10:
        return 0.0
    return float(s[0] / s.sum())


def add_subparser(subparsers):
    p = subparsers.add_parser("heads", help="Analyze attention heads — rank and sharpness")
    p.add_argument("model", nargs="?", default="gpt2", help="Model name")
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="Cache directory")
    p.add_argument("--layer", type=int, default=None, help="Show only this layer")
    p.set_defaults(func=cmd_heads)


def cmd_heads(args):
    cache = Path(args.cache)
    path = cache / f"{args.model}.safetensors"

    if not path.exists():
        console.print(f"[red]✗[/red] Not found: {path}")
        console.print("Run [bold]psychic download[/bold] first.")
        raise SystemExit(1)

    console.print(f"Loading [bold]{path.name}[/bold]...")
    weights = load_safetensors(path)

    cfg = GPT2_CONFIG
    n_layers = cfg["n_layers"]
    n_heads = cfg["n_heads"]
    d_model = cfg["d_model"]
    d_head = cfg["d_head"]

    table = Table(title=f"Attention Heads: {args.model}")
    table.add_column("Layer", style="cyan", justify="right")
    table.add_column("Head", style="cyan", justify="right")
    table.add_column("Q rank", style="yellow", justify="right")
    table.add_column("K rank", style="yellow", justify="right")
    table.add_column("Q top-sv", style="green", justify="right")
    table.add_column("K top-sv", style="green", justify="right")
    table.add_column("Type hint", style="magenta")

    layers = [args.layer] if args.layer is not None else range(n_layers)

    for layer in layers:
        key = f"h.{layer}.attn.c_attn.weight"
        if key not in weights:
            console.print(f"[red]✗[/red] Missing: {key}")
            continue

        # c_attn.weight is [d_model, 3 * d_model] = [768, 2304]
        # split into Q, K, V along axis 1
        w = weights[key]  # [768, 2304]
        W_Q = w[:, :d_model]           # [768, 768]
        W_K = w[:, d_model:2*d_model]  # [768, 768]

        for head in range(n_heads):
            # each head slice is d_head=64 columns
            q = W_Q[:, head*d_head:(head+1)*d_head]  # [768, 64]
            k = W_K[:, head*d_head:(head+1)*d_head]  # [768, 64]

            q_rank = effective_rank(q)
            k_rank = effective_rank(k)
            q_top = top_singular_ratio(q)
            k_top = top_singular_ratio(k)

            # simple heuristic type hint
            avg_top = (q_top + k_top) / 2
            if avg_top > 0.4:
                hint = "sharp (BP-like)"
            elif avg_top > 0.2:
                hint = "medium"
            else:
                hint = "diffuse"

            table.add_row(
                str(layer),
                str(head),
                f"{q_rank:.1f}",
                f"{k_rank:.1f}",
                f"{q_top:.3f}",
                f"{k_top:.3f}",
                hint,
            )

    console.print(table)