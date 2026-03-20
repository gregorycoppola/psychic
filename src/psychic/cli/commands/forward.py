"""Run a single forward pass on real text and time it."""
import time
import numpy as np
from pathlib import Path
from rich.console import Console

console = Console()

DEFAULT_CACHE = Path.home() / ".cache" / "psychic"


def add_subparser(subparsers):
    p = subparsers.add_parser("forward", help="Run a single forward pass on real text")
    p.add_argument("model", nargs="?", default="gpt2", help="Model name")
    p.add_argument("--text", type=str, default="The cat sat on the", help="Input text")
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="Cache directory")
    p.set_defaults(func=cmd_forward)


def cmd_forward(args):
    from psychic.core.loader import load_safetensors
    from psychic.core.forward import forward_pass
    from psychic.core.tokenizer import BPETokenizer
    from psychic.core.models import get_config

    cache = Path(args.cache)

    try:
        cfg = get_config(args.model)
    except FileNotFoundError as e:
        console.print(f"[red]✗[/red] {e}")
        raise SystemExit(1)

    weights_path = cache / cfg["safetensors_filename"]
    vocab_path = cache / cfg["vocab_filename"]
    merges_path = cache / cfg["merges_filename"]

    if not weights_path.exists():
        console.print(f"[red]✗[/red] Weights not found.")
        raise SystemExit(1)
    if not vocab_path.exists() or not merges_path.exists():
        console.print(f"[red]✗[/red] Vocab not found.")
        raise SystemExit(1)

    tokenizer = BPETokenizer(vocab_path, merges_path)
    console.print(f"Loading weights ({cfg['parameters_m']}M params)...")
    t0 = time.time()
    weights = load_safetensors(weights_path)
    console.print(f"  loaded in {time.time() - t0:.2f}s")

    token_ids = tokenizer.encode(args.text)
    decoded = [tokenizer.decode([t]) for t in token_ids]
    console.print(f"Input: [cyan]{args.text}[/cyan]")
    console.print(f"Tokens ({len(token_ids)}): {decoded}")

    console.print("Running forward pass...")
    t0 = time.time()
    logits, patterns = forward_pass(weights, token_ids, cfg)
    console.print(f"  [green]done in {time.time() - t0:.2f}s[/green]")

    next_logits = logits[-1]
    top5 = np.argsort(next_logits)[-5:][::-1]
    top5_tokens = [tokenizer.decode([i]) for i in top5]
    console.print(f"Top-5 next tokens: {top5_tokens}")