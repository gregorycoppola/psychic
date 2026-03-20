"""Analyze attention patterns across a set of prompts."""
import time
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_CACHE = Path.home() / ".cache" / "psychic"


def add_subparser(subparsers):
    p = subparsers.add_parser("patterns", help="Analyze attention patterns across prompts")
    p.add_argument("model", nargs="?", default="gpt2", help="Model name")
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="Cache directory")
    p.add_argument("--layer", type=int, default=None, help="Show only this layer")
    p.add_argument("--prompts", nargs="+", default=None, help="Prompt file names to use (default: all)")
    p.add_argument("--batch-size", type=int, default=10, help="Print progress every N prompts")
    p.set_defaults(func=cmd_patterns)


def cmd_patterns(args):
    from psychic.core.loader import load_safetensors
    from psychic.core.forward import forward_pass
    from psychic.core.analysis import ANALYSES, classify_head
    from psychic.core.prompts import load_prompts
    from psychic.core.tokenizer import BPETokenizer

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

    tokenizer = BPETokenizer(vocab_path, merges_path)
    console.print("Loading weights...")
    weights = load_safetensors(path)

    prompts = load_prompts(args.prompts)
    console.print(f"Loaded {len(prompts)} prompts")

    n_layers = 12
    n_heads = 12

    # accumulators: layer -> head -> analysis_name -> list of scores
    acc = {
        layer: {
            head: {name: [] for name in ANALYSES}
            for head in range(n_heads)
        }
        for layer in range(n_layers)
    }

    console.print(f"Running {len(prompts)} prompts...")
    t0 = time.time()

    for i, prompt in enumerate(prompts):
        token_ids = tokenizer.encode(prompt)
        _, patterns = forward_pass(weights, token_ids)

        for layer in range(n_layers):
            for head in range(n_heads):
                pat = patterns[layer][head]
                for name, fn in ANALYSES.items():
                    acc[layer][head][name].append(fn(pat))

        if (i + 1) % args.batch_size == 0 or (i + 1) == len(prompts):
            elapsed = time.time() - t0
            console.print(f"  [{i+1}/{len(prompts)}] {elapsed:.1f}s — {prompt[:50]}")

    console.print(f"[green]done[/green]\n")

    # display
    layers = [args.layer] if args.layer is not None else range(n_layers)

    table = Table(title=f"Attention Pattern Analysis: {args.model}")
    table.add_column("Layer", style="cyan", justify="right")
    table.add_column("Head", style="cyan", justify="right")
    for name in ANALYSES:
        table.add_column(name, justify="right")
    table.add_column("Type", style="bold magenta")

    for layer in layers:
        for head in range(n_heads):
            avg_scores = {
                name: float(np.mean(acc[layer][head][name]))
                for name in ANALYSES
            }
            hint = classify_head(avg_scores)
            row = [str(layer), str(head)]
            row += [f"{avg_scores[name]:.3f}" for name in ANALYSES]
            row += [hint]
            table.add_row(*row)

    console.print(table)