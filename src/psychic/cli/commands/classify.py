"""Per-prompt head classification — distribution of types across prompts."""
import time
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_CACHE = Path.home() / ".cache" / "psychic"


def add_subparser(subparsers):
    p = subparsers.add_parser("classify", help="Per-prompt head type distribution")
    p.add_argument("model", nargs="?", default="gpt2", help="Model name")
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="Cache directory")
    p.add_argument("--layer", type=int, default=None, help="Show only this layer")
    p.add_argument("--prompts", nargs="+", default=None, help="Prompt files to use")
    p.add_argument("--batch-size", type=int, default=10, help="Print progress every N prompts")
    p.set_defaults(func=cmd_classify)


def cmd_classify(args):
    from psychic.core.loader import load_safetensors
    from psychic.core.forward import forward_pass
    from psychic.core.tokenizer import BPETokenizer
    from psychic.core.prompts import load_prompts
    from psychic.core.classify import classify_all_prompts, dominant_type, HEAD_TYPES

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

    # collect all patterns
    all_patterns = []
    console.print(f"Running {len(prompts)} prompts...")
    t0 = time.time()

    for i, prompt in enumerate(prompts):
        token_ids = tokenizer.encode(prompt)
        _, patterns = forward_pass(weights, token_ids)
        all_patterns.append(patterns)

        if (i + 1) % args.batch_size == 0 or (i + 1) == len(prompts):
            elapsed = time.time() - t0
            console.print(f"  [{i+1}/{len(prompts)}] {elapsed:.1f}s — {prompt[:50]}")

    console.print(f"[green]done in {time.time() - t0:.1f}s[/green]\n")

    # classify
    counts = classify_all_prompts(all_patterns, n_layers, n_heads)
    n_prompts = len(prompts)

    # display
    layers = [args.layer] if args.layer is not None else range(n_layers)

    table = Table(title=f"Head Type Distribution: {args.model} ({n_prompts} prompts)")
    table.add_column("Layer", style="cyan", justify="right")
    table.add_column("Head", style="cyan", justify="right")
    for t in HEAD_TYPES:
        table.add_column(t, justify="right")
    table.add_column("Dominant", style="bold magenta")

    for layer in layers:
        for head in range(n_heads):
            c = counts[layer][head]
            dom = dominant_type(c)
            row = [str(layer), str(head)]
            for t in HEAD_TYPES:
                pct = 100 * c[t] / n_prompts
                # color by dominance
                if t == dom and pct > 50:
                    row.append(f"[green]{pct:.0f}%[/green]")
                elif t == dom:
                    row.append(f"[yellow]{pct:.0f}%[/yellow]")
                else:
                    row.append(f"{pct:.0f}%")
            row.append(dom)
            table.add_row(*row)

    console.print(table)