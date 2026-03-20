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
    p.add_argument("--index", default="all", help="Index name")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--no-cache", action="store_true", help="Re-run even if collected")
    p.set_defaults(func=cmd_classify)


def cmd_classify(args):
    from psychic.core.loader import load_safetensors
    from psychic.core.forward import forward_pass
    from psychic.core.tokenizer import BPETokenizer
    from psychic.core.prompts import load_prompts, list_indexes
    from psychic.core.classify import classify_all_prompts, dominant_type, HEAD_TYPES
    from psychic.core.models import get_config
    from psychic.cli.commands.collect import collection_dir, load_collection

    cache = Path(args.cache)
    patterns_dir = collection_dir(cache, args.model, args.index)

    if patterns_dir.exists() and not args.no_cache:
        console.print(f"Loading from collection: [cyan]{patterns_dir.name}[/cyan]")
        meta, all_patterns, _ = load_collection(patterns_dir)
        n_prompts = meta["n_prompts"]
        cfg = meta["cfg"]
        console.print(f"  {n_prompts} prompts, collected {meta['timestamp'][:10]}")
    else:
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

        try:
            prompts = load_prompts(args.index)
        except FileNotFoundError:
            console.print(f"[red]✗[/red] Index '{args.index}' not found.")
            raise SystemExit(1)

        tokenizer = BPETokenizer(vocab_path, merges_path)
        console.print("Loading weights...")
        weights = load_safetensors(weights_path)

        all_patterns = []
        t0 = time.time()
        for i, prompt in enumerate(prompts):
            token_ids = tokenizer.encode(prompt)
            _, patterns = forward_pass(weights, token_ids, cfg)
            all_patterns.append(patterns)
            if (i + 1) % args.batch_size == 0 or (i + 1) == len(prompts):
                console.print(f"  [{i+1}/{len(prompts)}] {time.time()-t0:.1f}s — {prompt[:50]}")
        n_prompts = len(prompts)

    n_layers = cfg["n_layers"]
    n_heads = cfg["n_heads"]
    counts = classify_all_prompts(all_patterns, n_layers, n_heads)

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
                if t == dom and pct > 50:
                    row.append(f"[green]{pct:.0f}%[/green]")
                elif t == dom:
                    row.append(f"[yellow]{pct:.0f}%[/yellow]")
                else:
                    row.append(f"{pct:.0f}%")
            row.append(dom)
            table.add_row(*row)

    console.print(table)