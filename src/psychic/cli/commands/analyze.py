"""Map head classifications to the hybrid BP framework."""
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

DEFAULT_CACHE = Path.home() / ".cache" / "psychic"

HYBRID_ROLES = {
    "prev-token":  "boolean-AND",
    "self-attn":   "boolean-AND",
    "sharp":       "boolean-AND",
    "diffuse":     "continuous-OR",
    "first-token": "hub",
}

ROLE_COLORS = {
    "boolean-AND":   "green",
    "continuous-OR": "yellow",
    "hub":           "cyan",
    "mixed":         "magenta",
}


def assign_role(counts: dict, n_prompts: int) -> str:
    dom_type = max(counts, key=counts.get)
    dom_pct = counts[dom_type] / n_prompts
    if dom_pct < 0.6:
        return "mixed"
    return HYBRID_ROLES.get(dom_type, "unknown")


def add_subparser(subparsers):
    p = subparsers.add_parser("analyze", help="Map head classifications to hybrid BP framework")
    p.add_argument("model", nargs="?", default="gpt2", help="Model name")
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="Cache directory")
    p.add_argument("--index", default="all", help="Index name")
    p.add_argument("--layer", type=int, default=None, help="Show only this layer")
    p.set_defaults(func=cmd_analyze)


def cmd_analyze(args):
    from psychic.core.classify import classify_all_prompts, dominant_type, HEAD_TYPES
    from psychic.cli.commands.collect import collection_dir, load_collection

    cache = Path(args.cache)
    patterns_dir = collection_dir(cache, args.model, args.index)

    if not patterns_dir.exists():
        console.print(f"[red]✗[/red] No collection found. Run psychic collect {args.model}")
        raise SystemExit(1)

    console.print(f"Loading collection: [cyan]{patterns_dir.name}[/cyan]")
    meta, all_patterns, _ = load_collection(patterns_dir)
    n_prompts = meta["n_prompts"]
    cfg = meta["cfg"]
    n_layers = cfg["n_layers"]
    n_heads = cfg["n_heads"]
    console.print(f"  {n_prompts} prompts, {n_layers} layers, {n_heads} heads\n")

    counts = classify_all_prompts(all_patterns, n_layers, n_heads)

    roles = {
        layer: {
            head: assign_role(counts[layer][head], n_prompts)
            for head in range(n_heads)
        }
        for layer in range(n_layers)
    }

    role_counts = {"boolean-AND": 0, "continuous-OR": 0, "hub": 0, "mixed": 0}
    total = n_layers * n_heads
    for layer in range(n_layers):
        for head in range(n_heads):
            role_counts[roles[layer][head]] += 1

    lines = []
    for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        color = ROLE_COLORS[role]
        lines.append(f"[{color}]{role:20s}[/{color}]  {count:3d} heads  ({pct:.0f}%)")
    lines.append("")
    lines.append("  boolean-AND   = sharp gather, discrete BP routing")
    lines.append("  continuous-OR = diffuse aggregation, real-valued estimation")
    lines.append("  hub           = global workspace, position 0 communication")
    lines.append("  mixed         = context-dependent, no stable role")

    console.print(Panel("\n".join(lines),
                        title=f"Hybrid BP Map: {args.model} ({cfg['parameters_m']}M)"))

    # layer summary
    console.print()
    layer_table = Table(title="Layer Summary")
    layer_table.add_column("Layer", style="cyan", justify="right")
    layer_table.add_column("Depth%", style="white", justify="right")
    layer_table.add_column("boolean-AND", style="green", justify="right")
    layer_table.add_column("continuous-OR", style="yellow", justify="right")
    layer_table.add_column("hub", style="cyan", justify="right")
    layer_table.add_column("mixed", style="magenta", justify="right")
    layer_table.add_column("character", justify="left")

    for layer in range(n_layers):
        lc = {"boolean-AND": 0, "continuous-OR": 0, "hub": 0, "mixed": 0}
        for head in range(n_heads):
            lc[roles[layer][head]] += 1

        depth_pct = f"{100 * layer / (n_layers - 1):.0f}%"

        if lc["boolean-AND"] >= max(2, n_heads // 4):
            char = "boolean-heavy"
        elif lc["continuous-OR"] >= max(2, n_heads // 4):
            char = "continuous-heavy"
        elif lc["hub"] >= n_heads * 2 // 3:
            char = "hub-dominated"
        else:
            char = "mixed"

        layer_table.add_row(
            str(layer), depth_pct,
            str(lc["boolean-AND"]),
            str(lc["continuous-OR"]),
            str(lc["hub"]),
            str(lc["mixed"]),
            char,
        )

    console.print(layer_table)