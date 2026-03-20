"""Map head classifications to the hybrid BP framework."""
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

DEFAULT_CACHE = Path.home() / ".cache" / "psychic"

# Mapping from head type to hybrid BP role
HYBRID_ROLES = {
    "prev-token": "boolean-AND",
    "self-attn":  "boolean-AND",
    "sharp":      "boolean-AND",
    "diffuse":    "continuous-OR",
    "first-token": "hub",
}

ROLE_COLORS = {
    "boolean-AND":   "green",
    "continuous-OR": "yellow",
    "hub":           "cyan",
    "mixed":         "magenta",
}


def assign_role(counts: dict, n_prompts: int) -> str:
    """
    Assign a hybrid BP role to a head based on its type distribution.
    A head is 'mixed' if no single type exceeds 60% of prompts.
    """
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
        console.print(f"[red]✗[/red] No collection found for {args.model}/{args.index}")
        console.print("Run [bold]psychic collect[/bold] first.")
        raise SystemExit(1)

    console.print(f"Loading collection: [cyan]{patterns_dir.name}[/cyan]")
    meta, all_patterns, _ = load_collection(patterns_dir)
    n_prompts = meta["n_prompts"]
    console.print(f"  {n_prompts} prompts, collected {meta['timestamp'][:10]}\n")

    n_layers = 12
    n_heads = 12

    counts = classify_all_prompts(all_patterns, n_layers, n_heads)

    # assign hybrid roles
    roles = {
        layer: {
            head: assign_role(counts[layer][head], n_prompts)
            for head in range(n_heads)
        }
        for layer in range(n_layers)
    }

    # aggregate counts
    role_counts = {"boolean-AND": 0, "continuous-OR": 0, "hub": 0, "mixed": 0}
    for layer in range(n_layers):
        for head in range(n_heads):
            role_counts[roles[layer][head]] += 1

    total = n_layers * n_heads

    # summary panel
    lines = []
    for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        color = ROLE_COLORS[role]
        lines.append(f"[{color}]{role:20s}[/{color}]  {count:3d} heads  ({pct:.0f}%)")

    lines.append("")
    lines.append("[bold]Hybrid BP interpretation:[/bold]")
    lines.append("  boolean-AND   = sharp gather, discrete concepts, BP routing")
    lines.append("  continuous-OR = diffuse aggregation, real-valued estimation")
    lines.append("  hub           = global workspace (position 0), reader or writer")
    lines.append("  mixed         = context-dependent, switches type across prompts")

    console.print(Panel("\n".join(lines), title=f"Hybrid BP Map: {args.model}"))

    # per-layer grid
    console.print()
    layers = [args.layer] if args.layer is not None else range(n_layers)

    table = Table(title=f"Head Roles by Layer: {args.model}")
    table.add_column("Layer", style="cyan", justify="right")
    table.add_column("Head", style="cyan", justify="right")
    table.add_column("Dom type", justify="left")
    table.add_column("Dom %", justify="right")
    table.add_column("Hybrid role", justify="left")
    table.add_column("BP function", justify="left")

    BP_FUNCTIONS = {
        "boolean-AND":   "gather step — sharp routing of discrete belief",
        "continuous-OR": "update step — weighted aggregation of real values",
        "hub":           "global workspace — position 0 communication",
        "mixed":         "context-dependent — no stable role",
    }

    for layer in layers:
        for head in range(n_heads):
            c = counts[layer][head]
            dom = dominant_type(c)
            dom_pct = 100 * c[dom] / n_prompts
            role = roles[layer][head]
            color = ROLE_COLORS[role]

            table.add_row(
                str(layer),
                str(head),
                dom,
                f"{dom_pct:.0f}%",
                f"[{color}]{role}[/{color}]",
                BP_FUNCTIONS[role],
            )

    console.print(table)

    # layer summary
    console.print()
    layer_table = Table(title="Layer Summary")
    layer_table.add_column("Layer", style="cyan", justify="right")
    layer_table.add_column("boolean-AND", style="green", justify="right")
    layer_table.add_column("continuous-OR", style="yellow", justify="right")
    layer_table.add_column("hub", style="cyan", justify="right")
    layer_table.add_column("mixed", style="magenta", justify="right")
    layer_table.add_column("character", justify="left")

    for layer in range(n_layers):
        lc = {"boolean-AND": 0, "continuous-OR": 0, "hub": 0, "mixed": 0}
        for head in range(n_heads):
            lc[roles[layer][head]] += 1

        # characterize the layer
        if lc["boolean-AND"] >= 4:
            char = "boolean-heavy"
        elif lc["continuous-OR"] >= 4:
            char = "continuous-heavy"
        elif lc["hub"] >= 8:
            char = "hub-dominated"
        else:
            char = "mixed"

        layer_table.add_row(
            str(layer),
            str(lc["boolean-AND"]),
            str(lc["continuous-OR"]),
            str(lc["hub"]),
            str(lc["mixed"]),
            char,
        )

    console.print(layer_table)