"""Inspect model weights — print all tensor names and shapes."""
import torch
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_CACHE = Path.home() / ".cache" / "psychic"


def add_subparser(subparsers):
    p = subparsers.add_parser("inspect", help="Print all weight tensor names and shapes")
    p.add_argument("model", nargs="?", default="gpt2", help="Model name or path to .bin file")
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="Cache directory")
    p.set_defaults(func=cmd_inspect)


def cmd_inspect(args):
    # Resolve path
    path = Path(args.model)
    if not path.exists():
        path = Path(args.cache) / f"{args.model}.bin"

    if not path.exists():
        console.print(f"[red]✗[/red] Not found: {path}")
        console.print("Run [bold]psychic download[/bold] first.")
        raise SystemExit(1)

    console.print(f"Loading [bold]{path}[/bold]...")
    weights = torch.load(path, map_location="cpu", weights_only=True)

    table = Table(title=f"Weights: {path.name}")
    table.add_column("Name", style="cyan")
    table.add_column("Shape", style="green")
    table.add_column("Params", style="yellow", justify="right")

    total = 0
    for name, tensor in weights.items():
        n = tensor.numel()
        total += n
        table.add_row(name, str(list(tensor.shape)), f"{n:,}")

    console.print(table)
    console.print(f"\nTotal parameters: [bold]{total:,}[/bold]")