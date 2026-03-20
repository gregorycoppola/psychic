"""List available model configs."""
from rich.console import Console
from rich.table import Table

console = Console()


def add_subparser(subparsers):
    p = subparsers.add_parser("list-models", help="List available model configs")
    p.set_defaults(func=cmd_list_models)


def cmd_list_models(args):
    from psychic.core.models import list_models, get_config

    table = Table(title="Available Models")
    table.add_column("Name", style="cyan")
    table.add_column("Family", style="green")
    table.add_column("Layers", justify="right")
    table.add_column("Heads", justify="right")
    table.add_column("d_model", justify="right")
    table.add_column("Params", justify="right")

    for name in list_models():
        cfg = get_config(name)
        table.add_row(
            name,
            cfg["family"],
            str(cfg["n_layers"]),
            str(cfg["n_heads"]),
            str(cfg["d_model"]),
            f"{cfg['parameters_m']}M",
        )

    console.print(table)