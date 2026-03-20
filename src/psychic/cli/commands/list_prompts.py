"""List available prompt indexes and data files."""
from rich.console import Console
from rich.table import Table

console = Console()


def add_subparser(subparsers):
    p = subparsers.add_parser("list-prompts", help="List available prompt indexes and data files")
    p.set_defaults(func=cmd_list)


def cmd_list(args):
    from psychic.core.prompts import list_indexes, list_data_files, load_index, DATA_DIR

    console.print("\n[bold]Indexes:[/bold]")
    table = Table()
    table.add_column("Index", style="cyan")
    table.add_column("Files", style="green")
    table.add_column("Prompts", style="yellow", justify="right")

    for idx in list_indexes():
        try:
            from psychic.core.prompts import load_prompts
            files = load_index(idx)
            n = len(load_prompts(idx))
            table.add_row(idx, ", ".join(files), str(n))
        except Exception as e:
            table.add_row(idx, f"[red]error: {e}[/red]", "?")

    console.print(table)

    console.print("\n[bold]Data files:[/bold]")
    for f in list_data_files():
        path = DATA_DIR / f
        n = sum(1 for line in path.read_text().splitlines()
                if line.strip() and not line.strip().startswith("#"))
        console.print(f"  {f} — {n} prompts")