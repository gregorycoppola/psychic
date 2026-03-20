"""Clear cached model files."""
from pathlib import Path
from rich.console import Console

console = Console()

DEFAULT_CACHE = Path.home() / ".cache" / "psychic"


def add_subparser(subparsers):
    p = subparsers.add_parser("clear", help="Clear cached model files")
    p.add_argument("model", nargs="?", default="gpt2", help="Model name")
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="Cache directory")
    p.add_argument("--weights", action="store_true", help="Clear weights only")
    p.add_argument("--vocab", action="store_true", help="Clear vocab only")
    p.set_defaults(func=cmd_clear)


def cmd_clear(args):
    cache = Path(args.cache)
    model = args.model

    # if neither flag set, clear everything
    clear_weights = args.weights or (not args.weights and not args.vocab)
    clear_vocab = args.vocab or (not args.weights and not args.vocab)

    files = []
    if clear_weights:
        files += [
            cache / f"{model}.safetensors",
            cache / f"{model}.bin",
        ]
    if clear_vocab:
        files += [
            cache / f"{model}_vocab.json",
            cache / f"{model}_merges.txt",
        ]

    any_found = False
    for f in files:
        if f.exists():
            f.unlink()
            console.print(f"[red]deleted[/red] {f}")
            any_found = True

    if not any_found:
        console.print(f"[yellow]nothing to clear for {model}[/yellow]")
    else:
        console.print(f"[green]✓[/green] done")