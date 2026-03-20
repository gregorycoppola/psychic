"""Inspect model weights — print all tensor names and shapes."""
import pickle
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_CACHE = Path.home() / ".cache" / "psychic"


def load_weights(path: Path):
    """Load a pytorch .bin file (legacy pickle format) without torch."""

    class TorchUnpickler(pickle.Unpickler):
        def persistent_load(self, pid):
            return pid

        def find_class(self, module, name):
            if name == "_rebuild_tensor_v2":
                return rebuild_tensor
            if name == "_rebuild_parameter":
                return rebuild_parameter
            if module in ("torch", "torch.storage") and "Storage" in name:
                return make_storage(name)
            return super().find_class(module, name)

    def make_storage(name):
        def storage_constructor(*args):
            return args
        return storage_constructor

    def rebuild_tensor(storage, offset, shape, stride, requires_grad=False, *args):
        return {"_storage": storage, "_shape": shape, "_offset": offset}

    def rebuild_parameter(data, requires_grad=False, *args):
        return data

    with open(path, "rb") as f:
        unpickler = TorchUnpickler(f)
        result = unpickler.load()

    return result


def add_subparser(subparsers):
    p = subparsers.add_parser("inspect", help="Print all weight tensor names and shapes")
    p.add_argument("model", nargs="?", default="gpt2", help="Model name or path to .bin file")
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="Cache directory")
    p.add_argument("--debug", action="store_true", help="Print raw loaded object for debugging")
    p.set_defaults(func=cmd_inspect)


def cmd_inspect(args):
    path = Path(args.model)
    if not path.exists():
        path = Path(args.cache) / f"{args.model}.bin"

    if not path.exists():
        console.print(f"[red]✗[/red] Not found: {path}")
        console.print("Run [bold]psychic download[/bold] first.")
        raise SystemExit(1)

    console.print(f"Loading [bold]{path}[/bold]...")
    result = load_weights(path)

    console.print(f"Type: [yellow]{type(result)}[/yellow]")

    if args.debug:
        if isinstance(result, dict):
            for k, v in list(result.items())[:5]:
                console.print(f"  {k}: {type(v)} = {repr(v)[:100]}")
        else:
            console.print(repr(result)[:500])