"""Inspect model weights — print all tensor names and shapes."""
import pickle
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_CACHE = Path.home() / ".cache" / "psychic"


def load_weights(path: Path) -> dict:
    """Load a pytorch .bin file (legacy pickle format) without torch."""

    class TorchUnpickler(pickle.Unpickler):
        def persistent_load(self, pid):
            # pid is (storage_type, key, location, size) or similar
            # In legacy format, storage is already embedded
            return pid

        def find_class(self, module, name):
            if name == "_rebuild_tensor_v2":
                return rebuild_tensor
            if name == "_rebuild_parameter":
                return rebuild_parameter
            # For storage types, return a dummy that just passes through
            if module in ("torch", "torch.storage") and "Storage" in name:
                return make_storage(name)
            return super().find_class(module, name)

    def make_storage(name):
        def storage_constructor(*args):
            return args
        return storage_constructor

    def rebuild_tensor(storage, offset, shape, stride, requires_grad=False, *args):
        # storage is whatever came back from persistent_load
        # Try to make a numpy array from it
        try:
            if isinstance(storage, np.ndarray):
                return storage.reshape(shape) if shape else storage
            elif isinstance(storage, (list, tuple)) and len(storage) > 0:
                arr = np.array(storage[0]) if not isinstance(storage[0], np.ndarray) else storage[0]
                return arr.reshape(shape) if shape else arr
            else:
                return np.zeros(shape, dtype=np.float32)
        except Exception:
            return np.zeros(shape if shape else (1,), dtype=np.float32)

    def rebuild_parameter(data, requires_grad=False, *args):
        return data

    with open(path, "rb") as f:
        unpickler = TorchUnpickler(f)
        weights = unpickler.load()

    return weights


def add_subparser(subparsers):
    p = subparsers.add_parser("inspect", help="Print all weight tensor names and shapes")
    p.add_argument("model", nargs="?", default="gpt2", help="Model name or path to .bin file")
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="Cache directory")
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
    weights = load_weights(path)

    table = Table(title=f"Weights: {path.name}")
    table.add_column("Name", style="cyan")
    table.add_column("Shape", style="green")
    table.add_column("Params", style="yellow", justify="right")

    total = 0
    for name, tensor in weights.items():
        if hasattr(tensor, "shape") and len(tensor.shape) > 0:
            shape = list(tensor.shape)
            n = int(np.prod(tensor.shape))
        else:
            shape = []
            n = 1
        total += n
        table.add_row(name, str(shape), f"{n:,}")

    console.print(table)
    console.print(f"\nTotal parameters: [bold]{total:,}[/bold]")