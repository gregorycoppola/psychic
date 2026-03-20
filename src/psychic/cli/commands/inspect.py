"""Inspect model weights — print all tensor names and shapes."""
import pickle
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_CACHE = Path.home() / ".cache" / "psychic"


class NumpyUnpickler(pickle.Unpickler):
    """Custom unpickler that loads torch tensors as numpy arrays."""

    def find_class(self, module, name):
        if module == "torch" and name == "FloatStorage":
            return np.float32
        if module == "torch" and name == "LongStorage":
            return np.int64
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return self._rebuild_tensor
        return super().find_class(module, name)

    @staticmethod
    def _rebuild_tensor(storage, offset, shape, stride, *args):
        arr = np.array(storage).reshape(shape)
        return arr


def load_weights(path: Path) -> dict:
    """Load a pytorch .bin file without torch."""
    import pickle
    import struct
    import zipfile

    # pytorch .bin files are zip archives containing a pickle
    with zipfile.ZipFile(path) as zf:
        # find the data.pkl entry
        names = zf.namelist()
        pkl_name = next(n for n in names if n.endswith("data.pkl"))
        archive_name = pkl_name.replace("data.pkl", "")

        class TorchUnpickler(pickle.Unpickler):
            def __init__(self, f, zf, archive_name):
                super().__init__(f)
                self.zf = zf
                self.archive_name = archive_name
                self.storages = {}

            def persistent_load(self, pid):
                storage_type, key, location, size = pid[1], pid[2], pid[3], pid[4]
                data_path = f"{self.archive_name}data/{key}"
                with self.zf.open(data_path) as df:
                    buf = df.read()
                if "float" in str(storage_type).lower():
                    arr = np.frombuffer(buf, dtype=np.float32)
                else:
                    arr = np.frombuffer(buf, dtype=np.int64)
                return arr

            def find_class(self, module, name):
                if name == "_rebuild_tensor_v2":
                    return rebuild_tensor
                if name == "_rebuild_parameter":
                    return rebuild_parameter
                return super().find_class(module, name)

        def rebuild_tensor(storage, offset, shape, stride, *args):
            if len(shape) == 0:
                return storage[offset:offset+1].reshape(())
            return storage[offset:offset+np.prod(shape)].reshape(shape)

        def rebuild_parameter(data, requires_grad, *args):
            return data

        with zf.open(pkl_name) as f:
            unpickler = TorchUnpickler(f, zf, archive_name)
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
        if hasattr(tensor, "shape"):
            shape = list(tensor.shape)
            n = int(np.prod(tensor.shape)) if len(tensor.shape) > 0 else 1
        else:
            shape = []
            n = 1
        total += n
        table.add_row(name, str(shape), f"{n:,}")

    console.print(table)
    console.print(f"\nTotal parameters: [bold]{total:,}[/bold]")