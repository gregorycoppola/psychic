"""Inspect model weights — print all tensor names and shapes."""
import json
import struct
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_CACHE = Path.home() / ".cache" / "psychic"


def load_safetensors(path: Path) -> dict:
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes)
        data_start = 8 + header_len
        f.seek(data_start)
        data = f.read()

    tensors = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        shape = meta["shape"]
        raw = data[start:end]
        if meta["dtype"] == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16)
            u32 = u16.astype(np.uint32) << 16
            arr = u32.view(np.float32).reshape(shape)
        elif meta["dtype"] == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
        else:
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        tensors[name] = arr
    return tensors


def add_subparser(subparsers):
    p = subparsers.add_parser("inspect", help="Print all weight tensor names and shapes")
    p.add_argument("model", nargs="?", default="gpt2", help="Model name")
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="Cache directory")
    p.add_argument("--filter", default=None, help="Only show keys containing this string")
    p.add_argument("--layer", default=None, help="Only show keys for this layer number")
    p.set_defaults(func=cmd_inspect)


def cmd_inspect(args):
    from psychic.core.models import get_config

    cache = Path(args.cache)

    try:
        cfg = get_config(args.model)
    except FileNotFoundError as e:
        console.print(f"[red]✗[/red] {e}")
        raise SystemExit(1)

    path = cache / cfg["safetensors_filename"]
    if not path.exists():
        console.print(f"[red]✗[/red] Not found: {path}")
        console.print(f"Run [bold]psychic download {args.model}[/bold] first.")
        raise SystemExit(1)

    console.print(f"Loading [bold]{path.name}[/bold] ({cfg['parameters_m']}M, family={cfg['family']})...")
    weights = load_safetensors(path)

    table = Table(title=f"Weights: {path.name}")
    table.add_column("Name", style="cyan")
    table.add_column("Shape", style="green")
    table.add_column("Params", style="yellow", justify="right")

    total = 0
    for name, tensor in sorted(weights.items()):
        # apply filters
        if args.filter and args.filter not in name:
            continue
        if args.layer and f".{args.layer}." not in name:
            continue

        shape = list(tensor.shape)
        n = int(np.prod(tensor.shape)) if tensor.shape else 1
        total += n
        table.add_row(name, str(shape), f"{n:,}")

    console.print(table)
    console.print(f"\nShowing params: [bold]{total:,}[/bold]")