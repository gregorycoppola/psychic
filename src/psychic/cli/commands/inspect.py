"""Inspect model weights — print all tensor names and shapes."""
import struct
import json
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_CACHE = Path.home() / ".cache" / "psychic"

DTYPE_MAP = {
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.float32,  # bfloat16 — load as float32
    "I32": np.int32,
    "I64": np.int64,
}


def load_safetensors(path: Path) -> dict:
    """
    Load a .safetensors file without any dependencies.
    Format: 8-byte little-endian uint64 header length,
    then JSON header, then raw tensor data.
    """
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
        dtype = DTYPE_MAP.get(meta["dtype"], np.float32)
        start, end = meta["data_offsets"]
        shape = meta["shape"]
        raw = data[start:end]
        if meta["dtype"] == "BF16":
            # bfloat16: reinterpret as uint16, shift to float32
            u16 = np.frombuffer(raw, dtype=np.uint16)
            u32 = u16.astype(np.uint32) << 16
            arr = u32.view(np.float32).reshape(shape)
        else:
            arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
        tensors[name] = arr

    return tensors


def add_subparser(subparsers):
    p = subparsers.add_parser("inspect", help="Print all weight tensor names and shapes")
    p.add_argument("model", nargs="?", default="gpt2", help="Model name")
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="Cache directory")
    p.set_defaults(func=cmd_inspect)


def cmd_inspect(args):
    cache = Path(args.cache)
    path = cache / f"{args.model}.safetensors"

    if not path.exists():
        console.print(f"[red]✗[/red] Not found: {path}")
        console.print("Run [bold]psychic download[/bold] first.")
        raise SystemExit(1)

    console.print(f"Loading [bold]{path.name}[/bold]...")
    weights = load_safetensors(path)

    table = Table(title=f"Weights: {path.name}")
    table.add_column("Name", style="cyan")
    table.add_column("Shape", style="green")
    table.add_column("Params", style="yellow", justify="right")

    total = 0
    for name, tensor in sorted(weights.items()):
        shape = list(tensor.shape)
        n = int(np.prod(tensor.shape)) if tensor.shape else 1
        total += n
        table.add_row(name, str(shape), f"{n:,}")

    console.print(table)
    console.print(f"\nTotal parameters: [bold]{total:,}[/bold]")