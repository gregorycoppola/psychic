"""Shared weight loading utilities."""
import json
import struct
import numpy as np
from pathlib import Path


def load_safetensors(path: Path) -> dict:
    """Load a .safetensors file into a dict of numpy arrays."""
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