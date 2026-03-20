"""Download GPT-2 weights directly from Hugging Face."""
import requests
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, DownloadColumn, BarColumn, TransferSpeedColumn

console = Console()

MODELS = {
    "gpt2": {
        "url": "https://huggingface.co/gpt2/resolve/main/model.safetensors",
        "filename": "gpt2.safetensors",
    },
    "gpt2-medium": {
        "url": "https://huggingface.co/gpt2-medium/resolve/main/model.safetensors",
        "filename": "gpt2-medium.safetensors",
    },
}

DEFAULT_CACHE = Path.home() / ".cache" / "psychic"


def add_subparser(subparsers):
    p = subparsers.add_parser("download", help="Download model weights from Hugging Face")
    p.add_argument("model", nargs="?", default="gpt2", choices=MODELS.keys(), help="Model to download")
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="Cache directory")
    p.set_defaults(func=cmd_download)


def cmd_download(args):
    model = args.model
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    info = MODELS[model]
    dest = cache / info["filename"]

    if dest.exists():
        console.print(f"[green]✓[/green] Already downloaded: {dest}")
        return

    console.print(f"Downloading [bold]{model}[/bold] from Hugging Face...")
    console.print(f"  URL: {info['url']}")
    console.print(f"  Dest: {dest}")

    response = requests.get(info["url"], stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    with Progress(BarColumn(), DownloadColumn(), TransferSpeedColumn()) as progress:
        task = progress.add_task("downloading", total=total)
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                progress.advance(task, len(chunk))

    console.print(f"[green]✓[/green] Saved to {dest}")