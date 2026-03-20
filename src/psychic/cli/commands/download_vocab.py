"""Download tokenizer vocab and merges files from Hugging Face."""
import requests
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, DownloadColumn, BarColumn, TransferSpeedColumn

console = Console()

VOCAB_FILES = {
    "gpt2": [
        {
            "url": "https://huggingface.co/gpt2/resolve/main/vocab.json",
            "filename": "gpt2_vocab.json",
        },
        {
            "url": "https://huggingface.co/gpt2/resolve/main/merges.txt",
            "filename": "gpt2_merges.txt",
        },
    ],
}

DEFAULT_CACHE = Path.home() / ".cache" / "psychic"


def add_subparser(subparsers):
    p = subparsers.add_parser("download-vocab", help="Download tokenizer vocab and merges files")
    p.add_argument("model", nargs="?", default="gpt2", choices=VOCAB_FILES.keys())
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="Cache directory")
    p.add_argument("--force", action="store_true", help="Re-download even if exists")
    p.set_defaults(func=cmd_download_vocab)


def cmd_download_vocab(args):
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    for info in VOCAB_FILES[args.model]:
        dest = cache / info["filename"]

        if dest.exists() and not args.force:
            console.print(f"[green]✓[/green] Already exists: {dest}")
            continue

        console.print(f"Downloading [bold]{info['filename']}[/bold]...")

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