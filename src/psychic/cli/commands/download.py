"""Download model weights from Hugging Face."""
import requests
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, DownloadColumn, BarColumn, TransferSpeedColumn

console = Console()

DEFAULT_CACHE = Path.home() / ".cache" / "psychic"


def download_file(url: str, dest: Path, force: bool = False):
    if dest.exists() and not force:
        console.print(f"[green]✓[/green] Already exists: {dest.name}")
        return
    console.print(f"Downloading [bold]{dest.name}[/bold]...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with Progress(BarColumn(), DownloadColumn(), TransferSpeedColumn()) as progress:
        task = progress.add_task("downloading", total=total)
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                progress.advance(task, len(chunk))
    console.print(f"[green]✓[/green] Saved to {dest}")


def add_subparser(subparsers):
    p = subparsers.add_parser("download", help="Download model weights from Hugging Face")
    p.add_argument("model", nargs="?", default="gpt2", help="Model name")
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="Cache directory")
    p.add_argument("--force", action="store_true", help="Re-download even if exists")
    p.set_defaults(func=cmd_download)


def cmd_download(args):
    from psychic.core.models import get_config
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    try:
        cfg = get_config(args.model)
    except FileNotFoundError as e:
        console.print(f"[red]✗[/red] {e}")
        raise SystemExit(1)

    console.print(f"Model: [bold]{cfg['name']}[/bold] ({cfg['parameters_m']}M params)")
    download_file(cfg["safetensors_url"], cache / cfg["safetensors_filename"], args.force)