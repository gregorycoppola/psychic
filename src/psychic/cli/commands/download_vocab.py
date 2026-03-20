"""Download tokenizer files from Hugging Face."""
from pathlib import Path
from rich.console import Console

console = Console()

DEFAULT_CACHE = Path.home() / ".cache" / "psychic"


def add_subparser(subparsers):
    p = subparsers.add_parser("download-vocab", help="Download tokenizer files")
    p.add_argument("model", nargs="?", default="gpt2", help="Model name")
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="Cache directory")
    p.add_argument("--force", action="store_true", help="Re-download even if exists")
    p.set_defaults(func=cmd_download_vocab)


def cmd_download_vocab(args):
    from psychic.core.models import get_config
    from psychic.cli.commands.download import download_file

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    try:
        cfg = get_config(args.model)
    except FileNotFoundError as e:
        console.print(f"[red]✗[/red] {e}")
        raise SystemExit(1)

    tokenizer_type = cfg.get("tokenizer_type", "bpe")

    if tokenizer_type == "tokenizer_json":
        # single unified tokenizer.json
        download_file(cfg["tokenizer_url"], cache / cfg["tokenizer_filename"], args.force)
    else:
        # legacy GPT-2 style: vocab.json + merges.txt
        download_file(cfg["vocab_url"], cache / cfg["vocab_filename"], args.force)
        download_file(cfg["merges_url"], cache / cfg["merges_filename"], args.force)