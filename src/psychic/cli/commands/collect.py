"""Run forward pass on all prompts and save attention patterns to disk."""
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from rich.console import Console

console = Console()

DEFAULT_CACHE = Path.home() / ".cache" / "psychic"


def collection_dir(cache: Path, model: str, index: str) -> Path:
    return cache / f"{model}_{index}_patterns"


def save_collection(patterns_dir: Path, model: str, index: str,
                    prompts: list, all_patterns: list, all_token_ids: list,
                    cfg: dict):
    patterns_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": model,
        "index": index,
        "n_prompts": len(prompts),
        "timestamp": datetime.now().isoformat(),
        "prompts": prompts,
        "cfg": cfg,
    }
    (patterns_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    for i, (patterns, token_ids) in enumerate(zip(all_patterns, all_token_ids)):
        data = {"token_ids": np.array(token_ids)}
        for layer, pat in enumerate(patterns):
            data[f"layer_{layer}"] = pat
        np.savez(patterns_dir / f"prompt_{i:04d}.npz", **data)


def load_collection(patterns_dir: Path) -> tuple:
    meta = json.loads((patterns_dir / "meta.json").read_text())
    n_prompts = meta["n_prompts"]
    n_layers = meta["cfg"]["n_layers"]
    all_patterns = []
    all_token_ids = []
    for i in range(n_prompts):
        path = patterns_dir / f"prompt_{i:04d}.npz"
        data = np.load(path)
        token_ids = data["token_ids"].tolist()
        patterns = [data[f"layer_{l}"] for l in range(n_layers)]
        all_patterns.append(patterns)
        all_token_ids.append(token_ids)
    return meta, all_patterns, all_token_ids


def add_subparser(subparsers):
    p = subparsers.add_parser("collect", help="Run forward pass and save patterns to disk")
    p.add_argument("model", nargs="?", default="gpt2", help="Model name")
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="Cache directory")
    p.add_argument("--index", default="all", help="Index name")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--force", action="store_true", help="Re-collect even if exists")
    p.set_defaults(func=cmd_collect)


def cmd_collect(args):
    from psychic.core.loader import load_safetensors
    from psychic.core.forward import forward_pass
    from psychic.core.tokenizer import load_tokenizer
    from psychic.core.prompts import load_prompts, list_indexes
    from psychic.core.models import get_config

    cache = Path(args.cache)

    try:
        cfg = get_config(args.model)
    except FileNotFoundError as e:
        console.print(f"[red]✗[/red] {e}")
        raise SystemExit(1)

    weights_path = cache / cfg["safetensors_filename"]
    if not weights_path.exists():
        console.print(f"[red]✗[/red] Weights not found. Run psychic download {args.model}")
        raise SystemExit(1)

    patterns_dir = collection_dir(cache, args.model, args.index)

    if patterns_dir.exists() and not args.force:
        meta = json.loads((patterns_dir / "meta.json").read_text())
        console.print(f"[green]✓[/green] Already collected: {patterns_dir.name}")
        console.print(f"  {meta['n_prompts']} prompts, {meta['timestamp'][:10]}")
        console.print(f"  Use --force to re-collect.")
        return

    try:
        prompts = load_prompts(args.index)
    except FileNotFoundError:
        console.print(f"[red]✗[/red] Index '{args.index}' not found.")
        console.print(f"Available: {list_indexes()}")
        raise SystemExit(1)

    tokenizer = load_tokenizer(cfg, cache)
    console.print(f"Loading weights ({cfg['parameters_m']}M params)...")
    weights = load_safetensors(weights_path)
    console.print(f"Loaded {len(prompts)} prompts from index '{args.index}'")

    all_patterns = []
    all_token_ids = []
    t0 = time.time()
    console.print(f"Running {len(prompts)} prompts...")

    for i, prompt in enumerate(prompts):
        token_ids = tokenizer.encode(prompt)
        _, patterns = forward_pass(weights, token_ids, cfg)
        all_patterns.append(patterns)
        all_token_ids.append(token_ids)

        if (i + 1) % args.batch_size == 0 or (i + 1) == len(prompts):
            elapsed = time.time() - t0
            console.print(f"  [{i+1}/{len(prompts)}] {elapsed:.1f}s — {prompt[:50]}")

    console.print(f"[green]done in {time.time() - t0:.1f}s[/green]")
    console.print(f"Saving to {patterns_dir}...")
    save_collection(patterns_dir, args.model, args.index,
                    prompts, all_patterns, all_token_ids, cfg)
    size_mb = sum(f.stat().st_size for f in patterns_dir.iterdir()) / 1e6
    console.print(f"[green]✓[/green] Saved {len(prompts)} prompts ({size_mb:.1f} MB)")