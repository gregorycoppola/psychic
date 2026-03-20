"""Load prompts from the prompts directory."""
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def load_prompts(names: list[str] | None = None) -> list[str]:
    """
    Load prompts from .txt files in the prompts directory.
    If names is None, load all .txt files.
    Each line in a file is one prompt. Blank lines and # comments are skipped.
    """
    if names is None:
        files = sorted(PROMPTS_DIR.glob("*.txt"))
    else:
        files = [PROMPTS_DIR / f"{name}.txt" for name in names]

    prompts = []
    for f in files:
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)

    return prompts