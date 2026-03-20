"""Load prompts from the prompts directory using index files."""
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
DATA_DIR = PROMPTS_DIR / "data"
INDEXES_DIR = PROMPTS_DIR / "indexes"


def load_index(index_name: str = "all") -> list[str]:
    """
    Load prompt filenames listed in an index file.
    Each line in the index file is a filename in data/.
    Blank lines and # comments are skipped.
    """
    index_path = INDEXES_DIR / f"{index_name}.txt"
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")

    filenames = []
    for line in index_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            filenames.append(line)
    return filenames


def load_prompts(index_name: str = "all") -> list[str]:
    """
    Load all prompts listed in the given index.
    Returns a flat list of prompt strings.
    """
    filenames = load_index(index_name)
    prompts = []
    for filename in filenames:
        path = DATA_DIR / filename
        if not path.exists():
            print(f"Warning: prompt file not found: {path}")
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts


def list_indexes() -> list[str]:
    """List all available index names."""
    return sorted(p.stem for p in INDEXES_DIR.glob("*.txt"))


def list_data_files() -> list[str]:
    """List all available data files."""
    return sorted(p.name for p in DATA_DIR.glob("*.txt"))