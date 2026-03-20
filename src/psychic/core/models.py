"""Model configuration registry."""
import json
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"


def get_config(model_name: str) -> dict:
    """Load model config by name."""
    path = MODELS_DIR / f"{model_name}.json"
    if not path.exists():
        available = [p.stem for p in MODELS_DIR.glob("*.json")]
        raise FileNotFoundError(
            f"No config for '{model_name}'. Available: {available}"
        )
    return json.loads(path.read_text())


def list_models() -> list[str]:
    """List all available model configs."""
    return sorted(p.stem for p in MODELS_DIR.glob("*.json"))