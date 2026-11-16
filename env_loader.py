"""
EduSolve AI - Lightweight Environment Loader
Loads environment variables from a .env file without requiring external dependencies.
"""
import os
from pathlib import Path


def load_env(env_path: str = None) -> None:
    """
    Read key-value pairs from .env and inject them into os.environ.
    Does not overwrite variables that are already set in the environment.
    """
    if env_path is None:
        # Default to .env in the same directory as this file or current working directory
        base_dir = Path(__file__).resolve().parent
        candidate = base_dir / ".env"
        env_path = str(candidate) if candidate.exists() else ".env"

    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    # Set only if not already populated in os.environ
                    if key and key not in os.environ:
                        os.environ[key] = val
    except Exception as e:
        print(f"[Warning] Failed to load {env_path}: {e}")
