# config.py — single place for all key management
from dotenv import load_dotenv
from pathlib import Path
import os

def load_keys():
    """
    Load API keys from environment or .env file.
    Returns keys if found, raises SystemExit if missing.
    """
    # Check if keys already in environment
    # This handles cases where keys are set at OS level
    # e.g. in production servers, CI/CD pipelines
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    voyage_key    = os.environ.get("VOYAGE_API_KEY")

    # Only load .env if keys not already present
    if not anthropic_key or not voyage_key:
        dotenv_path = Path(__file__).parent / ".env"

        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path)
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
            voyage_key    = os.environ.get("VOYAGE_API_KEY")

    # Validate after attempting to load
    missing = []
    if not anthropic_key:
        missing.append("ANTHROPIC_API_KEY")
    if not voyage_key:
        missing.append("VOYAGE_API_KEY")

    if missing:
        raise SystemExit(
            f"Missing API keys: {', '.join(missing)}\n"
            f"Add them to your .env file or set as environment variables"
        )

    return anthropic_key, voyage_key