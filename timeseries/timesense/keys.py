"""API key utilities with macOS Keychain fallback."""

import os
import subprocess
import logging

logger = logging.getLogger(__name__)


def get_api_key(key_name: str = "OPEN_ROUTER_KEY") -> str:
    """Get API key from environment or macOS Keychain.

    Args:
        key_name: Key name to look up (used for both env var and keychain service)

    Returns:
        The API key string

    Raises:
        ValueError: If key not found in environment or Keychain
    """
    key = os.getenv(key_name)
    if key:
        return key

    # Fall back to macOS Keychain (same key name used as service)
    try:
        username = os.getenv("USER")
        result = subprocess.run(
            ["security", "find-generic-password", "-s", key_name, "-a", username, "-w"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get API key from Keychain: {e}")
        raise ValueError(f"Cannot find {key_name} in either environment or keychain")