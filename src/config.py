import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()

logger: logging.Logger = logging.getLogger(__name__)


def _get_env_variable(name: str, default: str | None = None) -> str | None:
    """
    Retrieves an environment variable, with an optional default value.

    Args:
            name (str): The name of the environment variable to retrieve.
            default (str | None): An optional default value to return if the environment variable is not set.

    Returns:
            str | None: The value of the environment variable, or the default value if it is not set.
    """

    try:
        value: str = os.getenv(name, default)

        if value in (None, ""):
            logger.warning(
                f"Environment variable '{name}' is not set, using default value: '{default if default is not None else 'None'}'"
            )
            return default

        return value
    except Exception as e:
        logger.error(f"Error retrieving environment variable '{name}': {e}")
        return default


BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

HF_TOKEN: str = _get_env_variable("HF_TOKEN", "default")
