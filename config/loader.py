"""YAML config loader for poker solver settings."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class Config:
    """Configuration for poker solver.

    Attributes:
        name: Human-readable name for this configuration.
        stack_depth: Stack depth in big blinds.
        raise_sizes: List of raise sizes in big blinds.
    """
    name: str
    stack_depth: float
    raise_sizes: List[float]

    def get_legal_raise_sizes(self, current_bet: float, stack: float) -> List[float]:
        """Get raise sizes that are legal given current bet and stack.

        Args:
            current_bet: Current bet size in big blinds.
            stack: Remaining stack size in big blinds.

        Returns:
            List of legal raise sizes (greater than current_bet and <= stack).
        """
        return [size for size in self.raise_sizes if size > current_bet and size <= stack]


def load_config(path: str) -> Config:
    """Load configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Config object with loaded settings.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If required fields are missing or invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax: {e}")

    # Validate that data is a dict
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping (dict), not a scalar or list")

    # Validate required fields
    if 'stack_depth' not in data:
        raise ValueError("Missing required field: stack_depth")
    if 'raise_sizes' not in data:
        raise ValueError("Missing required field: raise_sizes")

    # Validate stack_depth type and value
    try:
        stack_depth = float(data['stack_depth'])
    except (TypeError, ValueError):
        raise ValueError(f"stack_depth must be a number, got: {type(data['stack_depth']).__name__}")

    if stack_depth <= 0:
        raise ValueError(f"stack_depth must be positive, got: {stack_depth}")

    # Validate raise_sizes is a list
    if not isinstance(data['raise_sizes'], list):
        raise ValueError(f"raise_sizes must be a list, got: {type(data['raise_sizes']).__name__}")

    # Validate and convert raise_sizes values
    raise_sizes = []
    for i, s in enumerate(data['raise_sizes']):
        try:
            size = float(s)
        except (TypeError, ValueError):
            raise ValueError(f"raise_sizes[{i}] must be a number, got: {type(s).__name__}")
        if size <= 0:
            raise ValueError(f"raise_sizes[{i}] must be positive, got: {size}")
        raise_sizes.append(size)

    # Default name to "Custom" if not provided
    name = data.get('name', 'Custom')

    return Config(
        name=name,
        stack_depth=stack_depth,
        raise_sizes=raise_sizes,
    )


def get_preset_path(name: str) -> str:
    """Get the path to a preset configuration file.

    Args:
        name: Name of the preset (without .yaml extension).

    Returns:
        Absolute path to the preset file.
    """
    # Get the directory containing this module
    module_dir = Path(__file__).parent
    preset_path = module_dir / "presets" / f"{name}.yaml"
    return str(preset_path.resolve())
