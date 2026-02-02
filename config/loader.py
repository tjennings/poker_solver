"""YAML config loader for poker solver settings."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import yaml


@dataclass
class Config:
    """Configuration for poker solver.

    Attributes:
        name: Human-readable name for this configuration.
        stack_depths: List of stack depths in big blinds.
        raise_sizes: List of raise sizes in big blinds.
        max_bets_per_round: Maximum bets/raises per betting round (default 4).
        postflop_raise_sizes: Raise sizes for postflop subgame solving (default: simplified).
        postflop_max_bets_per_round: Max bets for postflop (default: 2).
    """
    name: str
    stack_depths: List[float]
    raise_sizes: List[float]
    max_bets_per_round: int = 4
    postflop_raise_sizes: List[float] = None  # None means use [2/3 pot, pot] based on pot size
    postflop_max_bets_per_round: int = 1  # Keep small for tractable subgame solving

    @property
    def stack_depth(self) -> float:
        """Backward compatibility: return first stack depth."""
        return self.stack_depths[0]

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

    # Validate required fields - accept either stack_depth or stack_depths
    has_stack_depth = 'stack_depth' in data
    has_stack_depths = 'stack_depths' in data
    if not has_stack_depth and not has_stack_depths:
        raise ValueError("Missing required field: stack_depth or stack_depths")
    if 'raise_sizes' not in data:
        raise ValueError("Missing required field: raise_sizes")

    # Parse stack_depths - support both singular and plural forms
    if has_stack_depths:
        # New format: stack_depths: [25, 50, 100]
        if not isinstance(data['stack_depths'], list):
            raise ValueError(f"stack_depths must be a list, got: {type(data['stack_depths']).__name__}")
        stack_depths = []
        for i, s in enumerate(data['stack_depths']):
            try:
                depth = float(s)
            except (TypeError, ValueError):
                raise ValueError(f"stack_depths[{i}] must be a number, got: {type(s).__name__}")
            if depth <= 0:
                raise ValueError(f"stack_depths[{i}] must be positive, got: {depth}")
            stack_depths.append(depth)
        if not stack_depths:
            raise ValueError("stack_depths cannot be empty")
    else:
        # Old format: stack_depth: 100 (backward compatibility)
        try:
            stack_depth = float(data['stack_depth'])
        except (TypeError, ValueError):
            raise ValueError(f"stack_depth must be a number, got: {type(data['stack_depth']).__name__}")
        if stack_depth <= 0:
            raise ValueError(f"stack_depth must be positive, got: {stack_depth}")
        stack_depths = [stack_depth]

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

    # Parse max_bets_per_round (optional, default 4)
    max_bets_per_round = data.get('max_bets_per_round', 4)
    if not isinstance(max_bets_per_round, int) or max_bets_per_round < 1:
        raise ValueError(f"max_bets_per_round must be a positive integer, got: {max_bets_per_round}")

    # Parse postflop-specific settings (optional)
    postflop_raise_sizes = None
    if 'postflop_raise_sizes' in data:
        if not isinstance(data['postflop_raise_sizes'], list):
            raise ValueError(f"postflop_raise_sizes must be a list")
        postflop_raise_sizes = [float(s) for s in data['postflop_raise_sizes']]

    postflop_max_bets = data.get('postflop_max_bets_per_round', 1)
    if not isinstance(postflop_max_bets, int) or postflop_max_bets < 1:
        raise ValueError(f"postflop_max_bets_per_round must be a positive integer")

    return Config(
        name=name,
        stack_depths=stack_depths,
        raise_sizes=raise_sizes,
        max_bets_per_round=max_bets_per_round,
        postflop_raise_sizes=postflop_raise_sizes,
        postflop_max_bets_per_round=postflop_max_bets,
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


def load_flops(path: str = None) -> Dict[int, List[Tuple[str, ...]]]:
    """Load flop configurations from YAML file.

    Args:
        path: Path to flops YAML file. If None, uses default config/flops.yaml.

    Returns:
        Dict mapping category (e.g., 25, 49, 85, 184) to list of flop tuples.
        Each flop is a tuple of 3 card strings, e.g., ("Kh", "Kd", "Ts").

    Raises:
        FileNotFoundError: If the flops file doesn't exist.
        ValueError: If the file format is invalid.
    """
    if path is None:
        module_dir = Path(__file__).parent
        path = module_dir / "flops.yaml"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Flops file not found: {path}")

    with open(path, 'r') as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax: {e}")

    if not isinstance(data, dict) or 'flops' not in data:
        raise ValueError("Flops file must contain a 'flops' key")

    result = {}
    for category, flop_list in data['flops'].items():
        category_int = int(category)
        parsed_flops = []
        for flop_str in flop_list:
            # Parse "KhKdTs" into ("Kh", "Kd", "Ts")
            if len(flop_str) != 6:
                raise ValueError(f"Invalid flop string: {flop_str} (must be 6 chars)")
            cards = (flop_str[0:2], flop_str[2:4], flop_str[4:6])
            parsed_flops.append(cards)
        result[category_int] = parsed_flops

    return result


def parse_flop_string(flop_str: str) -> Tuple[str, str, str]:
    """Parse a flop string like 'KhKdTs' into tuple of cards.

    Args:
        flop_str: 6-character flop string (e.g., "AhKd2c")

    Returns:
        Tuple of 3 card strings (e.g., ("Ah", "Kd", "2c"))
    """
    flop_str = flop_str.replace(" ", "")
    if len(flop_str) != 6:
        raise ValueError(f"Invalid flop string: {flop_str}")
    return (flop_str[0:2], flop_str[2:4], flop_str[4:6])
