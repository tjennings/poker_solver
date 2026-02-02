"""
Matrix Display Module for HUNL Preflop Strategy Visualization.

Provides color-coded 13x13 strategy matrix display for terminal output
using ANSI escape codes.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from core.hands import get_matrix_layout


# ANSI escape codes for colors
ANSI_RESET = "\033[0m"

# Blue for fold
ANSI_DIM_BLUE = "\033[34m"
ANSI_BRIGHT_BLUE = "\033[94m"

# Green for call
ANSI_DIM_GREEN = "\033[32m"
ANSI_BRIGHT_GREEN = "\033[92m"

# Red color codes for raises (256-color mode)
# Gradient from light to dark: 203 -> 196 -> 160 -> 124
RED_GRADIENT = [203, 196, 167, 160, 124]  # Light to dark red codes
ANSI_DARKEST_RED = "\033[38;5;124m"  # Reserved for all-in

# Brightness threshold
BRIGHT_THRESHOLD = 0.85


def get_raise_color(raise_size: float, raise_sizes: List[float]) -> str:
    """Get the ANSI color for a raise size based on its position in the config.

    Args:
        raise_size: The raise size to get color for
        raise_sizes: List of configured raise sizes (sorted ascending)

    Returns:
        ANSI escape code for the appropriate red shade
    """
    if not raise_sizes:
        return f"\033[38;5;{RED_GRADIENT[0]}m"

    # Find the index of this raise size in the sorted list
    sorted_sizes = sorted(raise_sizes)
    try:
        idx = sorted_sizes.index(raise_size)
    except ValueError:
        # If exact match not found, find closest
        idx = min(range(len(sorted_sizes)), key=lambda i: abs(sorted_sizes[i] - raise_size))

    # Map index to color gradient
    num_sizes = len(sorted_sizes)
    if num_sizes == 1:
        color_idx = len(RED_GRADIENT) // 2
    else:
        # Interpolate position in gradient
        ratio = idx / (num_sizes - 1)
        color_idx = int(ratio * (len(RED_GRADIENT) - 1))

    color_code = RED_GRADIENT[min(color_idx, len(RED_GRADIENT) - 1)]
    return f"\033[38;5;{color_code}m"


@dataclass
class ActionDistribution:
    """
    Distribution of actions for a specific hand.

    Attributes:
        fold: Probability of folding (0.0 - 1.0)
        call: Probability of calling (0.0 - 1.0)
        raises: Dictionary mapping raise sizes to probabilities
        all_in: Probability of going all-in (0.0 - 1.0)
    """

    fold: float
    call: float
    raises: Dict[float, float]  # size -> probability
    all_in: float

    def dominant_action(self) -> Tuple[str, float, Optional[float]]:
        """
        Determine the dominant action and its frequency.

        Returns:
            Tuple of (action_type, frequency, raise_size) where action_type is one of
            'fold', 'call', 'raise', or 'all_in'. raise_size is the dominant raise
            size if action is 'raise', None otherwise.
        """
        total_raise = sum(self.raises.values())

        actions = [
            ("fold", self.fold),
            ("call", self.call),
            ("raise", total_raise),
            ("all_in", self.all_in),
        ]

        # Find maximum frequency
        max_freq = max(freq for _, freq in actions)

        # Return the first action with max frequency
        for action, freq in actions:
            if freq == max_freq:
                if action == "raise" and self.raises:
                    # Find the dominant raise size
                    dominant_size = max(self.raises.keys(), key=lambda k: self.raises[k])
                    return action, freq, dominant_size
                return action, freq, None

        # Fallback (should never reach)
        return "fold", self.fold, None


def get_color_for_action(
    action: str,
    frequency: float,
    raise_size: Optional[float] = None,
    raise_sizes: Optional[List[float]] = None,
) -> str:
    """
    Get the ANSI color code for a given action and frequency.

    Args:
        action: The action type ('fold', 'call', 'raise', 'all_in')
        frequency: The frequency of the action (0.0 - 1.0)
        raise_size: For raises, the size of the raise
        raise_sizes: List of configured raise sizes for color mapping

    Returns:
        ANSI escape code string for the appropriate color
    """
    # Determine if bright or dim based on frequency
    is_bright = frequency >= BRIGHT_THRESHOLD

    # Map action to color
    if action == "fold":
        return ANSI_BRIGHT_BLUE if is_bright else ANSI_DIM_BLUE
    elif action == "call":
        return ANSI_BRIGHT_GREEN if is_bright else ANSI_DIM_GREEN
    elif action == "raise":
        if raise_size is not None and raise_sizes:
            return get_raise_color(raise_size, raise_sizes)
        # Fallback to middle red if no size info
        return f"\033[38;5;{RED_GRADIENT[len(RED_GRADIENT) // 2]}m"
    elif action == "all_in":
        return ANSI_DARKEST_RED
    else:
        # Default to reset for unknown actions
        return ANSI_RESET


def render_legend(raise_sizes: Optional[List[float]] = None) -> List[str]:
    """
    Render the color legend.

    Args:
        raise_sizes: List of configured raise sizes to show in legend

    Returns:
        List of strings for each legend line
    """
    lines = [
        "",
        "  Legend:",
        f"  {ANSI_BRIGHT_BLUE}■{ANSI_RESET} Fold",
        f"  {ANSI_BRIGHT_GREEN}■{ANSI_RESET} Call",
    ]

    # Add raise sizes with their colors
    if raise_sizes:
        sorted_sizes = sorted(raise_sizes)
        for size in sorted_sizes:
            color = get_raise_color(size, sorted_sizes)
            # Format size: show as integer if whole number
            if size == int(size):
                size_str = f"{int(size)}"
            else:
                size_str = f"{size:g}"
            lines.append(f"  {color}■{ANSI_RESET} Raise {size_str}")

    lines.append(f"  {ANSI_DARKEST_RED}■{ANSI_RESET} All-in")

    return lines


def render_matrix(
    strategy: Dict[str, ActionDistribution],
    header: str,
    raise_sizes: Optional[List[float]] = None,
) -> str:
    """
    Render a 13x13 strategy matrix with color-coded cells and legend.

    Args:
        strategy: Dictionary mapping hand names to ActionDistribution
        header: Header text to display above the matrix
        raise_sizes: List of configured raise sizes for color mapping and legend

    Returns:
        String containing the formatted matrix with ANSI color codes
    """
    layout = get_matrix_layout()
    matrix_lines: List[str] = []
    legend_lines = render_legend(raise_sizes)

    # Add header
    matrix_lines.append(header)
    matrix_lines.append("")

    # Render each row of the matrix
    for row_idx, row in enumerate(layout):
        row_cells: List[str] = []
        for hand in row:
            # Get the action distribution for this hand
            dist = strategy.get(hand)
            if dist is not None:
                action, freq, raise_size = dist.dominant_action()
                color = get_color_for_action(action, freq, raise_size, raise_sizes)
            else:
                # Default to no color if no strategy for this hand
                color = ANSI_RESET

            # Format cell: left-aligned, 4-char width
            cell = f"{color}{hand:<4}{ANSI_RESET}"
            row_cells.append(cell)

        # Combine row with legend (if legend line exists for this row)
        row_str = "".join(row_cells)
        if row_idx < len(legend_lines):
            row_str += legend_lines[row_idx]

        matrix_lines.append(row_str)

    # Add any remaining legend lines
    for i in range(len(layout), len(legend_lines)):
        matrix_lines.append(" " * 52 + legend_lines[i])

    return "\n".join(matrix_lines)


def render_header(
    stack: float, pot: float, action_history: List[str], player: str
) -> str:
    """
    Render the header showing game state information.

    Args:
        stack: Stack size in big blinds
        pot: Current pot size in big blinds
        action_history: List of action strings (e.g., ["SBr2.5", "BBr8"])
        player: Player to act ("SB" or "BB")

    Returns:
        Formatted header string
    """
    lines: List[str] = []

    # Format stack - show as integer if it's a whole number
    if stack == int(stack):
        stack_str = f"{int(stack)}BB"
    else:
        stack_str = f"{stack}BB"

    # First line: game info
    line1 = f"HUNL Preflop | Stack: {stack_str} | Pot: {pot}BB"
    lines.append(line1)

    # Second line: action history and player to act
    if action_history:
        action_str = " ".join(action_history)
        line2 = f"Action: {action_str} | {player} to act"
    else:
        line2 = f"{player} to act"

    lines.append(line2)

    return "\n".join(lines)
