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

# Red shades for raises (using 256-color mode for better gradients)
ANSI_LIGHT_RED = "\033[38;5;203m"      # Light red - small raise
ANSI_MEDIUM_RED = "\033[38;5;196m"     # Medium red - medium raise
ANSI_DARK_RED = "\033[38;5;160m"       # Dark red - large raise
ANSI_DARKEST_RED = "\033[38;5;124m"    # Darkest red - all-in

# Brightness threshold
BRIGHT_THRESHOLD = 0.85


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
    max_raise: float = 100.0,
) -> str:
    """
    Get the ANSI color code for a given action and frequency.

    Args:
        action: The action type ('fold', 'call', 'raise', 'all_in')
        frequency: The frequency of the action (0.0 - 1.0)
        raise_size: For raises, the size of the raise
        max_raise: Maximum raise size for scaling (default 100 BB)

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
        # Gradient from light to dark red based on raise size
        if raise_size is not None:
            ratio = min(raise_size / max_raise, 1.0)
            if ratio < 0.25:
                return ANSI_LIGHT_RED
            elif ratio < 0.5:
                return ANSI_MEDIUM_RED
            else:
                return ANSI_DARK_RED
        return ANSI_LIGHT_RED
    elif action == "all_in":
        return ANSI_DARKEST_RED
    else:
        # Default to reset for unknown actions
        return ANSI_RESET


def render_legend() -> List[str]:
    """
    Render the color legend.

    Returns:
        List of strings for each legend line
    """
    return [
        "",
        "  Legend:",
        f"  {ANSI_BRIGHT_BLUE}■{ANSI_RESET} Fold",
        f"  {ANSI_BRIGHT_GREEN}■{ANSI_RESET} Call",
        f"  {ANSI_LIGHT_RED}■{ANSI_RESET} Raise (small)",
        f"  {ANSI_MEDIUM_RED}■{ANSI_RESET} Raise (medium)",
        f"  {ANSI_DARK_RED}■{ANSI_RESET} Raise (large)",
        f"  {ANSI_DARKEST_RED}■{ANSI_RESET} All-in",
    ]


def render_matrix(
    strategy: Dict[str, ActionDistribution],
    header: str,
    max_raise: float = 100.0,
) -> str:
    """
    Render a 13x13 strategy matrix with color-coded cells and legend.

    Args:
        strategy: Dictionary mapping hand names to ActionDistribution
        header: Header text to display above the matrix
        max_raise: Maximum raise size for color scaling

    Returns:
        String containing the formatted matrix with ANSI color codes
    """
    layout = get_matrix_layout()
    matrix_lines: List[str] = []
    legend_lines = render_legend()

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
                color = get_color_for_action(action, freq, raise_size, max_raise)
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
