"""
Matrix Display Module for HUNL Preflop Strategy Visualization.

Provides color-coded 13x13 strategy matrix display for terminal output
using ANSI escape codes.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from core.hands import get_matrix_layout


# ANSI escape codes for colors
ANSI_RESET = "\033[0m"

# Dim colors (31-37)
ANSI_DIM_RED = "\033[31m"
ANSI_DIM_GREEN = "\033[32m"
ANSI_DIM_YELLOW = "\033[33m"
ANSI_DIM_BLUE = "\033[34m"

# Bright colors (91-97)
ANSI_BRIGHT_RED = "\033[91m"
ANSI_BRIGHT_GREEN = "\033[92m"
ANSI_BRIGHT_YELLOW = "\033[93m"
ANSI_BRIGHT_BLUE = "\033[94m"

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

    def dominant_action(self) -> Tuple[str, float]:
        """
        Determine the dominant action and its frequency.

        Returns:
            Tuple of (action_type, frequency) where action_type is one of
            'fold', 'call', 'raise', or 'all_in'.
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
        # (in order: fold, call, raise, all_in)
        for action, freq in actions:
            if freq == max_freq:
                return action, freq

        # Fallback (should never reach)
        return "fold", self.fold


def get_color_for_action(action: str, frequency: float) -> str:
    """
    Get the ANSI color code for a given action and frequency.

    Args:
        action: The action type ('fold', 'call', 'raise', 'all_in')
        frequency: The frequency of the action (0.0 - 1.0)

    Returns:
        ANSI escape code string for the appropriate color
    """
    # Determine if bright or dim based on frequency
    is_bright = frequency >= BRIGHT_THRESHOLD

    # Map action to color
    if action == "fold":
        return ANSI_BRIGHT_RED if is_bright else ANSI_DIM_RED
    elif action == "call":
        return ANSI_BRIGHT_GREEN if is_bright else ANSI_DIM_GREEN
    elif action == "raise":
        return ANSI_BRIGHT_BLUE if is_bright else ANSI_DIM_BLUE
    elif action == "all_in":
        return ANSI_BRIGHT_YELLOW if is_bright else ANSI_DIM_YELLOW
    else:
        # Default to reset for unknown actions
        return ANSI_RESET


def render_matrix(strategy: Dict[str, ActionDistribution], header: str) -> str:
    """
    Render a 13x13 strategy matrix with color-coded cells.

    Args:
        strategy: Dictionary mapping hand names to ActionDistribution
        header: Header text to display above the matrix

    Returns:
        String containing the formatted matrix with ANSI color codes
    """
    layout = get_matrix_layout()
    lines: List[str] = []

    # Add header
    lines.append(header)
    lines.append("")

    # Render each row of the matrix
    for row in layout:
        row_cells: List[str] = []
        for hand in row:
            # Get the action distribution for this hand
            dist = strategy.get(hand)
            if dist is not None:
                action, freq = dist.dominant_action()
                color = get_color_for_action(action, freq)
            else:
                # Default to no color if no strategy for this hand
                color = ANSI_RESET

            # Format cell: left-aligned, 4-char width
            cell = f"{color}{hand:<4}{ANSI_RESET}"
            row_cells.append(cell)

        lines.append("".join(row_cells))

    return "\n".join(lines)


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
