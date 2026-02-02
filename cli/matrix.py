"""
Matrix Display Module for HUNL Preflop Strategy Visualization.

Provides color-coded 13x13 strategy matrix display for terminal output
using ANSI escape codes. Each cell shows action frequencies as colored
background segments with white text.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from core.hands import get_matrix_layout


# ANSI escape codes
ANSI_RESET = "\033[0m"
ANSI_WHITE_FG = "\033[97m"  # Bright white foreground

# Background colors (256-color mode)
BG_BLACK = "\033[48;5;16m"     # Black for cell borders
BG_BLUE = "\033[48;5;24m"      # Blue for fold
BG_GREEN = "\033[48;5;22m"     # Green for call
BG_LIGHT_RED = "\033[48;5;124m"   # Light red for small raises
BG_DARK_RED = "\033[48;5;88m"    # Dark red for large raises
BG_DARKEST_RED = "\033[48;5;52m"  # Darkest red for all-in
BG_GRAY = "\033[48;5;236m"     # Gray for no strategy

# Foreground colors for legend
ANSI_DIM_BLUE = "\033[34m"
ANSI_BRIGHT_BLUE = "\033[94m"
ANSI_DIM_GREEN = "\033[32m"
ANSI_BRIGHT_GREEN = "\033[92m"

# Red color codes for raises (256-color mode foreground)
RED_GRADIENT = [203, 196, 167, 160, 124]
ANSI_DARKEST_RED = "\033[38;5;124m"

# Brightness threshold
BRIGHT_THRESHOLD = 0.85

# Cell dimensions
CELL_WIDTH = 5
CELL_HEIGHT = 2


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

    def get_action_segments(self, raise_sizes: Optional[List[float]] = None) -> List[Tuple[str, float]]:
        """
        Get ordered list of (action, frequency) for rendering segments.

        Returns actions from strongest to weakest (left to right):
        all_in, raises (descending by size), call, fold.

        Actions below the minimum visible threshold are filtered out.
        With CELL_WIDTH=5, each column is 20% of the cell. An action needs
        at least 50% of a column (10%) to be visible, otherwise it's rounded
        to zero.
        """
        # Minimum frequency to display: 50% of one column width
        # With CELL_WIDTH=5, each column is 20%, so threshold is 10%
        min_threshold = 0.5 / CELL_WIDTH  # 0.10 for CELL_WIDTH=5

        segments = []

        # Strongest action first (leftmost)
        if self.all_in >= min_threshold:
            segments.append(("all_in", self.all_in))

        # Raises sorted by size descending (larger raises = more aggressive)
        if self.raises:
            sorted_raises = sorted(self.raises.items(), key=lambda x: x[0], reverse=True)
            for size, freq in sorted_raises:
                if freq >= min_threshold:
                    segments.append((f"r{size}", freq))

        if self.call >= min_threshold:
            segments.append(("call", self.call))

        # Weakest action last (rightmost)
        if self.fold >= min_threshold:
            segments.append(("fold", self.fold))

        return segments


def get_bg_color_for_action(action: str, raise_sizes: Optional[List[float]] = None) -> str:
    """Get background color for an action."""
    if action == "fold":
        return BG_BLUE
    elif action == "call":
        return BG_GREEN
    elif action == "all_in":
        return BG_DARKEST_RED
    elif action.startswith("r"):
        # Raise - gradient based on size
        if raise_sizes:
            try:
                size = float(action[1:])
                sorted_sizes = sorted(raise_sizes)
                idx = min(range(len(sorted_sizes)), key=lambda i: abs(sorted_sizes[i] - size))
                ratio = idx / max(1, len(sorted_sizes) - 1)
                if ratio < 0.5:
                    return BG_LIGHT_RED
                else:
                    return BG_DARK_RED
            except ValueError:
                pass
        return BG_LIGHT_RED
    return BG_GRAY


def render_cell(hand: str, dist: Optional[ActionDistribution], raise_sizes: Optional[List[float]] = None) -> List[str]:
    """
    Render a single cell with frequency-based background segments.

    Cell has no border (borders are added between cells by render_matrix).
    Hand name appears in the upper-left corner.

    Args:
        hand: Hand name (e.g., "AA", "AKs", "72o")
        dist: Action distribution for this hand
        raise_sizes: List of configured raise sizes

    Returns:
        List of ANSI-formatted strings, one per row of the cell
    """
    # Pad hand to cell width for the label row
    padded_hand = f"{hand:<{CELL_WIDTH}}"
    blank_row = " " * CELL_WIDTH

    # Determine content colors for each column
    if dist is None or not dist.get_action_segments(raise_sizes):
        # No strategy - gray background
        column_colors = [BG_GRAY] * CELL_WIDTH
    else:
        # Build cumulative frequency ranges
        segments = dist.get_action_segments(raise_sizes)
        cumulative = []
        total = 0.0
        for action, freq in segments:
            cumulative.append((action, total, total + freq))
            total += freq

        # Pre-compute background color for each column
        column_colors = []
        for i in range(CELL_WIDTH):
            midpoint = (i + 0.5) / CELL_WIDTH
            scaled_pos = midpoint * total if total > 0 else 0

            bg_color = BG_GRAY
            for action, start, end in cumulative:
                if start <= scaled_pos < end:
                    bg_color = get_bg_color_for_action(action, raise_sizes)
                    break
            else:
                if cumulative:
                    bg_color = get_bg_color_for_action(cumulative[-1][0], raise_sizes)
            column_colors.append(bg_color)

    # Generate rows
    rows = []
    for row_idx in range(CELL_HEIGHT):
        text = padded_hand if row_idx == 0 else blank_row
        row_chars = []
        for i in range(CELL_WIDTH):
            row_chars.append(f"{column_colors[i]}{ANSI_WHITE_FG}{text[i]}")
        rows.append("".join(row_chars) + ANSI_RESET)

    return rows


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
    Render the color legend with background color samples.

    Legend order matches cell rendering: strongest (left) to weakest (right).

    Args:
        raise_sizes: List of configured raise sizes to show in legend

    Returns:
        List of strings for each legend line
    """
    lines = [
        "",
        "  Legend (left to right):",
        f"  {BG_DARKEST_RED}{ANSI_WHITE_FG}  {ANSI_RESET} All-in",
    ]

    # Add raise sizes with their colors (descending - largest first)
    if raise_sizes:
        sorted_sizes = sorted(raise_sizes, reverse=True)
        for i, size in enumerate(sorted_sizes):
            # Ratio based on position in descending list
            ratio = i / max(1, len(sorted_sizes) - 1)
            bg = BG_DARK_RED if ratio < 0.5 else BG_LIGHT_RED
            # Format size: show as integer if whole number
            if size == int(size):
                size_str = f"{int(size)}"
            else:
                size_str = f"{size:g}"
            lines.append(f"  {bg}{ANSI_WHITE_FG}  {ANSI_RESET} Raise {size_str}")

    lines.append(f"  {BG_GREEN}{ANSI_WHITE_FG}  {ANSI_RESET} Call")
    lines.append(f"  {BG_BLUE}{ANSI_WHITE_FG}  {ANSI_RESET} Fold")

    return lines


def render_matrix(
    strategy: Dict[str, ActionDistribution],
    header: str,
    raise_sizes: Optional[List[float]] = None,
) -> str:
    """
    Render a 13x13 strategy matrix with color-coded cells and legend.

    Cells are separated by thin 1-character black borders.

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

    # Track which legend line to add next
    legend_idx = 0

    # Black separator between columns (1 character)
    col_sep = f"{BG_BLACK} {ANSI_RESET}"

    # Matrix width: 13 cells + 12 separators
    matrix_width = CELL_WIDTH * 13 + 12

    # Create horizontal separator row (all black, full width)
    h_sep_row = f"{BG_BLACK}{ANSI_WHITE_FG}{' ' * matrix_width}{ANSI_RESET}"

    # Render each row of the matrix
    for row_idx, row in enumerate(layout):
        # Add horizontal separator before each row (except first)
        if row_idx > 0:
            matrix_lines.append(h_sep_row)

        # Render all cells for this row
        cell_rows: List[List[str]] = []
        for hand in row:
            dist = strategy.get(hand)
            cell_lines = render_cell(hand, dist, raise_sizes)
            cell_rows.append(cell_lines)

        # Combine cells horizontally for each terminal line
        for line_idx in range(CELL_HEIGHT):
            # Join cells with black separator
            row_str = col_sep.join(cell[line_idx] for cell in cell_rows)

            # Add legend line if available (on first line of cell for alignment)
            if line_idx == 0 and legend_idx < len(legend_lines):
                row_str += legend_lines[legend_idx]
                legend_idx += 1

            matrix_lines.append(row_str)

    # Add any remaining legend lines
    while legend_idx < len(legend_lines):
        matrix_lines.append(" " * matrix_width + legend_lines[legend_idx])
        legend_idx += 1

    return "\n".join(matrix_lines)


def render_header(
    stack: float,
    pot: float,
    action_history: List[str],
    player: str,
    street: str = "preflop",
    board: Tuple[str, ...] = (),
) -> str:
    """
    Render the header showing game state information.

    Args:
        stack: Stack size in big blinds
        pot: Current pot size in big blinds
        action_history: List of action strings (e.g., ["SBr2.5", "BBr8"])
        player: Player to act ("SB" or "BB")
        street: Current street ("preflop", "flop", "turn", "river")
        board: Board cards as tuple of strings

    Returns:
        Formatted header string
    """
    lines: List[str] = []

    # Format stack - show as integer if it's a whole number
    if stack == int(stack):
        stack_str = f"{int(stack)}BB"
    else:
        stack_str = f"{stack}BB"

    # First line: game info with street
    street_display = street.capitalize()
    line1 = f"HUNL {street_display} | Stack: {stack_str} | Pot: {pot}BB"
    lines.append(line1)

    # Board line (if post-flop)
    if board:
        board_str = " ".join(board)
        lines.append(f"Board: {board_str}")

    # Action history and player to act
    if action_history:
        action_str = " ".join(action_history)
        line2 = f"Action: {action_str} | {player} to act"
    else:
        line2 = f"{player} to act"

    lines.append(line2)

    return "\n".join(lines)
