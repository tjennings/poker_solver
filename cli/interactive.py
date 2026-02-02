"""Interactive CLI session for exploring HUNL strategies.

Provides an interactive loop for navigating the game tree and viewing
strategy matrices for different action sequences. Supports both preflop
and post-flop play.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from config.loader import Config
from cli.matrix import ActionDistribution, render_matrix, render_header
from games.hunl_preflop import FIXED_BOARD

# Simple action aliases - maps user input to (command_type, action_code)
ACTION_ALIASES = {
    "b": ("back", None), "back": ("back", None),
    "q": ("quit", None), "quit": ("quit", None),
    "f": ("action", "f"), "fold": ("action", "f"),
    "c": ("action", "c"), "call": ("action", "c"),
    "x": ("action", "x"), "check": ("action", "x"),
    "a": ("action", "a"), "all-in": ("action", "a"),
}


def parse_user_input(user_input: str) -> Tuple[str, Optional[str]]:
    """Parse user input into a command type and optional value."""
    cleaned = user_input.strip().lower()

    # Check simple aliases first
    if cleaned in ACTION_ALIASES:
        return ACTION_ALIASES[cleaned]

    # Raise (rX or rX.X)
    if match := re.match(r"^r(\d+(?:\.\d+)?)$", cleaned):
        return ("action", cleaned)

    # Stack switch (sX or sX.X)
    if match := re.match(r"^s(\d+(?:\.\d+)?)$", cleaned):
        return ("switch_stack", match.group(1))

    return ("invalid", user_input.strip())


class InteractiveSession:
    """Interactive session for exploring HUNL strategy.

    Provides navigation through the game tree with back/forward capability,
    displays strategy matrices for the current position. Supports both
    preflop and post-flop play.

    Attributes:
        config: Game configuration
        raw_strategy: Full strategy dictionary from solver (keyed by info_set_key)
            Can be flat format {info_set: probs} or nested {stack: {info_set: probs}}
        history: List of actions taken in current exploration
        stack: Current stack size being explored
        available_stacks: List of stack depths available for exploration
        current_stack: Currently selected stack depth
        street: Current street ("preflop", "flop", "turn", "river")
        board: Current board cards as tuple of strings
    """

    def __init__(self, config: Config, raw_strategy: Dict):
        """Initialize an interactive session.

        Args:
            config: Configuration with stack_depths and raise_sizes
            raw_strategy: Full strategy dictionary from solver with keys like
                "SB:AA::r2.5-r8" mapping to action probabilities.
                Can be flat format or nested {stack: {info_set: probs}} format.
        """
        self.config = config
        self.history: List[str] = []
        self.street: str = "preflop"
        self.board: Tuple[str, ...] = ()
        self.street_contributions: List[float] = [0.0, 0.0]

        # Detect strategy format and set up multi-stack support
        self._is_multi_stack = self._detect_multi_stack(raw_strategy)
        if self._is_multi_stack:
            # Nested format: {stack: {info_set: probs}}
            self._multi_stack_strategy = raw_strategy
            self.available_stacks = sorted(raw_strategy.keys())
            self.current_stack = self.available_stacks[0]
            self.raw_strategy = raw_strategy[self.current_stack]
        else:
            # Flat format: {info_set: probs} - single stack
            self._multi_stack_strategy = None
            self.available_stacks = config.stack_depths.copy()
            self.current_stack = config.stack_depth
            self.raw_strategy = raw_strategy

        self.stack: float = self.current_stack

    def _detect_multi_stack(self, strategy: Dict) -> bool:
        """Detect if strategy is in multi-stack nested format."""
        if not strategy:
            return False
        first_key = next(iter(strategy.keys()))
        return isinstance(first_key, (int, float))

    def switch_stack(self, stack: float) -> bool:
        """Switch to a different stack depth.

        Args:
            stack: Stack depth to switch to

        Returns:
            True if switch was successful, False if stack not available
        """
        if not self._is_multi_stack:
            return False

        if stack not in self.available_stacks:
            return False

        self.current_stack = stack
        self.stack = stack
        self.raw_strategy = self._multi_stack_strategy[stack]
        self.history = []  # Reset history when switching stacks
        self.street = "preflop"
        self.board = ()
        self.street_contributions = [0.0, 0.0]
        return True

    def apply_action(self, action: str) -> None:
        """Apply an action to the current state.

        Args:
            action: Action string (e.g., "f", "c", "r2.5", "a")
        """
        # Track street contributions for post-flop
        player = self._get_current_player_idx()
        if action == "c":
            # Call/check - match opponent's contribution
            other = 1 - player
            amount = max(0, self.street_contributions[other] - self.street_contributions[player])
            self.street_contributions[player] += amount
        elif action == "a":
            self.street_contributions[player] = self.stack
        elif action.startswith("r"):
            raise_to = float(action[1:])
            self.street_contributions[player] = raise_to

        self.history.append(action)

        # Check for street transition (only on call/check, not fold)
        if action != "f" and self._betting_complete() and self.street != "river":
            self._advance_street()

    def _get_current_player_idx(self) -> int:
        """Get current player index (0=SB, 1=BB)."""
        if self.street == "preflop":
            return len(self.history) % 2
        else:
            # Post-flop: count actions since street marker
            street_actions = self._count_street_actions()
            return street_actions % 2

    def _count_street_actions(self) -> int:
        """Count actions taken on current street (after street marker)."""
        count = 0
        for action in reversed(self.history):
            if action.startswith("/"):
                break
            count += 1
        return count

    def _betting_complete(self) -> bool:
        """Check if betting is complete on current street."""
        if not self.history:
            return False

        last_action = self.history[-1]
        if last_action == "f":
            return True

        if last_action == "c":
            if self.street == "preflop":
                return len(self.history) >= 2
            else:
                # Post-flop: need 2 actions on this street
                return self._count_street_actions() >= 2

        return False

    def _advance_street(self) -> None:
        """Advance to next street."""
        next_streets = {"preflop": "flop", "flop": "turn", "turn": "river"}
        self.street = next_streets[self.street]
        self.board = FIXED_BOARD[self.street]
        self.street_contributions = [0.0, 0.0]
        # Add street marker to history
        self.history.append(f"/{self.street}")

    def go_back(self) -> bool:
        """Go back one action in the history.

        Returns:
            True if an action was removed, False if already at root
        """
        if not self.history:
            return False

        # Remove actions from the end (skip street markers, go back to last real action)
        old_history = self.history.copy()
        while old_history and old_history[-1].startswith("/"):
            old_history.pop()
        if old_history:
            old_history.pop()  # Remove the actual action

        # Replay from the beginning to rebuild state
        self._reset_state()
        for action in old_history:
            if action.startswith("/"):
                # Skip markers, they're added by _advance_street
                continue
            self.apply_action(action)
        return True

    def _reset_state(self) -> None:
        """Reset state to initial."""
        self.history = []
        self.street = "preflop"
        self.board = ()
        self.street_contributions = [0.0, 0.0]

    def get_current_player(self) -> str:
        """Get the current player to act.

        Returns:
            "SB" or "BB"
        """
        return "SB" if self._get_current_player_idx() == 0 else "BB"

    def get_pot(self) -> float:
        """Calculate the current pot size."""
        pot = 1.5  # Start with blinds
        committed = [0.5, 1.0]  # [SB, BB]

        for i, action in enumerate(self.history):
            player = i % 2
            if action == "c":
                bet = self._get_bet_at_index(i)
                pot += bet - committed[player]
                committed[player] = bet
            elif action == "a":
                pot += self.stack - committed[player]
                committed[player] = self.stack
            elif action.startswith("r"):
                raise_to = float(action[1:])
                pot += raise_to - committed[player]
                committed[player] = raise_to
        return pot

    def _get_bet_at_index(self, action_idx: int) -> float:
        """Get the bet amount before a specific action index."""
        for i in range(action_idx - 1, -1, -1):
            action = self.history[i]
            if action.startswith("r"):
                return float(action[1:])
            elif action == "a":
                return self.stack
        return 1.0

    def get_legal_actions(self) -> List[str]:
        """Get legal actions at the current state.

        Returns:
            List of action strings
        """
        actions = []

        # Determine current bet and what player has committed
        current_bet = self._get_current_bet()
        player_committed = self._get_player_committed()

        # Can fold only if facing a bet (amount to call > 0)
        if current_bet > player_committed:
            actions.append("f")

        # Can always call/check
        actions.append("c")

        # Check if facing all-in (can't raise)
        if current_bet >= self.stack:
            return actions

        # Get legal raise sizes from config
        legal_raises = self.config.get_legal_raise_sizes(current_bet, self.stack)
        for size in legal_raises:
            actions.append(f"r{size:g}")

        # All-in option
        if self.stack > current_bet:
            actions.append("a")

        return actions

    def _get_current_bet(self) -> float:
        """Get the current bet amount that must be matched."""
        return self._get_bet_at_index(len(self.history))

    def _get_player_committed(self) -> float:
        """Get amount the current player has committed to the pot."""
        player = len(self.history) % 2
        committed = 0.5 if player == 0 else 1.0

        for i, action in enumerate(self.history):
            if i % 2 == player:
                if action == "c":
                    committed = self._get_bet_at_index(i)
                elif action == "a":
                    committed = self.stack
                elif action.startswith("r"):
                    committed = float(action[1:])
        return committed

    def is_terminal(self) -> bool:
        """Check if current state is terminal.

        Returns:
            True if the game has ended (fold or showdown on river)
        """
        if not self.history:
            return False

        last_action = self.history[-1]

        # Fold is always terminal
        if last_action == "f":
            return True

        # Showdown: betting complete on river
        if self.street == "river" and self._betting_complete():
            return True

        return False

    def get_strategy_for_current_state(self) -> Dict[str, ActionDistribution]:
        """Get the strategy dictionary for the current game state.

        Filters the raw strategy to only include hands matching the current
        history and board, and converts to ActionDistribution objects.

        Returns:
            Dictionary mapping hand strings to ActionDistribution
        """
        history_str = "-".join(self.history) if self.history else ""
        board_str = "".join(self.board) if self.board else ""
        result = {}

        for info_set_key, probs in self.raw_strategy.items():
            # Parse "POSITION:HAND:BOARD:HISTORY" (new format)
            # or "POSITION:HAND:HISTORY" (old format for backward compat)
            parts = info_set_key.split(":")
            if len(parts) == 4:
                position, hand, board, history = parts
            elif len(parts) == 3:
                position, hand, history = parts
                board = ""
            else:
                continue

            # Only include strategies matching current history and board
            if history != history_str:
                continue
            if board != board_str:
                continue

            # Build action distribution from probabilities
            fold = probs.get("f", 0.0)
            call = probs.get("c", 0.0)
            all_in = probs.get("a", 0.0)

            # Collect raise actions (anything starting with 'r')
            raises = {}
            for action_key, prob in probs.items():
                if action_key.startswith("r"):
                    try:
                        size = float(action_key[1:])
                        raises[size] = prob
                    except ValueError:
                        pass

            result[hand] = ActionDistribution(
                fold=fold,
                call=call,
                raises=raises,
                all_in=all_in,
            )

        return result

    def get_strategy_for_hand(self, hand: str) -> Optional[ActionDistribution]:
        """Get the strategy for a specific hand.

        Args:
            hand: Hand string (e.g., "AA", "AKs", "72o")

        Returns:
            ActionDistribution if found, None otherwise
        """
        strategy = self.get_strategy_for_current_state()
        return strategy.get(hand)

    def render(self) -> str:
        """Render the current state as a string.

        Returns:
            Formatted string with header and strategy matrix
        """
        # Build action history with position prefixes (skip street markers)
        action_history_with_pos = []
        player_idx = 0
        for action in self.history:
            if action.startswith("/"):
                # Street marker - show it but don't increment player
                action_history_with_pos.append(action.upper())
                player_idx = 0  # Reset to OOP after street change
            else:
                pos = "SB" if player_idx == 0 else "BB"
                action_history_with_pos.append(f"{pos}{action}")
                player_idx = 1 - player_idx

        # Render header
        header = render_header(
            stack=self.stack,
            pot=self.get_pot(),
            action_history=action_history_with_pos,
            player=self.get_current_player(),
            street=self.street,
            board=self.board,
        )

        # Get strategy for current state (filtered by history)
        current_strategy = self.get_strategy_for_current_state()

        # Render matrix with raise sizes from config
        matrix = render_matrix(current_strategy, header, self.config.raise_sizes)

        return matrix


def run_interactive(
    config: Config,
    strategy: Dict,
    initial_actions: tuple = ()
) -> None:
    """Run the interactive exploration loop.

    Args:
        config: Game configuration
        strategy: Strategy dictionary (flat or nested multi-stack format)
        initial_actions: Optional tuple of actions to pre-apply to the session
    """
    session = InteractiveSession(config, strategy)
    for action in initial_actions:
        session.apply_action(action)

    while True:
        # Clear screen and show current state
        print("\033[2J\033[H")  # Clear screen, move cursor to top
        print(session.render())
        print()

        if session.is_terminal():
            print("Terminal state reached. Press 'b' to go back or 'q' to quit.")

        # Build prompt with stack switching info if multi-stack
        if session._is_multi_stack and len(session.available_stacks) > 1:
            stacks_str = ", ".join(f"s{int(s) if s == int(s) else s}" for s in session.available_stacks)
            print(f"Enter action (f/c/rX/a), 'b' to go back, 'q' to quit, or switch stack ({stacks_str}):")
        else:
            print("Enter action (f/c/rX/a), 'b' to go back, 'q' to quit:")
        user_input = input("> ")

        command, value = parse_user_input(user_input)

        if command == "quit":
            print("Goodbye!")
            break
        elif command == "back":
            if not session.go_back():
                print("Already at root position.")
        elif command == "switch_stack":
            if not session._is_multi_stack:
                print("Stack switching not available (single stack strategy).")
            else:
                try:
                    target_stack = float(value)
                    if session.switch_stack(target_stack):
                        print(f"Switched to {target_stack}BB stack.")
                    else:
                        available = ", ".join(str(int(s) if s == int(s) else s) for s in session.available_stacks)
                        print(f"Stack {target_stack}BB not available. Available: {available}")
                except ValueError:
                    print(f"Invalid stack value: {value}")
        elif command == "action":
            if session.is_terminal():
                print("Cannot take action in terminal state. Go back first.")
            else:
                # Validate action is legal
                legal = session.get_legal_actions()
                # Normalize action for comparison
                if value in legal or value in [a.replace(".0", "") for a in legal]:
                    session.apply_action(value)
                else:
                    print(f"Invalid action: {value}. Legal actions: {legal}")
        elif command == "invalid":
            print(f"Unknown command: {value}")
