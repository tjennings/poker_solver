"""Interactive CLI session for exploring HUNL preflop strategies.

Provides an interactive loop for navigating the game tree and viewing
strategy matrices for different action sequences.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from config.loader import Config
from cli.matrix import ActionDistribution, render_matrix, render_header

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
    """Interactive session for exploring HUNL preflop strategy.

    Provides navigation through the game tree with back/forward capability,
    displays strategy matrices for the current position.

    Attributes:
        config: Game configuration
        raw_strategy: Full strategy dictionary from solver (keyed by info_set_key)
            Can be flat format {info_set: probs} or nested {stack: {info_set: probs}}
        history: List of actions taken in current exploration
        stack: Current stack size being explored
        available_stacks: List of stack depths available for exploration
        current_stack: Currently selected stack depth
    """

    def __init__(self, config: Config, raw_strategy: Dict):
        """Initialize an interactive session.

        Args:
            config: Configuration with stack_depths and raise_sizes
            raw_strategy: Full strategy dictionary from solver with keys like
                "SB:AA:r2.5-r8" mapping to action probabilities.
                Can be flat format or nested {stack: {info_set: probs}} format.
        """
        self.config = config
        self.history: List[str] = []

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
        return True

    def apply_action(self, action: str) -> None:
        """Apply an action to the current state.

        Args:
            action: Action string (e.g., "f", "c", "r2.5", "a")
        """
        self.history.append(action)

    def go_back(self) -> bool:
        """Go back one action in the history.

        Returns:
            True if an action was removed, False if already at root
        """
        if not self.history:
            return False
        self.history.pop()
        return True

    def get_current_player(self) -> str:
        """Get the current player to act.

        Returns:
            "SB" or "BB"
        """
        # SB acts first (index 0), then BB (index 1), alternating
        return "SB" if len(self.history) % 2 == 0 else "BB"

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
            True if the game has ended (fold or showdown)
        """
        if not self.history:
            return False

        last_action = self.history[-1]

        # Fold is always terminal
        if last_action == "f":
            return True

        # Call is terminal if there's been any raise in the history
        # or if it's BB checking after SB limp (c, c)
        if last_action == "c":
            if len(self.history) >= 2:
                return True
            # Single 'c' = SB limp, not terminal
            return False

        return False

    def get_strategy_for_current_state(self) -> Dict[str, ActionDistribution]:
        """Get the strategy dictionary for the current game state.

        Filters the raw strategy to only include hands matching the current
        history, and converts to ActionDistribution objects.

        Returns:
            Dictionary mapping hand strings to ActionDistribution
        """
        history_str = "-".join(self.history) if self.history else ""
        result = {}

        for info_set_key, probs in self.raw_strategy.items():
            # Parse "POSITION:HAND:HISTORY"
            parts = info_set_key.split(":")
            if len(parts) != 3:
                continue
            position, hand, history = parts

            # Only include strategies matching current history
            if history != history_str:
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
        # Build action history with position prefixes
        action_history_with_pos = []
        for i, action in enumerate(self.history):
            pos = "SB" if i % 2 == 0 else "BB"
            action_history_with_pos.append(f"{pos}{action}")

        # Render header
        header = render_header(
            stack=self.stack,
            pot=self.get_pot(),
            action_history=action_history_with_pos,
            player=self.get_current_player(),
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
