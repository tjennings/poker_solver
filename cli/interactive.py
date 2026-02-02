"""Interactive CLI session for exploring HUNL preflop strategies.

Provides an interactive loop for navigating the game tree and viewing
strategy matrices for different action sequences.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from config.loader import Config
from cli.matrix import ActionDistribution, render_matrix, render_header


def parse_user_input(user_input: str) -> Tuple[str, Optional[str]]:
    """Parse user input into a command type and optional value.

    Args:
        user_input: Raw user input string

    Returns:
        Tuple of (command_type, value) where:
        - command_type is one of: "action", "back", "quit", "invalid"
        - value is the action string for "action" type, original input for
          "invalid", or None for "back"/"quit"

    Examples:
        >>> parse_user_input("f")
        ('action', 'f')
        >>> parse_user_input("r8")
        ('action', 'r8')
        >>> parse_user_input("b")
        ('back', None)
        >>> parse_user_input("xyz")
        ('invalid', 'xyz')
    """
    cleaned = user_input.strip().lower()

    # Back command
    if cleaned in ("b", "back"):
        return ("back", None)

    # Quit command
    if cleaned in ("q", "quit"):
        return ("quit", None)

    # Fold
    if cleaned in ("f", "fold"):
        return ("action", "f")

    # Call
    if cleaned in ("c", "call"):
        return ("action", "c")

    # Check
    if cleaned in ("x", "check"):
        return ("action", "x")

    # All-in
    if cleaned in ("a", "all-in"):
        return ("action", "a")

    # Raise (rX or rX.X)
    if cleaned.startswith("r"):
        # Validate it has a numeric value after 'r'
        raise_match = re.match(r"^r(\d+(?:\.\d+)?)$", cleaned)
        if raise_match:
            return ("action", cleaned)

    # Invalid input
    return ("invalid", user_input.strip())


class InteractiveSession:
    """Interactive session for exploring HUNL preflop strategy.

    Provides navigation through the game tree with back/forward capability,
    displays strategy matrices for the current position.

    Attributes:
        config: Game configuration
        raw_strategy: Full strategy dictionary from solver (keyed by info_set_key)
        history: List of actions taken in current exploration
        stack: Stack size from config
    """

    def __init__(self, config: Config, raw_strategy: Dict):
        """Initialize an interactive session.

        Args:
            config: Configuration with stack_depth and raise_sizes
            raw_strategy: Full strategy dictionary from solver with keys like
                "SB:AA:r2.5-r8" mapping to action probabilities
        """
        self.config = config
        self.raw_strategy = raw_strategy
        self.history: List[str] = []
        self.stack: float = config.stack_depth

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
        """Calculate the current pot size.

        Returns:
            Pot size in big blinds
        """
        # Start with blinds: SB 0.5, BB 1.0
        pot = 1.5
        sb_committed = 0.5
        bb_committed = 1.0

        for i, action in enumerate(self.history):
            current_player = i % 2  # 0 = SB, 1 = BB

            # Get the bet this player needs to match (look at prior history)
            current_bet = self._get_bet_at_index(i)

            if current_player == 0:
                # SB acting
                if action == "c":
                    # SB calls/checks - match current bet
                    amount_to_add = current_bet - sb_committed
                    pot += amount_to_add
                    sb_committed = current_bet
                elif action == "a":
                    # All-in
                    amount_to_add = self.stack - sb_committed
                    pot += amount_to_add
                    sb_committed = self.stack
                elif action.startswith("r"):
                    # Raise to X
                    raise_to = float(action[1:])
                    amount_to_add = raise_to - sb_committed
                    pot += amount_to_add
                    sb_committed = raise_to
                # fold adds nothing
            else:
                # BB acting
                if action == "c":
                    # BB calls/checks - match current bet
                    amount_to_add = current_bet - bb_committed
                    pot += amount_to_add
                    bb_committed = current_bet
                elif action == "a":
                    # All-in
                    amount_to_add = self.stack - bb_committed
                    pot += amount_to_add
                    bb_committed = self.stack
                elif action.startswith("r"):
                    # Raise to X
                    raise_to = float(action[1:])
                    amount_to_add = raise_to - bb_committed
                    pot += amount_to_add
                    bb_committed = raise_to
                # fold adds nothing

        return pot

    def _get_bet_at_index(self, action_idx: int) -> float:
        """Get the bet amount at a specific action index.

        Scans backward from the given action index to find the most recent
        raise or all-in that set the bet amount.

        Args:
            action_idx: Index into history of the action being considered.

        Returns:
            The bet amount in BB that was active at action_idx.
            Returns 1.0 (BB) if no prior raises exist.
        """
        for i in range(action_idx - 1, -1, -1):
            action = self.history[i]
            if action.startswith("r"):
                return float(action[1:])
            elif action == "a":
                return self.stack
        return 1.0  # Default is BB = 1

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
        """Get the current bet amount that must be matched.

        Scans the action history in reverse to find the most recent raise
        or all-in action. If no raises have occurred, returns 1.0 (the BB).

        Returns:
            The current bet size in big blinds that the acting player must match.
        """
        if not self.history:
            return 1.0  # BB is 1

        # Look for the most recent raise or all-in
        for action in reversed(self.history):
            if action.startswith("r"):
                return float(action[1:])
            elif action == "a":
                return self.stack

        # No raise found - bet is 1BB (the BB)
        return 1.0

    def _get_player_committed(self) -> float:
        """Get amount the current player has committed to the pot.

        Calculates the total amount the current player (determined by history
        length) has put into the pot, including their initial blind and any
        calls, raises, or all-in actions.

        Returns:
            The total amount committed by the current player in big blinds.
        """
        current_player = len(self.history) % 2
        # Initial blinds
        committed = 0.5 if current_player == 0 else 1.0

        # Process this player's actions in history
        for i, action in enumerate(self.history):
            if i % 2 == current_player:
                if action == "c":
                    # Called - matched previous bet
                    bet_at_call = self._get_bet_before_action(i)
                    committed = bet_at_call
                elif action == "a":
                    committed = self.stack
                elif action.startswith("r"):
                    committed = float(action[1:])

        return committed

    def _get_bet_before_action(self, action_idx: int) -> float:
        """Get the bet amount before a specific action index.

        Scans backward from the given action index to find the most recent
        raise or all-in that set the bet amount the player needed to match.

        Args:
            action_idx: Index into history of the action being considered.

        Returns:
            The bet amount in BB that was active before action_idx.
            Returns 1.0 (BB) if no prior raises exist.
        """
        for i in range(action_idx - 1, -1, -1):
            action = self.history[i]
            if action.startswith("r"):
                return float(action[1:])
            elif action == "a":
                return self.stack
        return 1.0  # Default is BB = 1

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
        strategy: Strategy dictionary
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

        print("Enter action (f/c/rX/a), 'b' to go back, 'q' to quit:")
        user_input = input("> ")

        command, value = parse_user_input(user_input)

        if command == "quit":
            print("Goodbye!")
            break
        elif command == "back":
            if not session.go_back():
                print("Already at root position.")
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
