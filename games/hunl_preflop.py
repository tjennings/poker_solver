"""
HUNL Preflop Game Implementation.

Implements the Game interface for Heads-Up No-Limit Texas Hold'em preflop.
"""

from dataclasses import dataclass
from typing import Tuple, List

from config.loader import Config
from core.hands import hand_to_string, HAND_COUNT
from games.base import Game


@dataclass(frozen=True)
class HUNLState:
    """
    State in HUNL preflop.

    Attributes:
        hands: (sb_hand_idx, bb_hand_idx), indices 0-168
        history: Tuple of actions taken, e.g., ("r2.5", "r8", "c")
        stack: Effective stack in BB
        pot: Current pot in BB
        to_act: 0=SB, 1=BB
    """
    hands: Tuple[int, int]
    history: Tuple[str, ...]
    stack: float
    pot: float
    to_act: int


class HUNLPreflop(Game):
    """
    HUNL Preflop: Heads-Up No-Limit preflop betting round.

    Rules:
    - SB posts 0.5 BB, BB posts 1 BB (initial pot = 1.5 BB)
    - SB acts first preflop
    - Actions: fold (if facing bet), call/check, raises from config, all-in
    - Terminal: fold, call after raise/all-in, check after limp
    """

    def __init__(self, config: Config):
        """
        Initialize HUNL preflop game.

        Args:
            config: Configuration with stack_depth and raise_sizes
        """
        self.config = config
        self.stack_depth = config.stack_depth
        self.raise_sizes = config.raise_sizes

    def initial_states(self) -> List[HUNLState]:
        """
        Generate all possible starting states (after dealing).

        Returns 169 * 168 = 28,392 states (all hand combinations,
        excluding cases where both players have the same hand type).
        """
        states = []
        for sb_hand in range(HAND_COUNT):
            for bb_hand in range(HAND_COUNT):
                if sb_hand != bb_hand:  # Skip same hand
                    states.append(HUNLState(
                        hands=(sb_hand, bb_hand),
                        history=(),
                        stack=self.stack_depth,
                        pot=1.5,  # SB 0.5 + BB 1.0
                        to_act=0  # SB acts first
                    ))
        return states

    def is_terminal(self, state: HUNLState) -> bool:
        """
        Check if state is terminal.

        Terminal conditions:
        - Fold: last action is 'f'
        - Call after raise: last action is 'c' and there was a raise
        - Check after limp: history is ('c', 'c')
        """
        if not state.history:
            return False

        last_action = state.history[-1]

        # Fold is always terminal
        if last_action == "f":
            return True

        # Call is terminal if there's been any raise in the history
        # or if it's BB checking after SB limp (c, c)
        if last_action == "c":
            if len(state.history) >= 2:
                # Either a raise was made, or it's check-check (limp-check)
                return True
            # Single 'c' = SB limp, not terminal
            return False

        return False

    def player(self, state: HUNLState) -> int:
        """Return player to act at state."""
        return state.to_act

    def actions(self, state: HUNLState) -> List[str]:
        """
        Available actions at state.

        Returns list of action strings:
        - 'f': fold (if facing a bet)
        - 'c': call/check
        - 'rX': raise to X BB
        - 'a': all-in
        """
        actions = []

        # Determine current bet to match
        current_bet = self._current_bet(state)
        player_committed = self._player_committed(state)

        # Can fold only if facing a bet (amount to call > 0)
        if current_bet > player_committed:
            actions.append("f")

        # Can always call/check
        actions.append("c")

        # Check if facing all-in (can't raise)
        if current_bet >= state.stack:
            return actions

        # Get legal raise sizes from config
        legal_raises = self.config.get_legal_raise_sizes(current_bet, state.stack)
        for size in legal_raises:
            actions.append(f"r{size:g}")

        # All-in option - always include if stack > current_bet
        # (even if stack happens to equal a raise size, 'a' is a distinct action)
        if state.stack > current_bet:
            actions.append("a")

        return actions

    def next_state(self, state: HUNLState, action: str) -> HUNLState:
        """
        Return state after taking action.

        Updates pot, history, and to_act.
        """
        new_history = state.history + (action,)
        player_committed = self._player_committed(state)

        if action == "f":
            # Fold - no pot change
            new_pot = state.pot
        elif action == "c":
            # Call/check - match current bet
            current_bet = self._current_bet(state)
            amount_to_add = current_bet - player_committed
            new_pot = state.pot + amount_to_add
        elif action == "a":
            # All-in - put entire stack in
            amount_to_add = state.stack - player_committed
            new_pot = state.pot + amount_to_add
        elif action.startswith("r"):
            # Raise to specified amount
            raise_to = float(action[1:])
            amount_to_add = raise_to - player_committed
            new_pot = state.pot + amount_to_add
        else:
            raise ValueError(f"Unknown action: {action}")

        # Switch player
        new_to_act = 1 - state.to_act

        return HUNLState(
            hands=state.hands,
            history=new_history,
            stack=state.stack,
            pot=new_pot,
            to_act=new_to_act
        )

    def utility(self, state: HUNLState, player: int) -> float:
        """
        Utility for player at terminal state.

        Returns profit/loss in BB.
        """
        if not self.is_terminal(state):
            raise ValueError(f"Cannot get utility of non-terminal state: {state}")

        last_action = state.history[-1]

        # Calculate each player's contribution to the pot
        sb_committed = self._total_committed(state, 0)
        bb_committed = self._total_committed(state, 1)

        if last_action == "f":
            # The player who folded is determined by who acted at the last action
            # Actions alternate starting with SB (player 0)
            # len(history) = number of actions taken
            # Last action was at index len(history)-1
            # Player who took action i is: i % 2 (0=SB, 1=BB)
            folder = (len(state.history) - 1) % 2

            if player == folder:
                # Return the loss (negative of what they put in)
                return -sb_committed if folder == 0 else -bb_committed
            else:
                # Return the win (what the other player put in)
                return sb_committed if folder == 0 else bb_committed
        else:
            # Showdown - compare hands
            # Lower index = better hand (0=AA, 1=KK, etc.)
            sb_hand = state.hands[0]
            bb_hand = state.hands[1]

            # Lower index wins
            if sb_hand < bb_hand:
                winner = 0
            else:
                winner = 1

            if player == winner:
                # Winner gets opponent's contribution
                return bb_committed if winner == 0 else sb_committed
            else:
                # Loser loses their contribution
                return -sb_committed if player == 0 else -bb_committed

    def info_set_key(self, state: HUNLState) -> str:
        """
        Map state to information set identifier.

        Format: "POSITION:HAND:HISTORY"
        e.g., "SB:AA:", "BB:KK:r3", "SB:AKs:r3-r10"
        """
        player = state.to_act
        position = "SB" if player == 0 else "BB"
        hand_idx = state.hands[player]
        hand_str = hand_to_string(hand_idx)
        history_str = "-".join(state.history)

        return f"{position}:{hand_str}:{history_str}"

    def num_players(self) -> int:
        """Number of players in the game."""
        return 2

    def _current_bet(self, state: HUNLState) -> float:
        """
        Get the current bet amount that must be matched.

        This is the highest bet/raise made so far.
        """
        if not state.history:
            # Initial state: BB has 1BB posted
            return 1.0

        # Look for the most recent raise or all-in
        for action in reversed(state.history):
            if action.startswith("r"):
                return float(action[1:])
            elif action == "a":
                return state.stack

        # No raise found - either limps or initial
        # If SB limped, current bet is 1BB (the BB)
        if "c" in state.history and not any(a.startswith("r") or a == "a" for a in state.history):
            return 1.0

        return 1.0

    def _player_committed(self, state: HUNLState) -> float:
        """
        Get amount the current player has committed to the pot.
        """
        player = state.to_act
        return self._total_committed(state, player)

    def _total_committed(self, state: HUNLState, player: int) -> float:
        """
        Calculate total amount a player has committed to the pot.

        Args:
            state: Current game state
            player: 0 for SB, 1 for BB
        """
        # Initial blinds
        if player == 0:
            committed = 0.5  # SB
        else:
            committed = 1.0  # BB

        # Process action history
        # Actions alternate: SB, BB, SB, BB, ...
        current_player = 0  # SB acts first
        for action in state.history:
            if current_player == player:
                if action == "c":
                    # Call - match current bet
                    current_bet = self._bet_at_action(state, current_player)
                    committed = current_bet
                elif action == "a":
                    # All-in
                    committed = state.stack
                elif action.startswith("r"):
                    # Raise to this amount
                    committed = float(action[1:])
                # Fold doesn't add anything

            current_player = 1 - current_player

        return committed

    def _bet_at_action(self, state: HUNLState, acting_player: int) -> float:
        """
        Get the bet amount player needs to match at their action point.

        This looks at the history up to but not including the acting player's action.
        """
        # Count actions to find where we are
        action_count = 0
        for i, action in enumerate(state.history):
            if action_count % 2 == acting_player:
                # This is this player's action - look at previous history
                for prev_action in reversed(state.history[:i]):
                    if prev_action.startswith("r"):
                        return float(prev_action[1:])
                    elif prev_action == "a":
                        return state.stack
                # No previous raise - bet is 1BB (the BB)
                return 1.0
            action_count += 1

        # Shouldn't reach here in normal use
        return 1.0
