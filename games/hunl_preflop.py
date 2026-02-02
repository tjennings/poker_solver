"""
HUNL Game Implementation.

Implements the Game interface for Heads-Up No-Limit Texas Hold'em
with support for preflop through river play.
"""

from typing import Tuple, List, NamedTuple, Dict, Optional

from config.loader import Config
from core.hands import hand_to_string, HAND_COUNT
from core.equity import get_preflop_equity
from games.base import Game


# Default board for backward compatibility
FIXED_BOARD = {
    "flop": ("Ah", "Th", "6c"),
    "turn": ("Ah", "Th", "6c", "Jc"),
    "river": ("Ah", "Th", "6c", "Jc", "8c"),
}


def make_board_config(flop: Tuple[str, str, str], turn: str = "Jc", river: str = "8c") -> Dict[str, Tuple[str, ...]]:
    """Create a board config dict from flop cards.

    Args:
        flop: Tuple of 3 flop cards, e.g., ("Kh", "Kd", "Ts")
        turn: Turn card (default "Jc")
        river: River card (default "8c")

    Returns:
        Dict with 'flop', 'turn', 'river' keys mapping to board tuples.
    """
    return {
        "flop": flop,
        "turn": flop + (turn,),
        "river": flop + (turn, river),
    }


class HUNLState(NamedTuple):
    """State in HUNL game."""
    hands: Tuple[int, int]
    history: Tuple[str, ...]
    stack: float
    pot: float
    to_act: int
    street: str = "preflop"
    board: Tuple[str, ...] = ()
    street_contributions: Tuple[float, float] = (0.0, 0.0)
    committed: Tuple[float, float] = (0.5, 1.0)
    current_bet: float = 1.0


class HUNLPreflop(Game):
    """
    HUNL Game: Heads-Up No-Limit Texas Hold'em.

    Supports preflop through river play with a fixed board.

    Rules:
    - SB posts 0.5 BB, BB posts 1 BB (initial pot = 1.5 BB)
    - SB acts first preflop
    - Post-flop: OOP (SB) acts first each street
    - Actions: fold (if facing bet), call/check, raises from config, all-in
    - Terminal: fold, or showdown after river betting complete
    """

    def __init__(self, config: Config, preflop_only: bool = False, board: Optional[Dict[str, Tuple[str, ...]]] = None, terminal_street: str = "river"):
        """
        Initialize HUNL game.

        Args:
            config: Configuration with stack_depth and raise_sizes
            preflop_only: If True, use preflop-only terminal conditions
            board: Custom board config dict with 'flop', 'turn', 'river' keys.
                   If None, uses FIXED_BOARD.
            terminal_street: Street at which showdown occurs ("flop", "turn", or "river").
                            Only used when preflop_only=False.
        """
        self.config = config
        self.stack_depth = config.stack_depth
        self.raise_sizes = config.raise_sizes
        self.preflop_only = preflop_only
        self.board_config = board if board is not None else FIXED_BOARD
        self.terminal_street = terminal_street

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
        - Preflop-only mode: call/check completes betting
        - Full game: showdown only on river after betting complete
        """
        if not state.history:
            return False

        last_action = state.history[-1]

        # Fold is always terminal
        if last_action == "f":
            return True

        # Check if betting is complete on current street
        betting_complete = self._betting_complete(state)

        if self.preflop_only:
            # Preflop-only mode: terminal when betting is complete
            return betting_complete
        else:
            # Full game: terminal on terminal_street when betting is complete
            if state.street == self.terminal_street and betting_complete:
                return True
            return False

    def player(self, state: HUNLState) -> int:
        """Return player to act at state."""
        return state.to_act

    def _count_bets_this_round(self, state: HUNLState) -> int:
        """Count the number of bets/raises on the current street.

        For preflop, the BB counts as the first bet.
        """
        count = 0
        # Find actions on current street (after last street marker, or from start for preflop)
        for action in reversed(state.history):
            if action.startswith("/"):
                # Hit a street marker, stop counting
                break
            if action.startswith("r") or action == "a":
                count += 1

        # For preflop, the BB counts as a bet
        if state.street == "preflop":
            count += 1

        return count

    def actions(self, state: HUNLState) -> List[str]:
        """
        Available actions at state.

        Returns list of action strings:
        - 'f': fold (if facing a bet)
        - 'c': call/check
        - 'rX': raise to X BB
        - 'a': all-in

        Bet sizing rules:
        - Each subsequent bet must use at least the next raise size up
        - Final bet (at max_bets_per_round) must be all-in
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

        # Count bets this round to determine available raise sizes
        bets_this_round = self._count_bets_this_round(state)

        # Use postflop-specific settings for non-preflop streets
        if state.street == "preflop":
            max_bets = self.config.max_bets_per_round
            raise_sizes = self.config.raise_sizes
        else:
            max_bets = self.config.postflop_max_bets_per_round
            # Postflop: use pot-relative sizing if not configured
            if self.config.postflop_raise_sizes:
                raise_sizes = self.config.postflop_raise_sizes
            else:
                # Default: pot-size bet only (simplest tree)
                raise_sizes = [state.pot]

        # At max bets, no more raises allowed
        if bets_this_round >= max_bets:
            return actions

        # At max_bets - 1, only all-in is allowed as the final bet
        if bets_this_round == max_bets - 1:
            if state.stack > current_bet:
                actions.append("a")
            return actions

        # Get legal raise sizes - must use at least raise_sizes[bets_this_round]
        # (escalating minimum bet size)
        min_size_idx = min(bets_this_round, len(raise_sizes) - 1)
        min_raise_size = raise_sizes[min_size_idx]

        for size in raise_sizes[min_size_idx:]:
            if size > current_bet and size <= state.stack:
                actions.append(f"r{size:g}")

        # All-in option
        if state.stack > current_bet:
            actions.append("a")

        return actions

    def next_state(self, state: HUNLState, action: str) -> HUNLState:
        """
        Return state after taking action.

        Updates pot, history, to_act, committed, current_bet, and handles street transitions.
        """
        new_history = state.history + (action,)
        player = state.to_act
        cur_bet = state.current_bet

        # For pot/raise calculation, use street contribution for post-flop
        if state.street == "preflop":
            player_street_committed = state.committed[player]
        else:
            player_street_committed = state.street_contributions[player]

        street_contribs = list(state.street_contributions)
        new_committed = list(state.committed)
        new_current_bet = cur_bet  # Default: unchanged

        if action == "f":
            new_pot = state.pot
        elif action == "c":
            amount_to_add = cur_bet - player_street_committed
            new_pot = state.pot + amount_to_add
            new_committed[player] += amount_to_add
            street_contribs[player] += amount_to_add
        elif action == "a":
            amount_to_add = state.stack - state.committed[player]
            new_pot = state.pot + amount_to_add
            new_committed[player] = state.stack
            street_contribs[player] += amount_to_add
            new_current_bet = state.stack
        elif action.startswith("r"):
            raise_to = float(action[1:])
            amount_to_add = raise_to - player_street_committed
            new_pot = state.pot + amount_to_add
            new_committed[player] += amount_to_add
            street_contribs[player] += amount_to_add
            new_current_bet = raise_to
        else:
            raise ValueError(f"Unknown action: {action}")

        new_state = HUNLState(
            hands=state.hands,
            history=new_history,
            stack=state.stack,
            pot=new_pot,
            to_act=1 - player,
            street=state.street,
            board=state.board,
            street_contributions=tuple(street_contribs),
            committed=tuple(new_committed),
            current_bet=new_current_bet,
        )

        # Check for street transition (not in preflop-only mode, and not at terminal street)
        if not self.preflop_only and self._betting_complete(new_state) and new_state.street != self.terminal_street:
            new_state = self._advance_street(new_state)

        return new_state

    def _betting_complete(self, state: HUNLState) -> bool:
        """Check if betting is complete on the current street."""
        if not state.history:
            return False

        last_action = state.history[-1]

        # Fold ends betting
        if last_action == "f":
            return True

        # Call/check completes betting
        if last_action == "c":
            if state.street == "preflop":
                # Preflop: need at least 2 actions for betting to be complete
                # (limp-check or raise-call)
                return len(state.history) >= 2
            else:
                # Post-flop: betting completes when:
                # 1. Both players have acted (contributions are equal after call)
                # 2. Check-check (both contributions are 0)
                # Use street_contributions to determine if both players have acted
                sb_contrib = state.street_contributions[0]
                bb_contrib = state.street_contributions[1]

                # If contributions are equal, check if at least one non-zero
                # (meaning bet-call) or if there's been action
                if sb_contrib == bb_contrib:
                    # Either both 0 (check-check) or equal (bet-call)
                    # Need to verify both players have had a chance to act
                    # by checking if this is the second "c" for this street
                    return self._both_players_acted_postflop(state)

        return False

    def _both_players_acted_postflop(self, state: HUNLState) -> bool:
        """Check if both players have acted on current post-flop street."""
        # Count actions since the last street marker
        street_actions = self._count_actions_this_street(state)

        # Need at least 2 actions for betting to be complete
        return street_actions >= 2

    def _count_actions_this_street(self, state: HUNLState) -> int:
        """Count actions taken on the current street (after street marker)."""
        count = 0
        for action in reversed(state.history):
            if action.startswith("/"):
                # Hit street marker, stop counting
                break
            count += 1
        return count

    def _advance_street(self, state: HUNLState) -> HUNLState:
        """Advance to the next street, dealing new board cards."""
        next_streets = {"preflop": "flop", "flop": "turn", "turn": "river"}
        new_street = next_streets[state.street]

        # Get new board from board config
        new_board = self.board_config[new_street]

        # Add street marker to history (e.g., "/flop", "/turn", "/river")
        new_history = state.history + (f"/{new_street}",)

        return HUNLState(
            hands=state.hands,
            history=new_history,
            stack=state.stack,
            pot=state.pot,
            to_act=0,  # OOP (SB) acts first post-flop
            street=new_street,
            board=new_board,
            street_contributions=(0.0, 0.0),  # Reset for new street
            committed=state.committed,  # Preserve total committed
            current_bet=0.0,  # Reset for new street
        )

    def utility(self, state: HUNLState, player: int) -> float:
        """
        Utility for player at terminal state.

        Returns profit/loss in BB.
        """
        last_action = state.history[-1]
        sb_committed = state.committed[0]
        bb_committed = state.committed[1]

        if last_action == "f":
            # The player who folded is determined by action count
            # In preflop: actions alternate SB, BB, SB, BB...
            # Action index 0 = SB, index 1 = BB, etc.
            # So folder = (len(history) - 1) % 2
            folder = (len(state.history) - 1) % 2

            if player == folder:
                # Return the loss (negative of what they put in)
                return -sb_committed if folder == 0 else -bb_committed
            else:
                # Return the win (what the other player put in)
                return sb_committed if folder == 0 else bb_committed
        else:
            # Showdown
            sb_hand = state.hands[0]
            bb_hand = state.hands[1]
            pot = sb_committed + bb_committed

            if self.preflop_only or state.street == "preflop":
                # Preflop showdown - use equity
                sb_equity = get_preflop_equity(sb_hand, bb_hand)
                bb_equity = 1.0 - sb_equity

                if player == 0:
                    return sb_equity * pot - sb_committed
                else:
                    return bb_equity * pot - bb_committed
            else:
                # Post-flop showdown - use hand evaluation
                from core.hand_eval import evaluate_hand, compare_hands

                sb_value = evaluate_hand(sb_hand, state.board)
                bb_value = evaluate_hand(bb_hand, state.board)

                result = compare_hands(sb_value, bb_value)
                # result: 1 = SB wins, -1 = BB wins, 0 = tie

                if result == 0:
                    # Split pot
                    if player == 0:
                        return pot / 2 - sb_committed
                    else:
                        return pot / 2 - bb_committed
                elif result == 1:
                    # SB wins
                    if player == 0:
                        return pot - sb_committed
                    else:
                        return -bb_committed
                else:
                    # BB wins
                    if player == 0:
                        return -sb_committed
                    else:
                        return pot - bb_committed

    def info_set_key(self, state: HUNLState) -> str:
        """
        Map state to information set identifier.

        Format: "POSITION:HAND:BOARD:HISTORY"
        e.g., "SB:AA::", "BB:KK::r3", "SB:AKs:AhTh6c:r3-c"
        """
        player = state.to_act
        position = "SB" if player == 0 else "BB"
        hand_idx = state.hands[player]
        hand_str = hand_to_string(hand_idx)
        board_str = "".join(state.board) if state.board else ""
        history_str = "-".join(state.history)

        return f"{position}:{hand_str}:{board_str}:{history_str}"

    def num_players(self) -> int:
        """Number of players in the game."""
        return 2

    def _current_bet(self, state: HUNLState) -> float:
        """
        Get the current bet amount that must be matched on this street.

        For preflop: Based on blinds and raises
        For post-flop: Based on street_contributions (resets each street)
        """
        if state.street == "preflop":
            return self._preflop_current_bet(state)

        # Post-flop: current bet is max of street contributions
        return max(state.street_contributions)

    def _preflop_current_bet(self, state: HUNLState) -> float:
        """Get current bet for preflop street."""
        return state.current_bet

        # No raise found - either limps or initial
        # If SB limped, current bet is 1BB (the BB)
        if "c" in state.history and not any(a.startswith("r") or a == "a" for a in state.history):
            return 1.0

        return 1.0

    def _player_committed(self, state: HUNLState) -> float:
        """
        Get amount the current player has committed on this street.
        """
        player = state.to_act

        if state.street == "preflop":
            return self._preflop_player_committed(state, player)

        # Post-flop: use street_contributions
        return state.street_contributions[player]

    def _preflop_player_committed(self, state: HUNLState, player: int) -> float:
        """Get amount player has committed during preflop."""
        return state.committed[player]

        return committed

    def _total_committed(self, state: HUNLState, player: int) -> float:
        """
        Calculate total amount a player has committed to the pot across all streets.

        Args:
            state: Current game state
            player: 0 for SB, 1 for BB
        """
        # For full game, we need to track total across streets
        # The pot tracks total contributions, so we can derive from pot and equity
        # But for utility calculation, we need the actual amount each player put in

        # Simpler approach: pot / 2 when contributions are equal,
        # otherwise track through the game state
        # For now, use the preflop logic for total committed

        # Initial blinds
        if player == 0:
            committed = 0.5  # SB
        else:
            committed = 1.0  # BB

        # For post-flop, we track via pot changes
        # The total committed is: initial_blind + all raises/calls above that
        # This is complex to track through history, so we use a different approach:
        # pot = sb_committed + bb_committed, and we know contributions are balanced after calls

        # For fold case: we need exact amounts
        # Rebuild committed amount by tracking all actions
        return self._calculate_total_committed(state, player)

    def _calculate_total_committed(self, state: HUNLState, player: int) -> float:
        """Calculate total amount committed by player across all streets."""
        committed = 0.5 if player == 0 else 1.0
        current_player = 0
        current_bet = 1.0  # Track current bet to avoid repeated scans

        for i, action in enumerate(state.history):
            if action == "f":
                break
            if action.startswith("r"):
                current_bet = float(action[1:])
            elif action == "a":
                current_bet = state.stack

            if current_player == player:
                if action == "c":
                    committed = current_bet
                elif action == "a":
                    committed = state.stack
                elif action.startswith("r"):
                    committed = float(action[1:])

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

    def _bet_at_action_idx(self, state: HUNLState, action_idx: int) -> float:
        """Get the bet amount at a specific action index."""
        for prev_action in reversed(state.history[:action_idx]):
            if prev_action.startswith("r"):
                return float(prev_action[1:])
            elif prev_action == "a":
                return state.stack
        return 1.0
