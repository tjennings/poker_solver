"""Postflop subgame for on-demand solving from a specific state."""

from dataclasses import dataclass
from typing import List, Tuple, Dict
from games.base import Game
from games.hunl_preflop import FIXED_BOARD
from core.hands import hand_to_string
from config.loader import Config


@dataclass(frozen=True)
class PostflopState:
    """State for postflop subgame."""
    hands: Tuple[int, int]  # (SB hand index, BB hand index)
    history: Tuple[str, ...]  # Actions since start of subgame
    pot: float  # Current pot
    stacks: Tuple[float, float]  # Remaining stacks (SB, BB)
    street: str  # "flop", "turn", or "river"
    board: Tuple[str, ...]  # Board cards
    to_act: int  # 0=SB, 1=BB (postflop SB acts first)
    street_bets: Tuple[float, float]  # Bets on current street


class PostflopSubgame(Game):
    """
    Postflop subgame starting from a specific state.

    This represents just the flop→turn→river portion of the game,
    starting from a known pot size and stack depth after preflop.
    """

    def __init__(
        self,
        config: Config,
        starting_pot: float,
        starting_stacks: Tuple[float, float],
        starting_street: str = "flop",
        board: Dict[str, Tuple[str, ...]] = None,
    ):
        """
        Initialize postflop subgame.

        Args:
            config: Game configuration (raise sizes, max bets)
            starting_pot: Pot size at start of postflop
            starting_stacks: (SB stack, BB stack) remaining after preflop
            starting_street: Which street to start from (default "flop")
            board: Custom board dict with keys 'flop', 'turn', 'river' (default: FIXED_BOARD)
        """
        self.config = config
        self.starting_pot = starting_pot
        self.starting_stacks = starting_stacks
        self.starting_street = starting_street
        self.board_config = board if board else FIXED_BOARD
        self.starting_board = self.board_config[starting_street]

    def initial_states(self) -> List[PostflopState]:
        """Generate all initial states (one per hand combination)."""
        from core.hands import HAND_COUNT
        states = []
        for sb_hand in range(HAND_COUNT):
            for bb_hand in range(HAND_COUNT):
                if sb_hand != bb_hand:
                    states.append(PostflopState(
                        hands=(sb_hand, bb_hand),
                        history=(),
                        pot=self.starting_pot,
                        stacks=self.starting_stacks,
                        street=self.starting_street,
                        board=self.starting_board,
                        to_act=0,  # SB acts first postflop
                        street_bets=(0.0, 0.0),
                    ))
        return states

    def is_terminal(self, state: PostflopState) -> bool:
        """Check if state is terminal."""
        if not state.history:
            return False

        last_action = state.history[-1]

        # Fold is terminal
        if last_action == "f":
            return True

        # Check for showdown on river
        if state.street == "river" and self._betting_complete(state):
            return True

        return False

    def _betting_complete(self, state: PostflopState) -> bool:
        """Check if betting is complete on current street."""
        if not state.history:
            return False

        # Find actions on current street
        street_actions = self._get_street_actions(state)
        if len(street_actions) < 2:
            return False

        last = street_actions[-1]
        # Complete if last action was call/check and both players acted
        return last == "c" and len(street_actions) >= 2

    def _get_street_actions(self, state: PostflopState) -> List[str]:
        """Get actions taken on current street."""
        actions = []
        for action in reversed(state.history):
            if action.startswith("/"):
                break
            actions.append(action)
        return list(reversed(actions))

    def _count_bets_this_street(self, state: PostflopState) -> int:
        """Count bets/raises on current street."""
        count = 0
        for action in self._get_street_actions(state):
            if action.startswith("r") or action == "a":
                count += 1
        return count

    def player(self, state: PostflopState) -> int:
        """Return player to act."""
        return state.to_act

    def actions(self, state: PostflopState) -> List[str]:
        """Get available actions at state."""
        actions = []

        player = state.to_act
        my_street_bet = state.street_bets[player]
        opp_street_bet = state.street_bets[1 - player]
        my_stack = state.stacks[player]

        current_bet = max(state.street_bets)
        facing_bet = opp_street_bet > my_street_bet

        # Fold only if facing a bet
        if facing_bet:
            actions.append("f")

        # Always can check/call
        actions.append("c")

        # Can't raise if all-in or facing all-in
        if my_stack <= current_bet - my_street_bet:
            return actions
        if opp_street_bet >= state.stacks[1 - player]:
            return actions

        # Check bet limits
        bets_this_street = self._count_bets_this_street(state)
        max_bets = self.config.max_bets_per_round

        if bets_this_street >= max_bets:
            return actions

        # Final bet must be all-in
        if bets_this_street == max_bets - 1:
            actions.append("a")
            return actions

        # Escalating bet sizes
        raise_sizes = self.config.raise_sizes
        min_idx = min(bets_this_street, len(raise_sizes) - 1)

        for size in raise_sizes[min_idx:]:
            if size > current_bet and size <= my_stack + my_street_bet:
                actions.append(f"r{size:g}")

        # All-in
        if my_stack + my_street_bet > current_bet:
            actions.append("a")

        return actions

    def next_state(self, state: PostflopState, action: str) -> PostflopState:
        """Return state after taking action."""
        player = state.to_act
        my_street_bet = state.street_bets[player]
        opp_street_bet = state.street_bets[1 - player]

        new_history = state.history + (action,)
        new_stacks = list(state.stacks)
        new_street_bets = list(state.street_bets)
        new_pot = state.pot

        if action == "f":
            pass  # No change to pot/stacks
        elif action == "c":
            # Call/check
            call_amount = opp_street_bet - my_street_bet
            new_stacks[player] -= call_amount
            new_street_bets[player] = opp_street_bet
            new_pot += call_amount
        elif action == "a":
            # All-in
            allin_amount = new_stacks[player]
            new_street_bets[player] = my_street_bet + allin_amount
            new_pot += allin_amount
            new_stacks[player] = 0
        elif action.startswith("r"):
            raise_to = float(action[1:])
            bet_amount = raise_to - my_street_bet
            new_stacks[player] -= bet_amount
            new_street_bets[player] = raise_to
            new_pot += bet_amount

        new_state = PostflopState(
            hands=state.hands,
            history=new_history,
            pot=new_pot,
            stacks=tuple(new_stacks),
            street=state.street,
            board=state.board,
            to_act=1 - player,
            street_bets=tuple(new_street_bets),
        )

        # Check for street transition
        if self._betting_complete(new_state) and new_state.street != "river":
            new_state = self._advance_street(new_state)

        return new_state

    def _advance_street(self, state: PostflopState) -> PostflopState:
        """Advance to next street."""
        next_streets = {"flop": "turn", "turn": "river"}
        new_street = next_streets[state.street]
        new_board = self.board_config[new_street]

        return PostflopState(
            hands=state.hands,
            history=state.history + (f"/{new_street}",),
            pot=state.pot,
            stacks=state.stacks,
            street=new_street,
            board=new_board,
            to_act=0,  # SB acts first each street
            street_bets=(0.0, 0.0),
        )

    def utility(self, state: PostflopState, player: int) -> float:
        """Compute utility for player at terminal state."""
        from core.hand_eval import evaluate_hand

        last_action = state.history[-1] if state.history else "c"

        if last_action == "f":
            # Folder loses what they put in
            folder = 1 - state.to_act  # Player who just folded
            if folder == player:
                # I folded - I lose my contribution
                pot_contribution = self.starting_stacks[player] - state.stacks[player]
                return -pot_contribution
            else:
                # Opponent folded - I win the pot minus my contribution
                pot_contribution = self.starting_stacks[player] - state.stacks[player]
                return state.pot - pot_contribution

        # Showdown
        sb_hand, bb_hand = state.hands
        sb_value = evaluate_hand(sb_hand, state.board)
        bb_value = evaluate_hand(bb_hand, state.board)

        sb_contribution = self.starting_stacks[0] - state.stacks[0]
        bb_contribution = self.starting_stacks[1] - state.stacks[1]

        if sb_value > bb_value:
            # SB wins
            if player == 0:
                return state.pot - sb_contribution
            else:
                return -bb_contribution
        elif bb_value > sb_value:
            # BB wins
            if player == 0:
                return -sb_contribution
            else:
                return state.pot - bb_contribution
        else:
            # Tie - split pot
            if player == 0:
                return state.pot / 2 - sb_contribution
            else:
                return state.pot / 2 - bb_contribution

    def info_set_key(self, state: PostflopState) -> str:
        """Generate info set key."""
        player = state.to_act
        position = "SB" if player == 0 else "BB"
        hand_str = hand_to_string(state.hands[player])
        board_str = "".join(state.board)
        history_str = "-".join(state.history)

        return f"{position}:{hand_str}:{board_str}:{history_str}"

    def num_players(self) -> int:
        return 2
