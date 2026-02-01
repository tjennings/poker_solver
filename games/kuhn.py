from dataclasses import dataclass
from typing import Tuple, List
from itertools import permutations

from games.base import Game


@dataclass(frozen=True)
class KuhnState:
    """
    State in Kuhn Poker.

    Attributes:
        cards: (player_0_card, player_1_card) where cards are 0=J, 1=Q, 2=K
        history: Tuple of actions taken, e.g., ("p", "b", "b")
    """
    cards: Tuple[int, int]
    history: Tuple[str, ...]


class KuhnPoker(Game):
    """
    Kuhn Poker: Simplified 3-card poker game.

    Rules:
    - 3 cards: Jack (0), Queen (1), King (2)
    - 2 players, each dealt 1 card, 1 chip ante
    - Actions: Pass (p) = check/fold, Bet (b) = bet/call
    - Higher card wins at showdown
    """

    PASS = "p"
    BET = "b"

    # Terminal histories
    TERMINALS = {
        ("p", "p"),      # check-check
        ("b", "p"),      # bet-fold
        ("b", "b"),      # bet-call
        ("p", "b", "p"), # check-bet-fold
        ("p", "b", "b"), # check-bet-call
    }

    def initial_states(self) -> List[KuhnState]:
        """All 6 possible card dealings."""
        return [
            KuhnState(cards=cards, history=())
            for cards in permutations([0, 1, 2], 2)
        ]

    def is_terminal(self, state: KuhnState) -> bool:
        """Check if game has ended."""
        return state.history in self.TERMINALS

    def player(self, state: KuhnState) -> int:
        """Player alternates: 0, 1, 0, ..."""
        return len(state.history) % 2

    def actions(self, state: KuhnState) -> List[str]:
        """Always pass or bet."""
        return [self.PASS, self.BET]

    def next_state(self, state: KuhnState, action: str) -> KuhnState:
        """Append action to history."""
        return KuhnState(
            cards=state.cards,
            history=state.history + (action,)
        )

    def utility(self, state: KuhnState, player: int) -> float:
        """
        Utility for player at terminal state.

        Payoffs:
        - pp (check-check): winner gets 1
        - bp (bet-fold): P0 gets 1
        - bb (bet-call): winner gets 2
        - pbp (check-bet-fold): P1 gets 1
        - pbb (check-bet-call): winner gets 2
        """
        if not self.is_terminal(state):
            raise ValueError(f"Cannot get utility of non-terminal state: {state}")

        h = state.history
        winner = 0 if state.cards[0] > state.cards[1] else 1

        if h == ("p", "p"):
            # Check-check: winner gets 1
            payoff = 1.0
            return payoff if player == winner else -payoff
        elif h == ("b", "p"):
            # Bet-fold: P0 wins 1
            return 1.0 if player == 0 else -1.0
        elif h == ("b", "b"):
            # Bet-call: winner gets 2
            payoff = 2.0
            return payoff if player == winner else -payoff
        elif h == ("p", "b", "p"):
            # Check-bet-fold: P1 wins 1
            return 1.0 if player == 1 else -1.0
        elif h == ("p", "b", "b"):
            # Check-bet-call: winner gets 2
            payoff = 2.0
            return payoff if player == winner else -payoff

        raise ValueError(f"Unknown terminal state: {state}")

    def info_set_key(self, state: KuhnState) -> str:
        """
        Map state to information set.

        Player sees only their own card and the action history.
        Format: "card:history" e.g., "1:pb" = Queen, check-bet
        """
        player = self.player(state)
        card = state.cards[player]
        history = "".join(state.history)
        return f"{card}:{history}"

    def num_players(self) -> int:
        return 2
