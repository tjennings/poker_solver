from collections import defaultdict
from typing import Dict, List, Tuple

from games.base import Game


class VanillaCFR:
    """
    Vanilla Counterfactual Regret Minimization.

    Implements the algorithm from Zinkevich et al. "Regret Minimization
    in Games with Incomplete Information" (2007).
    """

    def __init__(self, game: Game):
        self.game = game
        # R^T(I, a): cumulative counterfactual regret
        self.regret_sum: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # Cumulative strategy weighted by reach probability
        self.strategy_sum: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def get_strategy(self, info_set: str, actions: List[str]) -> Dict[str, float]:
        """
        Equation (8): Compute strategy proportional to positive regret.

        σ(I)(a) = R+(I,a) / Σ_a' R+(I,a')  if denominator > 0
                = 1/|A(I)|                   otherwise
        """
        regrets = self.regret_sum[info_set]
        positive_regrets = {a: max(regrets[a], 0) for a in actions}
        total = sum(positive_regrets.values())

        if total > 0:
            return {a: positive_regrets[a] / total for a in actions}
        else:
            uniform = 1.0 / len(actions)
            return {a: uniform for a in actions}

    def cfr(
        self,
        state,
        reach_probs: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        Recursive CFR traversal.

        Args:
            state: Current game state
            reach_probs: (π_0(h), π_1(h)) reach probabilities

        Returns:
            (u_0, u_1): Expected utilities for both players
        """
        # Terminal: return payoffs
        if self.game.is_terminal(state):
            return (
                self.game.utility(state, 0),
                self.game.utility(state, 1),
            )

        player = self.game.player(state)
        opponent = 1 - player
        info_set = self.game.info_set_key(state)
        actions = self.game.actions(state)
        strategy = self.get_strategy(info_set, actions)

        # Traverse each action
        action_utils: Dict[str, Tuple[float, float]] = {}
        node_util = [0.0, 0.0]

        for action in actions:
            next_state = self.game.next_state(state, action)

            # Update reach probability for acting player
            new_reach = list(reach_probs)
            new_reach[player] *= strategy[action]

            action_utils[action] = self.cfr(next_state, tuple(new_reach))

            # Expected utility weighted by strategy
            for p in [0, 1]:
                node_util[p] += strategy[action] * action_utils[action][p]

        # Counterfactual reach: opponent's contribution
        cf_reach = reach_probs[opponent]

        # Update regrets: Equation (7)
        for action in actions:
            regret = action_utils[action][player] - node_util[player]
            self.regret_sum[info_set][action] += cf_reach * regret

        # Accumulate strategy weighted by player's reach
        my_reach = reach_probs[player]
        for action in actions:
            self.strategy_sum[info_set][action] += my_reach * strategy[action]

        return tuple(node_util)

    def train(self, iterations: int) -> None:
        """
        Run CFR for specified iterations.

        Each iteration traverses all possible card dealings.
        """
        for _ in range(iterations):
            for initial_state in self.game.initial_states():
                self.cfr(initial_state, reach_probs=(1.0, 1.0))

    def get_average_strategy(self, info_set: str) -> Dict[str, float]:
        """
        Equation (4): Compute average strategy.

        σ̄(I)(a) = Σ_t π_i(I) σ^t(I)(a) / Σ_t π_i(I)
        """
        strat = self.strategy_sum[info_set]
        total = sum(strat.values())

        if total > 0:
            return {a: v / total for a, v in strat.items()}

        # Fallback: uniform over game actions
        actions = self.game.actions(self.game.initial_states()[0])
        return {a: 1.0 / len(actions) for a in actions}

    def get_all_average_strategies(self) -> Dict[str, Dict[str, float]]:
        """Get average strategy for all visited information sets."""
        return {
            info_set: self.get_average_strategy(info_set)
            for info_set in self.strategy_sum.keys()
        }
