# CFR Poker Solver Design

## Overview

A PyTorch-based Counterfactual Regret Minimization (CFR) solver for extensive-form games with incomplete information. Based on the paper "Regret Minimization in Games with Incomplete Information" by Zinkevich et al.

**Goals:**
1. Educational/readable implementation that closely follows the paper
2. GPU acceleration via PyTorch (CUDA and Apple Silicon MPS)
3. Foundation for scaling to larger games (Leduc, Hold'em)

**Starting point:** Kuhn Poker (12 information sets, known Nash equilibrium for verification)

---

## Project Structure

```
poker_solver/
├── games/
│   ├── base.py          # Abstract game interface
│   ├── kuhn.py          # Kuhn Poker implementation
│   └── leduc.py         # (future) Leduc Poker
├── cfr/
│   ├── vanilla.py       # Vanilla CFR algorithm
│   ├── plus.py          # (future) CFR+ variant
│   └── mccfr.py         # (future) Monte Carlo CFR
├── core/
│   ├── tree.py          # Explicit tree representation
│   ├── tensors.py       # Tensor compilation & GPU ops
│   └── strategy.py      # Regret/strategy storage
├── tests/
│   ├── test_kuhn.py     # Verify against known equilibrium
│   └── test_equivalence.py  # CPU/GPU equivalence
├── docs/
│   └── plans/
└── main.py              # Entry point & benchmarking
```

---

## Core Abstractions

### Game Interface

Mirrors Definition 1 from the paper:

```python
from abc import ABC, abstractmethod
from typing import List, Tuple, Any
from dataclasses import dataclass

State = Any  # Game-specific state type
Action = str

class Game(ABC):
    """Defines extensive game rules."""

    @abstractmethod
    def initial_states(self) -> List[State]:
        """All possible starting states (after chance)."""
        pass

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """Is this a terminal history? (z ∈ Z)"""
        pass

    @abstractmethod
    def player(self, state: State) -> int:
        """Player to act at this state. P(h) in paper."""
        pass

    @abstractmethod
    def actions(self, state: State) -> List[Action]:
        """Available actions. A(h) in paper."""
        pass

    @abstractmethod
    def next_state(self, state: State, action: Action) -> State:
        """State after taking action."""
        pass

    @abstractmethod
    def utility(self, state: State, player: int) -> float:
        """Utility for player at terminal state. u_i(z) in paper."""
        pass

    @abstractmethod
    def info_set_key(self, state: State) -> str:
        """Maps state to information set. h → I_i in paper."""
        pass

    @abstractmethod
    def num_players(self) -> int:
        """Number of players (2 for poker)."""
        pass
```

### Information Set Storage

Stores cumulative regrets and strategy weights per the paper's equations:

```python
from collections import defaultdict
from typing import Dict
import torch

class InfoSetStore:
    """Holds R^T(I,a) and cumulative strategy weights."""

    def __init__(self):
        # Σ_t π^σ_{-i}(I) * [u(σ|I→a) - u(σ,I)]  (Equation 7)
        self.regret_sum: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # Σ_t π^σ_i(I) * σ^t(I)(a)  (for Equation 4)
        self.strategy_sum: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
```

---

## Kuhn Poker Implementation

### Game Rules

- 3 cards: Jack (0), Queen (1), King (2)
- 2 players, each dealt 1 card, 1 chip ante
- Actions: Pass (check/fold) or Bet (bet/call)
- Higher card wins at showdown

### State Representation

```python
from dataclasses import dataclass
from typing import Tuple
from itertools import permutations

@dataclass(frozen=True)
class KuhnState:
    cards: Tuple[int, int]      # (player_0_card, player_1_card)
    history: Tuple[str, ...]    # Action history, e.g., ("p", "b", "b")

class KuhnPoker(Game):
    PASS = "p"  # check or fold
    BET = "b"   # bet or call

    def initial_states(self) -> List[KuhnState]:
        """All 6 possible deals."""
        return [
            KuhnState(cards=cards, history=())
            for cards in permutations([0, 1, 2], 2)
        ]

    def is_terminal(self, state: KuhnState) -> bool:
        h = state.history
        if len(h) < 2:
            return False
        # Terminal conditions:
        # - "pp" (check-check, showdown)
        # - "bp" (bet-fold)
        # - "bb" (bet-call, showdown)
        # - "pbp" (check-bet-fold)
        # - "pbb" (check-bet-call, showdown)
        return (
            h == ("p", "p") or
            h == ("b", "p") or
            h == ("b", "b") or
            h == ("p", "b", "p") or
            h == ("p", "b", "b")
        )

    def player(self, state: KuhnState) -> int:
        return len(state.history) % 2

    def actions(self, state: KuhnState) -> List[str]:
        return [self.PASS, self.BET]

    def next_state(self, state: KuhnState, action: str) -> KuhnState:
        return KuhnState(
            cards=state.cards,
            history=state.history + (action,)
        )

    def utility(self, state: KuhnState, player: int) -> float:
        h = state.history
        winner = 0 if state.cards[0] > state.cards[1] else 1

        if h == ("p", "p"):
            # Check-check: winner gets 1 chip
            return 1.0 if player == winner else -1.0
        elif h == ("b", "p"):
            # Bet-fold: player 0 wins 1 chip
            return 1.0 if player == 0 else -1.0
        elif h == ("b", "b"):
            # Bet-call: winner gets 2 chips
            return 2.0 if player == winner else -2.0
        elif h == ("p", "b", "p"):
            # Check-bet-fold: player 1 wins 1 chip
            return 1.0 if player == 1 else -1.0
        elif h == ("p", "b", "b"):
            # Check-bet-call: winner gets 2 chips
            return 2.0 if player == winner else -2.0

        raise ValueError(f"Non-terminal state: {state}")

    def info_set_key(self, state: KuhnState) -> str:
        """Player sees own card + action history."""
        player = self.player(state)
        card = state.cards[player]
        return f"{card}:{''.join(state.history)}"

    def num_players(self) -> int:
        return 2
```

### Information Sets

12 total (6 per player):

| Player 0 | Player 1 |
|----------|----------|
| `0:` (J, root) | `0:p` (J, after check) |
| `1:` (Q, root) | `1:p` (Q, after check) |
| `2:` (K, root) | `2:p` (K, after check) |
| `0:pb` (J, facing bet) | `0:b` (J, facing bet) |
| `1:pb` (Q, facing bet) | `1:b` (Q, facing bet) |
| `2:pb` (K, facing bet) | `2:b` (K, facing bet) |

---

## Vanilla CFR Algorithm

Direct implementation of the paper's equations (7) and (8):

```python
from typing import Dict, List, Tuple
from collections import defaultdict

class VanillaCFR:
    """
    Vanilla CFR as described in Section 3 of the paper.

    Key equations:
    - Equation (7): Cumulative counterfactual regret
    - Equation (8): Strategy from regret matching
    - Equation (4): Average strategy computation
    """

    def __init__(self, game: Game):
        self.game = game
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))

    def get_strategy(self, info_set: str, actions: List[str]) -> Dict[str, float]:
        """
        Equation (8): Strategy proportional to positive regret.

        σ^{T+1}(I)(a) = R^{T,+}(I,a) / Σ_a R^{T,+}(I,a)  if denominator > 0
                      = 1/|A(I)|                          otherwise
        """
        regrets = self.regret_sum[info_set]
        positive_sum = sum(max(regrets[a], 0) for a in actions)

        if positive_sum > 0:
            return {a: max(regrets[a], 0) / positive_sum for a in actions}
        else:
            return {a: 1.0 / len(actions) for a in actions}

    def cfr(
        self,
        state,
        reach_probs: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Recursive CFR traversal.

        Args:
            state: Current game state
            reach_probs: (π_0(h), π_1(h)) - each player's contribution to reach prob

        Returns:
            (u_0, u_1) - expected utilities for both players from this state
        """
        # Terminal: return payoffs
        if self.game.is_terminal(state):
            return (
                self.game.utility(state, 0),
                self.game.utility(state, 1)
            )

        player = self.game.player(state)
        opponent = 1 - player
        info_set = self.game.info_set_key(state)
        actions = self.game.actions(state)
        strategy = self.get_strategy(info_set, actions)

        # Compute counterfactual values for each action
        action_utils = {}
        node_util = [0.0, 0.0]

        for action in actions:
            next_state = self.game.next_state(state, action)

            # Update reach probability for acting player
            new_reach = list(reach_probs)
            new_reach[player] *= strategy[action]

            action_utils[action] = self.cfr(next_state, tuple(new_reach))

            # Expected utility is weighted by strategy
            for p in [0, 1]:
                node_util[p] += strategy[action] * action_utils[action][p]

        # Counterfactual reach probability: π^σ_{-i}(I)
        cf_reach = reach_probs[opponent]

        # Update regrets: Equation (7)
        # R^T(I,a) += π_{-i}(I) * [u(σ|I→a, I) - u(σ, I)]
        for action in actions:
            regret = action_utils[action][player] - node_util[player]
            self.regret_sum[info_set][action] += cf_reach * regret

        # Accumulate strategy for averaging: Equation (4)
        # Weight by player's reach probability π_i(I)
        my_reach = reach_probs[player]
        for action in actions:
            self.strategy_sum[info_set][action] += my_reach * strategy[action]

        return tuple(node_util)

    def train(self, iterations: int):
        """Run CFR for T iterations over all chance outcomes."""
        for _ in range(iterations):
            for initial_state in self.game.initial_states():
                self.cfr(initial_state, reach_probs=(1.0, 1.0))

    def get_average_strategy(self, info_set: str) -> Dict[str, float]:
        """
        Equation (4): Average strategy weighted by reach probability.

        σ̄^T(I)(a) = Σ_t π^σ_i(I) σ^t(I)(a) / Σ_t π^σ_i(I)
        """
        strat = self.strategy_sum[info_set]
        total = sum(strat.values())
        if total > 0:
            return {a: v / total for a, v in strat.items()}
        return {a: 1.0 / len(strat) for a in strat}

    def get_all_average_strategies(self) -> Dict[str, Dict[str, float]]:
        """Get average strategy for all information sets."""
        return {
            info_set: self.get_average_strategy(info_set)
            for info_set in self.strategy_sum.keys()
        }
```

---

## GPU Acceleration

### Device Selection

Support CUDA, Apple Silicon (MPS), and CPU:

```python
import torch

def get_device(preference: str = "auto") -> torch.device:
    """Select compute device."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(preference)
```

### Game Compilation

Convert explicit tree to tensor representation for GPU execution:

```python
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
from torch import Tensor

@dataclass
class CompiledGame:
    """Game tree encoded as tensors for GPU execution."""

    # Node topology
    node_player: Tensor       # [num_nodes] - player to act (-1 for terminal)
    node_info_set: Tensor     # [num_nodes] - info set index
    node_num_actions: Tensor  # [num_nodes] - valid action count

    # Transitions
    action_child: Tensor      # [num_nodes, max_actions] - child node indices
    action_mask: Tensor       # [num_nodes, max_actions] - valid action mask

    # Terminal payoffs
    terminal_mask: Tensor     # [num_nodes] - is terminal
    terminal_utils: Tensor    # [num_nodes, num_players] - utilities (0 for non-terminal)

    # Structure for vectorized traversal
    depth_slices: List[Tuple[int, int]]  # (start, end) per depth level
    max_depth: int

    # Mappings
    info_set_to_idx: Dict[str, int]
    idx_to_info_set: Dict[int, str]
    num_info_sets: int
    num_nodes: int
    max_actions: int
    num_players: int

    device: torch.device


def compile_game(game: Game, device: torch.device) -> CompiledGame:
    """
    Traverse game tree once, assign indices, build tensors.

    This creates a flattened representation where tree traversal
    becomes tensor indexing operations.
    """
    nodes = []
    info_set_to_idx = {}
    depth_buckets = defaultdict(list)

    def add_info_set(key: str) -> int:
        if key not in info_set_to_idx:
            info_set_to_idx[key] = len(info_set_to_idx)
        return info_set_to_idx[key]

    def enumerate_tree(state, depth: int) -> int:
        node_idx = len(nodes)
        depth_buckets[depth].append(node_idx)

        if game.is_terminal(state):
            nodes.append({
                "terminal": True,
                "utils": [game.utility(state, p) for p in range(game.num_players())],
                "player": -1,
                "info_set": -1,
                "children": [],
            })
            return node_idx

        info_set = game.info_set_key(state)
        info_idx = add_info_set(info_set)
        actions = game.actions(state)

        node = {
            "terminal": False,
            "utils": [0.0] * game.num_players(),
            "player": game.player(state),
            "info_set": info_idx,
            "children": [],
        }
        nodes.append(node)

        for action in actions:
            child_state = game.next_state(state, action)
            child_idx = enumerate_tree(child_state, depth + 1)
            node["children"].append(child_idx)

        return node_idx

    # Build tree for all initial states
    for initial_state in game.initial_states():
        enumerate_tree(initial_state, depth=0)

    # Convert to tensors
    num_nodes = len(nodes)
    max_actions = max(len(n["children"]) for n in nodes if not n["terminal"])
    num_players = game.num_players()

    node_player = torch.zeros(num_nodes, dtype=torch.long, device=device)
    node_info_set = torch.zeros(num_nodes, dtype=torch.long, device=device)
    node_num_actions = torch.zeros(num_nodes, dtype=torch.long, device=device)
    action_child = torch.zeros(num_nodes, max_actions, dtype=torch.long, device=device)
    action_mask = torch.zeros(num_nodes, max_actions, dtype=torch.bool, device=device)
    terminal_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    terminal_utils = torch.zeros(num_nodes, num_players, device=device)

    for idx, node in enumerate(nodes):
        node_player[idx] = node["player"]
        node_info_set[idx] = node["info_set"]
        terminal_mask[idx] = node["terminal"]
        terminal_utils[idx] = torch.tensor(node["utils"])

        if not node["terminal"]:
            num_actions = len(node["children"])
            node_num_actions[idx] = num_actions
            for a_idx, child_idx in enumerate(node["children"]):
                action_child[idx, a_idx] = child_idx
                action_mask[idx, a_idx] = True

    # Build depth slices for vectorized traversal
    max_depth = max(depth_buckets.keys()) + 1
    depth_slices = []
    current = 0
    for d in range(max_depth):
        count = len(depth_buckets[d])
        depth_slices.append((current, current + count))
        current += count

    return CompiledGame(
        node_player=node_player,
        node_info_set=node_info_set,
        node_num_actions=node_num_actions,
        action_child=action_child,
        action_mask=action_mask,
        terminal_mask=terminal_mask,
        terminal_utils=terminal_utils,
        depth_slices=depth_slices,
        max_depth=max_depth,
        info_set_to_idx=info_set_to_idx,
        idx_to_info_set={v: k for k, v in info_set_to_idx.items()},
        num_info_sets=len(info_set_to_idx),
        num_nodes=num_nodes,
        max_actions=max_actions,
        num_players=num_players,
        device=device,
    )
```

### Batched CFR

Run N independent iterations in parallel:

```python
class BatchedCFR:
    """
    GPU-accelerated CFR using batched parallel traversals.

    Strategy: Run batch_size independent CFR iterations simultaneously.
    Each uses the same current strategy but accumulates to shared regret sums.
    """

    def __init__(
        self,
        compiled: CompiledGame,
        batch_size: int = 1024,
    ):
        self.compiled = compiled
        self.batch_size = batch_size
        self.device = compiled.device

        # Regrets and strategy sums: [num_info_sets, max_actions]
        self.regret_sum = torch.zeros(
            compiled.num_info_sets,
            compiled.max_actions,
            device=self.device
        )
        self.strategy_sum = torch.zeros_like(self.regret_sum)
        self.iterations = 0

    def get_current_strategy(self) -> Tensor:
        """
        Equation (8) vectorized across all info sets.
        Returns: [num_info_sets, max_actions]
        """
        positive_regrets = torch.clamp(self.regret_sum, min=0)
        totals = positive_regrets.sum(dim=1, keepdim=True)

        uniform = torch.ones_like(positive_regrets) / self.compiled.max_actions
        strategy = torch.where(
            totals > 0,
            positive_regrets / totals.clamp(min=1e-10),
            uniform
        )
        return strategy

    def forward_reach(self, strategy: Tensor) -> Tensor:
        """
        Top-down pass: compute reach probabilities.

        Returns: [batch, num_nodes, num_players]
        """
        reach = torch.ones(
            self.batch_size,
            self.compiled.num_nodes,
            self.compiled.num_players,
            device=self.device
        )

        for depth in range(self.compiled.max_depth):
            start, end = self.compiled.depth_slices[depth]
            if start >= end:
                continue

            nodes = torch.arange(start, end, device=self.device)
            non_terminal = ~self.compiled.terminal_mask[nodes]

            if not non_terminal.any():
                continue

            active_nodes = nodes[non_terminal]
            players = self.compiled.node_player[active_nodes]
            info_sets = self.compiled.node_info_set[active_nodes]
            children = self.compiled.action_child[active_nodes]
            mask = self.compiled.action_mask[active_nodes]

            node_strats = strategy[info_sets]  # [active, max_actions]
            parent_reach = reach[:, active_nodes, :]  # [batch, active, players]

            for a in range(self.compiled.max_actions):
                valid = mask[:, a]
                if not valid.any():
                    continue

                valid_nodes = active_nodes[valid]
                valid_children = children[valid, a]
                valid_players = players[valid]
                probs = node_strats[valid, a]  # [valid_count]

                # Child inherits parent reach
                reach[:, valid_children, :] = reach[:, valid_nodes, :]

                # Multiply by action prob for acting player
                for p in range(self.compiled.num_players):
                    player_mask = (valid_players == p)
                    if player_mask.any():
                        child_indices = valid_children[player_mask]
                        action_probs = probs[player_mask]
                        reach[:, child_indices, p] *= action_probs

        return reach

    def backward_utils(self, reach: Tensor, strategy: Tensor) -> Tensor:
        """
        Bottom-up pass: compute expected utilities.

        Returns: [batch, num_nodes, num_players]
        """
        utils = torch.zeros(
            self.batch_size,
            self.compiled.num_nodes,
            self.compiled.num_players,
            device=self.device
        )

        # Initialize terminal nodes
        utils[:, self.compiled.terminal_mask, :] = \
            self.compiled.terminal_utils[self.compiled.terminal_mask].unsqueeze(0)

        # Process in reverse depth order
        for depth in reversed(range(self.compiled.max_depth)):
            start, end = self.compiled.depth_slices[depth]
            if start >= end:
                continue

            nodes = torch.arange(start, end, device=self.device)
            non_terminal = ~self.compiled.terminal_mask[nodes]

            if not non_terminal.any():
                continue

            active_nodes = nodes[non_terminal]
            info_sets = self.compiled.node_info_set[active_nodes]
            children = self.compiled.action_child[active_nodes]
            mask = self.compiled.action_mask[active_nodes]

            node_strats = strategy[info_sets]

            # Weighted sum of child utilities
            for a in range(self.compiled.max_actions):
                valid = mask[:, a]
                if not valid.any():
                    continue

                valid_nodes = active_nodes[valid]
                valid_children = children[valid, a]
                probs = node_strats[valid, a].unsqueeze(0).unsqueeze(-1)

                utils[:, valid_nodes, :] += probs * utils[:, valid_children, :]

        return utils

    def train_step(self):
        """One batched CFR iteration."""
        strategy = self.get_current_strategy()
        reach = self.forward_reach(strategy)
        utils = self.backward_utils(reach, strategy)

        # Update regrets for non-terminal nodes
        non_terminal = ~self.compiled.terminal_mask
        active_nodes = torch.where(non_terminal)[0]

        for node_idx in active_nodes:
            player = self.compiled.node_player[node_idx].item()
            opponent = 1 - player
            info_set = self.compiled.node_info_set[node_idx].item()

            cf_reach = reach[:, node_idx, opponent]  # [batch]
            node_util = utils[:, node_idx, player]   # [batch]

            for a in range(self.compiled.max_actions):
                if not self.compiled.action_mask[node_idx, a]:
                    continue

                child_idx = self.compiled.action_child[node_idx, a]
                action_util = utils[:, child_idx, player]

                # Regret = Σ_batch cf_reach * (action_util - node_util)
                regret = (cf_reach * (action_util - node_util)).sum()
                self.regret_sum[info_set, a] += regret

            # Strategy sum weighted by player reach
            my_reach = reach[:, node_idx, player].sum()
            for a in range(self.compiled.max_actions):
                if self.compiled.action_mask[node_idx, a]:
                    self.strategy_sum[info_set, a] += my_reach * strategy[info_set, a]

        self.iterations += self.batch_size

    def get_average_strategy(self) -> Dict[str, Dict[str, float]]:
        """Convert tensor strategy to dictionary format."""
        result = {}
        strategy_sum = self.strategy_sum.cpu().numpy()

        for info_set, idx in self.compiled.info_set_to_idx.items():
            total = strategy_sum[idx].sum()
            if total > 0:
                result[info_set] = {
                    "p": strategy_sum[idx, 0] / total,
                    "b": strategy_sum[idx, 1] / total,
                }
            else:
                result[info_set] = {"p": 0.5, "b": 0.5}

        return result
```

---

## Testing & Verification

### Known Kuhn Poker Nash Equilibrium

For α = 0 (simplest equilibrium):

| Info Set | Action Probabilities | Description |
|----------|---------------------|-------------|
| `0:` | p=1.0, b=0.0 | Jack at root: always check |
| `0:pb` | p=1.0, b=0.0 | Jack facing bet: always fold |
| `1:` | p=1.0, b=0.0 | Queen at root: always check |
| `1:pb` | p=2/3, b=1/3 | Queen facing bet: call 1/3 |
| `2:` | p=1.0, b=0.0 | King at root: always check |
| `2:pb` | p=0.0, b=1.0 | King facing bet: always call |
| `0:p` | p=2/3, b=1/3 | Jack after check: bet 1/3 |
| `0:b` | p=1.0, b=0.0 | Jack facing bet: always fold |
| `1:p` | p=1.0, b=0.0 | Queen after check: always check |
| `1:b` | p=2/3, b=1/3 | Queen facing bet: call 1/3 |
| `2:p` | p=2/3, b=1/3 | King after check: bet 1/3 |
| `2:b` | p=0.0, b=1.0 | King facing bet: always call |

### Test Cases

```python
import pytest
import torch

class TestKuhnCFR:
    def test_vanilla_convergence(self):
        """Verify vanilla CFR converges to Nash equilibrium."""
        game = KuhnPoker()
        cfr = VanillaCFR(game)
        cfr.train(iterations=10000)

        strategy = cfr.get_all_average_strategies()

        # Key equilibrium properties
        assert strategy["0:pb"]["b"] < 0.05  # Jack folds to bet
        assert strategy["2:pb"]["b"] > 0.95  # King calls bet
        assert strategy["2:b"]["b"] > 0.95   # King calls bet (as P1)

    def test_exploitability_near_zero(self):
        """Converged strategy should have near-zero exploitability."""
        game = KuhnPoker()
        cfr = VanillaCFR(game)
        cfr.train(iterations=10000)

        strategy = cfr.get_all_average_strategies()
        exploit = compute_exploitability(game, strategy)

        assert exploit < 0.01  # Less than 1% of pot

    def test_cpu_gpu_equivalence(self):
        """Batched GPU should match CPU implementation."""
        game = KuhnPoker()

        # CPU baseline
        cpu_cfr = VanillaCFR(game)
        cpu_cfr.train(iterations=1000)
        cpu_strategy = cpu_cfr.get_all_average_strategies()

        # GPU batched
        device = get_device()
        compiled = compile_game(game, device)
        gpu_cfr = BatchedCFR(compiled, batch_size=1000)
        gpu_cfr.train_step()  # 1000 iterations in one step
        gpu_strategy = gpu_cfr.get_average_strategy()

        # Compare strategies (allow some variance due to batching)
        for info_set in cpu_strategy:
            for action in cpu_strategy[info_set]:
                diff = abs(cpu_strategy[info_set][action] - gpu_strategy[info_set][action])
                assert diff < 0.1  # Within 10%

    def test_mps_device(self):
        """Test Apple Silicon MPS backend if available."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        device = torch.device("mps")
        game = KuhnPoker()
        compiled = compile_game(game, device)
        cfr = BatchedCFR(compiled, batch_size=256)

        # Should not raise
        cfr.train_step()
        strategy = cfr.get_average_strategy()

        assert len(strategy) == 12  # All info sets


def compute_exploitability(game: Game, strategy: Dict) -> float:
    """
    Compute exploitability: max utility opponent can achieve.

    For a Nash equilibrium, exploitability = 0.
    """
    # Best response for each player against fixed opponent strategy
    br_value_0 = best_response_value(game, strategy, exploiting_player=0)
    br_value_1 = best_response_value(game, strategy, exploiting_player=1)

    # Exploitability = sum of what each player can gain by deviating
    return (br_value_0 + br_value_1) / 2


def best_response_value(game: Game, strategy: Dict, exploiting_player: int) -> float:
    """Compute value of best response for exploiting_player."""
    # Dynamic programming over game tree
    # ... (implementation details)
    pass
```

---

## Public API

```python
class Solver:
    """High-level API for solving extensive-form games."""

    def __init__(
        self,
        game: Game,
        variant: str = "vanilla",
        device: str = "auto",
        batch_size: int = 1024,
    ):
        self.game = game
        self.device = get_device(device)
        self.variant = variant

        # Choose implementation based on game size and variant
        if variant == "vanilla" and batch_size == 1:
            # Pure CPU implementation
            self.engine = VanillaCFR(game)
            self.batched = False
        else:
            # GPU-accelerated batched implementation
            self.compiled = compile_game(game, self.device)
            self.engine = BatchedCFR(self.compiled, batch_size)
            self.batched = True

    def solve(
        self,
        iterations: int,
        verbose: bool = True,
        log_interval: int = 1000,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run CFR for specified iterations.

        Returns: Average strategy for all information sets
        """
        if self.batched:
            steps = iterations // self.engine.batch_size
            for step in range(steps):
                self.engine.train_step()

                if verbose and (step + 1) % (log_interval // self.engine.batch_size) == 0:
                    current_iter = (step + 1) * self.engine.batch_size
                    exploit = self.exploitability()
                    print(f"Iteration {current_iter}: exploitability = {exploit:.6f}")
        else:
            self.engine.train(iterations)
            if verbose:
                exploit = self.exploitability()
                print(f"Final exploitability = {exploit:.6f}")

        return self.get_strategy()

    def get_strategy(self) -> Dict[str, Dict[str, float]]:
        """Get current average strategy."""
        if self.batched:
            return self.engine.get_average_strategy()
        else:
            return self.engine.get_all_average_strategies()

    def exploitability(self) -> float:
        """Compute exploitability of current strategy."""
        return compute_exploitability(self.game, self.get_strategy())


# Usage example
if __name__ == "__main__":
    from games.kuhn import KuhnPoker

    solver = Solver(
        game=KuhnPoker(),
        device="auto",      # Uses MPS on Mac, CUDA on Linux/Windows
        batch_size=1024,
    )

    strategy = solver.solve(iterations=10000, verbose=True)

    print("\nFinal Strategy:")
    for info_set, probs in sorted(strategy.items()):
        print(f"  {info_set}: pass={probs['p']:.3f}, bet={probs['b']:.3f}")
```

---

## Future Extensions

### CFR+ Variant

Minor modification to vanilla - floor regrets at zero:

```python
class CFRPlus(VanillaCFR):
    def train_step(self):
        super().train_step()
        # Floor regrets at zero
        for info_set in self.regret_sum:
            for action in self.regret_sum[info_set]:
                self.regret_sum[info_set][action] = max(0, self.regret_sum[info_set][action])
```

### Monte Carlo CFR (MCCFR)

Sample chance outcomes instead of full traversal - required for larger games:

```python
class MCCFR(VanillaCFR):
    def train(self, iterations: int):
        for _ in range(iterations):
            # Sample ONE initial state instead of all
            initial_state = random.choice(self.game.initial_states())
            self.cfr(initial_state, reach_probs=(1.0, 1.0))
```

### Vectorized Tree Operations

For very large games, process entire depth levels as tensor operations:

```python
def forward_reach_vectorized(self, strategy: Tensor) -> Tensor:
    """Fully vectorized - no Python loops over nodes."""
    # Use scatter/gather operations on depth-sorted node tensors
    # Requires more complex indexing but eliminates sequential bottleneck
    pass
```

### Leduc Poker

6-card deck, 2 rounds, ~936 information sets:

```python
class LeducPoker(Game):
    # Cards: J♠, J♦, Q♠, Q♦, K♠, K♦
    # Round 1: private card, betting
    # Round 2: public card, betting
    # Pair beats high card
    pass
```

---

## Summary

| Component | Purpose |
|-----------|---------|
| `games/base.py` | Abstract game interface |
| `games/kuhn.py` | Kuhn Poker implementation |
| `cfr/vanilla.py` | Reference CPU implementation |
| `core/tensors.py` | Game tree → tensor compilation |
| `core/strategy.py` | Batched GPU CFR engine |
| `tests/` | Verification against known equilibrium |
| `main.py` | Public Solver API |

**Key design decisions:**
1. Dual representation: explicit tree for clarity, tensors for GPU
2. Batch parallelism: N independent traversals simultaneously
3. Device abstraction: seamless CUDA/MPS/CPU support
4. Modular variants: easy to swap vanilla → CFR+ → MCCFR
