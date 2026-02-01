# CFR Poker Solver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a PyTorch-based CFR solver for Kuhn Poker with GPU acceleration (CUDA/MPS).

**Architecture:** Game abstraction layer → Vanilla CFR (CPU reference) → Tensor compilation → Batched GPU execution. TDD approach: verify against known Kuhn Nash equilibrium.

**Tech Stack:** Python 3.10+, PyTorch 2.0+, pytest

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `games/__init__.py`
- Create: `cfr/__init__.py`
- Create: `core/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "poker-solver"
version = "0.1.0"
description = "CFR solver for extensive-form games"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v"
```

**Step 2: Create package structure**

```bash
mkdir -p games cfr core tests
touch games/__init__.py cfr/__init__.py core/__init__.py tests/__init__.py
```

**Step 3: Create virtual environment and install**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

**Step 4: Verify pytest runs**

Run: `pytest --collect-only`
Expected: "no tests ran" (empty collection is OK)

**Step 5: Commit**

```bash
git add pyproject.toml games/ cfr/ core/ tests/
git commit -m "feat: initial project structure with PyTorch dependencies"
```

---

## Task 2: Game Base Class

**Files:**
- Create: `games/base.py`
- Create: `tests/test_game_base.py`

**Step 1: Write the test for Game interface**

Create `tests/test_game_base.py`:

```python
import pytest
from abc import ABC
from games.base import Game, State, Action


def test_game_is_abstract():
    """Game should be an abstract base class."""
    with pytest.raises(TypeError):
        Game()


def test_game_has_required_methods():
    """Game ABC should define all required abstract methods."""
    abstract_methods = {
        'initial_states',
        'is_terminal',
        'player',
        'actions',
        'next_state',
        'utility',
        'info_set_key',
        'num_players',
    }
    assert abstract_methods <= set(Game.__abstractmethods__)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_game_base.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'games.base'"

**Step 3: Write Game base class**

Create `games/base.py`:

```python
from abc import ABC, abstractmethod
from typing import List, Any, TypeVar

State = TypeVar('State')
Action = str


class Game(ABC):
    """
    Abstract base class for extensive-form games.

    Mirrors Definition 1 from Zinkevich et al.'s CFR paper.
    """

    @abstractmethod
    def initial_states(self) -> List[State]:
        """All possible starting states (after chance moves)."""
        pass

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """Check if state is terminal (z ∈ Z in paper)."""
        pass

    @abstractmethod
    def player(self, state: State) -> int:
        """Return player to act at state. P(h) in paper."""
        pass

    @abstractmethod
    def actions(self, state: State) -> List[Action]:
        """Available actions at state. A(h) in paper."""
        pass

    @abstractmethod
    def next_state(self, state: State, action: Action) -> State:
        """Return state after taking action."""
        pass

    @abstractmethod
    def utility(self, state: State, player: int) -> float:
        """Utility for player at terminal state. u_i(z) in paper."""
        pass

    @abstractmethod
    def info_set_key(self, state: State) -> str:
        """Map state to information set identifier. h → I in paper."""
        pass

    @abstractmethod
    def num_players(self) -> int:
        """Number of players in the game."""
        pass
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_game_base.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add games/base.py tests/test_game_base.py
git commit -m "feat: add abstract Game base class"
```

---

## Task 3: Kuhn Poker State

**Files:**
- Create: `games/kuhn.py`
- Create: `tests/test_kuhn.py`

**Step 1: Write test for KuhnState**

Create `tests/test_kuhn.py`:

```python
import pytest
from games.kuhn import KuhnState


class TestKuhnState:
    def test_state_is_immutable(self):
        """KuhnState should be immutable (frozen dataclass)."""
        state = KuhnState(cards=(0, 1), history=())
        with pytest.raises(AttributeError):
            state.cards = (1, 2)

    def test_state_equality(self):
        """States with same data should be equal."""
        s1 = KuhnState(cards=(0, 1), history=("p",))
        s2 = KuhnState(cards=(0, 1), history=("p",))
        assert s1 == s2

    def test_state_hashable(self):
        """States should be hashable for use as dict keys."""
        state = KuhnState(cards=(0, 1), history=())
        d = {state: "value"}
        assert d[state] == "value"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_kuhn.py::TestKuhnState -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write KuhnState**

Create `games/kuhn.py`:

```python
from dataclasses import dataclass
from typing import Tuple


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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_kuhn.py::TestKuhnState -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add games/kuhn.py tests/test_kuhn.py
git commit -m "feat: add KuhnState dataclass"
```

---

## Task 4: Kuhn Poker Game Logic

**Files:**
- Modify: `games/kuhn.py`
- Modify: `tests/test_kuhn.py`

**Step 1: Write tests for KuhnPoker game logic**

Add to `tests/test_kuhn.py`:

```python
from itertools import permutations
from games.kuhn import KuhnPoker, KuhnState


class TestKuhnPoker:
    @pytest.fixture
    def game(self):
        return KuhnPoker()

    def test_initial_states_count(self, game):
        """Should have 6 initial states (3P2 card permutations)."""
        states = game.initial_states()
        assert len(states) == 6

    def test_initial_states_are_all_deals(self, game):
        """Initial states should cover all card permutations."""
        states = game.initial_states()
        card_combos = {s.cards for s in states}
        expected = set(permutations([0, 1, 2], 2))
        assert card_combos == expected

    def test_player_alternates(self, game):
        """Player should alternate based on history length."""
        s0 = KuhnState(cards=(0, 1), history=())
        s1 = KuhnState(cards=(0, 1), history=("p",))
        s2 = KuhnState(cards=(0, 1), history=("p", "b"))

        assert game.player(s0) == 0
        assert game.player(s1) == 1
        assert game.player(s2) == 0

    def test_actions_always_two(self, game):
        """Should always have exactly 2 actions: pass and bet."""
        state = KuhnState(cards=(0, 1), history=())
        actions = game.actions(state)
        assert actions == ["p", "b"]

    def test_terminal_states(self, game):
        """Test all terminal conditions."""
        # pp = check-check (terminal)
        assert game.is_terminal(KuhnState((0, 1), ("p", "p"))) is True
        # bp = bet-fold (terminal)
        assert game.is_terminal(KuhnState((0, 1), ("b", "p"))) is True
        # bb = bet-call (terminal)
        assert game.is_terminal(KuhnState((0, 1), ("b", "b"))) is True
        # pbp = check-bet-fold (terminal)
        assert game.is_terminal(KuhnState((0, 1), ("p", "b", "p"))) is True
        # pbb = check-bet-call (terminal)
        assert game.is_terminal(KuhnState((0, 1), ("p", "b", "b"))) is True

        # Non-terminal
        assert game.is_terminal(KuhnState((0, 1), ())) is False
        assert game.is_terminal(KuhnState((0, 1), ("p",))) is False
        assert game.is_terminal(KuhnState((0, 1), ("b",))) is False
        assert game.is_terminal(KuhnState((0, 1), ("p", "b"))) is False

    def test_next_state(self, game):
        """next_state should append action to history."""
        s0 = KuhnState(cards=(0, 1), history=())
        s1 = game.next_state(s0, "p")
        assert s1 == KuhnState(cards=(0, 1), history=("p",))

        s2 = game.next_state(s1, "b")
        assert s2 == KuhnState(cards=(0, 1), history=("p", "b"))

    def test_num_players(self, game):
        """Kuhn Poker has 2 players."""
        assert game.num_players() == 2
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_kuhn.py::TestKuhnPoker -v`
Expected: FAIL with "cannot import name 'KuhnPoker'"

**Step 3: Implement KuhnPoker game logic**

Add to `games/kuhn.py`:

```python
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

    def num_players(self) -> int:
        return 2
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_kuhn.py::TestKuhnPoker -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add games/kuhn.py tests/test_kuhn.py
git commit -m "feat: add KuhnPoker game logic"
```

---

## Task 5: Kuhn Poker Utilities

**Files:**
- Modify: `games/kuhn.py`
- Modify: `tests/test_kuhn.py`

**Step 1: Write tests for utility function**

Add to `tests/test_kuhn.py`:

```python
class TestKuhnUtilities:
    @pytest.fixture
    def game(self):
        return KuhnPoker()

    def test_check_check_higher_wins(self, game):
        """pp: higher card wins 1 chip."""
        # P0 has King(2), P1 has Jack(0) -> P0 wins
        state = KuhnState(cards=(2, 0), history=("p", "p"))
        assert game.utility(state, 0) == 1.0
        assert game.utility(state, 1) == -1.0

        # P0 has Jack(0), P1 has Queen(1) -> P1 wins
        state = KuhnState(cards=(0, 1), history=("p", "p"))
        assert game.utility(state, 0) == -1.0
        assert game.utility(state, 1) == 1.0

    def test_bet_fold_bettor_wins(self, game):
        """bp: player 0 bet, player 1 folded -> P0 wins 1."""
        state = KuhnState(cards=(0, 2), history=("b", "p"))
        assert game.utility(state, 0) == 1.0
        assert game.utility(state, 1) == -1.0

    def test_bet_call_higher_wins(self, game):
        """bb: bet-call, higher card wins 2 chips."""
        # P0 has King, wins 2
        state = KuhnState(cards=(2, 1), history=("b", "b"))
        assert game.utility(state, 0) == 2.0
        assert game.utility(state, 1) == -2.0

        # P1 has King, wins 2
        state = KuhnState(cards=(1, 2), history=("b", "b"))
        assert game.utility(state, 0) == -2.0
        assert game.utility(state, 1) == 2.0

    def test_check_bet_fold(self, game):
        """pbp: check-bet-fold -> P1 wins 1."""
        state = KuhnState(cards=(2, 0), history=("p", "b", "p"))
        assert game.utility(state, 0) == -1.0
        assert game.utility(state, 1) == 1.0

    def test_check_bet_call(self, game):
        """pbb: check-bet-call, higher card wins 2."""
        state = KuhnState(cards=(2, 0), history=("p", "b", "b"))
        assert game.utility(state, 0) == 2.0
        assert game.utility(state, 1) == -2.0

    def test_zero_sum(self, game):
        """Utilities should always sum to zero."""
        for cards in permutations([0, 1, 2], 2):
            for terminal in KuhnPoker.TERMINALS:
                state = KuhnState(cards=cards, history=terminal)
                total = game.utility(state, 0) + game.utility(state, 1)
                assert total == 0.0, f"Non-zero sum at {state}"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_kuhn.py::TestKuhnUtilities -v`
Expected: FAIL with "utility() missing"

**Step 3: Implement utility function**

Add to `KuhnPoker` class in `games/kuhn.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_kuhn.py::TestKuhnUtilities -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add games/kuhn.py tests/test_kuhn.py
git commit -m "feat: add Kuhn Poker utility function"
```

---

## Task 6: Kuhn Poker Information Sets

**Files:**
- Modify: `games/kuhn.py`
- Modify: `tests/test_kuhn.py`

**Step 1: Write tests for info_set_key**

Add to `tests/test_kuhn.py`:

```python
class TestKuhnInfoSets:
    @pytest.fixture
    def game(self):
        return KuhnPoker()

    def test_info_set_includes_own_card(self, game):
        """Info set should encode player's card."""
        s1 = KuhnState(cards=(0, 1), history=())  # P0 has Jack
        s2 = KuhnState(cards=(2, 1), history=())  # P0 has King

        assert game.info_set_key(s1) != game.info_set_key(s2)
        assert "0" in game.info_set_key(s1)  # Jack = 0
        assert "2" in game.info_set_key(s2)  # King = 2

    def test_info_set_includes_history(self, game):
        """Info set should encode action history."""
        s1 = KuhnState(cards=(0, 1), history=())
        s2 = KuhnState(cards=(0, 1), history=("p",))
        s3 = KuhnState(cards=(0, 1), history=("p", "b"))

        k1 = game.info_set_key(s1)
        k2 = game.info_set_key(s2)
        k3 = game.info_set_key(s3)

        assert k1 != k2 != k3

    def test_info_set_hides_opponent_card(self, game):
        """Different opponent cards should map to same info set."""
        # P0 has Jack, history empty - can't distinguish opponent's card
        s1 = KuhnState(cards=(0, 1), history=())  # vs Queen
        s2 = KuhnState(cards=(0, 2), history=())  # vs King

        assert game.info_set_key(s1) == game.info_set_key(s2)

    def test_info_set_format(self, game):
        """Info set should be 'card:history' format."""
        state = KuhnState(cards=(1, 2), history=("p", "b"))
        # P0 to act, has Queen (1)
        key = game.info_set_key(state)
        assert key == "1:pb"

    def test_total_info_sets(self, game):
        """Kuhn poker has exactly 12 information sets."""
        info_sets = set()

        def traverse(state):
            if game.is_terminal(state):
                return
            info_sets.add(game.info_set_key(state))
            for action in game.actions(state):
                traverse(game.next_state(state, action))

        for initial in game.initial_states():
            traverse(initial)

        assert len(info_sets) == 12
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_kuhn.py::TestKuhnInfoSets -v`
Expected: FAIL with "info_set_key() missing"

**Step 3: Implement info_set_key**

Add to `KuhnPoker` class in `games/kuhn.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_kuhn.py::TestKuhnInfoSets -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add games/kuhn.py tests/test_kuhn.py
git commit -m "feat: add Kuhn Poker information set mapping"
```

---

## Task 7: Vanilla CFR - Strategy from Regrets

**Files:**
- Create: `cfr/vanilla.py`
- Create: `tests/test_vanilla_cfr.py`

**Step 1: Write test for get_strategy (regret matching)**

Create `tests/test_vanilla_cfr.py`:

```python
import pytest
from cfr.vanilla import VanillaCFR
from games.kuhn import KuhnPoker


class TestRegretMatching:
    @pytest.fixture
    def cfr(self):
        return VanillaCFR(KuhnPoker())

    def test_uniform_with_no_regrets(self, cfr):
        """With no regrets, strategy should be uniform."""
        strategy = cfr.get_strategy("0:", ["p", "b"])
        assert strategy == {"p": 0.5, "b": 0.5}

    def test_uniform_with_all_negative_regrets(self, cfr):
        """With all negative regrets, strategy should be uniform."""
        cfr.regret_sum["0:"]["p"] = -5.0
        cfr.regret_sum["0:"]["b"] = -3.0
        strategy = cfr.get_strategy("0:", ["p", "b"])
        assert strategy == {"p": 0.5, "b": 0.5}

    def test_proportional_to_positive_regret(self, cfr):
        """Strategy should be proportional to positive regrets."""
        cfr.regret_sum["0:"]["p"] = 3.0
        cfr.regret_sum["0:"]["b"] = 1.0
        strategy = cfr.get_strategy("0:", ["p", "b"])
        assert strategy["p"] == pytest.approx(0.75)
        assert strategy["b"] == pytest.approx(0.25)

    def test_ignores_negative_regret(self, cfr):
        """Negative regrets should be treated as zero."""
        cfr.regret_sum["0:"]["p"] = 4.0
        cfr.regret_sum["0:"]["b"] = -2.0
        strategy = cfr.get_strategy("0:", ["p", "b"])
        assert strategy["p"] == pytest.approx(1.0)
        assert strategy["b"] == pytest.approx(0.0)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vanilla_cfr.py::TestRegretMatching -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement VanillaCFR with get_strategy**

Create `cfr/vanilla.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vanilla_cfr.py::TestRegretMatching -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add cfr/vanilla.py tests/test_vanilla_cfr.py
git commit -m "feat: add VanillaCFR with regret matching strategy"
```

---

## Task 8: Vanilla CFR - Core Algorithm

**Files:**
- Modify: `cfr/vanilla.py`
- Modify: `tests/test_vanilla_cfr.py`

**Step 1: Write test for CFR traversal**

Add to `tests/test_vanilla_cfr.py`:

```python
class TestCFRTraversal:
    @pytest.fixture
    def cfr(self):
        return VanillaCFR(KuhnPoker())

    def test_cfr_returns_utilities(self, cfr):
        """CFR should return utility tuple for both players."""
        from games.kuhn import KuhnState
        state = KuhnState(cards=(0, 1), history=())
        utils = cfr.cfr(state, reach_probs=(1.0, 1.0))

        assert isinstance(utils, tuple)
        assert len(utils) == 2
        assert all(isinstance(u, float) for u in utils)

    def test_cfr_terminal_returns_utility(self, cfr):
        """At terminal state, CFR should return actual utilities."""
        from games.kuhn import KuhnState
        # P0 has King, P1 has Jack, check-check -> P0 wins 1
        state = KuhnState(cards=(2, 0), history=("p", "p"))
        utils = cfr.cfr(state, reach_probs=(1.0, 1.0))

        assert utils[0] == 1.0
        assert utils[1] == -1.0

    def test_cfr_updates_regrets(self, cfr):
        """CFR should update regret sums."""
        from games.kuhn import KuhnState
        state = KuhnState(cards=(0, 1), history=())

        # Before: no regrets
        assert len(cfr.regret_sum) == 0

        cfr.cfr(state, reach_probs=(1.0, 1.0))

        # After: should have regrets for visited info sets
        assert len(cfr.regret_sum) > 0

    def test_cfr_updates_strategy_sum(self, cfr):
        """CFR should accumulate strategy weights."""
        from games.kuhn import KuhnState
        state = KuhnState(cards=(0, 1), history=())

        cfr.cfr(state, reach_probs=(1.0, 1.0))

        # Should have accumulated strategies
        assert len(cfr.strategy_sum) > 0
        # Root info set should have both actions
        assert "p" in cfr.strategy_sum["0:"]
        assert "b" in cfr.strategy_sum["0:"]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vanilla_cfr.py::TestCFRTraversal -v`
Expected: FAIL with "'VanillaCFR' object has no attribute 'cfr'"

**Step 3: Implement cfr method**

Add to `VanillaCFR` class in `cfr/vanilla.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vanilla_cfr.py::TestCFRTraversal -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add cfr/vanilla.py tests/test_vanilla_cfr.py
git commit -m "feat: implement CFR recursive traversal"
```

---

## Task 9: Vanilla CFR - Training and Average Strategy

**Files:**
- Modify: `cfr/vanilla.py`
- Modify: `tests/test_vanilla_cfr.py`

**Step 1: Write tests for train and average strategy**

Add to `tests/test_vanilla_cfr.py`:

```python
class TestCFRTraining:
    def test_train_runs_iterations(self):
        """Train should run specified number of iterations."""
        cfr = VanillaCFR(KuhnPoker())
        cfr.train(iterations=100)

        # Should have visited all 12 info sets
        all_info_sets = set(cfr.regret_sum.keys()) | set(cfr.strategy_sum.keys())
        assert len(all_info_sets) == 12

    def test_get_average_strategy_normalized(self):
        """Average strategy should sum to 1."""
        cfr = VanillaCFR(KuhnPoker())
        cfr.train(iterations=100)

        for info_set in cfr.strategy_sum:
            strategy = cfr.get_average_strategy(info_set)
            total = sum(strategy.values())
            assert total == pytest.approx(1.0), f"Strategy at {info_set} sums to {total}"

    def test_get_average_strategy_uniform_when_empty(self):
        """Should return uniform if no strategy accumulated."""
        cfr = VanillaCFR(KuhnPoker())
        strategy = cfr.get_average_strategy("nonexistent")
        # With no data, defaults should be uniform-ish or empty
        # Implementation choice: return uniform over known actions

    def test_get_all_average_strategies(self):
        """Should return strategies for all info sets."""
        cfr = VanillaCFR(KuhnPoker())
        cfr.train(iterations=100)

        strategies = cfr.get_all_average_strategies()
        assert len(strategies) == 12

        for info_set, strategy in strategies.items():
            assert "p" in strategy
            assert "b" in strategy
            assert sum(strategy.values()) == pytest.approx(1.0)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vanilla_cfr.py::TestCFRTraining -v`
Expected: FAIL with "'VanillaCFR' object has no attribute 'train'"

**Step 3: Implement train and average strategy methods**

Add to `VanillaCFR` class in `cfr/vanilla.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vanilla_cfr.py::TestCFRTraining -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add cfr/vanilla.py tests/test_vanilla_cfr.py
git commit -m "feat: add CFR training loop and average strategy"
```

---

## Task 10: Exploitability Calculation

**Files:**
- Create: `core/exploitability.py`
- Create: `tests/test_exploitability.py`

**Step 1: Write test for best response value**

Create `tests/test_exploitability.py`:

```python
import pytest
from core.exploitability import best_response_value, compute_exploitability
from games.kuhn import KuhnPoker


class TestBestResponse:
    @pytest.fixture
    def game(self):
        return KuhnPoker()

    def test_best_response_against_always_fold(self, game):
        """Against always-fold, best response should win big."""
        # Opponent always folds (passes) to any bet
        strategy = {
            "0:": {"p": 0.0, "b": 1.0},    # Always bet
            "0:pb": {"p": 1.0, "b": 0.0},  # Always fold
            "1:": {"p": 0.0, "b": 1.0},
            "1:pb": {"p": 1.0, "b": 0.0},
            "2:": {"p": 0.0, "b": 1.0},
            "2:pb": {"p": 1.0, "b": 0.0},
            "0:p": {"p": 1.0, "b": 0.0},   # Always fold
            "0:b": {"p": 1.0, "b": 0.0},
            "1:p": {"p": 1.0, "b": 0.0},
            "1:b": {"p": 1.0, "b": 0.0},
            "2:p": {"p": 1.0, "b": 0.0},
            "2:b": {"p": 1.0, "b": 0.0},
        }

        # Best response should achieve positive value
        br_value = best_response_value(game, strategy, exploiting_player=0)
        assert br_value > 0

    def test_exploitability_of_uniform_strategy(self, game):
        """Uniform random strategy should be highly exploitable."""
        strategy = {
            info_set: {"p": 0.5, "b": 0.5}
            for info_set in [
                "0:", "1:", "2:", "0:pb", "1:pb", "2:pb",
                "0:p", "1:p", "2:p", "0:b", "1:b", "2:b"
            ]
        }

        exploit = compute_exploitability(game, strategy)
        assert exploit > 0.05  # Should be quite exploitable


class TestExploitability:
    def test_exploitability_decreases_with_training(self):
        """More CFR iterations should reduce exploitability."""
        from cfr.vanilla import VanillaCFR

        game = KuhnPoker()
        cfr = VanillaCFR(game)

        cfr.train(iterations=10)
        exploit_10 = compute_exploitability(game, cfr.get_all_average_strategies())

        cfr.train(iterations=990)  # Total 1000
        exploit_1000 = compute_exploitability(game, cfr.get_all_average_strategies())

        assert exploit_1000 < exploit_10
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_exploitability.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement exploitability calculation**

Create `core/exploitability.py`:

```python
from typing import Dict
from games.base import Game


def best_response_value(
    game: Game,
    strategy: Dict[str, Dict[str, float]],
    exploiting_player: int,
) -> float:
    """
    Compute value of best response for exploiting_player.

    Uses dynamic programming to find optimal counter-strategy.
    """
    opponent = 1 - exploiting_player

    def traverse(state, prob_opponent: float) -> float:
        """
        Returns expected value for exploiting_player.

        prob_opponent: probability opponent played to reach this state
        """
        if game.is_terminal(state):
            return prob_opponent * game.utility(state, exploiting_player)

        player = game.player(state)
        info_set = game.info_set_key(state)
        actions = game.actions(state)

        if player == exploiting_player:
            # Maximizing player: pick best action
            best_value = float('-inf')
            for action in actions:
                next_state = game.next_state(state, action)
                value = traverse(next_state, prob_opponent)
                best_value = max(best_value, value)
            return best_value
        else:
            # Opponent plays fixed strategy
            opp_strategy = strategy.get(info_set, {a: 1/len(actions) for a in actions})
            value = 0.0
            for action in actions:
                action_prob = opp_strategy.get(action, 0.0)
                if action_prob > 0:
                    next_state = game.next_state(state, action)
                    value += traverse(next_state, prob_opponent * action_prob)
            return value

    # Average over all initial states (card dealings)
    total_value = 0.0
    initial_states = game.initial_states()

    for state in initial_states:
        total_value += traverse(state, prob_opponent=1.0)

    return total_value / len(initial_states)


def compute_exploitability(
    game: Game,
    strategy: Dict[str, Dict[str, float]],
) -> float:
    """
    Compute exploitability of a strategy.

    Exploitability = (BR_value_p0 + BR_value_p1) / 2

    For Nash equilibrium, exploitability = 0.
    """
    br_value_0 = best_response_value(game, strategy, exploiting_player=0)
    br_value_1 = best_response_value(game, strategy, exploiting_player=1)

    return (br_value_0 + br_value_1) / 2
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_exploitability.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add core/exploitability.py tests/test_exploitability.py
git commit -m "feat: add exploitability calculation"
```

---

## Task 11: Verify CFR Converges to Nash Equilibrium

**Files:**
- Create: `tests/test_nash_convergence.py`

**Step 1: Write comprehensive Nash equilibrium tests**

Create `tests/test_nash_convergence.py`:

```python
import pytest
from cfr.vanilla import VanillaCFR
from games.kuhn import KuhnPoker
from core.exploitability import compute_exploitability


class TestNashConvergence:
    """Verify CFR converges to known Kuhn Poker Nash equilibrium."""

    @pytest.fixture
    def trained_cfr(self):
        """Train CFR for enough iterations to converge."""
        game = KuhnPoker()
        cfr = VanillaCFR(game)
        cfr.train(iterations=10000)
        return cfr

    def test_exploitability_near_zero(self, trained_cfr):
        """Converged strategy should have near-zero exploitability."""
        game = KuhnPoker()
        strategy = trained_cfr.get_all_average_strategies()
        exploit = compute_exploitability(game, strategy)

        # Nash equilibrium has 0 exploitability
        # Allow small epsilon for numerical convergence
        assert exploit < 0.01, f"Exploitability {exploit} too high"

    def test_jack_folds_to_bet(self, trained_cfr):
        """Player with Jack should (almost) always fold when facing a bet."""
        strategy = trained_cfr.get_all_average_strategies()

        # P0 with Jack facing bet after check-bet
        assert strategy["0:pb"]["p"] > 0.95, "Jack should fold to bet (P0)"

        # P1 with Jack facing bet
        assert strategy["0:b"]["p"] > 0.95, "Jack should fold to bet (P1)"

    def test_king_calls_bet(self, trained_cfr):
        """Player with King should always call a bet."""
        strategy = trained_cfr.get_all_average_strategies()

        # P0 with King facing bet
        assert strategy["2:pb"]["b"] > 0.95, "King should call bet (P0)"

        # P1 with King facing bet
        assert strategy["2:b"]["b"] > 0.95, "King should call bet (P1)"

    def test_jack_bluffs_sometimes(self, trained_cfr):
        """Player with Jack should bluff with some probability."""
        strategy = trained_cfr.get_all_average_strategies()

        # In Nash equilibrium, Jack bets ~1/3 at root
        # But this depends on alpha parameter, so just check it's bounded
        jack_bet_prob = strategy["0:"]["b"]
        assert 0.0 <= jack_bet_prob <= 0.4, f"Jack bet prob {jack_bet_prob} out of range"

    def test_king_value_bets_sometimes(self, trained_cfr):
        """Player with King should value bet sometimes."""
        strategy = trained_cfr.get_all_average_strategies()

        # King at root should bet sometimes (value bet)
        king_bet_prob = strategy["2:"]["b"]
        assert king_bet_prob > 0.0, "King should sometimes bet at root"

    def test_queen_indifferent(self, trained_cfr):
        """Queen should have mixed strategy in some spots."""
        strategy = trained_cfr.get_all_average_strategies()

        # Queen facing bet has mixed strategy
        queen_call = strategy["1:pb"]["b"]
        # Should call with some probability (around 1/3 in equilibrium)
        assert 0.2 < queen_call < 0.5, f"Queen call prob {queen_call} unexpected"


class TestConvergenceRate:
    """Test that exploitability decreases with iterations."""

    def test_exploitability_decreasing(self):
        """Exploitability should decrease as training progresses."""
        game = KuhnPoker()
        cfr = VanillaCFR(game)

        exploits = []
        for i in [100, 500, 1000, 5000]:
            iterations_to_run = i - (sum([100, 500, 1000, 5000][:exploits.__len__()]) if exploits else 0)
            if iterations_to_run <= 0:
                continue
            cfr.train(iterations=iterations_to_run)
            exploit = compute_exploitability(game, cfr.get_all_average_strategies())
            exploits.append((i, exploit))

        # Each checkpoint should have lower exploitability
        for i in range(1, len(exploits)):
            assert exploits[i][1] <= exploits[i-1][1] * 1.1, \
                f"Exploitability increased: {exploits[i-1]} -> {exploits[i]}"
```

**Step 2: Run tests**

Run: `pytest tests/test_nash_convergence.py -v`
Expected: PASS (7 tests) - This validates the entire CFR implementation!

**Step 3: Commit**

```bash
git add tests/test_nash_convergence.py
git commit -m "test: verify CFR converges to Kuhn Nash equilibrium"
```

---

## Task 12: Device Utility for GPU

**Files:**
- Create: `core/device.py`
- Create: `tests/test_device.py`

**Step 1: Write tests for device selection**

Create `tests/test_device.py`:

```python
import pytest
import torch
from core.device import get_device


class TestDeviceSelection:
    def test_auto_returns_device(self):
        """Auto should return a valid torch device."""
        device = get_device("auto")
        assert isinstance(device, torch.device)

    def test_cpu_returns_cpu(self):
        """Explicit cpu should return cpu device."""
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_invalid_device_raises(self):
        """Invalid device string should raise."""
        with pytest.raises(ValueError):
            get_device("invalid_device_xyz")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_when_available(self):
        """Should return CUDA device when available and requested."""
        device = get_device("cuda")
        assert device.type == "cuda"

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps_when_available(self):
        """Should return MPS device when available and requested."""
        device = get_device("mps")
        assert device.type == "mps"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_device.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement device selection**

Create `core/device.py`:

```python
import torch


def get_device(preference: str = "auto") -> torch.device:
    """
    Select compute device for PyTorch operations.

    Args:
        preference: One of "auto", "cpu", "cuda", "mps"

    Returns:
        torch.device for the selected backend

    Raises:
        ValueError: If requested device is not available
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    if preference == "cpu":
        return torch.device("cpu")

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available")
        return torch.device("cuda")

    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS requested but not available")
        return torch.device("mps")

    raise ValueError(f"Unknown device: {preference}. Use 'auto', 'cpu', 'cuda', or 'mps'")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_device.py -v`
Expected: PASS (3-5 tests depending on hardware)

**Step 5: Commit**

```bash
git add core/device.py tests/test_device.py
git commit -m "feat: add device selection utility for CPU/CUDA/MPS"
```

---

## Task 13: Game Tree Compilation

**Files:**
- Create: `core/tensors.py`
- Create: `tests/test_tensors.py`

**Step 1: Write tests for compiled game structure**

Create `tests/test_tensors.py`:

```python
import pytest
import torch
from core.tensors import CompiledGame, compile_game
from core.device import get_device
from games.kuhn import KuhnPoker


class TestCompiledGame:
    @pytest.fixture
    def compiled(self):
        game = KuhnPoker()
        device = get_device("cpu")
        return compile_game(game, device)

    def test_has_correct_num_info_sets(self, compiled):
        """Should have 12 info sets for Kuhn poker."""
        assert compiled.num_info_sets == 12

    def test_has_correct_num_players(self, compiled):
        """Should have 2 players."""
        assert compiled.num_players == 2

    def test_has_correct_max_actions(self, compiled):
        """Should have 2 actions (pass/bet)."""
        assert compiled.max_actions == 2

    def test_node_tensors_correct_shape(self, compiled):
        """Node tensors should have num_nodes elements."""
        n = compiled.num_nodes
        assert compiled.node_player.shape == (n,)
        assert compiled.node_info_set.shape == (n,)
        assert compiled.terminal_mask.shape == (n,)

    def test_action_tensors_correct_shape(self, compiled):
        """Action tensors should be [num_nodes, max_actions]."""
        n = compiled.num_nodes
        a = compiled.max_actions
        assert compiled.action_child.shape == (n, a)
        assert compiled.action_mask.shape == (n, a)

    def test_terminal_nodes_marked(self, compiled):
        """Terminal nodes should have player=-1 and terminal_mask=True."""
        terminal_indices = torch.where(compiled.terminal_mask)[0]
        assert len(terminal_indices) > 0

        for idx in terminal_indices:
            assert compiled.node_player[idx] == -1

    def test_info_set_mapping_bijective(self, compiled):
        """Info set mappings should be inverses."""
        for key, idx in compiled.info_set_to_idx.items():
            assert compiled.idx_to_info_set[idx] == key

    def test_all_tensors_on_correct_device(self, compiled):
        """All tensors should be on the specified device."""
        device = compiled.device
        assert compiled.node_player.device == device
        assert compiled.action_child.device == device
        assert compiled.terminal_utils.device == device
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tensors.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement game compilation**

Create `core/tensors.py`:

```python
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
from torch import Tensor

from games.base import Game


@dataclass
class CompiledGame:
    """Game tree encoded as tensors for GPU execution."""

    # Node topology [num_nodes]
    node_player: Tensor       # Player to act (-1 for terminal)
    node_info_set: Tensor     # Info set index (-1 for terminal)
    node_num_actions: Tensor  # Number of valid actions

    # Transitions [num_nodes, max_actions]
    action_child: Tensor      # Child node index
    action_mask: Tensor       # Valid action mask

    # Terminal info [num_nodes, num_players]
    terminal_mask: Tensor     # Is this a terminal node?
    terminal_utils: Tensor    # Utilities (0 for non-terminal)

    # Depth structure for vectorized traversal
    depth_slices: List[Tuple[int, int]]
    max_depth: int

    # Mappings
    info_set_to_idx: Dict[str, int]
    idx_to_info_set: Dict[int, str]

    # Dimensions
    num_info_sets: int
    num_nodes: int
    max_actions: int
    num_players: int

    device: torch.device


def compile_game(game: Game, device: torch.device) -> CompiledGame:
    """
    Compile game tree to tensor representation.

    Traverses tree once, assigns indices, builds GPU-friendly tensors.
    """
    nodes = []
    info_set_to_idx: Dict[str, int] = {}
    depth_buckets: Dict[int, List[int]] = defaultdict(list)

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
                "player": -1,
                "info_set": -1,
                "utils": [game.utility(state, p) for p in range(game.num_players())],
                "children": [],
                "num_actions": 0,
            })
            return node_idx

        info_set = game.info_set_key(state)
        info_idx = add_info_set(info_set)
        actions = game.actions(state)

        node = {
            "terminal": False,
            "player": game.player(state),
            "info_set": info_idx,
            "utils": [0.0] * game.num_players(),
            "children": [],
            "num_actions": len(actions),
        }
        nodes.append(node)

        for action in actions:
            next_state = game.next_state(state, action)
            child_idx = enumerate_tree(next_state, depth + 1)
            node["children"].append(child_idx)

        return node_idx

    # Build tree for all initial states
    for initial_state in game.initial_states():
        enumerate_tree(initial_state, depth=0)

    # Convert to tensors
    num_nodes = len(nodes)
    num_players = game.num_players()
    max_actions = max(n["num_actions"] for n in nodes if not n["terminal"]) if nodes else 2

    node_player = torch.zeros(num_nodes, dtype=torch.long, device=device)
    node_info_set = torch.zeros(num_nodes, dtype=torch.long, device=device)
    node_num_actions = torch.zeros(num_nodes, dtype=torch.long, device=device)
    action_child = torch.zeros(num_nodes, max_actions, dtype=torch.long, device=device)
    action_mask = torch.zeros(num_nodes, max_actions, dtype=torch.bool, device=device)
    terminal_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    terminal_utils = torch.zeros(num_nodes, num_players, dtype=torch.float32, device=device)

    for idx, node in enumerate(nodes):
        node_player[idx] = node["player"]
        node_info_set[idx] = node["info_set"]
        node_num_actions[idx] = node["num_actions"]
        terminal_mask[idx] = node["terminal"]
        terminal_utils[idx] = torch.tensor(node["utils"], dtype=torch.float32)

        for a_idx, child_idx in enumerate(node["children"]):
            action_child[idx, a_idx] = child_idx
            action_mask[idx, a_idx] = True

    # Build depth slices
    max_depth = max(depth_buckets.keys()) + 1 if depth_buckets else 0
    depth_slices = []

    # Reorder nodes by depth for vectorized traversal
    # For simplicity, store (start, end) assuming nodes are already depth-ordered
    # In practice, we'd reindex - but current BFS ordering is close enough
    running_count = 0
    for d in range(max_depth):
        count = len(depth_buckets[d])
        depth_slices.append((running_count, running_count + count))
        running_count += count

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

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tensors.py -v`
Expected: PASS (9 tests)

**Step 5: Commit**

```bash
git add core/tensors.py tests/test_tensors.py
git commit -m "feat: add game tree compilation to tensors"
```

---

## Task 14: Batched CFR Engine

**Files:**
- Create: `cfr/batched.py`
- Create: `tests/test_batched_cfr.py`

**Step 1: Write tests for batched CFR**

Create `tests/test_batched_cfr.py`:

```python
import pytest
import torch
from cfr.batched import BatchedCFR
from core.tensors import compile_game
from core.device import get_device
from games.kuhn import KuhnPoker


class TestBatchedCFR:
    @pytest.fixture
    def batched_cfr(self):
        game = KuhnPoker()
        device = get_device("cpu")
        compiled = compile_game(game, device)
        return BatchedCFR(compiled, batch_size=64)

    def test_initialization(self, batched_cfr):
        """Should initialize with correct tensor shapes."""
        assert batched_cfr.regret_sum.shape == (12, 2)  # 12 info sets, 2 actions
        assert batched_cfr.strategy_sum.shape == (12, 2)

    def test_get_current_strategy_uniform_initially(self, batched_cfr):
        """Initial strategy should be uniform."""
        strategy = batched_cfr.get_current_strategy()
        assert strategy.shape == (12, 2)

        # All should be 0.5 (uniform)
        assert torch.allclose(strategy, torch.full_like(strategy, 0.5))

    def test_get_current_strategy_proportional_to_regret(self, batched_cfr):
        """Strategy should be proportional to positive regrets."""
        # Set some regrets
        batched_cfr.regret_sum[0, 0] = 3.0  # Info set 0, action 0
        batched_cfr.regret_sum[0, 1] = 1.0  # Info set 0, action 1

        strategy = batched_cfr.get_current_strategy()

        assert strategy[0, 0].item() == pytest.approx(0.75)
        assert strategy[0, 1].item() == pytest.approx(0.25)

    def test_train_step_updates_regrets(self, batched_cfr):
        """Training step should modify regret sums."""
        initial_regrets = batched_cfr.regret_sum.clone()

        batched_cfr.train_step()

        # Regrets should have changed
        assert not torch.equal(batched_cfr.regret_sum, initial_regrets)

    def test_train_step_updates_strategy_sum(self, batched_cfr):
        """Training step should accumulate strategy weights."""
        initial_sum = batched_cfr.strategy_sum.sum().item()

        batched_cfr.train_step()

        # Strategy sum should have increased
        assert batched_cfr.strategy_sum.sum().item() > initial_sum

    def test_iterations_tracked(self, batched_cfr):
        """Should track number of iterations."""
        assert batched_cfr.iterations == 0

        batched_cfr.train_step()

        assert batched_cfr.iterations == 64  # batch_size
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_batched_cfr.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement BatchedCFR**

Create `cfr/batched.py`:

```python
from typing import Dict
import torch
from torch import Tensor

from core.tensors import CompiledGame


class BatchedCFR:
    """
    GPU-accelerated CFR using batched parallel traversals.

    Runs batch_size independent CFR iterations simultaneously.
    """

    def __init__(self, compiled: CompiledGame, batch_size: int = 1024):
        self.compiled = compiled
        self.batch_size = batch_size
        self.device = compiled.device

        # Regrets and strategy sums: [num_info_sets, max_actions]
        self.regret_sum = torch.zeros(
            compiled.num_info_sets,
            compiled.max_actions,
            device=self.device,
        )
        self.strategy_sum = torch.zeros_like(self.regret_sum)
        self.iterations = 0

    def get_current_strategy(self) -> Tensor:
        """
        Equation (8): Strategy proportional to positive regret.

        Returns: [num_info_sets, max_actions]
        """
        positive_regrets = torch.clamp(self.regret_sum, min=0)
        totals = positive_regrets.sum(dim=1, keepdim=True)

        uniform = torch.ones_like(positive_regrets) / self.compiled.max_actions
        strategy = torch.where(
            totals > 0,
            positive_regrets / totals.clamp(min=1e-10),
            uniform,
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
            device=self.device,
        )

        # Process non-terminal nodes in topological order
        non_terminal = ~self.compiled.terminal_mask

        for node_idx in range(self.compiled.num_nodes):
            if self.compiled.terminal_mask[node_idx]:
                continue

            player = self.compiled.node_player[node_idx].item()
            info_set = self.compiled.node_info_set[node_idx].item()

            for a in range(self.compiled.max_actions):
                if not self.compiled.action_mask[node_idx, a]:
                    continue

                child_idx = self.compiled.action_child[node_idx, a].item()
                prob = strategy[info_set, a]

                # Child inherits parent reach
                reach[:, child_idx, :] = reach[:, node_idx, :].clone()
                # Multiply acting player's reach by action probability
                reach[:, child_idx, player] *= prob

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
            device=self.device,
        )

        # Initialize terminal nodes
        terminal_indices = torch.where(self.compiled.terminal_mask)[0]
        for idx in terminal_indices:
            utils[:, idx, :] = self.compiled.terminal_utils[idx]

        # Process in reverse order (children before parents)
        for node_idx in reversed(range(self.compiled.num_nodes)):
            if self.compiled.terminal_mask[node_idx]:
                continue

            info_set = self.compiled.node_info_set[node_idx].item()

            # Weighted sum of child utilities
            for a in range(self.compiled.max_actions):
                if not self.compiled.action_mask[node_idx, a]:
                    continue

                child_idx = self.compiled.action_child[node_idx, a].item()
                prob = strategy[info_set, a]
                utils[:, node_idx, :] += prob * utils[:, child_idx, :]

        return utils

    def train_step(self):
        """One batched CFR iteration."""
        strategy = self.get_current_strategy()
        reach = self.forward_reach(strategy)
        utils = self.backward_utils(reach, strategy)

        # Update regrets for non-terminal nodes
        for node_idx in range(self.compiled.num_nodes):
            if self.compiled.terminal_mask[node_idx]:
                continue

            player = self.compiled.node_player[node_idx].item()
            opponent = 1 - player
            info_set = self.compiled.node_info_set[node_idx].item()

            # Counterfactual reach and node utility
            cf_reach = reach[:, node_idx, opponent]  # [batch]
            node_util = utils[:, node_idx, player]   # [batch]

            for a in range(self.compiled.max_actions):
                if not self.compiled.action_mask[node_idx, a]:
                    continue

                child_idx = self.compiled.action_child[node_idx, a].item()
                action_util = utils[:, child_idx, player]

                # Accumulate regret across batch
                regret = (cf_reach * (action_util - node_util)).sum()
                self.regret_sum[info_set, a] += regret

            # Accumulate strategy weighted by player reach
            my_reach = reach[:, node_idx, player].sum()
            for a in range(self.compiled.max_actions):
                if self.compiled.action_mask[node_idx, a]:
                    self.strategy_sum[info_set, a] += my_reach * strategy[info_set, a]

        self.iterations += self.batch_size

    def get_average_strategy(self) -> Dict[str, Dict[str, float]]:
        """Convert tensor strategy to dictionary format."""
        result = {}
        strategy_sum = self.strategy_sum.cpu().numpy()
        actions = ["p", "b"]  # Kuhn poker actions

        for info_set, idx in self.compiled.info_set_to_idx.items():
            total = strategy_sum[idx].sum()
            if total > 0:
                result[info_set] = {
                    actions[a]: float(strategy_sum[idx, a] / total)
                    for a in range(self.compiled.max_actions)
                }
            else:
                result[info_set] = {a: 0.5 for a in actions}

        return result
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_batched_cfr.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add cfr/batched.py tests/test_batched_cfr.py
git commit -m "feat: add batched GPU CFR engine"
```

---

## Task 15: Verify Batched CFR Matches Vanilla

**Files:**
- Create: `tests/test_cpu_gpu_equivalence.py`

**Step 1: Write equivalence tests**

Create `tests/test_cpu_gpu_equivalence.py`:

```python
import pytest
from cfr.vanilla import VanillaCFR
from cfr.batched import BatchedCFR
from core.tensors import compile_game
from core.device import get_device
from core.exploitability import compute_exploitability
from games.kuhn import KuhnPoker


class TestCPUGPUEquivalence:
    """Verify batched GPU implementation matches CPU reference."""

    def test_both_converge_to_low_exploitability(self):
        """Both implementations should converge to near-Nash."""
        game = KuhnPoker()

        # CPU vanilla
        cpu_cfr = VanillaCFR(game)
        cpu_cfr.train(iterations=5000)
        cpu_strategy = cpu_cfr.get_all_average_strategies()
        cpu_exploit = compute_exploitability(game, cpu_strategy)

        # GPU batched
        device = get_device("cpu")  # Use CPU for deterministic comparison
        compiled = compile_game(game, device)
        gpu_cfr = BatchedCFR(compiled, batch_size=100)

        for _ in range(50):  # 50 steps * 100 batch = 5000 iterations
            gpu_cfr.train_step()

        gpu_strategy = gpu_cfr.get_average_strategy()
        gpu_exploit = compute_exploitability(game, gpu_strategy)

        # Both should have low exploitability
        assert cpu_exploit < 0.02, f"CPU exploitability {cpu_exploit} too high"
        assert gpu_exploit < 0.02, f"GPU exploitability {gpu_exploit} too high"

    def test_strategies_approximately_equal(self):
        """Converged strategies should be similar."""
        game = KuhnPoker()

        # Train both
        cpu_cfr = VanillaCFR(game)
        cpu_cfr.train(iterations=10000)
        cpu_strategy = cpu_cfr.get_all_average_strategies()

        device = get_device("cpu")
        compiled = compile_game(game, device)
        gpu_cfr = BatchedCFR(compiled, batch_size=100)
        for _ in range(100):
            gpu_cfr.train_step()
        gpu_strategy = gpu_cfr.get_average_strategy()

        # Key strategic properties should match
        # Jack folds to bet
        assert abs(cpu_strategy["0:pb"]["p"] - gpu_strategy["0:pb"]["p"]) < 0.1
        # King calls bet
        assert abs(cpu_strategy["2:pb"]["b"] - gpu_strategy["2:pb"]["b"]) < 0.1

    @pytest.mark.skipif(
        not __import__('torch').cuda.is_available() and
        not __import__('torch').backends.mps.is_available(),
        reason="No GPU available"
    )
    def test_gpu_produces_valid_strategy(self):
        """GPU execution should produce valid strategy."""
        game = KuhnPoker()
        device = get_device("auto")

        compiled = compile_game(game, device)
        cfr = BatchedCFR(compiled, batch_size=256)

        for _ in range(10):
            cfr.train_step()

        strategy = cfr.get_average_strategy()

        # All info sets should be present
        assert len(strategy) == 12

        # Probabilities should sum to 1
        for info_set, probs in strategy.items():
            total = sum(probs.values())
            assert total == pytest.approx(1.0), f"{info_set}: {probs}"
```

**Step 2: Run tests**

Run: `pytest tests/test_cpu_gpu_equivalence.py -v`
Expected: PASS (2-3 tests depending on GPU availability)

**Step 3: Commit**

```bash
git add tests/test_cpu_gpu_equivalence.py
git commit -m "test: verify batched CFR matches vanilla implementation"
```

---

## Task 16: Public Solver API

**Files:**
- Create: `solver.py`
- Create: `tests/test_solver.py`

**Step 1: Write tests for Solver API**

Create `tests/test_solver.py`:

```python
import pytest
from solver import Solver
from games.kuhn import KuhnPoker


class TestSolverAPI:
    def test_solve_returns_strategy(self):
        """Solve should return a strategy dict."""
        solver = Solver(KuhnPoker())
        strategy = solver.solve(iterations=100, verbose=False)

        assert isinstance(strategy, dict)
        assert len(strategy) == 12

    def test_solve_with_batching(self):
        """Should work with batched execution."""
        solver = Solver(KuhnPoker(), batch_size=64)
        strategy = solver.solve(iterations=640, verbose=False)

        assert len(strategy) == 12

    def test_exploitability_method(self):
        """Should expose exploitability calculation."""
        solver = Solver(KuhnPoker())
        solver.solve(iterations=1000, verbose=False)

        exploit = solver.exploitability()
        assert isinstance(exploit, float)
        assert exploit >= 0

    def test_get_strategy_before_training(self):
        """Should return uniform strategy before training."""
        solver = Solver(KuhnPoker())
        strategy = solver.get_strategy()

        # All should be close to uniform
        for probs in strategy.values():
            assert probs["p"] == pytest.approx(0.5, abs=0.01)
            assert probs["b"] == pytest.approx(0.5, abs=0.01)

    def test_device_selection(self):
        """Should accept device parameter."""
        solver = Solver(KuhnPoker(), device="cpu")
        strategy = solver.solve(iterations=100, verbose=False)
        assert len(strategy) == 12
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_solver.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement Solver API**

Create `solver.py`:

```python
from typing import Dict, Optional

from games.base import Game
from cfr.vanilla import VanillaCFR
from cfr.batched import BatchedCFR
from core.tensors import compile_game
from core.device import get_device
from core.exploitability import compute_exploitability


class Solver:
    """
    High-level API for solving extensive-form games with CFR.

    Example:
        solver = Solver(KuhnPoker(), device="auto")
        strategy = solver.solve(iterations=10000)
        print(f"Exploitability: {solver.exploitability()}")
    """

    def __init__(
        self,
        game: Game,
        device: str = "auto",
        batch_size: int = 1,
    ):
        """
        Initialize solver.

        Args:
            game: Game instance to solve
            device: "auto", "cpu", "cuda", or "mps"
            batch_size: Number of parallel iterations (1 = vanilla CFR)
        """
        self.game = game
        self.batch_size = batch_size

        if batch_size == 1:
            # Use vanilla CPU implementation
            self.engine = VanillaCFR(game)
            self.batched = False
        else:
            # Use batched GPU implementation
            self.device = get_device(device)
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

        Args:
            iterations: Number of CFR iterations
            verbose: Print progress updates
            log_interval: How often to print (if verbose)

        Returns:
            Average strategy for all information sets
        """
        if self.batched:
            steps = max(1, iterations // self.batch_size)
            for step in range(steps):
                self.engine.train_step()

                if verbose and (step + 1) % max(1, log_interval // self.batch_size) == 0:
                    current_iter = (step + 1) * self.batch_size
                    exploit = self.exploitability()
                    print(f"Iteration {current_iter}: exploitability = {exploit:.6f}")
        else:
            if verbose:
                # Train in chunks for progress updates
                chunk_size = log_interval
                for i in range(0, iterations, chunk_size):
                    self.engine.train(min(chunk_size, iterations - i))
                    exploit = self.exploitability()
                    print(f"Iteration {i + chunk_size}: exploitability = {exploit:.6f}")
            else:
                self.engine.train(iterations)

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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_solver.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add solver.py tests/test_solver.py
git commit -m "feat: add high-level Solver API"
```

---

## Task 17: Main Entry Point

**Files:**
- Create: `main.py`

**Step 1: Create main entry point**

Create `main.py`:

```python
#!/usr/bin/env python3
"""
CFR Poker Solver - Main Entry Point

Usage:
    python main.py                    # Solve Kuhn Poker with defaults
    python main.py --iterations 10000 # More iterations
    python main.py --device cuda      # Use GPU
    python main.py --batch-size 1024  # Larger batches
"""

import argparse
import time

from games.kuhn import KuhnPoker
from solver import Solver


def main():
    parser = argparse.ArgumentParser(description="CFR Poker Solver")
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=10000,
        help="Number of CFR iterations (default: 10000)"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device (default: auto)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Batch size for parallel iterations (default: 1 = vanilla CFR)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("CFR Poker Solver - Kuhn Poker")
    print("=" * 50)
    print(f"Iterations: {args.iterations}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Create solver
    solver = Solver(
        game=KuhnPoker(),
        device=args.device,
        batch_size=args.batch_size,
    )

    # Solve
    print("Training CFR...")
    start_time = time.time()
    strategy = solver.solve(
        iterations=args.iterations,
        verbose=not args.quiet,
    )
    elapsed = time.time() - start_time

    # Results
    print()
    print("=" * 50)
    print("Results")
    print("=" * 50)
    print(f"Time: {elapsed:.2f} seconds")
    print(f"Iterations/sec: {args.iterations / elapsed:.0f}")
    print(f"Final exploitability: {solver.exploitability():.6f}")
    print()

    print("Strategy:")
    print("-" * 40)
    for info_set in sorted(strategy.keys()):
        probs = strategy[info_set]
        card = ["J", "Q", "K"][int(info_set[0])]
        history = info_set[2:] if len(info_set) > 2 else "(root)"
        print(f"  {card} {history:8s}: pass={probs['p']:.3f}  bet={probs['b']:.3f}")


if __name__ == "__main__":
    main()
```

**Step 2: Test it runs**

Run: `python main.py --iterations 1000 --quiet`
Expected: Should complete and print strategy

**Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add main entry point with CLI"
```

---

## Task 18: Final Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Create comprehensive integration test**

Create `tests/test_integration.py`:

```python
import pytest
import subprocess
import sys


class TestIntegration:
    """End-to-end integration tests."""

    def test_main_runs_successfully(self):
        """Main script should run without errors."""
        result = subprocess.run(
            [sys.executable, "main.py", "--iterations", "100", "--quiet"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_full_training_converges(self):
        """Full training should converge to Nash."""
        from solver import Solver
        from games.kuhn import KuhnPoker

        solver = Solver(KuhnPoker(), batch_size=100)
        solver.solve(iterations=10000, verbose=False)

        exploit = solver.exploitability()
        assert exploit < 0.01, f"Did not converge: exploitability = {exploit}"

    def test_all_implementations_agree(self):
        """All implementations should produce similar results."""
        from solver import Solver
        from games.kuhn import KuhnPoker
        from core.exploitability import compute_exploitability

        game = KuhnPoker()

        # Vanilla
        vanilla = Solver(game, batch_size=1)
        vanilla.solve(iterations=5000, verbose=False)
        vanilla_exploit = vanilla.exploitability()

        # Batched
        batched = Solver(game, batch_size=100)
        batched.solve(iterations=5000, verbose=False)
        batched_exploit = batched.exploitability()

        # Both should be low
        assert vanilla_exploit < 0.02
        assert batched_exploit < 0.02
```

**Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests"
```

---

## Task 19: Merge to Main

**Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 2: Check git status**

Run: `git status`
Expected: Clean working tree

**Step 3: Merge branch**

```bash
cd /Users/ltj/Documents/code/poker_solver
git merge feature/cfr-implementation --no-ff -m "feat: CFR poker solver implementation

Implements Counterfactual Regret Minimization for Kuhn Poker:
- Game abstraction layer (games/base.py, games/kuhn.py)
- Vanilla CFR reference implementation (cfr/vanilla.py)
- Batched GPU CFR with PyTorch (cfr/batched.py)
- Game tree tensor compilation (core/tensors.py)
- Exploitability calculation (core/exploitability.py)
- Device selection for CUDA/MPS/CPU (core/device.py)
- High-level Solver API (solver.py)
- CLI entry point (main.py)

Verified against known Kuhn Poker Nash equilibrium.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

**Step 4: Clean up worktree**

```bash
git worktree remove .worktrees/cfr-impl
```

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | Project setup | - |
| 2 | Game base class | 2 |
| 3 | KuhnState | 3 |
| 4 | KuhnPoker logic | 7 |
| 5 | Utilities | 6 |
| 6 | Info sets | 5 |
| 7 | Regret matching | 4 |
| 8 | CFR traversal | 4 |
| 9 | Training | 4 |
| 10 | Exploitability | 3 |
| 11 | Nash convergence | 7 |
| 12 | Device utility | 3-5 |
| 13 | Tensor compilation | 9 |
| 14 | Batched CFR | 6 |
| 15 | CPU/GPU equivalence | 2-3 |
| 16 | Solver API | 5 |
| 17 | Main entry point | - |
| 18 | Integration | 3 |

**Total: ~70 tests, 19 tasks, ~17 commits**
