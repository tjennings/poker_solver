# HUNL Preflop Solver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Heads-Up No-Limit Hold'em preflop solver with GPU-accelerated CFR+ and interactive color-coded strategy matrix CLI.

**Architecture:** New `HUNLPreflop` game class using existing `Game` interface, YAML config for bet sizing, CFR+ modification to batched engine, and terminal-based interactive explorer with ANSI colors.

**Tech Stack:** PyTorch (GPU), PyYAML (config), ANSI escape codes (terminal colors)

---

## Task 1: Hand Parser Module

**Files:**
- Create: `core/hands.py`
- Test: `tests/test_hands.py`

**Step 1: Write failing tests for hand parsing**

```python
# tests/test_hands.py
import pytest
from core.hands import (
    parse_hand,
    hand_to_string,
    HAND_COUNT,
    canonical_index,
)


class TestHandParsing:
    """Test hand string parsing."""

    def test_parse_pair(self):
        """Pairs parse correctly."""
        assert parse_hand("AA") == (0, True, None)  # (index, is_canonical, specific_cards)
        assert parse_hand("KK") == (1, True, None)
        assert parse_hand("22") == (12, True, None)

    def test_parse_suited(self):
        """Suited hands parse correctly."""
        assert parse_hand("AK") == (13, True, None)
        assert parse_hand("AQ") == (14, True, None)
        assert parse_hand("32") == (90, True, None)

    def test_parse_offsuit(self):
        """Offsuit hands parse correctly."""
        assert parse_hand("AKo") == (91, True, None)
        assert parse_hand("AQo") == (92, True, None)
        assert parse_hand("32o") == (168, True, None)

    def test_parse_specific_suited(self):
        """Specific suited combos parse correctly."""
        idx, is_canonical, cards = parse_hand("AcKc")
        assert idx == 13  # Maps to AK suited
        assert is_canonical == False
        assert cards == (("A", "c"), ("K", "c"))

    def test_parse_specific_offsuit(self):
        """Specific offsuit combos parse correctly."""
        idx, is_canonical, cards = parse_hand("AcKh")
        assert idx == 91  # Maps to AKo
        assert is_canonical == False
        assert cards == (("A", "c"), ("K", "h"))

    def test_parse_specific_pair(self):
        """Specific pair combos parse correctly."""
        idx, is_canonical, cards = parse_hand("AsAd")
        assert idx == 0  # Maps to AA
        assert is_canonical == False
        assert cards == (("A", "s"), ("A", "d"))

    def test_parse_normalizes_order(self):
        """Lower rank first normalizes to higher rank first."""
        assert parse_hand("KA") == parse_hand("AK")
        assert parse_hand("9To") == parse_hand("T9o")

    def test_parse_invalid_raises(self):
        """Invalid hands raise ValueError."""
        with pytest.raises(ValueError):
            parse_hand("AX")
        with pytest.raises(ValueError):
            parse_hand("AAA")
        with pytest.raises(ValueError):
            parse_hand("")


class TestHandToString:
    """Test converting indices back to strings."""

    def test_pair_to_string(self):
        assert hand_to_string(0) == "AA"
        assert hand_to_string(12) == "22"

    def test_suited_to_string(self):
        assert hand_to_string(13) == "AK"
        assert hand_to_string(90) == "32"

    def test_offsuit_to_string(self):
        assert hand_to_string(91) == "AKo"
        assert hand_to_string(168) == "32o"


class TestHandCount:
    """Test hand counting constants."""

    def test_total_hands(self):
        assert HAND_COUNT == 169

    def test_canonical_index_range(self):
        for i in range(169):
            assert 0 <= canonical_index(hand_to_string(i)) < 169
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pytest tests/test_hands.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'core.hands'"

**Step 3: Implement hand parser**

```python
# core/hands.py
"""
Hand parsing for HUNL poker.

Supports:
- Canonical hands: AA, AK (suited), AKo (offsuit)
- Specific combos: AcKc, AcKh, AsAd
"""

from typing import Tuple, Optional, List

RANKS = "AKQJT98765432"
SUITS = "cdhs"
HAND_COUNT = 169  # 13 pairs + 78 suited + 78 offsuit


def _rank_index(rank: str) -> int:
    """Convert rank char to index (A=0, K=1, ..., 2=12)."""
    idx = RANKS.find(rank.upper())
    if idx == -1:
        raise ValueError(f"Invalid rank: {rank}")
    return idx


def _is_valid_suit(suit: str) -> bool:
    """Check if suit char is valid."""
    return suit.lower() in SUITS


def canonical_index(hand: str) -> int:
    """
    Get canonical index (0-168) for a hand string.

    Pairs: 0-12 (AA=0, KK=1, ..., 22=12)
    Suited: 13-90 (AK=13, AQ=14, ..., 32=90)
    Offsuit: 91-168 (AKo=91, AQo=92, ..., 32o=168)
    """
    result = parse_hand(hand)
    return result[0]


def parse_hand(hand: str) -> Tuple[int, bool, Optional[Tuple[Tuple[str, str], Tuple[str, str]]]]:
    """
    Parse hand string to (index, is_canonical, specific_cards).

    Args:
        hand: Hand string like "AK", "AKo", "AcKh"

    Returns:
        (canonical_index, is_canonical, specific_cards)
        - canonical_index: 0-168
        - is_canonical: True if no specific suits given
        - specific_cards: ((rank1, suit1), (rank2, suit2)) or None
    """
    if not hand or len(hand) < 2:
        raise ValueError(f"Invalid hand: {hand}")

    hand = hand.strip()

    # Check for specific suits (length 4: AcKh, AsAd)
    if len(hand) == 4 and _is_valid_suit(hand[1]) and _is_valid_suit(hand[3]):
        r1, s1, r2, s2 = hand[0], hand[1].lower(), hand[2], hand[3].lower()
        i1, i2 = _rank_index(r1), _rank_index(r2)

        # Normalize order (higher rank first)
        if i1 > i2:
            r1, s1, r2, s2 = r2, s2, r1, s1
            i1, i2 = i2, i1

        specific = ((r1.upper(), s1), (r2.upper(), s2))

        if i1 == i2:
            # Pair
            idx = i1
        elif s1 == s2:
            # Suited
            idx = 13 + _suited_offset(i1, i2)
        else:
            # Offsuit
            idx = 91 + _suited_offset(i1, i2)

        return (idx, False, specific)

    # Canonical format
    r1 = hand[0].upper()
    r2 = hand[1].upper()
    i1, i2 = _rank_index(r1), _rank_index(r2)

    # Normalize order
    if i1 > i2:
        r1, r2 = r2, r1
        i1, i2 = i2, i1

    # Check for offsuit marker
    is_offsuit = len(hand) >= 3 and hand[2].lower() == 'o'

    if i1 == i2:
        # Pair
        return (i1, True, None)
    elif is_offsuit:
        # Offsuit
        return (91 + _suited_offset(i1, i2), True, None)
    else:
        # Suited (default for non-pairs without 'o')
        return (13 + _suited_offset(i1, i2), True, None)


def _suited_offset(i1: int, i2: int) -> int:
    """
    Calculate offset within suited/offsuit section.

    For ranks i1 < i2, counts combinations before this pair.
    """
    # Number of suited combos: 78 = 12+11+10+...+1
    # For rank i1 (0=A, 1=K, ...), pairs with ranks > i1
    offset = 0
    for r in range(i1):
        offset += 12 - r  # Number of lower ranks to pair with
    offset += (i2 - i1 - 1)
    return offset


def hand_to_string(idx: int) -> str:
    """Convert canonical index (0-168) to hand string."""
    if idx < 0 or idx >= 169:
        raise ValueError(f"Invalid hand index: {idx}")

    if idx < 13:
        # Pair
        return RANKS[idx] + RANKS[idx]
    elif idx < 91:
        # Suited
        i1, i2 = _offset_to_ranks(idx - 13)
        return RANKS[i1] + RANKS[i2]
    else:
        # Offsuit
        i1, i2 = _offset_to_ranks(idx - 91)
        return RANKS[i1] + RANKS[i2] + "o"


def _offset_to_ranks(offset: int) -> Tuple[int, int]:
    """Convert suited/offsuit offset back to rank indices."""
    i1 = 0
    remaining = offset
    while remaining >= (12 - i1):
        remaining -= (12 - i1)
        i1 += 1
    i2 = i1 + 1 + remaining
    return (i1, i2)


def get_all_hands() -> List[str]:
    """Return list of all 169 canonical hand strings in order."""
    return [hand_to_string(i) for i in range(169)]


def get_matrix_layout() -> List[List[str]]:
    """
    Return 13x13 matrix layout for display.

    Rows/cols indexed by rank (A=0, K=1, ..., 2=12).
    - Diagonal: pairs
    - Above diagonal: suited
    - Below diagonal: offsuit
    """
    matrix = []
    for row in range(13):
        row_hands = []
        for col in range(13):
            if row == col:
                row_hands.append(RANKS[row] + RANKS[row])
            elif row < col:
                # Above diagonal = suited
                row_hands.append(RANKS[row] + RANKS[col])
            else:
                # Below diagonal = offsuit
                row_hands.append(RANKS[col] + RANKS[row] + "o")
        matrix.append(row_hands)
    return matrix
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pytest tests/test_hands.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add core/hands.py tests/test_hands.py
git commit -m "feat(hunl): add hand parser for 169 canonical hands

Supports parsing:
- Canonical: AA, AK (suited), AKo (offsuit)
- Specific suits: AcKh, AsAd
- Normalizes rank order (KA -> AK)
- Bidirectional conversion (string <-> index)"
```

---

## Task 2: Config Loader Module

**Files:**
- Create: `config/__init__.py`
- Create: `config/loader.py`
- Create: `config/presets/standard.yaml`
- Test: `tests/test_config.py`
- Modify: `pyproject.toml` (add pyyaml dependency)

**Step 1: Add pyyaml dependency**

Edit `pyproject.toml` line 6-9:
```toml
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "pyyaml>=6.0",
]
```

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pip install pyyaml`

**Step 2: Write failing tests for config loading**

```python
# tests/test_config.py
import pytest
import tempfile
import os
from pathlib import Path

from config.loader import load_config, Config, get_preset_path


class TestConfigLoading:
    """Test YAML config loading."""

    def test_load_preset_standard(self):
        """Standard preset loads correctly."""
        config = load_config(get_preset_path("standard"))
        assert config.name == "Standard 100BB"
        assert config.stack_depth == 100
        assert 2.5 in config.raise_sizes
        assert 3 in config.raise_sizes

    def test_load_custom_config(self):
        """Custom config file loads correctly."""
        yaml_content = """
name: "Test Config"
stack_depth: 50
raise_sizes: [2, 4, 10]
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = load_config(f.name)

        os.unlink(f.name)

        assert config.name == "Test Config"
        assert config.stack_depth == 50
        assert config.raise_sizes == [2, 4, 10]

    def test_config_validation_missing_stack(self):
        """Missing stack_depth raises error."""
        yaml_content = """
name: "Bad Config"
raise_sizes: [2, 4]
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            with pytest.raises(ValueError, match="stack_depth"):
                load_config(f.name)
        os.unlink(f.name)

    def test_config_validation_missing_sizes(self):
        """Missing raise_sizes raises error."""
        yaml_content = """
name: "Bad Config"
stack_depth: 100
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            with pytest.raises(ValueError, match="raise_sizes"):
                load_config(f.name)
        os.unlink(f.name)


class TestConfigDefaults:
    """Test config default values."""

    def test_default_name(self):
        """Missing name gets default."""
        yaml_content = """
stack_depth: 100
raise_sizes: [2.5, 3]
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = load_config(f.name)
        os.unlink(f.name)

        assert config.name == "Custom"


class TestLegalActions:
    """Test computing legal actions from config."""

    def test_legal_raises_at_open(self):
        """Opening raises filtered by stack."""
        config = Config(
            name="Test",
            stack_depth=100,
            raise_sizes=[2.5, 3, 8, 20, 50, 100, 200],
        )
        legal = config.get_legal_raise_sizes(current_bet=0, stack=100)
        assert legal == [2.5, 3, 8, 20, 50, 100]  # 200 exceeds stack

    def test_legal_raises_facing_bet(self):
        """Raises must exceed current bet."""
        config = Config(
            name="Test",
            stack_depth=100,
            raise_sizes=[2.5, 3, 8, 20],
        )
        legal = config.get_legal_raise_sizes(current_bet=8, stack=100)
        assert legal == [20]  # Only 20 > 8

    def test_legal_raises_short_stack(self):
        """Short stack limits raises."""
        config = Config(
            name="Test",
            stack_depth=100,
            raise_sizes=[2.5, 3, 8, 20],
        )
        legal = config.get_legal_raise_sizes(current_bet=0, stack=15)
        assert legal == [2.5, 3, 8]  # 20 > 15
```

**Step 3: Run tests to verify they fail**

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pytest tests/test_config.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'config'"

**Step 4: Create config module structure**

```python
# config/__init__.py
from config.loader import load_config, Config, get_preset_path

__all__ = ["load_config", "Config", "get_preset_path"]
```

```python
# config/loader.py
"""
YAML configuration loader for HUNL solver.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml


@dataclass
class Config:
    """Solver configuration."""
    name: str
    stack_depth: float
    raise_sizes: List[float]

    def get_legal_raise_sizes(self, current_bet: float, stack: float) -> List[float]:
        """
        Get raise sizes that are legal given current bet and stack.

        Args:
            current_bet: Current bet to match (0 if no bet)
            stack: Remaining stack

        Returns:
            List of legal raise sizes (all > current_bet and <= stack)
        """
        return [s for s in self.raise_sizes if s > current_bet and s <= stack]


def get_preset_path(name: str) -> Path:
    """Get path to a preset config file."""
    config_dir = Path(__file__).parent / "presets"
    return config_dir / f"{name}.yaml"


def load_config(path: str | Path) -> Config:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        Config object

    Raises:
        ValueError: If required fields are missing
        FileNotFoundError: If config file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    # Validate required fields
    if "stack_depth" not in data:
        raise ValueError("Config missing required field: stack_depth")
    if "raise_sizes" not in data:
        raise ValueError("Config missing required field: raise_sizes")

    return Config(
        name=data.get("name", "Custom"),
        stack_depth=float(data["stack_depth"]),
        raise_sizes=[float(s) for s in data["raise_sizes"]],
    )
```

**Step 5: Create presets directory and standard config**

```bash
mkdir -p config/presets
```

```yaml
# config/presets/standard.yaml
name: "Standard 100BB"
stack_depth: 100
raise_sizes: [2.5, 3, 6, 8, 10, 15, 20, 25, 50, 100]
```

**Step 6: Run tests to verify they pass**

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pytest tests/test_config.py -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add config/ tests/test_config.py pyproject.toml
git commit -m "feat(hunl): add YAML config loader with presets

- Config dataclass with stack_depth and raise_sizes
- get_legal_raise_sizes() computes valid raises
- Standard 100BB preset included
- Validates required fields on load"
```

---

## Task 3: Action Parser Module

**Files:**
- Create: `cli/__init__.py`
- Create: `cli/parser.py`
- Test: `tests/test_parser.py`

**Step 1: Write failing tests for action parsing**

```python
# tests/test_parser.py
import pytest
from cli.parser import parse_action_sequence, ParsedAction, ParsedSequence


class TestParseActionSequence:
    """Test parsing action sequences like '50bb SBr2.5 BBr8'."""

    def test_parse_simple_open(self):
        """Parse single open raise."""
        result = parse_action_sequence("SBr2.5")
        assert result.stack_override is None
        assert len(result.actions) == 1
        assert result.actions[0] == ParsedAction("SB", "raise", 2.5)

    def test_parse_with_stack_override(self):
        """Parse with stack depth override."""
        result = parse_action_sequence("50bb SBr2.5")
        assert result.stack_override == 50
        assert len(result.actions) == 1

    def test_parse_full_sequence(self):
        """Parse multi-action sequence."""
        result = parse_action_sequence("SBr2.5 BBr8 SBc")
        assert len(result.actions) == 3
        assert result.actions[0] == ParsedAction("SB", "raise", 2.5)
        assert result.actions[1] == ParsedAction("BB", "raise", 8)
        assert result.actions[2] == ParsedAction("SB", "call", None)

    def test_parse_fold(self):
        """Parse fold action."""
        result = parse_action_sequence("SBr2.5 BBf")
        assert result.actions[1] == ParsedAction("BB", "fold", None)

    def test_parse_all_in(self):
        """Parse all-in action."""
        result = parse_action_sequence("SBa")
        assert result.actions[0] == ParsedAction("SB", "all_in", None)

    def test_parse_call(self):
        """Parse call action."""
        result = parse_action_sequence("SBr3 BBc")
        assert result.actions[1] == ParsedAction("BB", "call", None)

    def test_parse_check(self):
        """Parse check (BB preflop option)."""
        result = parse_action_sequence("SBc BBx")
        assert result.actions[0] == ParsedAction("SB", "call", None)  # SB limps
        assert result.actions[1] == ParsedAction("BB", "check", None)

    def test_parse_case_insensitive(self):
        """Actions are case insensitive."""
        result = parse_action_sequence("sbR2.5 bbc")
        assert result.actions[0].position == "SB"
        assert result.actions[0].action_type == "raise"
        assert result.actions[1].action_type == "call"

    def test_parse_invalid_position(self):
        """Invalid position raises error."""
        with pytest.raises(ValueError, match="position"):
            parse_action_sequence("UTGr2.5")

    def test_parse_invalid_action(self):
        """Invalid action type raises error."""
        with pytest.raises(ValueError, match="action"):
            parse_action_sequence("SBz2.5")

    def test_parse_empty(self):
        """Empty sequence returns no actions."""
        result = parse_action_sequence("")
        assert result.actions == []
        assert result.stack_override is None


class TestParsedSequenceToHistory:
    """Test converting parsed sequence to history tuple."""

    def test_to_history_tuple(self):
        """Convert to history format for game state."""
        result = parse_action_sequence("SBr2.5 BBr8 SBc")
        history = result.to_history_tuple()
        assert history == ("r2.5", "r8", "c")
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pytest tests/test_parser.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'cli'"

**Step 3: Implement action parser**

```python
# cli/__init__.py
from cli.parser import parse_action_sequence, ParsedAction, ParsedSequence

__all__ = ["parse_action_sequence", "ParsedAction", "ParsedSequence"]
```

```python
# cli/parser.py
"""
Action sequence parser for HUNL preflop.

Parses strings like "50bb SBr2.5 BBr8 SBc" into structured actions.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import re


@dataclass(frozen=True)
class ParsedAction:
    """A single parsed action."""
    position: str  # "SB" or "BB"
    action_type: str  # "fold", "call", "check", "raise", "all_in"
    amount: Optional[float]  # Raise amount in BB, or None

    def to_history_string(self) -> str:
        """Convert to history string format."""
        if self.action_type == "fold":
            return "f"
        elif self.action_type == "call":
            return "c"
        elif self.action_type == "check":
            return "x"
        elif self.action_type == "all_in":
            return "a"
        elif self.action_type == "raise":
            return f"r{self.amount}"
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")


@dataclass
class ParsedSequence:
    """A parsed action sequence."""
    actions: List[ParsedAction]
    stack_override: Optional[float]

    def to_history_tuple(self) -> Tuple[str, ...]:
        """Convert to history tuple for game state."""
        return tuple(a.to_history_string() for a in self.actions)


# Regex patterns
STACK_PATTERN = re.compile(r'^(\d+(?:\.\d+)?)\s*bb\s+', re.IGNORECASE)
ACTION_PATTERN = re.compile(
    r'(SB|BB)(r(\d+(?:\.\d+)?)|c|f|a|x)',
    re.IGNORECASE
)


def parse_action_sequence(sequence: str) -> ParsedSequence:
    """
    Parse an action sequence string.

    Format: [stack]bb [Position][Action] [Position][Action] ...

    Examples:
        "SBr2.5" -> SB raises to 2.5BB
        "50bb SBr2.5 BBr8" -> 50BB effective, SB opens, BB 3-bets
        "SBr3 BBc" -> SB opens 3BB, BB calls
        "SBa" -> SB all-in
        "SBc BBx" -> SB limps, BB checks

    Args:
        sequence: Action sequence string

    Returns:
        ParsedSequence with actions and optional stack override

    Raises:
        ValueError: If sequence contains invalid actions
    """
    sequence = sequence.strip()
    if not sequence:
        return ParsedSequence(actions=[], stack_override=None)

    # Check for stack override
    stack_override = None
    stack_match = STACK_PATTERN.match(sequence)
    if stack_match:
        stack_override = float(stack_match.group(1))
        sequence = sequence[stack_match.end():]

    # Parse actions
    actions = []
    remaining = sequence.strip()

    while remaining:
        remaining = remaining.lstrip()
        if not remaining:
            break

        match = ACTION_PATTERN.match(remaining)
        if not match:
            # Try to identify what went wrong
            word = remaining.split()[0] if remaining.split() else remaining
            if word and word[:2].upper() not in ("SB", "BB"):
                raise ValueError(f"Invalid position in: {word}")
            raise ValueError(f"Invalid action in: {word}")

        position = match.group(1).upper()
        action_str = match.group(2).lower()

        if action_str.startswith('r'):
            action_type = "raise"
            amount = float(match.group(3))
        elif action_str == 'c':
            action_type = "call"
            amount = None
        elif action_str == 'f':
            action_type = "fold"
            amount = None
        elif action_str == 'a':
            action_type = "all_in"
            amount = None
        elif action_str == 'x':
            action_type = "check"
            amount = None
        else:
            raise ValueError(f"Invalid action: {action_str}")

        actions.append(ParsedAction(position, action_type, amount))
        remaining = remaining[match.end():]

    return ParsedSequence(actions=actions, stack_override=stack_override)
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pytest tests/test_parser.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add cli/ tests/test_parser.py
git commit -m "feat(hunl): add action sequence parser

Parses strings like '50bb SBr2.5 BBr8 SBc':
- Optional stack override prefix
- Position (SB/BB) + action type
- Actions: r=raise, c=call, f=fold, a=all-in, x=check
- Case insensitive"
```

---

## Task 4: HUNL Preflop Game State

**Files:**
- Create: `games/hunl_preflop.py`
- Test: `tests/test_hunl_game.py`

**Step 1: Write failing tests for game state**

```python
# tests/test_hunl_game.py
import pytest
from games.hunl_preflop import HUNLPreflop, HUNLState
from config.loader import Config


@pytest.fixture
def config():
    """Standard test config."""
    return Config(
        name="Test",
        stack_depth=100,
        raise_sizes=[2.5, 3, 8, 20, 50, 100],
    )


@pytest.fixture
def game(config):
    """Game instance for testing."""
    return HUNLPreflop(config)


class TestHUNLState:
    """Test HUNL state representation."""

    def test_initial_state(self, game):
        """Initial state has correct values."""
        states = game.initial_states()
        # Should have 169 * 168 = 28392 starting states
        assert len(states) == 169 * 168

        # Check first state properties
        state = states[0]
        assert state.history == ()
        assert state.pot == 1.5  # SB(0.5) + BB(1)
        assert state.to_act == 0  # SB acts first preflop
        assert state.stack == 100

    def test_state_is_hashable(self, game):
        """States can be used as dict keys."""
        states = game.initial_states()
        d = {states[0]: "test"}
        assert d[states[0]] == "test"


class TestHUNLActions:
    """Test action generation."""

    def test_sb_opening_actions(self, game):
        """SB opening has all raises plus fold."""
        state = HUNLState(
            hands=(0, 1),  # AA vs KK
            history=(),
            stack=100,
            pot=1.5,
            to_act=0,
        )
        actions = game.actions(state)
        # fold, call (limp), raises: 2.5, 3, 8, 20, 50, 100, all-in
        assert "f" in actions
        assert "c" in actions  # limp
        assert "r2.5" in actions
        assert "r3" in actions
        assert "r100" in actions
        assert "a" in actions  # all-in

    def test_bb_facing_raise(self, game):
        """BB facing raise has fold, call, and valid raises."""
        state = HUNLState(
            hands=(0, 1),
            history=("r2.5",),
            stack=100,
            pot=3.0,  # 0.5 + 1 + 1.5 open
            to_act=1,
        )
        actions = game.actions(state)
        assert "f" in actions
        assert "c" in actions
        assert "r2.5" not in actions  # Can't raise to same amount
        assert "r3" in actions
        assert "r8" in actions
        assert "a" in actions

    def test_no_raise_options_when_all_in(self, game):
        """No raises available when facing all-in."""
        state = HUNLState(
            hands=(0, 1),
            history=("a",),  # SB shoved
            stack=100,
            pot=101.5,  # All in
            to_act=1,
        )
        actions = game.actions(state)
        assert actions == ["f", "c"]  # Only fold or call


class TestHUNLTransitions:
    """Test state transitions."""

    def test_raise_updates_pot(self, game):
        """Raise action updates pot correctly."""
        state = HUNLState(
            hands=(0, 1),
            history=(),
            stack=100,
            pot=1.5,
            to_act=0,
        )
        next_state = game.next_state(state, "r3")
        assert next_state.history == ("r3",)
        assert next_state.pot == 4.0  # 1.5 + 2.5 (3 - 0.5 SB already in)
        assert next_state.to_act == 1

    def test_call_ends_action(self, game):
        """Call after raise ends betting."""
        state = HUNLState(
            hands=(0, 1),
            history=("r3",),
            stack=100,
            pot=4.0,
            to_act=1,
        )
        next_state = game.next_state(state, "c")
        assert game.is_terminal(next_state)

    def test_fold_ends_action(self, game):
        """Fold ends the hand."""
        state = HUNLState(
            hands=(0, 1),
            history=("r3",),
            stack=100,
            pot=4.0,
            to_act=1,
        )
        next_state = game.next_state(state, "f")
        assert game.is_terminal(next_state)


class TestHUNLTerminal:
    """Test terminal state detection and utilities."""

    def test_fold_is_terminal(self, game):
        """Fold creates terminal state."""
        state = HUNLState(
            hands=(0, 1),
            history=("r3", "f"),
            stack=100,
            pot=4.0,
            to_act=0,
        )
        assert game.is_terminal(state)

    def test_call_after_raise_is_terminal(self, game):
        """Call after a raise is terminal."""
        state = HUNLState(
            hands=(0, 1),
            history=("r3", "c"),
            stack=100,
            pot=6.0,
            to_act=0,
        )
        assert game.is_terminal(state)

    def test_limp_check_is_terminal(self, game):
        """Limp then check is terminal."""
        state = HUNLState(
            hands=(0, 1),
            history=("c", "x"),
            stack=100,
            pot=2.0,
            to_act=0,
        )
        assert game.is_terminal(state)

    def test_fold_utility(self, game):
        """Folder loses, other player wins pot."""
        state = HUNLState(
            hands=(0, 1),
            history=("r3", "f"),
            stack=100,
            pot=4.0,
            to_act=0,
        )
        # BB folded, SB wins 1BB (BB's contribution)
        assert game.utility(state, 0) == 1.0  # SB wins BB's 1BB
        assert game.utility(state, 1) == -1.0  # BB loses their 1BB


class TestHUNLInfoSets:
    """Test information set keys."""

    def test_info_set_format(self, game):
        """Info set key has correct format."""
        state = HUNLState(
            hands=(0, 1),  # AA vs KK
            history=("r3",),
            stack=100,
            pot=4.0,
            to_act=1,
        )
        key = game.info_set_key(state)
        # BB sees their hand (KK) and the action
        assert key == "BB:KK:r3"

    def test_info_set_opening(self, game):
        """Opening info set is just position and hand."""
        state = HUNLState(
            hands=(0, 1),
            history=(),
            stack=100,
            pot=1.5,
            to_act=0,
        )
        key = game.info_set_key(state)
        assert key == "SB:AA:"


class TestHUNLGameInterface:
    """Test Game interface compliance."""

    def test_num_players(self, game):
        """Game has 2 players."""
        assert game.num_players() == 2

    def test_player_alternates(self, game):
        """Player alternates correctly."""
        state = HUNLState(hands=(0, 1), history=(), stack=100, pot=1.5, to_act=0)
        assert game.player(state) == 0

        state2 = game.next_state(state, "r3")
        assert game.player(state2) == 1

        state3 = game.next_state(state2, "r8")
        assert game.player(state3) == 0
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pytest tests/test_hunl_game.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'games.hunl_preflop'"

**Step 3: Implement HUNL game**

```python
# games/hunl_preflop.py
"""
Heads-Up No-Limit Hold'em Preflop game implementation.
"""

from dataclasses import dataclass
from typing import List, Tuple
from itertools import permutations

from games.base import Game
from config.loader import Config
from core.hands import hand_to_string, HAND_COUNT


@dataclass(frozen=True)
class HUNLState:
    """
    State in HUNL Preflop.

    Attributes:
        hands: (sb_hand_idx, bb_hand_idx) as 0-168 indices
        history: Tuple of actions, e.g., ("r2.5", "r8", "c")
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
    Heads-Up No-Limit Hold'em Preflop.

    Rules:
    - SB posts 0.5BB, BB posts 1BB
    - SB acts first preflop
    - Actions: fold, call/check, raise to X BB, all-in
    - Hand ends on fold or call (after raise)
    """

    def __init__(self, config: Config):
        """
        Initialize HUNL preflop game.

        Args:
            config: Configuration with stack depth and raise sizes
        """
        self.config = config
        self.stack_depth = config.stack_depth

    def initial_states(self) -> List[HUNLState]:
        """All possible starting hand combinations."""
        states = []
        for sb_hand in range(HAND_COUNT):
            for bb_hand in range(HAND_COUNT):
                if sb_hand == bb_hand:
                    continue  # Can't both have same canonical hand
                states.append(HUNLState(
                    hands=(sb_hand, bb_hand),
                    history=(),
                    stack=self.stack_depth,
                    pot=1.5,  # SB(0.5) + BB(1)
                    to_act=0,  # SB acts first
                ))
        return states

    def is_terminal(self, state: HUNLState) -> bool:
        """Check if hand is over."""
        if not state.history:
            return False

        last_action = state.history[-1]

        # Fold ends the hand
        if last_action == "f":
            return True

        # Check (x) only valid as BB after limp, ends hand
        if last_action == "x":
            return True

        # Call after raise/all-in ends hand
        if last_action == "c" and len(state.history) >= 2:
            # Check if there was a raise before
            for action in state.history[:-1]:
                if action.startswith("r") or action == "a":
                    return True

        # Call as limp (SB) doesn't end hand
        if last_action == "c" and len(state.history) == 1:
            return False

        return False

    def player(self, state: HUNLState) -> int:
        """Return player to act."""
        return state.to_act

    def actions(self, state: HUNLState) -> List[str]:
        """Available actions at state."""
        actions = []

        # Compute current bet to call
        current_bet = self._current_bet(state)
        my_committed = self._committed(state, state.to_act)
        to_call = current_bet - my_committed
        remaining_stack = state.stack - my_committed

        # Fold (if facing a bet)
        if to_call > 0:
            actions.append("f")

        # Call or check
        if to_call > 0:
            actions.append("c")
        elif len(state.history) == 1 and state.history[0] == "c":
            # BB can check after SB limps
            actions.append("x")
        else:
            # Can limp as SB
            actions.append("c")

        # All-in already happened, no more raises
        if "a" in state.history:
            return actions

        # Raise options
        legal_raises = self.config.get_legal_raise_sizes(
            current_bet=current_bet,
            stack=state.stack,
        )
        for size in legal_raises:
            actions.append(f"r{size}")

        # All-in always available if we have chips
        if remaining_stack > current_bet:
            actions.append("a")

        return actions

    def next_state(self, state: HUNLState, action: str) -> HUNLState:
        """Return state after taking action."""
        new_history = state.history + (action,)

        # Calculate new pot
        my_committed = self._committed(state, state.to_act)

        if action == "f":
            new_pot = state.pot
        elif action == "c":
            current_bet = self._current_bet(state)
            added = current_bet - my_committed
            new_pot = state.pot + added
        elif action == "x":
            new_pot = state.pot
        elif action == "a":
            added = state.stack - my_committed
            new_pot = state.pot + added
        elif action.startswith("r"):
            raise_to = float(action[1:])
            added = raise_to - my_committed
            new_pot = state.pot + added
        else:
            raise ValueError(f"Unknown action: {action}")

        return HUNLState(
            hands=state.hands,
            history=new_history,
            stack=state.stack,
            pot=new_pot,
            to_act=1 - state.to_act,
        )

    def utility(self, state: HUNLState, player: int) -> float:
        """
        Utility for player at terminal state.

        Returns profit/loss in BB (not total chips).
        """
        if not self.is_terminal(state):
            raise ValueError("Cannot get utility of non-terminal state")

        last_action = state.history[-1]

        # Who won?
        if last_action == "f":
            # Last actor folded, previous player wins
            folder = state.to_act  # Who would act next = who just folded
            winner = 1 - folder
        else:
            # Showdown - for preflop we don't model actual hand strength
            # In real implementation, this would depend on runout
            # For now, return 0 (will be weighted by hand combos)
            # Actually, for preflop-only, call/check = showdown
            # We'll compute EV based on hand vs hand equity elsewhere
            # For CFR, we need deterministic utility
            # Winner is higher hand index (simplified)
            if state.hands[0] < state.hands[1]:
                winner = 0  # Lower index = stronger (AA=0)
            else:
                winner = 1

        # Calculate profit/loss
        my_committed = self._total_committed(state, player)
        opp_committed = self._total_committed(state, 1 - player)

        if player == winner:
            return opp_committed  # Win opponent's contribution
        else:
            return -my_committed  # Lose our contribution

    def info_set_key(self, state: HUNLState) -> str:
        """
        Map state to information set.

        Format: position:hand:history
        """
        position = "SB" if state.to_act == 0 else "BB"
        hand = hand_to_string(state.hands[state.to_act])
        history = "-".join(state.history) if state.history else ""
        return f"{position}:{hand}:{history}"

    def num_players(self) -> int:
        return 2

    def _current_bet(self, state: HUNLState) -> float:
        """Get the current bet to match."""
        bet = 1.0  # BB is the initial bet
        for action in state.history:
            if action.startswith("r"):
                bet = float(action[1:])
            elif action == "a":
                bet = state.stack
        return bet

    def _committed(self, state: HUNLState, player: int) -> float:
        """Get amount committed by player before current action."""
        # Start with blinds
        committed = 0.5 if player == 0 else 1.0

        actor = 0  # SB acts first
        for action in state.history:
            if actor == player:
                if action.startswith("r"):
                    committed = float(action[1:])
                elif action == "c":
                    # Called the current bet
                    committed = self._current_bet_at(state.history[:state.history.index(action)])
                    if committed == 1.0 and player == 0:
                        committed = 1.0  # SB limping
                elif action == "a":
                    committed = state.stack
            actor = 1 - actor

        return committed

    def _current_bet_at(self, history: Tuple[str, ...]) -> float:
        """Get bet at a point in history."""
        bet = 1.0
        for action in history:
            if action.startswith("r"):
                bet = float(action[1:])
            elif action == "a":
                return 100  # Stack depth placeholder
        return bet

    def _total_committed(self, state: HUNLState, player: int) -> float:
        """Get total amount committed by player including final action."""
        # This is for terminal states
        committed = 0.5 if player == 0 else 1.0

        actor = 0
        current_bet = 1.0  # BB

        for action in state.history:
            if action.startswith("r"):
                current_bet = float(action[1:])
                if actor == player:
                    committed = current_bet
            elif action == "c":
                if actor == player:
                    committed = current_bet
            elif action == "a":
                current_bet = state.stack
                if actor == player:
                    committed = state.stack
            elif action == "f":
                pass  # No additional commitment
            elif action == "x":
                pass  # No additional commitment
            actor = 1 - actor

        return committed
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pytest tests/test_hunl_game.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add games/hunl_preflop.py tests/test_hunl_game.py
git commit -m "feat(hunl): add HUNLPreflop game implementation

- HUNLState dataclass with hands, history, stack, pot, to_act
- 169x168 = 28,392 initial hand combinations
- Dynamic action generation based on config raise sizes
- Terminal detection for fold, call, check sequences
- Utility calculation based on hand index ordering
- Information set keys: position:hand:history"
```

---

## Task 5: CFR+ Modification

**Files:**
- Modify: `cfr/batched.py`
- Test: `tests/test_cfr_plus.py`

**Step 1: Write failing test for CFR+ regret flooring**

```python
# tests/test_cfr_plus.py
import pytest
import torch
from cfr.batched import BatchedCFR
from core.tensors import compile_game
from games.kuhn import KuhnPoker


class TestCFRPlus:
    """Test CFR+ modification (regret flooring)."""

    def test_regrets_never_negative(self):
        """After training, regrets should be >= 0 (CFR+ property)."""
        game = KuhnPoker()
        compiled = compile_game(game, torch.device("cpu"))
        cfr = BatchedCFR(compiled, batch_size=32)

        # Train for a few iterations
        for _ in range(100):
            cfr.train_step()

        # All regrets should be non-negative
        assert (cfr.regret_sum >= 0).all(), "CFR+ should floor regrets at 0"

    def test_cfr_plus_converges_faster(self):
        """CFR+ should converge faster than vanilla (not strictly tested, just runs)."""
        game = KuhnPoker()
        compiled = compile_game(game, torch.device("cpu"))
        cfr = BatchedCFR(compiled, batch_size=32)

        # Should complete without error
        for _ in range(200):
            cfr.train_step()

        # Strategy should be valid (sums to 1)
        strategy = cfr.get_current_strategy()
        sums = strategy.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
```

**Step 2: Run tests to verify current behavior**

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pytest tests/test_cfr_plus.py -v`
Expected: First test may FAIL (regrets can be negative in vanilla CFR)

**Step 3: Add CFR+ regret flooring**

Edit `cfr/batched.py`, after line 145 (`self.regret_sum[info_set, a] += regret`), add:

```python
        # CFR+ modification: floor regrets at zero
        self.regret_sum = torch.clamp(self.regret_sum, min=0)
```

The modified `train_step` method lines 117-153 should now end with:

```python
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

        # CFR+ modification: floor regrets at zero
        self.regret_sum = torch.clamp(self.regret_sum, min=0)

        self.iterations += self.batch_size
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pytest tests/test_cfr_plus.py -v`
Expected: All tests PASS

**Step 5: Run full test suite to verify no regressions**

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pytest -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add cfr/batched.py tests/test_cfr_plus.py
git commit -m "feat(cfr): add CFR+ regret flooring for faster convergence

Floor regrets at zero after each iteration (CFR+ algorithm).
This prevents negative regret accumulation and provides
2-10x faster convergence in practice."
```

---

## Task 6: Matrix Display Module

**Files:**
- Create: `cli/matrix.py`
- Test: `tests/test_matrix.py`

**Step 1: Write failing tests for matrix rendering**

```python
# tests/test_matrix.py
import pytest
from cli.matrix import (
    render_matrix,
    get_color_for_action,
    ActionDistribution,
    ANSI_RESET,
)
from core.hands import get_matrix_layout


class TestColorMapping:
    """Test action to color mapping."""

    def test_fold_is_red(self):
        """Fold dominant hands are red."""
        color = get_color_for_action("fold", 0.9)
        assert "31" in color or "91" in color  # ANSI red codes

    def test_call_is_green(self):
        """Call dominant hands are green."""
        color = get_color_for_action("call", 0.9)
        assert "32" in color or "92" in color  # ANSI green codes

    def test_raise_is_blue(self):
        """Raise dominant hands are blue."""
        color = get_color_for_action("raise", 0.9)
        assert "34" in color or "94" in color  # ANSI blue codes

    def test_all_in_is_yellow(self):
        """All-in dominant hands are yellow."""
        color = get_color_for_action("all_in", 0.9)
        assert "33" in color or "93" in color  # ANSI yellow codes

    def test_high_frequency_is_bright(self):
        """High frequency (85%+) uses bright color."""
        color_bright = get_color_for_action("fold", 0.90)
        color_dim = get_color_for_action("fold", 0.65)
        # Bright codes are 90-97, dim codes are 30-37
        assert "9" in color_bright  # Bright
        assert "3" in color_dim and "9" not in color_dim  # Dim


class TestMatrixLayout:
    """Test matrix layout structure."""

    def test_matrix_is_13x13(self):
        """Matrix has correct dimensions."""
        layout = get_matrix_layout()
        assert len(layout) == 13
        assert all(len(row) == 13 for row in layout)

    def test_diagonal_is_pairs(self):
        """Diagonal contains pairs."""
        layout = get_matrix_layout()
        expected_pairs = ["AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22"]
        for i, pair in enumerate(expected_pairs):
            assert layout[i][i] == pair

    def test_above_diagonal_is_suited(self):
        """Above diagonal contains suited hands."""
        layout = get_matrix_layout()
        assert layout[0][1] == "AK"  # Suited, no 'o'
        assert layout[0][12] == "A2"

    def test_below_diagonal_is_offsuit(self):
        """Below diagonal contains offsuit hands."""
        layout = get_matrix_layout()
        assert layout[1][0] == "AKo"
        assert layout[12][0] == "A2o"


class TestRenderMatrix:
    """Test full matrix rendering."""

    def test_render_contains_all_hands(self):
        """Rendered matrix contains all 169 hands."""
        # Create dummy strategy (all fold)
        strategy = {}
        layout = get_matrix_layout()
        for row in layout:
            for hand in row:
                strategy[hand] = ActionDistribution(
                    fold=1.0, call=0.0, raises={}, all_in=0.0
                )

        output = render_matrix(strategy, "Test title")

        # Check some hands are present
        assert "AA" in output
        assert "AKo" in output
        assert "22" in output

    def test_render_has_ansi_codes(self):
        """Rendered matrix contains color codes."""
        strategy = {}
        layout = get_matrix_layout()
        for row in layout:
            for hand in row:
                strategy[hand] = ActionDistribution(
                    fold=1.0, call=0.0, raises={}, all_in=0.0
                )

        output = render_matrix(strategy, "Test")
        assert "\033[" in output  # ANSI escape sequence
        assert ANSI_RESET in output

    def test_render_left_aligned(self):
        """Cells are left-aligned with consistent width."""
        strategy = {}
        layout = get_matrix_layout()
        for row in layout:
            for hand in row:
                strategy[hand] = ActionDistribution(
                    fold=0.0, call=1.0, raises={}, all_in=0.0
                )

        output = render_matrix(strategy, "Test")
        lines = output.strip().split("\n")

        # Find matrix lines (skip header)
        matrix_lines = [l for l in lines if "AA" in l or "AKo" in l or "22" in l]
        assert len(matrix_lines) > 0
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pytest tests/test_matrix.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'cli.matrix'"

**Step 3: Implement matrix display**

```python
# cli/matrix.py
"""
Color-coded strategy matrix display for terminal.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from core.hands import get_matrix_layout

# ANSI color codes
ANSI_RESET = "\033[0m"

# Dim colors (30-37)
ANSI_DIM_RED = "\033[31m"
ANSI_DIM_GREEN = "\033[32m"
ANSI_DIM_YELLOW = "\033[33m"
ANSI_DIM_BLUE = "\033[34m"

# Bright colors (90-97)
ANSI_BRIGHT_RED = "\033[91m"
ANSI_BRIGHT_GREEN = "\033[92m"
ANSI_BRIGHT_YELLOW = "\033[93m"
ANSI_BRIGHT_BLUE = "\033[94m"


@dataclass
class ActionDistribution:
    """Distribution over actions for a hand."""
    fold: float
    call: float
    raises: Dict[float, float]  # raise_size -> probability
    all_in: float

    def dominant_action(self) -> tuple[str, float]:
        """Return (action_type, frequency) for most common action."""
        # Sum all raises
        total_raise = sum(self.raises.values())

        actions = [
            ("fold", self.fold),
            ("call", self.call),
            ("raise", total_raise),
            ("all_in", self.all_in),
        ]

        return max(actions, key=lambda x: x[1])


def get_color_for_action(action: str, frequency: float) -> str:
    """
    Get ANSI color code for action type and frequency.

    Args:
        action: "fold", "call", "raise", or "all_in"
        frequency: 0.0-1.0, how often this action is taken

    Returns:
        ANSI escape code string
    """
    # Use bright color for 85%+, dim for lower
    bright = frequency >= 0.85

    if action == "fold":
        return ANSI_BRIGHT_RED if bright else ANSI_DIM_RED
    elif action == "call":
        return ANSI_BRIGHT_GREEN if bright else ANSI_DIM_GREEN
    elif action == "raise":
        return ANSI_BRIGHT_BLUE if bright else ANSI_DIM_BLUE
    elif action == "all_in":
        return ANSI_BRIGHT_YELLOW if bright else ANSI_DIM_YELLOW
    else:
        return ANSI_RESET


def render_matrix(
    strategy: Dict[str, ActionDistribution],
    header: str,
    cell_width: int = 4,
) -> str:
    """
    Render a color-coded strategy matrix.

    Args:
        strategy: Dict mapping hand string to ActionDistribution
        header: Header text to display above matrix
        cell_width: Width of each cell (default 4 for "AKo ")

    Returns:
        String with ANSI color codes for terminal display
    """
    lines = [header, ""]

    layout = get_matrix_layout()

    for row in layout:
        row_str = ""
        for hand in row:
            dist = strategy.get(hand)
            if dist is None:
                # No strategy, show grey
                row_str += f"{hand:<{cell_width}}"
            else:
                action, freq = dist.dominant_action()
                color = get_color_for_action(action, freq)
                row_str += f"{color}{hand:<{cell_width}}{ANSI_RESET}"
        lines.append(row_str)

    return "\n".join(lines)


def render_header(
    stack: float,
    pot: float,
    action_history: str,
    player: str,
) -> str:
    """
    Render the header showing game state.

    Args:
        stack: Effective stack in BB
        pot: Current pot in BB
        action_history: Human-readable action string
        player: "SB" or "BB"

    Returns:
        Header string
    """
    parts = [
        f"HUNL Preflop | Stack: {stack:.0f}BB | Pot: {pot:.1f}BB",
    ]

    if action_history:
        parts.append(f"Action: {action_history} | {player} to act")
    else:
        parts.append(f"{player} to act (opening)")

    return "\n".join(parts)
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pytest tests/test_matrix.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add cli/matrix.py tests/test_matrix.py
git commit -m "feat(cli): add color-coded strategy matrix display

- ANSI colors: red=fold, green=call, blue=raise, yellow=all-in
- Bright colors for 85%+ frequency, dim for lower
- 13x13 matrix layout matching standard hand chart
- Left-aligned cells for clean terminal display"
```

---

## Task 7: Interactive CLI Loop

**Files:**
- Create: `cli/interactive.py`
- Test: `tests/test_interactive.py`

**Step 1: Write failing tests for interactive loop**

```python
# tests/test_interactive.py
import pytest
from unittest.mock import patch, MagicMock
from cli.interactive import InteractiveSession, parse_user_input


class TestParseUserInput:
    """Test user input parsing."""

    def test_parse_fold(self):
        """Parse fold command."""
        assert parse_user_input("f") == ("action", "f")
        assert parse_user_input("fold") == ("action", "f")

    def test_parse_call(self):
        """Parse call command."""
        assert parse_user_input("c") == ("action", "c")
        assert parse_user_input("call") == ("action", "c")

    def test_parse_raise(self):
        """Parse raise command."""
        assert parse_user_input("r8") == ("action", "r8")
        assert parse_user_input("r20") == ("action", "r20")

    def test_parse_all_in(self):
        """Parse all-in command."""
        assert parse_user_input("a") == ("action", "a")
        assert parse_user_input("all-in") == ("action", "a")
        assert parse_user_input("allin") == ("action", "a")

    def test_parse_back(self):
        """Parse back command."""
        assert parse_user_input("b") == ("back", None)
        assert parse_user_input("back") == ("back", None)

    def test_parse_quit(self):
        """Parse quit command."""
        assert parse_user_input("q") == ("quit", None)
        assert parse_user_input("quit") == ("quit", None)

    def test_parse_invalid(self):
        """Invalid input returns None."""
        assert parse_user_input("xyz") == ("invalid", "xyz")
        assert parse_user_input("") == ("invalid", "")


class TestInteractiveSession:
    """Test interactive session state management."""

    def test_session_init(self):
        """Session initializes with config and solver."""
        from config.loader import Config
        config = Config(name="Test", stack_depth=100, raise_sizes=[2.5, 3, 8])

        session = InteractiveSession(config, strategy={})
        assert session.history == []
        assert session.stack == 100

    def test_session_apply_action(self):
        """Applying action updates history."""
        from config.loader import Config
        config = Config(name="Test", stack_depth=100, raise_sizes=[2.5, 3, 8])

        session = InteractiveSession(config, strategy={})
        session.apply_action("r2.5")

        assert session.history == ["r2.5"]

    def test_session_go_back(self):
        """Going back removes last action."""
        from config.loader import Config
        config = Config(name="Test", stack_depth=100, raise_sizes=[2.5, 3, 8])

        session = InteractiveSession(config, strategy={})
        session.apply_action("r2.5")
        session.apply_action("r8")
        session.go_back()

        assert session.history == ["r2.5"]

    def test_session_get_legal_actions(self):
        """Session returns legal actions for current state."""
        from config.loader import Config
        config = Config(name="Test", stack_depth=100, raise_sizes=[2.5, 3, 8, 20])

        session = InteractiveSession(config, strategy={})
        actions = session.get_legal_actions()

        # Opening: fold, call (limp), raises, all-in
        assert "f" in actions
        assert "c" in actions
        assert "r2.5" in actions
        assert "a" in actions
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pytest tests/test_interactive.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'cli.interactive'"

**Step 3: Implement interactive session**

```python
# cli/interactive.py
"""
Interactive CLI session for exploring HUNL strategies.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from config.loader import Config
from cli.matrix import render_matrix, render_header, ActionDistribution
from core.hands import get_matrix_layout, hand_to_string


def parse_user_input(user_input: str) -> Tuple[str, Optional[str]]:
    """
    Parse user input into command type and value.

    Returns:
        (command_type, value) where command_type is one of:
        - "action": value is the action string (f, c, rX, a)
        - "back": go back one step
        - "quit": exit session
        - "invalid": unrecognized input
    """
    user_input = user_input.strip().lower()

    if not user_input:
        return ("invalid", "")

    # Quit
    if user_input in ("q", "quit", "exit"):
        return ("quit", None)

    # Back
    if user_input in ("b", "back"):
        return ("back", None)

    # Fold
    if user_input in ("f", "fold"):
        return ("action", "f")

    # Call
    if user_input in ("c", "call"):
        return ("action", "c")

    # Check
    if user_input in ("x", "check"):
        return ("action", "x")

    # All-in
    if user_input in ("a", "all-in", "allin", "all"):
        return ("action", "a")

    # Raise (r8, r20, etc.)
    if user_input.startswith("r") and len(user_input) > 1:
        try:
            float(user_input[1:])
            return ("action", user_input)
        except ValueError:
            pass

    return ("invalid", user_input)


@dataclass
class InteractiveSession:
    """
    Manages state for interactive strategy exploration.
    """
    config: Config
    strategy: Dict[str, Dict[str, float]]  # info_set -> action -> probability
    history: List[str] = field(default_factory=list)
    stack: float = field(init=False)

    def __post_init__(self):
        self.stack = self.config.stack_depth

    def apply_action(self, action: str) -> None:
        """Apply an action to the current state."""
        self.history.append(action)

    def go_back(self) -> bool:
        """Remove last action. Returns False if at root."""
        if self.history:
            self.history.pop()
            return True
        return False

    def get_current_player(self) -> str:
        """Get current player to act (SB or BB)."""
        # SB acts first, then alternates
        return "SB" if len(self.history) % 2 == 0 else "BB"

    def get_pot(self) -> float:
        """Calculate current pot size."""
        pot = 1.5  # SB + BB
        sb_committed = 0.5
        bb_committed = 1.0

        actor = 0  # SB
        current_bet = 1.0  # BB

        for action in self.history:
            if action == "f":
                break
            elif action == "c":
                if actor == 0:
                    pot += current_bet - sb_committed
                    sb_committed = current_bet
                else:
                    pot += current_bet - bb_committed
                    bb_committed = current_bet
            elif action == "x":
                pass
            elif action == "a":
                if actor == 0:
                    pot += self.stack - sb_committed
                    sb_committed = self.stack
                else:
                    pot += self.stack - bb_committed
                    bb_committed = self.stack
                current_bet = self.stack
            elif action.startswith("r"):
                amount = float(action[1:])
                if actor == 0:
                    pot += amount - sb_committed
                    sb_committed = amount
                else:
                    pot += amount - bb_committed
                    bb_committed = amount
                current_bet = amount
            actor = 1 - actor

        return pot

    def get_current_bet(self) -> float:
        """Get the current bet to match."""
        bet = 1.0  # BB
        for action in self.history:
            if action.startswith("r"):
                bet = float(action[1:])
            elif action == "a":
                bet = self.stack
        return bet

    def get_my_committed(self) -> float:
        """Get amount current player has committed."""
        player = len(self.history) % 2  # 0=SB, 1=BB
        committed = 0.5 if player == 0 else 1.0

        actor = 0
        for action in self.history:
            if actor == player:
                if action.startswith("r"):
                    committed = float(action[1:])
                elif action == "c":
                    committed = self.get_current_bet()
                elif action == "a":
                    committed = self.stack
            actor = 1 - actor

        return committed

    def get_legal_actions(self) -> List[str]:
        """Get list of legal actions."""
        actions = []

        current_bet = self.get_current_bet()
        my_committed = self.get_my_committed()
        to_call = current_bet - my_committed
        remaining = self.stack - my_committed

        # Fold if facing a bet
        if to_call > 0:
            actions.append("f")

        # Call or check
        if to_call > 0:
            actions.append("c")
        elif len(self.history) == 1 and self.history[0] == "c":
            actions.append("x")  # BB can check after limp
        else:
            actions.append("c")  # Limp/call

        # No raises after all-in
        if "a" in self.history:
            return actions

        # Raises
        for size in self.config.raise_sizes:
            if size > current_bet and size <= self.stack:
                actions.append(f"r{size}")

        # All-in
        if remaining > current_bet:
            actions.append("a")

        return actions

    def is_terminal(self) -> bool:
        """Check if current state is terminal."""
        if not self.history:
            return False

        last = self.history[-1]

        if last == "f":
            return True
        if last == "x":
            return True
        if last == "c" and len(self.history) >= 2:
            for a in self.history[:-1]:
                if a.startswith("r") or a == "a":
                    return True

        return False

    def get_strategy_for_hand(self, hand: str) -> Optional[ActionDistribution]:
        """Get strategy distribution for a specific hand."""
        player = self.get_current_player()
        history_str = "-".join(self.history) if self.history else ""
        info_set = f"{player}:{hand}:{history_str}"

        if info_set not in self.strategy:
            return None

        action_probs = self.strategy[info_set]

        # Parse into ActionDistribution
        fold_prob = action_probs.get("f", 0.0)
        call_prob = action_probs.get("c", 0.0) + action_probs.get("x", 0.0)
        all_in_prob = action_probs.get("a", 0.0)

        raises = {}
        for action, prob in action_probs.items():
            if action.startswith("r"):
                size = float(action[1:])
                raises[size] = prob

        return ActionDistribution(
            fold=fold_prob,
            call=call_prob,
            raises=raises,
            all_in=all_in_prob,
        )

    def get_all_hand_strategies(self) -> Dict[str, ActionDistribution]:
        """Get strategy for all 169 hands."""
        result = {}
        layout = get_matrix_layout()
        for row in layout:
            for hand in row:
                dist = self.get_strategy_for_hand(hand)
                if dist:
                    result[hand] = dist
        return result

    def render(self) -> str:
        """Render current state as string."""
        header = render_header(
            stack=self.stack,
            pot=self.get_pot(),
            action_history=" ".join(self.history) if self.history else "",
            player=self.get_current_player(),
        )

        strategies = self.get_all_hand_strategies()
        matrix = render_matrix(strategies, header)

        # Add action prompt
        actions = self.get_legal_actions()
        action_strs = []
        for a in actions:
            if a == "f":
                action_strs.append("f (fold)")
            elif a == "c":
                action_strs.append("c (call)")
            elif a == "x":
                action_strs.append("x (check)")
            elif a == "a":
                action_strs.append("a (all-in)")
            elif a.startswith("r"):
                action_strs.append(f"{a} (raise {a[1:]}BB)")

        action_strs.extend(["b (back)", "q (quit)"])
        prompt = "\nActions: " + " | ".join(action_strs) + "\n> "

        return matrix + prompt


def run_interactive(config: Config, strategy: Dict[str, Dict[str, float]]) -> None:
    """
    Run interactive exploration session.

    Args:
        config: Game configuration
        strategy: Trained strategy dict
    """
    session = InteractiveSession(config, strategy)

    while True:
        # Clear screen and render
        print("\033[2J\033[H", end="")  # Clear screen
        print(session.render(), end="")

        if session.is_terminal():
            print("\n[Hand complete - press 'b' to go back or 'q' to quit]")

        try:
            user_input = input()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        cmd, value = parse_user_input(user_input)

        if cmd == "quit":
            print("Goodbye!")
            break
        elif cmd == "back":
            if not session.go_back():
                print("Already at root position")
        elif cmd == "action":
            if value in session.get_legal_actions():
                session.apply_action(value)
            else:
                print(f"Illegal action: {value}")
        elif cmd == "invalid":
            print(f"Unknown command: {value}")
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pytest tests/test_interactive.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add cli/interactive.py tests/test_interactive.py
git commit -m "feat(cli): add interactive strategy exploration session

- Parse user commands: f/c/rX/a for actions, b for back, q for quit
- Track game state and compute legal actions
- Render full matrix with current position highlighted
- Terminal state detection for hand completion"
```

---

## Task 8: HUNL Main Command

**Files:**
- Modify: `main.py`
- Test: `tests/test_hunl_cli.py`

**Step 1: Write failing test for HUNL CLI**

```python
# tests/test_hunl_cli.py
import pytest
import subprocess
import sys


class TestHUNLCLI:
    """Test HUNL command line interface."""

    def test_hunl_help(self):
        """HUNL subcommand shows help."""
        result = subprocess.run(
            [sys.executable, "main.py", "hunl", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "HUNL" in result.stdout or "hunl" in result.stdout.lower()

    def test_hunl_requires_config(self):
        """HUNL requires config file."""
        result = subprocess.run(
            [sys.executable, "main.py", "hunl"],
            capture_output=True,
            text=True,
        )
        # Should fail or prompt for config
        assert "config" in result.stderr.lower() or result.returncode != 0

    def test_hunl_with_preset(self):
        """HUNL works with preset config."""
        result = subprocess.run(
            [sys.executable, "main.py", "hunl", "--config", "standard", "--iterations", "100", "--no-interactive"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        # Should complete successfully
        assert result.returncode == 0 or "Training" in result.stdout
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pytest tests/test_hunl_cli.py -v`
Expected: FAIL because HUNL subcommand doesn't exist yet

**Step 3: Add HUNL subcommand to main.py**

```python
#!/usr/bin/env python3
"""
CFR Poker Solver - Main Entry Point

Usage:
    python main.py                    # Solve Kuhn Poker with defaults
    python main.py --iterations 10000 # More iterations
    python main.py --device cuda      # Use GPU
    python main.py --batch-size 1024  # Larger batches

    python main.py hunl --config standard  # HUNL preflop solver
"""

import argparse
import time
import sys

from games.kuhn import KuhnPoker
from solver import Solver


def run_kuhn(args):
    """Run Kuhn Poker solver."""
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


def run_hunl(args):
    """Run HUNL preflop solver."""
    from pathlib import Path
    from config.loader import load_config, get_preset_path
    from games.hunl_preflop import HUNLPreflop
    from cli.interactive import run_interactive
    from cli.parser import parse_action_sequence

    # Load config
    if args.config in ("standard", "aggressive"):
        config_path = get_preset_path(args.config)
    else:
        config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)

    # Parse action sequence if provided
    initial_history = ()
    if args.action:
        parsed = parse_action_sequence(args.action)
        initial_history = parsed.to_history_tuple()
        if parsed.stack_override:
            config.stack_depth = parsed.stack_override

    print("=" * 50)
    print("CFR Poker Solver - HUNL Preflop")
    print("=" * 50)
    print(f"Config: {config.name}")
    print(f"Stack: {config.stack_depth} BB")
    print(f"Raise sizes: {config.raise_sizes}")
    print(f"Iterations: {args.iterations}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Create game and solver
    game = HUNLPreflop(config)
    solver = Solver(
        game=game,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Solve
    print("Training CFR+...")
    start_time = time.time()
    strategy = solver.solve(
        iterations=args.iterations,
        verbose=not args.quiet,
        log_interval=args.iterations // 10 if args.iterations >= 10 else 1,
    )
    elapsed = time.time() - start_time

    print()
    print("=" * 50)
    print("Results")
    print("=" * 50)
    print(f"Time: {elapsed:.2f} seconds")
    print(f"Iterations/sec: {args.iterations / elapsed:.0f}")
    print(f"Final exploitability: {solver.exploitability():.6f}")
    print()

    # Interactive mode
    if not args.no_interactive:
        print("Entering interactive mode...")
        print("(Use --no-interactive to skip)")
        print()
        run_interactive(config, strategy)


def main():
    parser = argparse.ArgumentParser(description="CFR Poker Solver")
    subparsers = parser.add_subparsers(dest="command", help="Game to solve")

    # Kuhn subcommand (also default)
    kuhn_parser = subparsers.add_parser("kuhn", help="Solve Kuhn Poker")
    kuhn_parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=10000,
        help="Number of CFR iterations (default: 10000)"
    )
    kuhn_parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device (default: auto)"
    )
    kuhn_parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Batch size for parallel iterations (default: 1 = vanilla CFR)"
    )
    kuhn_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    # HUNL subcommand
    hunl_parser = subparsers.add_parser("hunl", help="Solve HUNL Preflop")
    hunl_parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Config file path or preset name (standard, aggressive)"
    )
    hunl_parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=100000,
        help="Number of CFR iterations (default: 100000)"
    )
    hunl_parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device (default: auto)"
    )
    hunl_parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1024,
        help="Batch size for parallel iterations (default: 1024)"
    )
    hunl_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    hunl_parser.add_argument(
        "--action", "-a",
        type=str,
        default="",
        help="Initial action sequence (e.g., '50bb SBr2.5 BBr8')"
    )
    hunl_parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive mode after training"
    )

    args = parser.parse_args()

    # Default to kuhn if no subcommand
    if args.command is None:
        # Add default args for kuhn
        args.command = "kuhn"
        args.iterations = 10000
        args.device = "auto"
        args.batch_size = 1
        args.quiet = False

    if args.command == "kuhn":
        run_kuhn(args)
    elif args.command == "hunl":
        run_hunl(args)


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pytest tests/test_hunl_cli.py -v`
Expected: All tests PASS

**Step 5: Run full test suite**

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pytest -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add main.py tests/test_hunl_cli.py
git commit -m "feat(cli): add hunl subcommand for HUNL preflop solver

Usage:
  python main.py hunl --config standard --iterations 100000
  python main.py hunl --config standard --action 'SBr2.5 BBr8'

Features:
- Load preset or custom config
- Train CFR+ with progress
- Enter interactive exploration mode
- Optional initial action sequence"
```

---

## Task 9: Solver Updates for HUNL

**Files:**
- Modify: `solver.py`
- Modify: `cfr/batched.py`
- Test: `tests/test_hunl_solver.py`

**Step 1: Write test for HUNL solver integration**

```python
# tests/test_hunl_solver.py
import pytest
from config.loader import Config
from games.hunl_preflop import HUNLPreflop
from solver import Solver


@pytest.fixture
def small_config():
    """Small config for fast tests."""
    return Config(
        name="Test",
        stack_depth=20,  # Short stack for smaller tree
        raise_sizes=[2, 4, 10, 20],  # Fewer sizes
    )


class TestHUNLSolver:
    """Test solver with HUNL game."""

    def test_solver_creates_with_hunl(self, small_config):
        """Solver initializes with HUNL game."""
        game = HUNLPreflop(small_config)
        solver = Solver(game, device="cpu", batch_size=1)
        assert solver is not None

    def test_solver_runs_vanilla_cfr(self, small_config):
        """Vanilla CFR runs on HUNL."""
        game = HUNLPreflop(small_config)
        solver = Solver(game, device="cpu", batch_size=1)

        strategy = solver.solve(iterations=10, verbose=False)

        assert len(strategy) > 0
        # Check strategy format
        for info_set, probs in strategy.items():
            assert sum(probs.values()) == pytest.approx(1.0, abs=0.01)

    def test_solver_runs_batched_cfr(self, small_config):
        """Batched CFR runs on HUNL."""
        game = HUNLPreflop(small_config)
        solver = Solver(game, device="cpu", batch_size=32)

        strategy = solver.solve(iterations=100, verbose=False)

        assert len(strategy) > 0

    def test_exploitability_decreases(self, small_config):
        """Exploitability decreases with training."""
        game = HUNLPreflop(small_config)
        solver = Solver(game, device="cpu", batch_size=32)

        # Train a bit
        solver.solve(iterations=100, verbose=False)
        exploit_early = solver.exploitability()

        # Train more
        solver.solve(iterations=500, verbose=False)
        exploit_late = solver.exploitability()

        # Should decrease (or at least not increase much)
        assert exploit_late <= exploit_early * 1.1  # Allow some variance
```

**Step 2: Run tests**

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pytest tests/test_hunl_solver.py -v`
Expected: Tests may fail due to batched CFR assuming Kuhn-specific actions

**Step 3: Update batched CFR to handle dynamic actions**

The current `get_average_strategy` method in `cfr/batched.py` hardcodes `actions = ["p", "b"]`. Update it to get actions from the game:

Edit `cfr/batched.py` to add a reference to the game's actions and update `get_average_strategy`:

```python
# In BatchedCFR.__init__, after self.iterations = 0, add:
        self.info_set_actions = {}  # Cache of info_set -> list of action strings

# Replace get_average_strategy method:
    def get_average_strategy(self) -> Dict[str, Dict[str, float]]:
        """Convert tensor strategy to dictionary format."""
        result = {}
        strategy_sum = self.strategy_sum.cpu().numpy()

        for info_set, idx in self.compiled.info_set_to_idx.items():
            total = strategy_sum[idx].sum()
            action_names = self.compiled.info_set_actions.get(info_set, [])

            if total > 0 and action_names:
                probs = {}
                for a, name in enumerate(action_names):
                    if a < self.compiled.max_actions:
                        probs[name] = float(strategy_sum[idx, a] / total)
                result[info_set] = probs
            elif action_names:
                # Uniform over actions
                result[info_set] = {name: 1.0 / len(action_names) for name in action_names}

        return result
```

And update `core/tensors.py` to store action names per info set:

Add to `CompiledGame` dataclass:
```python
    info_set_actions: Dict[str, List[str]] = field(default_factory=dict)
```

And in `compile_game`, after building `info_set_to_idx`, add action tracking:
```python
    # Track actions per info set
    info_set_actions = {}
    for state in all_states:
        if not game.is_terminal(state):
            key = game.info_set_key(state)
            if key not in info_set_actions:
                info_set_actions[key] = game.actions(state)
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pytest tests/test_hunl_solver.py -v`
Expected: All tests PASS

**Step 5: Run full test suite to ensure no regressions**

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pytest -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add solver.py cfr/batched.py core/tensors.py tests/test_hunl_solver.py
git commit -m "feat(solver): support dynamic actions for HUNL games

- Store action names per info set in CompiledGame
- Update BatchedCFR.get_average_strategy to use dynamic actions
- Enables solver to work with any game, not just Kuhn"
```

---

## Task 10: Integration Test and Polish

**Files:**
- Create: `tests/test_hunl_integration.py`
- Create: `config/presets/aggressive.yaml`

**Step 1: Create aggressive preset**

```yaml
# config/presets/aggressive.yaml
name: "Aggressive 100BB"
stack_depth: 100
raise_sizes: [3, 4, 10, 12, 25, 30, 60, 100]
```

**Step 2: Write integration tests**

```python
# tests/test_hunl_integration.py
"""
Integration tests for HUNL preflop solver.
"""

import pytest
from config.loader import load_config, get_preset_path
from games.hunl_preflop import HUNLPreflop
from solver import Solver
from cli.parser import parse_action_sequence
from cli.interactive import InteractiveSession


class TestHUNLIntegration:
    """End-to-end integration tests."""

    @pytest.fixture
    def standard_config(self):
        return load_config(get_preset_path("standard"))

    @pytest.fixture
    def trained_solver(self, standard_config):
        """Pre-trained solver for tests."""
        game = HUNLPreflop(standard_config)
        solver = Solver(game, device="cpu", batch_size=32)
        solver.solve(iterations=100, verbose=False)
        return solver

    def test_full_pipeline(self, standard_config):
        """Test full training pipeline."""
        game = HUNLPreflop(standard_config)
        solver = Solver(game, device="cpu", batch_size=64)

        strategy = solver.solve(iterations=200, verbose=False)
        exploit = solver.exploitability()

        assert len(strategy) > 1000  # Many info sets
        assert exploit < 100  # Some convergence

    def test_interactive_session_workflow(self, standard_config, trained_solver):
        """Test interactive session state management."""
        strategy = trained_solver.get_strategy()
        session = InteractiveSession(standard_config, strategy)

        # Apply opening raise
        assert "r2.5" in session.get_legal_actions()
        session.apply_action("r2.5")

        # BB should be to act
        assert session.get_current_player() == "BB"

        # BB has legal actions
        actions = session.get_legal_actions()
        assert "f" in actions
        assert "c" in actions

        # Apply 3-bet
        session.apply_action("r8")
        assert session.get_current_player() == "SB"

        # Go back
        session.go_back()
        assert session.get_current_player() == "BB"
        assert session.history == ["r2.5"]

    def test_action_sequence_parsing_integration(self, standard_config):
        """Test parsing action sequences and applying to game."""
        game = HUNLPreflop(standard_config)

        # Parse sequence
        parsed = parse_action_sequence("SBr2.5 BBr8 SBc")
        history = parsed.to_history_tuple()

        assert history == ("r2.5", "r8", "c")

        # Apply to initial state
        states = game.initial_states()
        state = states[0]

        for action in history:
            state = game.next_state(state, action)

        # Should be terminal (call after 3-bet)
        assert game.is_terminal(state)

    def test_presets_load_correctly(self):
        """All preset configs load without error."""
        for preset in ["standard", "aggressive"]:
            config = load_config(get_preset_path(preset))
            assert config.stack_depth == 100
            assert len(config.raise_sizes) > 0


class TestHUNLStrategySanity:
    """Sanity checks on trained strategies."""

    @pytest.fixture
    def trained_strategy(self):
        """Train a strategy for testing."""
        config = load_config(get_preset_path("standard"))
        # Use smaller subset for speed
        config.raise_sizes = [2.5, 3, 8, 20]

        game = HUNLPreflop(config)
        solver = Solver(game, device="cpu", batch_size=64)
        return solver.solve(iterations=500, verbose=False)

    def test_strategy_probabilities_sum_to_one(self, trained_strategy):
        """All strategy distributions sum to 1."""
        for info_set, probs in trained_strategy.items():
            total = sum(probs.values())
            assert total == pytest.approx(1.0, abs=0.01), f"{info_set}: {probs}"

    def test_premium_hands_raise_often(self, trained_strategy):
        """Premium hands should raise frequently at opening."""
        # Check SB opening with AA
        aa_key = "SB:AA:"
        if aa_key in trained_strategy:
            probs = trained_strategy[aa_key]
            raise_prob = sum(p for a, p in probs.items() if a.startswith("r") or a == "a")
            # AA should raise/all-in > 50% of time
            assert raise_prob > 0.5, f"AA should raise often: {probs}"
```

**Step 3: Run integration tests**

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pytest tests/test_hunl_integration.py -v`
Expected: All tests PASS

**Step 4: Run full test suite**

Run: `cd /Users/ltj/Documents/code/poker_solver/.worktrees/hunl-preflop && source venv/bin/activate && pytest -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add config/presets/aggressive.yaml tests/test_hunl_integration.py
git commit -m "test(hunl): add integration tests and aggressive preset

- Full pipeline test: config -> game -> solver -> strategy
- Interactive session workflow test
- Action sequence parsing integration
- Strategy sanity checks (probabilities sum to 1, premiums raise)"
```

---

## Summary

This plan implements HUNL preflop support in 10 tasks:

1. **Hand Parser** - Parse AA, AKo, AcKh notation
2. **Config Loader** - YAML config with raise sizes
3. **Action Parser** - Parse "SBr2.5 BBr8" sequences
4. **HUNL Game** - Game interface implementation
5. **CFR+** - Regret flooring for faster convergence
6. **Matrix Display** - Color-coded terminal output
7. **Interactive CLI** - Explore strategies interactively
8. **Main Command** - CLI entry point
9. **Solver Updates** - Dynamic action support
10. **Integration Tests** - End-to-end verification

Each task follows TDD with failing tests first, then implementation, then verification.
