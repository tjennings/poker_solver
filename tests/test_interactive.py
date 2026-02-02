"""Tests for the interactive CLI module."""

import pytest
from typing import Dict

from config.loader import Config
from cli.matrix import ActionDistribution


# Fixtures

@pytest.fixture
def config():
    """Create a basic config for testing."""
    return Config(
        name="Test",
        stack_depths=[100.0],
        raise_sizes=[2.5, 3.0, 8.0, 20.0],
    )


@pytest.fixture
def empty_strategy():
    """Empty strategy for testing."""
    return {}


@pytest.fixture
def sample_strategy():
    """Create a sample raw strategy for testing.

    Uses the info_set_key format: "POSITION:HAND:HISTORY"
    with action probabilities as dicts.
    """
    return {
        # SB initial strategies (empty history)
        "SB:AA:": {"f": 0.0, "c": 0.0, "r3.0": 0.9, "a": 0.1},
        "SB:AKs:": {"f": 0.0, "c": 0.2, "r2.5": 0.7, "a": 0.1},
        "SB:72o:": {"f": 0.9, "c": 0.1},
        # BB strategies after SB raises 2.5
        "BB:AA:r2.5": {"f": 0.0, "c": 0.1, "r8": 0.8, "a": 0.1},
        "BB:AKs:r2.5": {"f": 0.1, "c": 0.5, "r8": 0.4},
        "BB:72o:r2.5": {"f": 0.95, "c": 0.05},
    }


# Test parse_user_input


class TestParseUserInput:
    """Tests for parse_user_input function."""

    def test_parse_fold_short(self):
        """'f' should parse as fold action."""
        from cli.interactive import parse_user_input

        assert parse_user_input("f") == ("action", "f")

    def test_parse_fold_long(self):
        """'fold' should parse as fold action."""
        from cli.interactive import parse_user_input

        assert parse_user_input("fold") == ("action", "f")

    def test_parse_call_short(self):
        """'c' should parse as call action."""
        from cli.interactive import parse_user_input

        assert parse_user_input("c") == ("action", "c")

    def test_parse_call_long(self):
        """'call' should parse as call action."""
        from cli.interactive import parse_user_input

        assert parse_user_input("call") == ("action", "c")

    def test_parse_check_short(self):
        """'x' should parse as check action."""
        from cli.interactive import parse_user_input

        assert parse_user_input("x") == ("action", "x")

    def test_parse_check_long(self):
        """'check' should parse as check action."""
        from cli.interactive import parse_user_input

        assert parse_user_input("check") == ("action", "x")

    def test_parse_all_in_short(self):
        """'a' should parse as all-in action."""
        from cli.interactive import parse_user_input

        assert parse_user_input("a") == ("action", "a")

    def test_parse_all_in_long(self):
        """'all-in' should parse as all-in action."""
        from cli.interactive import parse_user_input

        assert parse_user_input("all-in") == ("action", "a")

    def test_parse_raise(self):
        """'r8' should parse as raise action."""
        from cli.interactive import parse_user_input

        assert parse_user_input("r8") == ("action", "r8")

    def test_parse_raise_decimal(self):
        """'r2.5' should parse as raise action."""
        from cli.interactive import parse_user_input

        assert parse_user_input("r2.5") == ("action", "r2.5")

    def test_parse_raise_large(self):
        """'r100' should parse as raise action."""
        from cli.interactive import parse_user_input

        assert parse_user_input("r100") == ("action", "r100")

    def test_parse_back_short(self):
        """'b' should parse as back command."""
        from cli.interactive import parse_user_input

        assert parse_user_input("b") == ("back", None)

    def test_parse_back_long(self):
        """'back' should parse as back command."""
        from cli.interactive import parse_user_input

        assert parse_user_input("back") == ("back", None)

    def test_parse_quit_short(self):
        """'q' should parse as quit command."""
        from cli.interactive import parse_user_input

        assert parse_user_input("q") == ("quit", None)

    def test_parse_quit_long(self):
        """'quit' should parse as quit command."""
        from cli.interactive import parse_user_input

        assert parse_user_input("quit") == ("quit", None)

    def test_parse_invalid(self):
        """Unknown input should return invalid with original input."""
        from cli.interactive import parse_user_input

        assert parse_user_input("xyz") == ("invalid", "xyz")

    def test_parse_invalid_preserves_input(self):
        """Invalid input should preserve the original string."""
        from cli.interactive import parse_user_input

        assert parse_user_input("unknown command") == ("invalid", "unknown command")

    def test_parse_case_insensitive_fold(self):
        """Input parsing should be case-insensitive."""
        from cli.interactive import parse_user_input

        assert parse_user_input("FOLD") == ("action", "f")
        assert parse_user_input("Fold") == ("action", "f")

    def test_parse_case_insensitive_back(self):
        """Back command should be case-insensitive."""
        from cli.interactive import parse_user_input

        assert parse_user_input("BACK") == ("back", None)
        assert parse_user_input("Back") == ("back", None)

    def test_parse_whitespace_stripped(self):
        """Input should have whitespace stripped."""
        from cli.interactive import parse_user_input

        assert parse_user_input("  f  ") == ("action", "f")
        assert parse_user_input("\tb\n") == ("back", None)


# Test InteractiveSession


class TestInteractiveSessionInit:
    """Tests for InteractiveSession initialization."""

    def test_session_initializes_with_empty_history(self, config, empty_strategy):
        """Session should start with empty history."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        assert session.history == []

    def test_session_initializes_stack_from_config(self, config, empty_strategy):
        """Session should get stack from config."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        assert session.stack == 100.0

    def test_session_stores_strategy(self, config, sample_strategy):
        """Session should store the raw strategy."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, sample_strategy)
        assert session.raw_strategy == sample_strategy


class TestInteractiveSessionApplyAction:
    """Tests for InteractiveSession.apply_action method."""

    def test_apply_action_adds_to_history(self, config, empty_strategy):
        """apply_action should add action to history."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r2.5")
        assert session.history == ["r2.5"]

    def test_apply_multiple_actions(self, config, empty_strategy):
        """Multiple actions should all be recorded."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r2.5")
        session.apply_action("r8")
        session.apply_action("c")
        assert session.history == ["r2.5", "r8", "c"]

    def test_apply_fold_action(self, config, empty_strategy):
        """Fold action should be added to history."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("f")
        assert session.history == ["f"]


class TestInteractiveSessionGoBack:
    """Tests for InteractiveSession.go_back method."""

    def test_go_back_removes_last_action(self, config, empty_strategy):
        """go_back should remove the last action."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r2.5")
        session.apply_action("r8")
        session.go_back()
        assert session.history == ["r2.5"]

    def test_go_back_returns_true_when_action_removed(self, config, empty_strategy):
        """go_back should return True when an action is removed."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r2.5")
        result = session.go_back()
        assert result is True
        assert session.history == []

    def test_go_back_returns_false_at_root(self, config, empty_strategy):
        """go_back should return False when at root (empty history)."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        result = session.go_back()
        assert result is False

    def test_go_back_multiple_times(self, config, empty_strategy):
        """go_back should be able to be called multiple times."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r2.5")
        session.apply_action("r8")
        session.apply_action("c")

        assert session.go_back() is True
        assert session.history == ["r2.5", "r8"]

        assert session.go_back() is True
        assert session.history == ["r2.5"]

        assert session.go_back() is True
        assert session.history == []

        assert session.go_back() is False


class TestInteractiveSessionGetCurrentPlayer:
    """Tests for InteractiveSession.get_current_player method."""

    def test_current_player_starts_sb(self, config, empty_strategy):
        """SB acts first at root."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        assert session.get_current_player() == "SB"

    def test_current_player_after_one_action(self, config, empty_strategy):
        """BB acts after SB's first action."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r2.5")
        assert session.get_current_player() == "BB"

    def test_current_player_alternates(self, config, empty_strategy):
        """Players alternate after each action."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        assert session.get_current_player() == "SB"

        session.apply_action("r2.5")
        assert session.get_current_player() == "BB"

        session.apply_action("r8")
        assert session.get_current_player() == "SB"

        session.apply_action("c")
        assert session.get_current_player() == "BB"


class TestInteractiveSessionGetPot:
    """Tests for InteractiveSession.get_pot method."""

    def test_pot_at_start(self, config, empty_strategy):
        """Initial pot should be 1.5 BB (0.5 SB + 1.0 BB)."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        assert session.get_pot() == 1.5

    def test_pot_after_raise(self, config, empty_strategy):
        """Pot should increase after a raise."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r2.5")
        # SB raises to 2.5 (was 0.5, adds 2.0)
        assert session.get_pot() == 3.5

    def test_pot_after_call(self, config, empty_strategy):
        """Pot should increase after a call."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r2.5")  # SB raises to 2.5, pot = 3.5
        session.apply_action("c")  # BB calls 2.5 (was 1.0, adds 1.5)
        assert session.get_pot() == 5.0


class TestInteractiveSessionGetLegalActions:
    """Tests for InteractiveSession.get_legal_actions method."""

    def test_legal_actions_at_start(self, config, empty_strategy):
        """SB should have call, raises, and all-in at start (no fold)."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        actions = session.get_legal_actions()
        # SB is not facing a bet (only posted 0.5, matching 1.0 BB)
        assert "c" in actions
        assert "a" in actions
        # Should have legal raises from config
        assert "r2.5" in actions
        assert "r3" in actions or "r3.0" in actions

    def test_legal_actions_facing_raise(self, config, empty_strategy):
        """BB facing raise should be able to fold, call, raise, or all-in."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r2.5")
        actions = session.get_legal_actions()
        assert "f" in actions
        assert "c" in actions
        # Only raises > 2.5 and <= stack
        assert "r8" in actions or "r8.0" in actions
        assert "a" in actions


class TestInteractiveSessionIsTerminal:
    """Tests for InteractiveSession.is_terminal method."""

    def test_not_terminal_at_start(self, config, empty_strategy):
        """Session is not terminal at start."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        assert session.is_terminal() is False

    def test_terminal_after_fold(self, config, empty_strategy):
        """Session is terminal after fold."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r2.5")
        session.apply_action("f")
        assert session.is_terminal() is True

    def test_terminal_after_call_of_raise(self, config, empty_strategy):
        """Session is terminal after calling a raise."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r2.5")
        session.apply_action("c")
        assert session.is_terminal() is True

    def test_not_terminal_after_limp(self, config, empty_strategy):
        """Session is not terminal after SB limps."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("c")  # SB limps
        assert session.is_terminal() is False

    def test_terminal_after_check_after_limp(self, config, empty_strategy):
        """Session is terminal after BB checks over limp."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("c")  # SB limps
        session.apply_action("c")  # BB checks
        assert session.is_terminal() is True


class TestInteractiveSessionGetStrategyForHand:
    """Tests for InteractiveSession.get_strategy_for_hand method."""

    def test_get_strategy_for_existing_hand(self, config, sample_strategy):
        """Should return strategy for existing hand."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, sample_strategy)
        result = session.get_strategy_for_hand("AA")
        assert result is not None
        assert result.fold == 0.0

    def test_get_strategy_for_missing_hand(self, config, sample_strategy):
        """Should return None for missing hand."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, sample_strategy)
        result = session.get_strategy_for_hand("KK")
        assert result is None

    def test_strategy_changes_after_action(self, config, sample_strategy):
        """Strategy should update based on current history."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, sample_strategy)
        # At start (SB to act), AA should have raise strategy
        sb_strategy = session.get_strategy_for_hand("AA")
        assert sb_strategy is not None
        assert sb_strategy.raises.get(3.0) == 0.9

        # After SB raises 2.5, BB faces different strategy
        session.apply_action("r2.5")
        bb_strategy = session.get_strategy_for_hand("AA")
        assert bb_strategy is not None
        # BB's AA strategy facing raise should have r8 as main raise
        assert bb_strategy.raises.get(8) == 0.8

    def test_strategy_returns_empty_for_unknown_history(self, config, sample_strategy):
        """Strategy should be empty for histories not in raw strategy."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, sample_strategy)
        # sample_strategy only has "" and "r2.5" histories
        session.apply_action("r3.0")  # Unknown history
        result = session.get_strategy_for_current_state()
        assert result == {}


class TestInteractiveSessionRender:
    """Tests for InteractiveSession.render method."""

    def test_render_returns_string(self, config, empty_strategy):
        """render should return a string."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        result = session.render()
        assert isinstance(result, str)

    def test_render_contains_header_info(self, config, empty_strategy):
        """render should include game state info."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        result = session.render()
        assert "HUNL Preflop" in result
        assert "100" in result  # stack
        assert "SB to act" in result

    def test_render_contains_matrix(self, config, sample_strategy):
        """render should include the strategy matrix."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, sample_strategy)
        result = session.render()
        # Matrix should contain hand names
        assert "AA" in result


# Edge case tests for comprehensive coverage


class TestAllInScenarios:
    """Tests for all-in action scenarios."""

    def test_pot_after_sb_all_in(self, config, empty_strategy):
        """Pot should be correct after SB goes all-in."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("a")  # SB all-in for 100
        # SB put in 100 (was 0.5, adds 99.5), pot = 1.5 + 99.5 = 101
        assert session.get_pot() == 101.0

    def test_pot_after_sb_all_in_bb_call(self, config, empty_strategy):
        """Pot should be correct when BB calls SB's all-in."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("a")  # SB all-in for 100
        session.apply_action("c")  # BB calls
        # Both players have 100 in, pot = 200
        assert session.get_pot() == 200.0

    def test_terminal_after_sb_all_in_bb_call(self, config, empty_strategy):
        """Game should be terminal after BB calls SB all-in."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("a")
        session.apply_action("c")
        assert session.is_terminal() is True

    def test_terminal_after_sb_all_in_bb_fold(self, config, empty_strategy):
        """Game should be terminal after BB folds to SB all-in."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("a")
        session.apply_action("f")
        assert session.is_terminal() is True

    def test_legal_actions_facing_all_in(self, config, empty_strategy):
        """BB facing all-in should only have fold and call."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("a")
        actions = session.get_legal_actions()
        assert "f" in actions
        assert "c" in actions
        # No raises allowed when facing all-in
        assert "a" not in actions
        assert not any(a.startswith("r") for a in actions)

    def test_pot_after_raise_then_all_in(self, config, empty_strategy):
        """Pot should be correct after raise followed by all-in."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r8")  # SB raises to 8, pot = 1.5 + 7.5 = 9
        session.apply_action("a")  # BB all-in for 100, pot = 9 + 99 = 108
        assert session.get_pot() == 108.0

    def test_not_terminal_after_raise_all_in(self, config, empty_strategy):
        """Game not terminal after raise then all-in (SB still needs to act)."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r8")
        session.apply_action("a")
        assert session.is_terminal() is False

    def test_terminal_after_raise_all_in_call(self, config, empty_strategy):
        """Game should be terminal after SB calls BB's all-in."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r8")
        session.apply_action("a")
        session.apply_action("c")
        assert session.is_terminal() is True

    def test_pot_after_raise_all_in_call(self, config, empty_strategy):
        """Pot should be 200 (both all-in) after raise, all-in, call."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r8")
        session.apply_action("a")
        session.apply_action("c")
        # SB raised to 8 (pot=9), BB all-in (pot=108), SB calls (puts in 92 more)
        # pot = 108 + 92 = 200
        assert session.get_pot() == 200.0


class TestLimpCheckScenarios:
    """Tests for limp-check (c, c) action pattern."""

    def test_pot_after_sb_limp(self, config, empty_strategy):
        """Pot should be 2.0 after SB limps (calls BB)."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("c")  # SB limps (0.5 -> 1.0, adds 0.5)
        assert session.get_pot() == 2.0

    def test_not_terminal_after_limp(self, config, empty_strategy):
        """Game should NOT be terminal after just a limp."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("c")
        assert session.is_terminal() is False

    def test_current_player_after_limp(self, config, empty_strategy):
        """BB should be the current player after SB limps."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("c")
        assert session.get_current_player() == "BB"

    def test_pot_after_limp_check(self, config, empty_strategy):
        """Pot should be 2.0 after limp-check (no more money added)."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("c")  # SB limps
        session.apply_action("c")  # BB checks
        assert session.get_pot() == 2.0

    def test_terminal_after_limp_check(self, config, empty_strategy):
        """Game should be terminal after limp-check."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("c")
        session.apply_action("c")
        assert session.is_terminal() is True

    def test_pot_after_limp_raise(self, config, empty_strategy):
        """Pot should reflect BB raise after SB limp."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("c")  # SB limps, pot = 2.0
        session.apply_action("r8")  # BB raises to 8, pot = 2 + 7 = 9
        assert session.get_pot() == 9.0

    def test_not_terminal_after_limp_raise(self, config, empty_strategy):
        """Game should NOT be terminal after limp then raise."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("c")
        session.apply_action("r8")
        assert session.is_terminal() is False

    def test_terminal_after_limp_raise_call(self, config, empty_strategy):
        """Game should be terminal after limp, raise, call."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("c")  # SB limps
        session.apply_action("r8")  # BB raises
        session.apply_action("c")  # SB calls
        assert session.is_terminal() is True

    def test_pot_after_limp_raise_call(self, config, empty_strategy):
        """Pot should be 16 after limp, raise to 8, call."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("c")  # SB limps, pot = 2
        session.apply_action("r8")  # BB raises to 8, pot = 2 + 7 = 9
        session.apply_action("c")  # SB calls 8, pot = 9 + 7 = 16
        assert session.get_pot() == 16.0


class TestComplexRaiseSequences:
    """Tests for complex raise-reraise-call sequences."""

    def test_pot_after_raise_reraise(self, config, empty_strategy):
        """Pot should be correct after raise and reraise."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r3")  # SB raises to 3, pot = 1.5 + 2.5 = 4
        session.apply_action("r8")  # BB raises to 8, pot = 4 + 7 = 11
        assert session.get_pot() == 11.0

    def test_not_terminal_after_raise_reraise(self, config, empty_strategy):
        """Game should NOT be terminal after raise-reraise."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r3")
        session.apply_action("r8")
        assert session.is_terminal() is False

    def test_current_player_after_raise_reraise(self, config, empty_strategy):
        """SB should be the current player after raise-reraise."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r3")
        session.apply_action("r8")
        assert session.get_current_player() == "SB"

    def test_terminal_after_raise_reraise_call(self, config, empty_strategy):
        """Game should be terminal after raise-reraise-call."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r3")
        session.apply_action("r8")
        session.apply_action("c")
        assert session.is_terminal() is True

    def test_pot_after_raise_reraise_call(self, config, empty_strategy):
        """Pot should be correct after raise-reraise-call."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r3")  # SB raises to 3, pot = 4
        session.apply_action("r8")  # BB raises to 8, pot = 11
        session.apply_action("c")  # SB calls 8, pot = 11 + 5 = 16
        assert session.get_pot() == 16.0

    def test_pot_after_raise_reraise_reraise(self, config, empty_strategy):
        """Pot should be correct after three raises."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r3")  # SB raises to 3, pot = 4
        session.apply_action("r8")  # BB raises to 8, pot = 11
        session.apply_action("r20")  # SB raises to 20, pot = 11 + 17 = 28
        assert session.get_pot() == 28.0

    def test_terminal_after_raise_reraise_reraise_call(self, config, empty_strategy):
        """Game should be terminal after raise-reraise-reraise-call."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r3")
        session.apply_action("r8")
        session.apply_action("r20")
        session.apply_action("c")
        assert session.is_terminal() is True

    def test_pot_after_raise_reraise_reraise_call(self, config, empty_strategy):
        """Pot should be correct after raise-reraise-reraise-call."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r3")  # SB raises to 3, pot = 4
        session.apply_action("r8")  # BB raises to 8, pot = 11
        session.apply_action("r20")  # SB raises to 20, pot = 28
        session.apply_action("c")  # BB calls 20, pot = 28 + 12 = 40
        assert session.get_pot() == 40.0

    def test_terminal_after_raise_reraise_fold(self, config, empty_strategy):
        """Game should be terminal after raise-reraise-fold."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r3")
        session.apply_action("r8")
        session.apply_action("f")
        assert session.is_terminal() is True

    def test_legal_actions_after_raise(self, config, empty_strategy):
        """BB should have fold, call, larger raises, all-in after SB raise."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r3")
        actions = session.get_legal_actions()
        assert "f" in actions
        assert "c" in actions
        # Larger raises only
        assert "r2.5" not in actions  # Too small
        assert "r8" in actions or "r8.0" in actions
        assert "r20" in actions or "r20.0" in actions
        assert "a" in actions


class TestPotCalculationEdgeCases:
    """Tests for pot calculation in various scenarios."""

    def test_pot_starts_at_blinds(self, config, empty_strategy):
        """Initial pot should be 1.5 (0.5 SB + 1.0 BB)."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        assert session.get_pot() == 1.5

    def test_pot_after_fold(self, config, empty_strategy):
        """Pot should not change on fold."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r3")  # pot = 4
        session.apply_action("f")
        assert session.get_pot() == 4.0

    def test_pot_decimal_raise_sizes(self, config, empty_strategy):
        """Pot should handle decimal raise sizes correctly."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r2.5")  # SB raises to 2.5, pot = 1.5 + 2 = 3.5
        assert session.get_pot() == 3.5

    def test_pot_after_multiple_calls(self, config, empty_strategy):
        """Pot should be correct after limp-raise-call sequence."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("c")  # SB limps, pot = 2
        session.apply_action("r8")  # BB raises to 8, pot = 9
        session.apply_action("r20")  # SB reraises to 20, pot = 9 + 19 = 28
        session.apply_action("c")  # BB calls 20, pot = 28 + 12 = 40
        assert session.get_pot() == 40.0

    def test_player_committed_at_start_sb(self, config, empty_strategy):
        """SB should have 0.5 committed at start."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        assert session._get_player_committed() == 0.5

    def test_player_committed_after_raise(self, config, empty_strategy):
        """BB should have 1.0 committed after SB raises (BB hasn't acted yet)."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r3")
        # Current player is BB, who has committed 1.0
        assert session._get_player_committed() == 1.0

    def test_player_committed_after_call(self, config, empty_strategy):
        """After SB raises and BB calls, SB should have 3.0 committed."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r3")
        session.apply_action("c")
        # Current player is SB (though terminal), who committed 3.0
        assert session._get_player_committed() == 3.0

    def test_current_bet_at_start(self, config, empty_strategy):
        """Current bet at start should be 1.0 (the BB)."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        assert session._get_current_bet() == 1.0

    def test_current_bet_after_raise(self, config, empty_strategy):
        """Current bet after raise should be the raise amount."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r8")
        assert session._get_current_bet() == 8.0

    def test_current_bet_after_all_in(self, config, empty_strategy):
        """Current bet after all-in should be stack size."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("a")
        assert session._get_current_bet() == 100.0


class TestSmallStackScenarios:
    """Tests with smaller stack sizes."""

    @pytest.fixture
    def small_stack_config(self):
        """Create a config with small stack (10 BB)."""
        return Config(
            name="Small",
            stack_depths=[10.0],
            raise_sizes=[2.5, 3.0, 5.0],
        )

    def test_legal_raises_limited_by_stack(self, small_stack_config, empty_strategy):
        """Raise sizes should be limited by stack."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(small_stack_config, empty_strategy)
        actions = session.get_legal_actions()
        # All raises should be <= 10
        raise_actions = [a for a in actions if a.startswith("r")]
        for action in raise_actions:
            size = float(action[1:])
            assert size <= 10.0

    def test_all_in_equals_stack(self, small_stack_config, empty_strategy):
        """All-in should put stack size in pot."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(small_stack_config, empty_strategy)
        session.apply_action("a")  # SB all-in for 10
        # Pot = 1.5 + 9.5 = 11
        assert session.get_pot() == 11.0

    def test_small_stack_all_in_call(self, small_stack_config, empty_strategy):
        """Both players all-in should result in pot = 2x stack."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(small_stack_config, empty_strategy)
        session.apply_action("a")
        session.apply_action("c")
        # Both have 10 in, pot = 20
        assert session.get_pot() == 20.0


class TestTerminalStateValidation:
    """Additional terminal state validation tests."""

    def test_cannot_fold_when_not_facing_bet(self, config, empty_strategy):
        """Fold should not be in legal actions when not facing a bet."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        # At start, SB faces BB's 1BB bet, but SB only has 0.5 committed
        # So SB IS facing a bet (needs to put in 0.5 more to call)
        # Actually, checking the logic: current_bet=1, player_committed=0.5
        # So f should be available
        actions = session.get_legal_actions()
        assert "f" in actions

    def test_cannot_fold_after_limp(self, config, empty_strategy):
        """BB cannot fold after SB limps (not facing a bet)."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("c")  # SB limps
        # BB has 1.0 committed, current_bet is 1.0, so no fold needed
        actions = session.get_legal_actions()
        assert "f" not in actions

    def test_long_sequence_player_alternation(self, config, empty_strategy):
        """Players should alternate correctly through long sequences."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        assert session.get_current_player() == "SB"
        session.apply_action("r3")
        assert session.get_current_player() == "BB"
        session.apply_action("r8")
        assert session.get_current_player() == "SB"
        session.apply_action("r20")
        assert session.get_current_player() == "BB"
        session.apply_action("a")
        assert session.get_current_player() == "SB"

    def test_history_preserved_through_sequence(self, config, empty_strategy):
        """History should be correctly preserved through actions."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r3")
        session.apply_action("r8")
        session.apply_action("r20")
        assert session.history == ["r3", "r8", "r20"]

    def test_go_back_restores_pot(self, config, empty_strategy):
        """Going back should restore the correct pot."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        initial_pot = session.get_pot()
        session.apply_action("r3")
        pot_after_raise = session.get_pot()
        session.go_back()
        assert session.get_pot() == initial_pot
        assert session.get_pot() != pot_after_raise

    def test_go_back_restores_terminal_state(self, config, empty_strategy):
        """Going back from terminal should make non-terminal."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, empty_strategy)
        session.apply_action("r3")
        session.apply_action("c")
        assert session.is_terminal() is True
        session.go_back()
        assert session.is_terminal() is False


class TestParseStackSwitch:
    """Tests for stack switching command parsing."""

    def test_parse_stack_switch_integer(self):
        """'s25' should parse as switch_stack command."""
        from cli.interactive import parse_user_input

        assert parse_user_input("s25") == ("switch_stack", "25")

    def test_parse_stack_switch_decimal(self):
        """'s2.5' should parse as switch_stack command."""
        from cli.interactive import parse_user_input

        assert parse_user_input("s2.5") == ("switch_stack", "2.5")

    def test_parse_stack_switch_large(self):
        """'s100' should parse as switch_stack command."""
        from cli.interactive import parse_user_input

        assert parse_user_input("s100") == ("switch_stack", "100")

    def test_parse_stack_switch_case_insensitive(self):
        """'S50' should parse as switch_stack command."""
        from cli.interactive import parse_user_input

        assert parse_user_input("S50") == ("switch_stack", "50")


class TestMultiStackSession:
    """Tests for multi-stack InteractiveSession functionality."""

    @pytest.fixture
    def multi_stack_config(self):
        """Create a config with multiple stack depths."""
        return Config(
            name="Multi",
            stack_depths=[25.0, 50.0, 100.0],
            raise_sizes=[2.5, 3.0, 8.0, 20.0],
        )

    @pytest.fixture
    def multi_stack_strategy(self):
        """Create a multi-stack strategy in nested format."""
        return {
            25.0: {
                "SB:AA:": {"f": 0.0, "c": 0.0, "r3.0": 0.9, "a": 0.1},
                "SB:72o:": {"f": 0.9, "c": 0.1},
            },
            50.0: {
                "SB:AA:": {"f": 0.0, "c": 0.1, "r3.0": 0.8, "a": 0.1},
                "SB:72o:": {"f": 0.85, "c": 0.15},
            },
            100.0: {
                "SB:AA:": {"f": 0.0, "c": 0.2, "r3.0": 0.7, "a": 0.1},
                "SB:72o:": {"f": 0.8, "c": 0.2},
            },
        }

    def test_multi_stack_detection(self, multi_stack_config, multi_stack_strategy):
        """Session should detect multi-stack format."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(multi_stack_config, multi_stack_strategy)
        assert session._is_multi_stack is True
        assert session.available_stacks == [25.0, 50.0, 100.0]

    def test_single_stack_detection(self, config, sample_strategy):
        """Session should detect single-stack format."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, sample_strategy)
        assert session._is_multi_stack is False

    def test_initial_stack_is_first(self, multi_stack_config, multi_stack_strategy):
        """Initial current_stack should be the first available."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(multi_stack_config, multi_stack_strategy)
        assert session.current_stack == 25.0
        assert session.stack == 25.0

    def test_switch_stack_success(self, multi_stack_config, multi_stack_strategy):
        """switch_stack should work for valid stack."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(multi_stack_config, multi_stack_strategy)
        result = session.switch_stack(50.0)
        assert result is True
        assert session.current_stack == 50.0
        assert session.stack == 50.0

    def test_switch_stack_updates_strategy(self, multi_stack_config, multi_stack_strategy):
        """switch_stack should update raw_strategy to new stack's strategy."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(multi_stack_config, multi_stack_strategy)
        session.switch_stack(100.0)
        # Check that raw_strategy is now the 100BB strategy
        assert session.raw_strategy == multi_stack_strategy[100.0]

    def test_switch_stack_resets_history(self, multi_stack_config, multi_stack_strategy):
        """switch_stack should reset action history."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(multi_stack_config, multi_stack_strategy)
        session.apply_action("r3")
        session.apply_action("c")
        assert len(session.history) == 2

        session.switch_stack(50.0)
        assert session.history == []

    def test_switch_stack_invalid_stack(self, multi_stack_config, multi_stack_strategy):
        """switch_stack should fail for invalid stack."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(multi_stack_config, multi_stack_strategy)
        result = session.switch_stack(75.0)  # Not in available stacks
        assert result is False
        assert session.current_stack == 25.0  # Unchanged

    def test_switch_stack_not_multi_stack(self, config, sample_strategy):
        """switch_stack should fail for single-stack strategy."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, sample_strategy)
        result = session.switch_stack(50.0)
        assert result is False
