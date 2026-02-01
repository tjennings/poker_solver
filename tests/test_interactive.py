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
        stack_depth=100.0,
        raise_sizes=[2.5, 3.0, 8.0, 20.0],
    )


@pytest.fixture
def empty_strategy():
    """Empty strategy for testing."""
    return {}


@pytest.fixture
def sample_strategy():
    """Create a sample strategy for testing."""
    return {
        "AA": ActionDistribution(fold=0.0, call=0.0, raises={3.0: 0.9}, all_in=0.1),
        "AKs": ActionDistribution(fold=0.0, call=0.2, raises={2.5: 0.7}, all_in=0.1),
        "72o": ActionDistribution(fold=0.9, call=0.1, raises={}, all_in=0.0),
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
        """Session should store the strategy."""
        from cli.interactive import InteractiveSession

        session = InteractiveSession(config, sample_strategy)
        assert session.strategy == sample_strategy


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
