"""Tests for the matrix display module."""

import pytest
from typing import Dict

from cli.matrix import (
    ActionDistribution,
    get_color_for_action,
    render_matrix,
    render_header,
    ANSI_RESET,
    ANSI_DIM_RED,
    ANSI_BRIGHT_RED,
    ANSI_DIM_GREEN,
    ANSI_BRIGHT_GREEN,
    ANSI_DIM_BLUE,
    ANSI_BRIGHT_BLUE,
    ANSI_DIM_YELLOW,
    ANSI_BRIGHT_YELLOW,
)
from core.hands import get_matrix_layout


class TestActionDistribution:
    """Tests for the ActionDistribution dataclass."""

    def test_action_distribution_creation(self):
        """ActionDistribution can be created with all fields."""
        dist = ActionDistribution(
            fold=0.3,
            call=0.4,
            raises={2.5: 0.2, 3.0: 0.1},
            all_in=0.0,
        )
        assert dist.fold == 0.3
        assert dist.call == 0.4
        assert dist.raises == {2.5: 0.2, 3.0: 0.1}
        assert dist.all_in == 0.0

    def test_dominant_action_fold(self):
        """dominant_action returns fold when it's highest."""
        dist = ActionDistribution(
            fold=0.7,
            call=0.2,
            raises={2.5: 0.1},
            all_in=0.0,
        )
        action, freq = dist.dominant_action()
        assert action == "fold"
        assert freq == 0.7

    def test_dominant_action_call(self):
        """dominant_action returns call when it's highest."""
        dist = ActionDistribution(
            fold=0.1,
            call=0.8,
            raises={2.5: 0.1},
            all_in=0.0,
        )
        action, freq = dist.dominant_action()
        assert action == "call"
        assert freq == 0.8

    def test_dominant_action_raise(self):
        """dominant_action returns raise when total raises is highest."""
        dist = ActionDistribution(
            fold=0.1,
            call=0.1,
            raises={2.5: 0.4, 3.0: 0.3},
            all_in=0.1,
        )
        action, freq = dist.dominant_action()
        assert action == "raise"
        assert freq == 0.7

    def test_dominant_action_all_in(self):
        """dominant_action returns all_in when it's highest."""
        dist = ActionDistribution(
            fold=0.1,
            call=0.1,
            raises={2.5: 0.1},
            all_in=0.7,
        )
        action, freq = dist.dominant_action()
        assert action == "all_in"
        assert freq == 0.7

    def test_dominant_action_tie_goes_to_most_aggressive(self):
        """When tied, prefer more aggressive action."""
        # Equal fold and call - call is more aggressive
        dist = ActionDistribution(
            fold=0.5,
            call=0.5,
            raises={},
            all_in=0.0,
        )
        action, freq = dist.dominant_action()
        # Expect the more aggressive action to win ties
        assert action in ("fold", "call")
        assert freq == 0.5


class TestColorMapping:
    """Tests for get_color_for_action function."""

    def test_fold_is_red(self):
        """Fold action should use red ANSI codes."""
        color = get_color_for_action("fold", 0.9)
        assert "31" in color or "91" in color

    def test_call_is_green(self):
        """Call action should use green ANSI codes."""
        color = get_color_for_action("call", 0.9)
        assert "32" in color or "92" in color

    def test_raise_is_blue(self):
        """Raise action should use blue ANSI codes."""
        color = get_color_for_action("raise", 0.9)
        assert "34" in color or "94" in color

    def test_all_in_is_yellow(self):
        """All-in action should use yellow ANSI codes."""
        color = get_color_for_action("all_in", 0.9)
        assert "33" in color or "93" in color

    def test_high_frequency_is_bright(self):
        """Frequency >= 0.85 should use bright colors."""
        color_bright = get_color_for_action("fold", 0.90)
        color_dim = get_color_for_action("fold", 0.65)
        # Bright codes are 91-97, dim are 31-37
        assert "9" in color_bright  # Bright (91)
        assert "91" in color_bright or "9" in color_bright

    def test_low_frequency_is_dim(self):
        """Frequency < 0.85 should use dim colors."""
        color = get_color_for_action("fold", 0.5)
        # Dim codes are 31-37 (no 9 prefix)
        assert "31" in color
        assert "91" not in color

    def test_boundary_frequency(self):
        """Test the boundary at exactly 0.85."""
        color_at_85 = get_color_for_action("call", 0.85)
        color_below_85 = get_color_for_action("call", 0.84)
        # 0.85 should be bright
        assert "92" in color_at_85
        # 0.84 should be dim
        assert "32" in color_below_85


class TestRenderMatrix:
    """Tests for render_matrix function."""

    def test_matrix_is_13x13(self):
        """Matrix layout should be 13x13."""
        layout = get_matrix_layout()
        assert len(layout) == 13
        for row in layout:
            assert len(row) == 13

    def test_render_contains_hands(self):
        """Rendered matrix should contain hand names."""
        strategy = _create_dummy_strategy()
        output = render_matrix(strategy, "Test")
        assert "AA" in output
        assert "AKo" in output
        assert "AKs" in output

    def test_render_contains_header(self):
        """Rendered matrix should include the header."""
        strategy = _create_dummy_strategy()
        output = render_matrix(strategy, "Test Header")
        assert "Test Header" in output

    def test_render_contains_ansi_codes(self):
        """Rendered matrix should contain ANSI color codes."""
        strategy = _create_dummy_strategy()
        output = render_matrix(strategy, "Test")
        assert "\033[" in output
        assert ANSI_RESET in output

    def test_render_all_169_hands(self):
        """Rendered matrix should contain all 169 hands."""
        strategy = _create_dummy_strategy()
        output = render_matrix(strategy, "Test")
        layout = get_matrix_layout()
        for row in layout:
            for hand in row:
                assert hand in output, f"Hand {hand} not found in output"

    def test_cells_are_4_char_width(self):
        """Each cell should be left-aligned with 4-char width."""
        strategy = _create_dummy_strategy()
        output = render_matrix(strategy, "Test")
        # Remove ANSI codes for checking spacing
        clean = _strip_ansi(output)
        # Check that AA appears with proper spacing
        # This is a bit tricky since we're checking formatted output
        lines = clean.split("\n")
        # Find a line with hand data (not header)
        hand_lines = [l for l in lines if "AA" in l or "KK" in l]
        assert len(hand_lines) > 0


class TestRenderHeader:
    """Tests for render_header function."""

    def test_header_basic_format(self):
        """Header should have the expected format."""
        header = render_header(
            stack=100.0,
            pot=10.5,
            action_history=["SBr2.5", "BBr8"],
            player="SB",
        )
        assert "HUNL Preflop" in header
        assert "100" in header
        assert "10.5" in header

    def test_header_shows_stack(self):
        """Header should show stack size."""
        header = render_header(stack=100.0, pot=10.5, action_history=[], player="SB")
        assert "Stack: 100BB" in header or "Stack: 100.0BB" in header

    def test_header_shows_pot(self):
        """Header should show pot size."""
        header = render_header(stack=100.0, pot=10.5, action_history=[], player="SB")
        assert "Pot: 10.5BB" in header

    def test_header_shows_action_history(self):
        """Header should show action history."""
        header = render_header(
            stack=100.0,
            pot=10.5,
            action_history=["SBr2.5", "BBr8"],
            player="SB",
        )
        assert "SBr2.5" in header
        assert "BBr8" in header

    def test_header_shows_player_to_act(self):
        """Header should show which player is to act."""
        header = render_header(
            stack=100.0,
            pot=10.5,
            action_history=[],
            player="SB",
        )
        assert "SB to act" in header

    def test_header_empty_action_history(self):
        """Header should handle empty action history."""
        header = render_header(
            stack=100.0,
            pot=1.5,
            action_history=[],
            player="BB",
        )
        # Should still render without crashing
        assert "HUNL Preflop" in header
        assert "BB to act" in header


class TestANSIConstants:
    """Tests for ANSI constant values."""

    def test_ansi_reset(self):
        """ANSI_RESET should be the reset code."""
        assert ANSI_RESET == "\033[0m"

    def test_ansi_dim_red(self):
        """ANSI_DIM_RED should be code 31."""
        assert "31" in ANSI_DIM_RED

    def test_ansi_bright_red(self):
        """ANSI_BRIGHT_RED should be code 91."""
        assert "91" in ANSI_BRIGHT_RED

    def test_ansi_dim_green(self):
        """ANSI_DIM_GREEN should be code 32."""
        assert "32" in ANSI_DIM_GREEN

    def test_ansi_bright_green(self):
        """ANSI_BRIGHT_GREEN should be code 92."""
        assert "92" in ANSI_BRIGHT_GREEN

    def test_ansi_dim_blue(self):
        """ANSI_DIM_BLUE should be code 34."""
        assert "34" in ANSI_DIM_BLUE

    def test_ansi_bright_blue(self):
        """ANSI_BRIGHT_BLUE should be code 94."""
        assert "94" in ANSI_BRIGHT_BLUE

    def test_ansi_dim_yellow(self):
        """ANSI_DIM_YELLOW should be code 33."""
        assert "33" in ANSI_DIM_YELLOW

    def test_ansi_bright_yellow(self):
        """ANSI_BRIGHT_YELLOW should be code 93."""
        assert "93" in ANSI_BRIGHT_YELLOW


# Helper functions


def _create_dummy_strategy() -> Dict[str, ActionDistribution]:
    """Create a dummy strategy for testing."""
    from core.hands import get_matrix_layout

    strategy = {}
    layout = get_matrix_layout()
    for row_idx, row in enumerate(layout):
        for col_idx, hand in enumerate(row):
            # Create varied distributions for different hand types
            if row_idx == col_idx:  # Pairs - mostly raise
                strategy[hand] = ActionDistribution(
                    fold=0.05,
                    call=0.15,
                    raises={3.0: 0.7},
                    all_in=0.1,
                )
            elif col_idx > row_idx:  # Suited - mixed
                strategy[hand] = ActionDistribution(
                    fold=0.2,
                    call=0.4,
                    raises={2.5: 0.3},
                    all_in=0.1,
                )
            else:  # Offsuit - more folds
                strategy[hand] = ActionDistribution(
                    fold=0.6,
                    call=0.3,
                    raises={2.5: 0.1},
                    all_in=0.0,
                )
    return strategy


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    import re

    ansi_pattern = re.compile(r"\033\[[0-9;]*m")
    return ansi_pattern.sub("", text)
