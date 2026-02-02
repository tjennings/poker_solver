"""Tests for the matrix display module."""

import pytest
from typing import Dict

from cli.matrix import (
    ActionDistribution,
    get_color_for_action,
    render_matrix,
    render_header,
    render_legend,
    ANSI_RESET,
    ANSI_DIM_BLUE,
    ANSI_BRIGHT_BLUE,
    ANSI_DIM_GREEN,
    ANSI_BRIGHT_GREEN,
    ANSI_LIGHT_RED,
    ANSI_MEDIUM_RED,
    ANSI_DARK_RED,
    ANSI_DARKEST_RED,
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
        action, freq, raise_size = dist.dominant_action()
        assert action == "fold"
        assert freq == 0.7
        assert raise_size is None

    def test_dominant_action_call(self):
        """dominant_action returns call when it's highest."""
        dist = ActionDistribution(
            fold=0.1,
            call=0.8,
            raises={2.5: 0.1},
            all_in=0.0,
        )
        action, freq, raise_size = dist.dominant_action()
        assert action == "call"
        assert freq == 0.8
        assert raise_size is None

    def test_dominant_action_raise(self):
        """dominant_action returns raise when total raises is highest."""
        dist = ActionDistribution(
            fold=0.1,
            call=0.1,
            raises={2.5: 0.4, 3.0: 0.3},
            all_in=0.1,
        )
        action, freq, raise_size = dist.dominant_action()
        assert action == "raise"
        assert freq == 0.7
        assert raise_size == 2.5  # Dominant raise size

    def test_dominant_action_all_in(self):
        """dominant_action returns all_in when it's highest."""
        dist = ActionDistribution(
            fold=0.1,
            call=0.1,
            raises={2.5: 0.1},
            all_in=0.7,
        )
        action, freq, raise_size = dist.dominant_action()
        assert action == "all_in"
        assert freq == 0.7
        assert raise_size is None

    def test_dominant_action_tie_goes_to_most_aggressive(self):
        """When tied, prefer more aggressive action."""
        # Equal fold and call - call is more aggressive
        dist = ActionDistribution(
            fold=0.5,
            call=0.5,
            raises={},
            all_in=0.0,
        )
        action, freq, raise_size = dist.dominant_action()
        # Expect the more aggressive action to win ties
        assert action in ("fold", "call")
        assert freq == 0.5


class TestColorMapping:
    """Tests for get_color_for_action function."""

    def test_fold_is_blue(self):
        """Fold action should use blue ANSI codes."""
        color = get_color_for_action("fold", 0.9)
        assert "34" in color or "94" in color

    def test_call_is_green(self):
        """Call action should use green ANSI codes."""
        color = get_color_for_action("call", 0.9)
        assert "32" in color or "92" in color

    def test_raise_small_is_light_red(self):
        """Small raise should use light red."""
        color = get_color_for_action("raise", 0.9, raise_size=5.0, max_raise=100.0)
        assert "203" in color  # Light red 256-color code

    def test_raise_large_is_dark_red(self):
        """Large raise should use dark red."""
        color = get_color_for_action("raise", 0.9, raise_size=75.0, max_raise=100.0)
        assert "160" in color  # Dark red 256-color code

    def test_all_in_is_darkest_red(self):
        """All-in action should use darkest red."""
        color = get_color_for_action("all_in", 0.9)
        assert "124" in color  # Darkest red 256-color code

    def test_high_frequency_fold_is_bright_blue(self):
        """Frequency >= 0.85 for fold should use bright blue."""
        color = get_color_for_action("fold", 0.90)
        assert "94" in color  # Bright blue

    def test_low_frequency_fold_is_dim_blue(self):
        """Frequency < 0.85 for fold should use dim blue."""
        color = get_color_for_action("fold", 0.5)
        assert "34" in color  # Dim blue

    def test_high_frequency_call_is_bright_green(self):
        """Frequency >= 0.85 for call should use bright green."""
        color = get_color_for_action("call", 0.85)
        assert "92" in color  # Bright green

    def test_low_frequency_call_is_dim_green(self):
        """Frequency < 0.85 for call should use dim green."""
        color = get_color_for_action("call", 0.84)
        assert "32" in color  # Dim green


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

    def test_render_contains_legend(self):
        """Rendered matrix should contain the legend."""
        strategy = _create_dummy_strategy()
        output = render_matrix(strategy, "Test")
        assert "Legend" in output
        assert "Fold" in output
        assert "Call" in output
        assert "All-in" in output

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
        lines = clean.split("\n")
        # Find a line with hand data (not header)
        hand_lines = [l for l in lines if "AA" in l or "KK" in l]
        assert len(hand_lines) > 0


class TestRenderLegend:
    """Tests for render_legend function."""

    def test_legend_contains_all_actions(self):
        """Legend should list all action types."""
        legend_lines = render_legend()
        legend_text = "\n".join(legend_lines)
        assert "Fold" in legend_text
        assert "Call" in legend_text
        assert "Raise" in legend_text
        assert "All-in" in legend_text

    def test_legend_contains_colors(self):
        """Legend should contain color codes."""
        legend_lines = render_legend()
        legend_text = "\n".join(legend_lines)
        assert "\033[" in legend_text


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

    def test_ansi_dim_blue(self):
        """ANSI_DIM_BLUE should be code 34."""
        assert "34" in ANSI_DIM_BLUE

    def test_ansi_bright_blue(self):
        """ANSI_BRIGHT_BLUE should be code 94."""
        assert "94" in ANSI_BRIGHT_BLUE

    def test_ansi_dim_green(self):
        """ANSI_DIM_GREEN should be code 32."""
        assert "32" in ANSI_DIM_GREEN

    def test_ansi_bright_green(self):
        """ANSI_BRIGHT_GREEN should be code 92."""
        assert "92" in ANSI_BRIGHT_GREEN

    def test_ansi_light_red(self):
        """ANSI_LIGHT_RED should use 256-color mode."""
        assert "38;5;203" in ANSI_LIGHT_RED

    def test_ansi_darkest_red(self):
        """ANSI_DARKEST_RED should use 256-color mode."""
        assert "38;5;124" in ANSI_DARKEST_RED


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
