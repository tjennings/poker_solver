"""Comprehensive integration tests for HUNL preflop solver.

Tests the full pipeline from config loading through game creation,
solving, and interactive exploration.
"""

import pytest
from pathlib import Path

from config.loader import Config, load_config, get_preset_path
from games.hunl_preflop import HUNLPreflop
from solver import Solver
from cli.interactive import InteractiveSession
from cli.parser import parse_action_sequence


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def standard_config():
    """Load the standard preset configuration."""
    return load_config(get_preset_path("standard"))


@pytest.fixture
def aggressive_config():
    """Load the aggressive preset configuration."""
    return load_config(get_preset_path("aggressive"))


@pytest.fixture
def tiny_config():
    """Load the tiny preset configuration."""
    return load_config(get_preset_path("tiny"))


@pytest.fixture
def small_config():
    """Minimal config for fast integration tests."""
    return Config(
        name="Integration Test",
        stack_depths=[10.0],
        raise_sizes=[2.5, 5.0, 10.0],
    )


@pytest.fixture
def mock_strategy():
    """Mock strategy for testing session behavior without solver training."""
    # Create a minimal mock strategy with valid probability distributions
    return {
        "SB:AA:": {"c": 0.3, "r2.5": 0.5, "a": 0.2},
        "SB:KK:": {"c": 0.4, "r2.5": 0.4, "a": 0.2},
        "SB:72o:": {"c": 0.8, "r2.5": 0.1, "a": 0.1},
        "BB:AA:r2.5": {"f": 0.0, "c": 0.2, "r5": 0.6, "a": 0.2},
        "BB:KK:r2.5": {"f": 0.0, "c": 0.3, "r5": 0.5, "a": 0.2},
        "BB:72o:r2.5": {"f": 0.7, "c": 0.2, "r5": 0.05, "a": 0.05},
    }


# =============================================================================
# Full Pipeline Tests
# =============================================================================


class TestFullPipeline:
    """Tests for the complete config -> game -> solver -> strategy pipeline."""

    @pytest.mark.slow
    def test_full_pipeline_with_small_config(self, small_config):
        """Full pipeline test with small config (still needs full tree compilation)."""
        # Create game
        game = HUNLPreflop(small_config)
        initial_states = game.initial_states()
        assert len(initial_states) == 169 * 168  # All hand combinations

        # Create and run solver
        solver = Solver(game, device="cpu", batch_size=64)
        strategy = solver.solve(iterations=200, verbose=False)

        # Verify strategy has many info sets
        assert len(strategy) > 1000  # Many info sets

        # Verify all strategies are valid
        for info_set, probs in strategy.items():
            assert sum(probs.values()) == pytest.approx(1.0, abs=0.01)

    @pytest.mark.slow
    def test_full_pipeline_standard(self):
        """Full pipeline test with standard config."""
        # Load config
        config = load_config(get_preset_path("standard"))
        assert config.name == "Standard 100BB"
        assert config.stack_depth == 100

        # Create game
        game = HUNLPreflop(config)
        initial_states = game.initial_states()
        assert len(initial_states) == 169 * 168  # All hand combinations

        # Create and run solver (fewer iterations for speed)
        solver = Solver(game, device="cpu", batch_size=64)
        strategy = solver.solve(iterations=50, verbose=False)

        # Verify strategy has many info sets
        assert len(strategy) > 500  # Should have many info sets

        # Verify all strategies are valid
        for info_set, probs in strategy.items():
            assert sum(probs.values()) == pytest.approx(1.0, abs=0.01)

    @pytest.mark.slow
    def test_full_pipeline_aggressive(self):
        """Full pipeline test with aggressive config."""
        config = load_config(get_preset_path("aggressive"))
        assert config.name == "Aggressive 100BB"
        assert config.stack_depth == 100
        assert 3 in config.raise_sizes
        assert 60 in config.raise_sizes

        game = HUNLPreflop(config)
        solver = Solver(game, device="cpu", batch_size=64)
        strategy = solver.solve(iterations=50, verbose=False)

        assert len(strategy) > 500

    @pytest.mark.slow
    def test_full_pipeline_tiny(self):
        """Full pipeline test with tiny config."""
        config = load_config(get_preset_path("tiny"))
        assert config.name == "Tiny Test"
        assert config.stack_depth == 10

        game = HUNLPreflop(config)
        solver = Solver(game, device="cpu", batch_size=64)
        strategy = solver.solve(iterations=100, verbose=False)

        assert len(strategy) > 100

    @pytest.mark.slow
    def test_full_pipeline_with_vanilla_cfr(self, small_config):
        """Full pipeline with vanilla CFR (batch_size=1)."""
        game = HUNLPreflop(small_config)
        solver = Solver(game, device="cpu", batch_size=1)
        strategy = solver.solve(iterations=50, verbose=False)

        assert len(strategy) > 0
        for info_set, probs in strategy.items():
            assert sum(probs.values()) == pytest.approx(1.0, abs=0.01)


# =============================================================================
# Interactive Session Workflow Tests
# =============================================================================


class TestInteractiveSessionWorkflow:
    """Tests for the interactive session workflow."""

    def test_basic_session_workflow(self, small_config, mock_strategy):
        """Test basic interactive session workflow."""
        session = InteractiveSession(small_config, mock_strategy)

        # Initial state: SB to act
        assert session.get_current_player() == "SB"
        assert session.history == []

        # Apply SB raise
        session.apply_action("r2.5")
        assert session.get_current_player() == "BB"
        assert session.history == ["r2.5"]

        # Apply BB raise
        session.apply_action("r5")
        assert session.get_current_player() == "SB"
        assert session.history == ["r2.5", "r5"]

        # Go back
        session.go_back()
        assert session.history == ["r2.5"]
        assert session.get_current_player() == "BB"

    def test_session_pot_calculation(self, small_config, mock_strategy):
        """Test pot calculation in session."""
        session = InteractiveSession(small_config, mock_strategy)

        # Initial pot: 1.5 (0.5 SB + 1.0 BB)
        assert session.get_pot() == 1.5

        # SB raises to 2.5 (adds 2.0 more)
        session.apply_action("r2.5")
        assert session.get_pot() == 3.5

        # BB raises to 5 (adds 4.0 more)
        session.apply_action("r5")
        assert session.get_pot() == 7.5

    def test_session_terminal_detection(self, small_config, mock_strategy):
        """Test terminal state detection."""
        session = InteractiveSession(small_config, mock_strategy)

        # Not terminal initially
        assert not session.is_terminal()

        # SB limp, not terminal
        session.apply_action("c")
        assert not session.is_terminal()

        # BB check, terminal
        session.apply_action("c")
        assert session.is_terminal()

    def test_session_fold_terminal(self, small_config, mock_strategy):
        """Test fold creates terminal state."""
        session = InteractiveSession(small_config, mock_strategy)

        # SB raise
        session.apply_action("r2.5")
        assert not session.is_terminal()

        # BB fold - terminal
        session.apply_action("f")
        assert session.is_terminal()

    def test_session_call_after_raise_terminal(self, small_config, mock_strategy):
        """Test call after raise creates terminal state."""
        session = InteractiveSession(small_config, mock_strategy)

        # SB raise
        session.apply_action("r2.5")
        # BB call - terminal
        session.apply_action("c")
        assert session.is_terminal()

    def test_session_navigation_back_to_root(self, small_config, mock_strategy):
        """Test navigating back to root position."""
        session = InteractiveSession(small_config, mock_strategy)

        session.apply_action("r2.5")
        session.apply_action("r5")
        session.apply_action("c")

        # Go back to root
        assert session.go_back()  # Remove call
        assert session.go_back()  # Remove BB raise
        assert session.go_back()  # Remove SB raise

        assert session.history == []
        assert not session.go_back()  # Already at root

    def test_session_legal_actions(self, small_config, mock_strategy):
        """Test legal actions at various positions."""
        session = InteractiveSession(small_config, mock_strategy)

        # Initial: SB can fold (facing BB), limp (c), raise, or all-in
        actions = session.get_legal_actions()
        assert "c" in actions
        assert any(a.startswith("r") for a in actions)
        assert "a" in actions
        # SB CAN fold because they're facing the BB (1BB > 0.5BB already committed)
        assert "f" in actions

        # After SB raise, BB can fold, call, reraise
        session.apply_action("r2.5")
        actions = session.get_legal_actions()
        assert "f" in actions
        assert "c" in actions


# =============================================================================
# Action Sequence Parsing Integration Tests
# =============================================================================


class TestActionSequenceParsingIntegration:
    """Tests for action sequence parsing integrated with the game."""

    def test_action_sequence_simple(self, small_config):
        """Test parsing and executing a simple action sequence."""
        game = HUNLPreflop(small_config)

        parsed = parse_action_sequence("SBr2.5 BBc")
        history = parsed.to_history_tuple()

        # Should parse to ("r2.5", "c")
        assert history == ("r2.5", "c")

        # Execute in game
        state = game.initial_states()[0]
        for action in history:
            assert not game.is_terminal(state)
            state = game.next_state(state, action)

        # Should be terminal (call after raise)
        assert game.is_terminal(state)

    def test_action_sequence_3bet_pot(self, small_config):
        """Test 3-bet action sequence."""
        game = HUNLPreflop(small_config)

        parsed = parse_action_sequence("SBr2.5 BBr5 SBc")
        history = parsed.to_history_tuple()

        assert history == ("r2.5", "r5", "c")

        state = game.initial_states()[0]
        for action in history:
            state = game.next_state(state, action)

        assert game.is_terminal(state)

    def test_action_sequence_fold(self, small_config):
        """Test fold action sequence."""
        game = HUNLPreflop(small_config)

        parsed = parse_action_sequence("SBr2.5 BBf")
        history = parsed.to_history_tuple()

        state = game.initial_states()[0]
        for action in history:
            state = game.next_state(state, action)

        assert game.is_terminal(state)

    def test_action_sequence_limp(self, small_config):
        """Test limp action sequence."""
        game = HUNLPreflop(small_config)

        parsed = parse_action_sequence("SBc BBc")
        history = parsed.to_history_tuple()

        assert history == ("c", "c")

        state = game.initial_states()[0]
        for action in history:
            state = game.next_state(state, action)

        assert game.is_terminal(state)

    def test_action_sequence_with_stack_override(self, small_config):
        """Test action sequence with stack override."""
        parsed = parse_action_sequence("50bb SBr2.5 BBr8 SBc")

        assert parsed.stack_override == 50.0
        assert parsed.to_history_tuple() == ("r2.5", "r8", "c")


# =============================================================================
# Preset Loading Tests
# =============================================================================


class TestPresetsLoadCorrectly:
    """Tests for preset configuration loading."""

    def test_standard_preset_loads(self):
        """Standard preset should load correctly."""
        config = load_config(get_preset_path("standard"))
        assert config.stack_depth > 0
        assert config.name == "Standard 100BB"
        assert len(config.raise_sizes) > 0

    def test_aggressive_preset_loads(self):
        """Aggressive preset should load correctly."""
        config = load_config(get_preset_path("aggressive"))
        assert config.stack_depth > 0
        assert config.name == "Aggressive 100BB"
        assert len(config.raise_sizes) > 0
        # Aggressive has larger raise sizes
        assert 60 in config.raise_sizes
        assert 100 in config.raise_sizes

    def test_tiny_preset_loads(self):
        """Tiny preset should load correctly."""
        config = load_config(get_preset_path("tiny"))
        assert config.stack_depth > 0
        assert config.name == "Tiny Test"
        assert len(config.raise_sizes) > 0

    def test_all_presets_have_valid_configs(self):
        """All presets should have valid configurations."""
        presets = ["standard", "aggressive", "tiny"]

        for preset in presets:
            config = load_config(get_preset_path(preset))
            assert config.stack_depth > 0, f"{preset} has invalid stack_depth"
            assert len(config.raise_sizes) > 0, f"{preset} has no raise_sizes"

            # All raise sizes should be positive
            for size in config.raise_sizes:
                assert size > 0, f"{preset} has invalid raise_size {size}"

    def test_preset_paths_exist(self):
        """Preset files should exist."""
        presets = ["standard", "aggressive", "tiny"]

        for preset in presets:
            path = get_preset_path(preset)
            assert Path(path).exists(), f"Preset file not found: {path}"


# =============================================================================
# Strategy Sanity Checks
# =============================================================================


class TestStrategySanityChecks:
    """Sanity checks for computed strategies."""

    def test_strategy_probabilities_sum_to_one(self, mock_strategy):
        """All strategy probabilities should sum to 1."""
        for info_set, probs in mock_strategy.items():
            total = sum(probs.values())
            assert total == pytest.approx(1.0, abs=0.01), (
                f"Info set {info_set}: probabilities sum to {total}"
            )

    def test_strategy_probabilities_non_negative(self, mock_strategy):
        """All strategy probabilities should be non-negative."""
        for info_set, probs in mock_strategy.items():
            for action, prob in probs.items():
                assert prob >= 0, (
                    f"Info set {info_set}: action {action} has negative prob {prob}"
                )

    def test_strategy_has_valid_actions(self, mock_strategy):
        """Strategy actions should be valid HUNL actions."""
        valid_action_prefixes = {"c", "f", "r", "a"}

        for info_set, probs in mock_strategy.items():
            for action in probs.keys():
                first_char = action[0] if action else ""
                assert first_char in valid_action_prefixes, (
                    f"Info set {info_set}: invalid action {action}"
                )

    def test_strategy_info_set_format(self, mock_strategy):
        """Info set keys should have correct format (POSITION:HAND:HISTORY)."""
        for info_set in mock_strategy.keys():
            parts = info_set.split(":")
            assert len(parts) == 3, f"Invalid info set format: {info_set}"

            position, hand, history = parts
            assert position in {"SB", "BB"}, f"Invalid position in {info_set}"
            # Hand should be 2-3 chars (e.g., "AA", "AKs", "72o")
            assert 2 <= len(hand) <= 3, f"Invalid hand in {info_set}"


# =============================================================================
# End-to-End Tests
# =============================================================================


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_explore_with_mock_strategy(self, small_config, mock_strategy):
        """Test exploring with mock strategy (fast)."""
        # Create session with mock strategy
        session = InteractiveSession(small_config, mock_strategy)

        # Navigate through some positions
        session.apply_action("r2.5")
        assert session.get_current_player() == "BB"

        # Session should be able to render (even if we don't check output)
        output = session.render()
        assert len(output) > 0

        # Navigate back
        session.go_back()
        assert session.get_current_player() == "SB"

    @pytest.mark.slow
    def test_solve_and_explore(self, small_config):
        """Test solving and then exploring the strategy."""
        # Solve
        game = HUNLPreflop(small_config)
        solver = Solver(game, device="cpu", batch_size=64)
        strategy = solver.solve(iterations=200, verbose=False)

        # Create session with solved strategy
        session = InteractiveSession(small_config, strategy)

        # Navigate through some positions
        session.apply_action("r2.5")
        assert session.get_current_player() == "BB"

        # Session should be able to render (even if we don't check output)
        output = session.render()
        assert len(output) > 0

        # Navigate back
        session.go_back()
        assert session.get_current_player() == "SB"

    @pytest.mark.slow
    def test_exploitability_decreases_with_iterations(self, small_config):
        """Exploitability should decrease with more iterations."""
        game = HUNLPreflop(small_config)
        solver = Solver(game, device="cpu", batch_size=64)

        # Train briefly
        solver.solve(iterations=100, verbose=False)
        early_exploit = solver.exploitability()

        # Train more
        solver.solve(iterations=400, verbose=False)
        late_exploit = solver.exploitability()

        # Should decrease (with some tolerance for noise)
        assert late_exploit <= early_exploit * 1.2

    def test_different_configs_produce_different_game_trees(self):
        """Different configs should produce different game trees."""
        tiny_config = load_config(get_preset_path("tiny"))
        aggressive_config = load_config(get_preset_path("aggressive"))

        tiny_game = HUNLPreflop(tiny_config)
        aggressive_game = HUNLPreflop(aggressive_config)

        # Get initial state actions
        tiny_state = tiny_game.initial_states()[0]
        aggressive_state = aggressive_game.initial_states()[0]

        tiny_actions = set(tiny_game.actions(tiny_state))
        aggressive_actions = set(aggressive_game.actions(aggressive_state))

        # Different configs should have different action sets
        assert tiny_actions != aggressive_actions

    @pytest.mark.slow
    def test_consistent_results_across_runs(self, small_config):
        """Multiple solver runs should produce valid strategies."""
        game = HUNLPreflop(small_config)

        for _ in range(3):
            solver = Solver(game, device="cpu", batch_size=64)
            strategy = solver.solve(iterations=100, verbose=False)

            # Each run should produce valid strategy
            assert len(strategy) > 0
            for info_set, probs in strategy.items():
                assert sum(probs.values()) == pytest.approx(1.0, abs=0.01)
