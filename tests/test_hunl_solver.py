"""Tests for HUNL solver integration with dynamic actions."""

import pytest
from config.loader import Config
from games.hunl_preflop import HUNLPreflop
from solver import Solver


@pytest.fixture
def small_config():
    """Minimal config for fast tests."""
    return Config(
        name="Test",
        stack_depths=[5.0],  # Very shallow for fast iteration
        raise_sizes=[2.5, 3.0],  # Only 2 raise sizes
    )


@pytest.mark.slow
class TestHUNLSolver:
    """These tests require full HUNL game tree compilation, which is slow."""

    def test_solver_with_hunl(self, small_config):
        """Solver should work with HUNL preflop game."""
        game = HUNLPreflop(small_config)
        solver = Solver(game, device="cpu", batch_size=32)
        strategy = solver.solve(iterations=100, verbose=False)

        assert len(strategy) > 0

        # Check strategy format - probabilities should sum to 1
        for info_set, probs in strategy.items():
            assert sum(probs.values()) == pytest.approx(1.0, abs=0.01)

    def test_exploitability_decreases(self, small_config):
        """Exploitability should decrease (or stay stable) with more iterations."""
        game = HUNLPreflop(small_config)
        solver = Solver(game, device="cpu", batch_size=32)

        solver.solve(iterations=100, verbose=False)
        exploit_early = solver.exploitability()

        solver.solve(iterations=500, verbose=False)
        exploit_late = solver.exploitability()

        # Allow some variance, but should trend down
        assert exploit_late <= exploit_early * 1.1

    def test_strategy_has_correct_action_names(self, small_config):
        """Strategy should use game-specific action names, not hardcoded Kuhn actions."""
        game = HUNLPreflop(small_config)
        solver = Solver(game, device="cpu", batch_size=32)
        strategy = solver.solve(iterations=100, verbose=False)

        # HUNL actions include things like 'c', 'f', 'r2.5', 'a' - not 'p', 'b'
        for info_set, probs in strategy.items():
            action_names = list(probs.keys())
            # Should have HUNL-specific actions
            # 'c' for call/check is common, 'p'/'b' are Kuhn-specific
            assert "p" not in action_names or "c" in action_names  # If 'p' exists, it's coincidence
            # At minimum, check or call should be available
            assert "c" in action_names

    def test_all_info_sets_have_valid_strategies(self, small_config):
        """Every info set should have a valid probability distribution."""
        game = HUNLPreflop(small_config)
        solver = Solver(game, device="cpu", batch_size=32)
        strategy = solver.solve(iterations=100, verbose=False)

        for info_set, probs in strategy.items():
            # All probabilities should be non-negative
            for action, prob in probs.items():
                assert prob >= 0, f"{info_set}: {action} has negative prob {prob}"

            # Sum should be 1
            total = sum(probs.values())
            assert total == pytest.approx(1.0, abs=0.01), f"{info_set}: probs sum to {total}"

    def test_vanilla_cfr_still_works_with_hunl(self, small_config):
        """Vanilla CFR (batch_size=1) should also work with HUNL."""
        game = HUNLPreflop(small_config)
        solver = Solver(game, device="cpu", batch_size=1)
        strategy = solver.solve(iterations=50, verbose=False)

        assert len(strategy) > 0
        for info_set, probs in strategy.items():
            assert sum(probs.values()) == pytest.approx(1.0, abs=0.01)
