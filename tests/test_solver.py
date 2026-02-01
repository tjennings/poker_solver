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
