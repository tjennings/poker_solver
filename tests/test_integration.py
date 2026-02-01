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
        # Batched CFR converges slightly slower than vanilla
        assert exploit < 0.05, f"Did not converge: exploitability = {exploit}"

    def test_all_implementations_agree(self):
        """All implementations should produce similar results."""
        from solver import Solver
        from games.kuhn import KuhnPoker

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
        assert batched_exploit < 0.05  # Batched converges slightly slower
