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
        # GPU may converge slightly slower due to iteration counting differences
        assert cpu_exploit < 0.02, f"CPU exploitability {cpu_exploit} too high"
        assert gpu_exploit < 0.05, f"GPU exploitability {gpu_exploit} too high"

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
