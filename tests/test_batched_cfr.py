import pytest
import torch
from cfr.batched import BatchedCFR
from core.tensors import compile_game
from core.device import get_device
from games.kuhn import KuhnPoker


class TestBatchedCFR:
    @pytest.fixture
    def batched_cfr(self):
        game = KuhnPoker()
        device = get_device("cpu")
        compiled = compile_game(game, device)
        return BatchedCFR(compiled, batch_size=64)

    def test_initialization(self, batched_cfr):
        """Should initialize with correct tensor shapes."""
        assert batched_cfr.regret_sum.shape == (12, 2)  # 12 info sets, 2 actions
        assert batched_cfr.strategy_sum.shape == (12, 2)

    def test_get_current_strategy_uniform_initially(self, batched_cfr):
        """Initial strategy should be uniform."""
        strategy = batched_cfr.get_current_strategy()
        assert strategy.shape == (12, 2)

        # All should be 0.5 (uniform)
        assert torch.allclose(strategy, torch.full_like(strategy, 0.5))

    def test_get_current_strategy_proportional_to_regret(self, batched_cfr):
        """Strategy should be proportional to positive regrets."""
        # Set some regrets
        batched_cfr.regret_sum[0, 0] = 3.0  # Info set 0, action 0
        batched_cfr.regret_sum[0, 1] = 1.0  # Info set 0, action 1

        strategy = batched_cfr.get_current_strategy()

        assert strategy[0, 0].item() == pytest.approx(0.75)
        assert strategy[0, 1].item() == pytest.approx(0.25)

    def test_train_step_updates_regrets(self, batched_cfr):
        """Training step should modify regret sums."""
        initial_regrets = batched_cfr.regret_sum.clone()

        batched_cfr.train_step()

        # Regrets should have changed
        assert not torch.equal(batched_cfr.regret_sum, initial_regrets)

    def test_train_step_updates_strategy_sum(self, batched_cfr):
        """Training step should accumulate strategy weights."""
        initial_sum = batched_cfr.strategy_sum.sum().item()

        batched_cfr.train_step()

        # Strategy sum should have increased
        assert batched_cfr.strategy_sum.sum().item() > initial_sum

    def test_iterations_tracked(self, batched_cfr):
        """Should track number of iterations."""
        assert batched_cfr.iterations == 0

        batched_cfr.train_step()

        assert batched_cfr.iterations == 64  # batch_size
