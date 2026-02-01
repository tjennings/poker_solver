import pytest
import torch
from cfr.batched import BatchedCFR
from core.tensors import compile_game
from games.kuhn import KuhnPoker


def test_regrets_never_negative():
    """After training, regrets should be >= 0 (CFR+ property)."""
    game = KuhnPoker()
    compiled = compile_game(game, torch.device("cpu"))
    cfr = BatchedCFR(compiled, batch_size=32)

    for _ in range(100):
        cfr.train_step()

    assert (cfr.regret_sum >= 0).all()


def test_cfr_plus_converges():
    """CFR+ should produce valid strategies."""
    game = KuhnPoker()
    compiled = compile_game(game, torch.device("cpu"))
    cfr = BatchedCFR(compiled, batch_size=32)

    for _ in range(200):
        cfr.train_step()

    strategy = cfr.get_current_strategy()
    sums = strategy.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
