import pytest
from cfr.vanilla import VanillaCFR
from games.kuhn import KuhnPoker, KuhnState


class TestRegretMatching:
    @pytest.fixture
    def cfr(self):
        return VanillaCFR(KuhnPoker())

    def test_uniform_with_no_regrets(self, cfr):
        """With no regrets, strategy should be uniform."""
        strategy = cfr.get_strategy("0:", ["p", "b"])
        assert strategy == {"p": 0.5, "b": 0.5}

    def test_uniform_with_all_negative_regrets(self, cfr):
        """With all negative regrets, strategy should be uniform."""
        cfr.regret_sum["0:"]["p"] = -5.0
        cfr.regret_sum["0:"]["b"] = -3.0
        strategy = cfr.get_strategy("0:", ["p", "b"])
        assert strategy == {"p": 0.5, "b": 0.5}

    def test_proportional_to_positive_regret(self, cfr):
        """Strategy should be proportional to positive regrets."""
        cfr.regret_sum["0:"]["p"] = 3.0
        cfr.regret_sum["0:"]["b"] = 1.0
        strategy = cfr.get_strategy("0:", ["p", "b"])
        assert strategy["p"] == pytest.approx(0.75)
        assert strategy["b"] == pytest.approx(0.25)

    def test_ignores_negative_regret(self, cfr):
        """Negative regrets should be treated as zero."""
        cfr.regret_sum["0:"]["p"] = 4.0
        cfr.regret_sum["0:"]["b"] = -2.0
        strategy = cfr.get_strategy("0:", ["p", "b"])
        assert strategy["p"] == pytest.approx(1.0)
        assert strategy["b"] == pytest.approx(0.0)


class TestCFRTraversal:
    @pytest.fixture
    def cfr(self):
        return VanillaCFR(KuhnPoker())

    def test_cfr_returns_utilities(self, cfr):
        """CFR should return utility tuple for both players."""
        state = KuhnState(cards=(0, 1), history=())
        utils = cfr.cfr(state, reach_probs=(1.0, 1.0))

        assert isinstance(utils, tuple)
        assert len(utils) == 2
        assert all(isinstance(u, float) for u in utils)

    def test_cfr_terminal_returns_utility(self, cfr):
        """At terminal state, CFR should return actual utilities."""
        # P0 has King, P1 has Jack, check-check -> P0 wins 1
        state = KuhnState(cards=(2, 0), history=("p", "p"))
        utils = cfr.cfr(state, reach_probs=(1.0, 1.0))

        assert utils[0] == 1.0
        assert utils[1] == -1.0

    def test_cfr_updates_regrets(self, cfr):
        """CFR should update regret sums."""
        state = KuhnState(cards=(0, 1), history=())

        # Before: no regrets
        assert len(cfr.regret_sum) == 0

        cfr.cfr(state, reach_probs=(1.0, 1.0))

        # After: should have regrets for visited info sets
        assert len(cfr.regret_sum) > 0

    def test_cfr_updates_strategy_sum(self, cfr):
        """CFR should accumulate strategy weights."""
        state = KuhnState(cards=(0, 1), history=())

        cfr.cfr(state, reach_probs=(1.0, 1.0))

        # Should have accumulated strategies
        assert len(cfr.strategy_sum) > 0
        # Root info set should have both actions
        assert "p" in cfr.strategy_sum["0:"]
        assert "b" in cfr.strategy_sum["0:"]


class TestCFRTraining:
    def test_train_runs_iterations(self):
        """Train should run specified number of iterations."""
        cfr = VanillaCFR(KuhnPoker())
        cfr.train(iterations=100)

        # Should have visited all 12 info sets
        all_info_sets = set(cfr.regret_sum.keys()) | set(cfr.strategy_sum.keys())
        assert len(all_info_sets) == 12

    def test_get_average_strategy_normalized(self):
        """Average strategy should sum to 1."""
        cfr = VanillaCFR(KuhnPoker())
        cfr.train(iterations=100)

        for info_set in cfr.strategy_sum:
            strategy = cfr.get_average_strategy(info_set)
            total = sum(strategy.values())
            assert total == pytest.approx(1.0), f"Strategy at {info_set} sums to {total}"

    def test_get_average_strategy_uniform_when_empty(self):
        """Should return uniform if no strategy accumulated."""
        cfr = VanillaCFR(KuhnPoker())
        strategy = cfr.get_average_strategy("nonexistent")
        # With no data, defaults should be uniform over game actions
        assert sum(strategy.values()) == pytest.approx(1.0)

    def test_get_all_average_strategies(self):
        """Should return strategies for all info sets."""
        cfr = VanillaCFR(KuhnPoker())
        cfr.train(iterations=100)

        strategies = cfr.get_all_average_strategies()
        assert len(strategies) == 12

        for info_set, strategy in strategies.items():
            assert "p" in strategy
            assert "b" in strategy
            assert sum(strategy.values()) == pytest.approx(1.0)
