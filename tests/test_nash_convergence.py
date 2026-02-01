import pytest
from cfr.vanilla import VanillaCFR
from games.kuhn import KuhnPoker
from core.exploitability import compute_exploitability


class TestNashConvergence:
    """Verify CFR converges to known Kuhn Poker Nash equilibrium."""

    @pytest.fixture
    def trained_cfr(self):
        """Train CFR for enough iterations to converge."""
        game = KuhnPoker()
        cfr = VanillaCFR(game)
        cfr.train(iterations=10000)
        return cfr

    def test_exploitability_near_zero(self, trained_cfr):
        """Converged strategy should have near-zero exploitability."""
        game = KuhnPoker()
        strategy = trained_cfr.get_all_average_strategies()
        exploit = compute_exploitability(game, strategy)

        # Nash equilibrium has 0 exploitability
        # Allow small epsilon for numerical convergence
        assert exploit < 0.01, f"Exploitability {exploit} too high"

    def test_jack_folds_to_bet(self, trained_cfr):
        """Player with Jack should (almost) always fold when facing a bet."""
        strategy = trained_cfr.get_all_average_strategies()

        # P0 with Jack facing bet after check-bet
        assert strategy["0:pb"]["p"] > 0.95, "Jack should fold to bet (P0)"

        # P1 with Jack facing bet
        assert strategy["0:b"]["p"] > 0.95, "Jack should fold to bet (P1)"

    def test_king_calls_bet(self, trained_cfr):
        """Player with King should always call a bet."""
        strategy = trained_cfr.get_all_average_strategies()

        # P0 with King facing bet
        assert strategy["2:pb"]["b"] > 0.95, "King should call bet (P0)"

        # P1 with King facing bet
        assert strategy["2:b"]["b"] > 0.95, "King should call bet (P1)"

    def test_jack_bluffs_sometimes(self, trained_cfr):
        """Player with Jack should bluff with some probability."""
        strategy = trained_cfr.get_all_average_strategies()

        # In Nash equilibrium, Jack bets ~1/3 at root
        # But this depends on alpha parameter, so just check it's bounded
        jack_bet_prob = strategy["0:"]["b"]
        assert 0.0 <= jack_bet_prob <= 0.4, f"Jack bet prob {jack_bet_prob} out of range"

    def test_king_value_bets_sometimes(self, trained_cfr):
        """Player with King should value bet sometimes."""
        strategy = trained_cfr.get_all_average_strategies()

        # King at root should bet sometimes (value bet)
        king_bet_prob = strategy["2:"]["b"]
        assert king_bet_prob > 0.0, "King should sometimes bet at root"

    def test_queen_indifferent(self, trained_cfr):
        """Queen should have mixed strategy in some spots."""
        strategy = trained_cfr.get_all_average_strategies()

        # Queen facing bet has mixed strategy
        queen_call = strategy["1:pb"]["b"]
        # Should call with some probability (around 1/3 in equilibrium)
        assert 0.2 < queen_call < 0.7, f"Queen call prob {queen_call} unexpected"


class TestConvergenceRate:
    """Test that exploitability decreases with iterations."""

    def test_exploitability_decreasing(self):
        """Exploitability should generally decrease as training progresses."""
        game = KuhnPoker()

        checkpoints = [100, 500, 1000, 5000]
        exploits = []

        for iters in checkpoints:
            cfr = VanillaCFR(game)
            cfr.train(iterations=iters)
            exploit = compute_exploitability(game, cfr.get_all_average_strategies())
            exploits.append((iters, exploit))

        # Final should be much lower than initial
        assert exploits[-1][1] < exploits[0][1] * 0.5, \
            f"Exploitability didn't decrease enough: {exploits[0]} -> {exploits[-1]}"

        # Final exploitability should be small
        assert exploits[-1][1] < 0.02, \
            f"Final exploitability too high: {exploits[-1][1]}"
