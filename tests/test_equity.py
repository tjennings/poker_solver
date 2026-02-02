"""Tests for preflop equity calculations."""

import pytest
from core.equity import get_preflop_equity, get_preflop_equity_by_name


class TestPreflopEquity:
    """Tests for preflop equity functions."""

    def test_same_hand_is_fifty_percent(self):
        """Same hand vs itself should be 50% equity."""
        # AA vs AA
        eq = get_preflop_equity_by_name("AA", "AA")
        assert eq == 0.5

        # 72o vs 72o
        eq = get_preflop_equity_by_name("72o", "72o")
        assert eq == 0.5

    def test_aa_vs_kk_is_dominant(self):
        """AA vs KK should be heavily favored (~80%)."""
        eq = get_preflop_equity_by_name("AA", "KK")
        assert 0.78 <= eq <= 0.85

    def test_kk_vs_aa_is_underdog(self):
        """KK vs AA should be ~20%."""
        eq = get_preflop_equity_by_name("KK", "AA")
        assert 0.15 <= eq <= 0.22

    def test_equities_sum_to_one(self):
        """Hand1 equity + Hand2 equity should equal 1.0."""
        eq1 = get_preflop_equity_by_name("AA", "KK")
        eq2 = get_preflop_equity_by_name("KK", "AA")
        assert abs(eq1 + eq2 - 1.0) < 0.001

    def test_overpair_vs_underpair(self):
        """Higher pair should beat lower pair ~80%."""
        eq = get_preflop_equity_by_name("QQ", "77")
        assert 0.78 <= eq <= 0.85

    def test_pair_vs_overcards(self):
        """Pair vs two overcards should be ~52-55%."""
        eq = get_preflop_equity_by_name("77", "AKo")
        assert 0.50 <= eq <= 0.58

    def test_pair_vs_undercards(self):
        """Overpair vs two undercards should be ~80%+."""
        eq = get_preflop_equity_by_name("AA", "KQo")
        assert 0.78 <= eq <= 0.88

    def test_dominated_hand(self):
        """AK vs AQ (domination) should be ~70%."""
        eq = get_preflop_equity_by_name("AKo", "AQo")
        assert 0.65 <= eq <= 0.75

    def test_suited_bonus(self):
        """Suited hand should have slightly more equity than offsuit."""
        eq_suited = get_preflop_equity_by_name("AKs", "QQ")
        eq_offsuit = get_preflop_equity_by_name("AKo", "QQ")
        assert eq_suited > eq_offsuit

    def test_index_based_lookup(self):
        """Index-based lookup should match name-based lookup."""
        from core.hands import canonical_index

        aa_idx = canonical_index("AA")
        kk_idx = canonical_index("KK")

        eq_by_idx = get_preflop_equity(aa_idx, kk_idx)
        eq_by_name = get_preflop_equity_by_name("AA", "KK")

        assert eq_by_idx == eq_by_name

    def test_all_equities_valid_range(self):
        """All equities should be between 0 and 1."""
        for i in range(169):
            for j in range(169):
                eq = get_preflop_equity(i, j)
                assert 0.0 <= eq <= 1.0, f"Invalid equity {eq} for ({i}, {j})"
