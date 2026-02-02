"""
Preflop equity calculations for heads-up poker.

Provides equity values for all 169 canonical hand matchups.
Equity represents the probability of winning (plus half the probability of tying).
"""

from typing import Dict, Tuple
from core.hands import hand_to_string, canonical_index, HAND_COUNT


# Precomputed preflop equities for canonical hand matchups.
# Format: (hand1_idx, hand2_idx) -> equity for hand1
# These are approximate values based on standard preflop equity calculations.
# Equity includes win probability + 0.5 * tie probability.

def _compute_equity_table() -> Dict[Tuple[int, int], float]:
    """
    Build equity lookup table for all hand matchups.

    Uses hand strength categories and known equity relationships:
    - Pairs vs pairs: higher pair ~82% (unless very close)
    - Pairs vs unpaired: pair ~50-85% depending on overcards
    - Unpaired vs unpaired: based on high card and connectivity
    """
    equity = {}

    for i in range(HAND_COUNT):
        for j in range(HAND_COUNT):
            if i == j:
                # Same hand category - equity depends on suit blocking
                equity[(i, j)] = 0.5
            else:
                hand1 = hand_to_string(i)
                hand2 = hand_to_string(j)
                eq = _calculate_matchup_equity(hand1, hand2)
                equity[(i, j)] = eq

    return equity


def _parse_hand(hand: str) -> Tuple[int, int, bool]:
    """Parse hand string into (high_rank, low_rank, suited).

    Ranks: A=14, K=13, Q=12, J=11, T=10, 9-2=9-2
    """
    ranks = "23456789TJQKA"

    r1 = ranks.index(hand[0]) + 2
    r2 = ranks.index(hand[1]) + 2

    high = max(r1, r2)
    low = min(r1, r2)

    # Check if pair, suited, or offsuit
    if len(hand) == 2:
        suited = False  # Pair
    elif hand[2] == 's':
        suited = True
    else:
        suited = False

    is_pair = (r1 == r2)

    return high, low, suited, is_pair


def _calculate_matchup_equity(hand1: str, hand2: str) -> float:
    """
    Calculate approximate preflop equity for hand1 vs hand2.

    Uses simplified equity model based on:
    - Pair vs pair matchups
    - Pair vs unpaired (dominated, overcards, etc.)
    - Unpaired vs unpaired
    """
    h1_high, h1_low, h1_suited, h1_pair = _parse_hand(hand1)
    h2_high, h2_low, h2_suited, h2_pair = _parse_hand(hand2)

    # Pair vs Pair
    if h1_pair and h2_pair:
        if h1_high > h2_high:
            # Higher pair vs lower pair: ~80-82%
            gap = h1_high - h2_high
            return 0.80 + min(gap * 0.005, 0.02)
        else:
            gap = h2_high - h1_high
            return 0.20 - min(gap * 0.005, 0.02)

    # Pair vs Unpaired
    if h1_pair and not h2_pair:
        return _pair_vs_unpaired_equity(h1_high, h2_high, h2_low, h2_suited)

    if h2_pair and not h1_pair:
        return 1.0 - _pair_vs_unpaired_equity(h2_high, h1_high, h1_low, h1_suited)

    # Unpaired vs Unpaired
    return _unpaired_vs_unpaired_equity(
        h1_high, h1_low, h1_suited,
        h2_high, h2_low, h2_suited
    )


def _pair_vs_unpaired_equity(pair_rank: int, other_high: int, other_low: int, other_suited: bool) -> float:
    """Equity for a pair against an unpaired hand."""

    # Overpair (pair higher than both cards)
    if pair_rank > other_high:
        # Overpair vs two undercards: ~80-85%
        base = 0.82
        if other_suited:
            base -= 0.03  # Suited has more outs
        if other_high == other_low - 1:
            base -= 0.02  # Connected has straight outs
        return base

    # Pair vs one overcard
    if pair_rank > other_low and pair_rank < other_high:
        # Pair vs overcard + undercard: ~55-70%
        base = 0.55
        gap = other_high - pair_rank
        base += gap * 0.02
        if other_suited:
            base -= 0.03
        return min(max(base, 0.50), 0.70)

    # Underpair (both cards higher than pair)
    if pair_rank < other_low:
        # Underpair vs two overcards: ~45-55%
        base = 0.52
        if other_suited:
            base -= 0.03
        if other_high - other_low == 1:
            base -= 0.02  # Connected
        return max(base, 0.45)

    # Pair vs one card of same rank (dominated)
    if pair_rank == other_high or pair_rank == other_low:
        # Set mining situation: ~70%
        return 0.70

    return 0.55  # Default


def _unpaired_vs_unpaired_equity(
    h1_high: int, h1_low: int, h1_suited: bool,
    h2_high: int, h2_low: int, h2_suited: bool
) -> float:
    """Equity for unpaired hand vs unpaired hand."""

    # Check for domination (shared high card)
    if h1_high == h2_high:
        if h1_low > h2_low:
            # Dominating kicker: ~70-75%
            base = 0.70
            gap = h1_low - h2_low
            base += min(gap * 0.01, 0.05)
            if h1_suited and not h2_suited:
                base += 0.03
            elif h2_suited and not h1_suited:
                base -= 0.03
            return base
        elif h1_low < h2_low:
            base = 0.30
            gap = h2_low - h1_low
            base -= min(gap * 0.01, 0.05)
            if h1_suited and not h2_suited:
                base += 0.03
            elif h2_suited and not h1_suited:
                base -= 0.03
            return base
        else:
            # Same hand, different suits
            if h1_suited and not h2_suited:
                return 0.53
            elif h2_suited and not h1_suited:
                return 0.47
            return 0.50

    # Check for domination (shared low card)
    if h1_low == h2_low:
        if h1_high > h2_high:
            return 0.70
        else:
            return 0.30

    # One hand dominates the other (one card matches)
    if h1_high == h2_low:
        # h2 has higher high card (h2_high > h1_high since h2_low == h1_high)
        return 0.35
    if h2_high == h1_low:
        # h1 has higher high card
        return 0.65

    # No shared cards - compare high cards
    if h1_high > h2_high:
        base = 0.55
        # Adjust for second card
        if h1_low > h2_high:
            base = 0.65  # Both cards higher
        elif h1_low > h2_low:
            base = 0.58

        # Suited bonus
        if h1_suited and not h2_suited:
            base += 0.03
        elif h2_suited and not h1_suited:
            base -= 0.03

        # Connectivity bonus for underdog
        if h2_high - h2_low <= 4:
            base -= 0.02

        return min(max(base, 0.45), 0.70)
    else:
        # h2 has higher high card - mirror the calculation
        base = 0.45
        if h2_low > h1_high:
            base = 0.35
        elif h2_low > h1_low:
            base = 0.42

        if h1_suited and not h2_suited:
            base += 0.03
        elif h2_suited and not h1_suited:
            base -= 0.03

        if h1_high - h1_low <= 4:
            base += 0.02

        return min(max(base, 0.30), 0.55)


# Build the equity table at module load time
_EQUITY_TABLE = _compute_equity_table()


def get_preflop_equity(hand1_idx: int, hand2_idx: int) -> float:
    """
    Get preflop equity for hand1 vs hand2.

    Args:
        hand1_idx: Index of first hand (0-168)
        hand2_idx: Index of second hand (0-168)

    Returns:
        Equity for hand1 (probability of winning + 0.5 * probability of tie)
    """
    return _EQUITY_TABLE[(hand1_idx, hand2_idx)]


def get_preflop_equity_by_name(hand1: str, hand2: str) -> float:
    """
    Get preflop equity for hand1 vs hand2 by hand name.

    Args:
        hand1: Hand string (e.g., "AA", "AKs", "72o")
        hand2: Hand string (e.g., "KK", "QJs", "T9o")

    Returns:
        Equity for hand1
    """
    idx1 = canonical_index(hand1)
    idx2 = canonical_index(hand2)
    return get_preflop_equity(idx1, idx2)
