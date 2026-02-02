"""Hand evaluation for poker.

Evaluates 5-card poker hands and finds best hand from 7 cards (2 hole + 5 board).
"""

from enum import IntEnum
from itertools import combinations
from typing import List, Tuple, Set
from collections import Counter

from core.cards import Card, parse_card, expand_canonical_hand, get_blocked_cards


class HandRank(IntEnum):
    """Poker hand rankings from lowest to highest."""
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_KIND = 7
    STRAIGHT_FLUSH = 8


# Hand value is (HandRank, tiebreaker_tuple)
HandValue = Tuple[HandRank, Tuple[int, ...]]


def evaluate_5cards(cards: List[Card]) -> HandValue:
    """Evaluate a 5-card poker hand.

    Args:
        cards: List of exactly 5 cards as (rank_value, suit_value) tuples

    Returns:
        (HandRank, tiebreaker_tuple) for comparison
    """
    if len(cards) != 5:
        raise ValueError(f"Expected 5 cards, got {len(cards)}")

    ranks = sorted([c[0] for c in cards], reverse=True)
    suits = [c[1] for c in cards]

    is_flush = len(set(suits)) == 1
    is_straight, straight_high = _check_straight(ranks)

    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)

    # Straight flush
    if is_flush and is_straight:
        return (HandRank.STRAIGHT_FLUSH, (straight_high,))

    # Four of a kind
    if counts == [4, 1]:
        quad_rank = _get_rank_with_count(rank_counts, 4)
        kicker = _get_rank_with_count(rank_counts, 1)
        return (HandRank.FOUR_KIND, (quad_rank, kicker))

    # Full house
    if counts == [3, 2]:
        trips_rank = _get_rank_with_count(rank_counts, 3)
        pair_rank = _get_rank_with_count(rank_counts, 2)
        return (HandRank.FULL_HOUSE, (trips_rank, pair_rank))

    # Flush
    if is_flush:
        return (HandRank.FLUSH, tuple(ranks))

    # Straight
    if is_straight:
        return (HandRank.STRAIGHT, (straight_high,))

    # Three of a kind
    if counts == [3, 1, 1]:
        trips_rank = _get_rank_with_count(rank_counts, 3)
        kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        return (HandRank.THREE_KIND, (trips_rank,) + tuple(kickers))

    # Two pair
    if counts == [2, 2, 1]:
        pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
        kicker = _get_rank_with_count(rank_counts, 1)
        return (HandRank.TWO_PAIR, (pairs[0], pairs[1], kicker))

    # One pair
    if counts == [2, 1, 1, 1]:
        pair_rank = _get_rank_with_count(rank_counts, 2)
        kickers = sorted([r for r, c in rank_counts.items() if c == 1], reverse=True)
        return (HandRank.PAIR, (pair_rank,) + tuple(kickers))

    # High card
    return (HandRank.HIGH_CARD, tuple(ranks))


def _check_straight(ranks: List[int]) -> Tuple[bool, int]:
    """Check if sorted ranks form a straight.

    Args:
        ranks: Sorted ranks (high to low)

    Returns:
        (is_straight, high_card) tuple
    """
    unique_ranks = sorted(set(ranks), reverse=True)
    if len(unique_ranks) != 5:
        return False, 0

    # Regular straight
    if unique_ranks[0] - unique_ranks[4] == 4:
        return True, unique_ranks[0]

    # Wheel (A-2-3-4-5): ranks would be [12, 3, 2, 1, 0]
    if unique_ranks == [12, 3, 2, 1, 0]:
        return True, 3  # 5-high straight

    return False, 0


def _get_rank_with_count(rank_counts: Counter, count: int) -> int:
    """Get the rank that appears 'count' times."""
    for rank, c in rank_counts.items():
        if c == count:
            return rank
    raise ValueError(f"No rank with count {count}")


def evaluate_hand(hand_idx: int, board: Tuple[str, ...]) -> HandValue:
    """Evaluate best 5-card hand from hole cards + board.

    Args:
        hand_idx: Canonical hand index (0-168)
        board: Board cards as strings ("Ah", "Th", "6c", ...)

    Returns:
        (HandRank, tiebreaker_tuple) for comparison

    Raises:
        ValueError: If no valid card combo exists (all blocked)
    """
    # Parse board cards
    board_cards = [parse_card(c) for c in board]
    blocked = get_blocked_cards(board)

    # Expand canonical hand to specific cards (excluding blocked)
    valid_combos = expand_canonical_hand(hand_idx, blocked)

    if not valid_combos:
        raise ValueError(f"No valid card combos for hand {hand_idx} with board {board}")

    # Use first valid combo (they're all equivalent for evaluation purposes
    # in terms of expected outcome when we don't know opponent's specific cards)
    hole_cards = list(valid_combos[0])

    # Combine hole cards with board
    all_cards = hole_cards + board_cards

    # Find best 5-card hand from 7 cards
    best_value = None
    for five_cards in combinations(all_cards, 5):
        value = evaluate_5cards(list(five_cards))
        if best_value is None or value > best_value:
            best_value = value

    return best_value


def evaluate_specific_hand(hole_cards: Tuple[Card, Card], board: Tuple[str, ...]) -> HandValue:
    """Evaluate best 5-card hand from specific hole cards + board.

    Args:
        hole_cards: Tuple of two Card tuples
        board: Board cards as strings

    Returns:
        (HandRank, tiebreaker_tuple) for comparison
    """
    board_cards = [parse_card(c) for c in board]
    all_cards = list(hole_cards) + board_cards

    best_value = None
    for five_cards in combinations(all_cards, 5):
        value = evaluate_5cards(list(five_cards))
        if best_value is None or value > best_value:
            best_value = value

    return best_value


def compare_hands(hand1: HandValue, hand2: HandValue) -> int:
    """Compare two hand values.

    Args:
        hand1: First hand value
        hand2: Second hand value

    Returns:
        1 if hand1 wins, -1 if hand2 wins, 0 if tie
    """
    if hand1 > hand2:
        return 1
    elif hand1 < hand2:
        return -1
    return 0
