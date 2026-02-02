"""Card utilities for poker hand evaluation.

Provides functions for parsing cards, converting between representations,
and expanding canonical hands to specific card combinations.
"""

from typing import List, Tuple, Set

# Rank order: 2=0, 3=1, ..., A=12 (for hand evaluation)
RANKS = "23456789TJQKA"
RANK_VALUES = {r: i for i, r in enumerate(RANKS)}

# Suits
SUITS = "cdhs"
SUIT_VALUES = {s: i for i, s in enumerate(SUITS)}

# Card type alias: (rank_value, suit_value)
Card = Tuple[int, int]


def parse_card(s: str) -> Card:
    """Parse a card string to (rank_value, suit_value).

    Args:
        s: Card string like "Ah", "Tc", "2d"

    Returns:
        Tuple of (rank_value, suit_value) where rank 2=0, A=12

    Raises:
        ValueError: If card string is invalid
    """
    if len(s) != 2:
        raise ValueError(f"Invalid card string: {s}")

    rank = s[0].upper()
    suit = s[1].lower()

    if rank not in RANK_VALUES:
        raise ValueError(f"Invalid rank: {rank}")
    if suit not in SUIT_VALUES:
        raise ValueError(f"Invalid suit: {suit}")

    return (RANK_VALUES[rank], SUIT_VALUES[suit])


def card_to_string(card: Card) -> str:
    """Convert (rank_value, suit_value) to card string.

    Args:
        card: Tuple of (rank_value, suit_value)

    Returns:
        Card string like "Ah", "Tc"
    """
    rank_val, suit_val = card
    return RANKS[rank_val] + SUITS[suit_val]


def cards_to_string(cards: List[Card]) -> str:
    """Convert list of cards to space-separated string."""
    return " ".join(card_to_string(c) for c in cards)


def expand_canonical_hand(
    hand_idx: int,
    blocked_cards: Set[Card] = None
) -> List[Tuple[Card, Card]]:
    """Expand canonical hand index to all valid specific card combos.

    Args:
        hand_idx: Canonical hand index (0-168)
        blocked_cards: Set of cards that cannot be used (e.g., board cards)

    Returns:
        List of (card1, card2) tuples representing valid combos

    Example:
        - Index for 'AKs' -> [(Ac,Kc), (Ad,Kd), (Ah,Kh), (As,Ks)]
        - With Ah blocked -> [(Ac,Kc), (Ad,Kd), (As,Ks)]
    """
    if blocked_cards is None:
        blocked_cards = set()

    # Import here to avoid circular dependency
    from core.hands import hand_to_string

    hand_str = hand_to_string(hand_idx)

    # Parse the canonical hand
    if len(hand_str) == 2:
        # Pair: AA, KK, etc.
        rank = hand_str[0]
        return _expand_pair(rank, blocked_cards)
    elif hand_str[2] == 's':
        # Suited: AKs, QJs, etc.
        rank1, rank2 = hand_str[0], hand_str[1]
        return _expand_suited(rank1, rank2, blocked_cards)
    else:
        # Offsuit: AKo, QJo, etc.
        rank1, rank2 = hand_str[0], hand_str[1]
        return _expand_offsuit(rank1, rank2, blocked_cards)


def _expand_pair(rank: str, blocked: Set[Card]) -> List[Tuple[Card, Card]]:
    """Expand a pair to all valid combos (6 max)."""
    rank_val = RANK_VALUES[rank.upper()]
    combos = []

    # All suit combinations for a pair
    for s1 in range(4):
        for s2 in range(s1 + 1, 4):
            card1 = (rank_val, s1)
            card2 = (rank_val, s2)
            if card1 not in blocked and card2 not in blocked:
                combos.append((card1, card2))

    return combos


def _expand_suited(rank1: str, rank2: str, blocked: Set[Card]) -> List[Tuple[Card, Card]]:
    """Expand a suited hand to all valid combos (4 max)."""
    r1_val = RANK_VALUES[rank1.upper()]
    r2_val = RANK_VALUES[rank2.upper()]
    combos = []

    # Same suit for both cards
    for suit in range(4):
        card1 = (r1_val, suit)
        card2 = (r2_val, suit)
        if card1 not in blocked and card2 not in blocked:
            combos.append((card1, card2))

    return combos


def _expand_offsuit(rank1: str, rank2: str, blocked: Set[Card]) -> List[Tuple[Card, Card]]:
    """Expand an offsuit hand to all valid combos (12 max)."""
    r1_val = RANK_VALUES[rank1.upper()]
    r2_val = RANK_VALUES[rank2.upper()]
    combos = []

    # Different suits for the two cards
    for s1 in range(4):
        for s2 in range(4):
            if s1 != s2:
                card1 = (r1_val, s1)
                card2 = (r2_val, s2)
                if card1 not in blocked and card2 not in blocked:
                    combos.append((card1, card2))

    return combos


def get_blocked_cards(board: Tuple[str, ...]) -> Set[Card]:
    """Convert board cards to a set of blocked Card tuples.

    Args:
        board: Tuple of board card strings ("Ah", "Th", "6c", ...)

    Returns:
        Set of Card tuples that are blocked
    """
    return {parse_card(c) for c in board}
