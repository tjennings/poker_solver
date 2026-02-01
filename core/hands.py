"""
HUNL Preflop Hand Parser Module.

Parses and converts between hand representations:
- Canonical hands (169 total): AA, AKs, AKo, etc.
- Specific card combos: AcKh, AsAd, etc.

Index Layout:
- Pairs: 0-12 (AA, KK, QQ, ..., 22)
- Suited: 13-90 (78 hands)
- Offsuit: 91-168 (78 hands)
"""

from typing import Tuple, Optional, List

# Constants
HAND_COUNT = 169

# Rank order: A=0 (highest), K=1, Q=2, ..., 2=12 (lowest)
RANKS = "AKQJT98765432"
RANK_TO_INDEX = {r: i for i, r in enumerate(RANKS)}

# Suits
SUITS = "cdhs"


def _normalize_rank_order(rank1: str, rank2: str) -> Tuple[str, str]:
    """Ensure higher rank comes first."""
    r1 = rank1.upper()
    r2 = rank2.upper()
    if RANK_TO_INDEX[r1] > RANK_TO_INDEX[r2]:
        return r2, r1
    return r1, r2


def _parse_canonical(hand: str) -> Tuple[int, bool, None]:
    """Parse a canonical hand string (e.g., AA, AKs, AKo)."""
    hand = hand.upper()

    if len(hand) < 2 or len(hand) > 3:
        raise ValueError(f"Invalid hand string: {hand}")

    rank1 = hand[0]
    rank2 = hand[1]

    if rank1 not in RANK_TO_INDEX or rank2 not in RANK_TO_INDEX:
        raise ValueError(f"Invalid rank in hand: {hand}")

    # Normalize rank order
    rank1, rank2 = _normalize_rank_order(rank1, rank2)

    # Determine hand type
    if len(hand) == 2:
        # No suffix - pair or suited
        if rank1 == rank2:
            # Pair
            idx = RANK_TO_INDEX[rank1]
            return idx, True, None
        else:
            # Default to suited for 2-char non-pair
            return _suited_index(rank1, rank2), True, None
    else:
        suffix = hand[2].lower()
        if rank1 == rank2:
            # Pair with suffix (invalid but tolerate)
            idx = RANK_TO_INDEX[rank1]
            return idx, True, None
        elif suffix == "s":
            return _suited_index(rank1, rank2), True, None
        elif suffix == "o":
            return _offsuit_index(rank1, rank2), True, None
        else:
            raise ValueError(f"Invalid suffix in hand: {hand}")


def _suited_index(rank1: str, rank2: str) -> int:
    """Calculate index for a suited hand (13-90)."""
    # Ensure proper order (higher rank first)
    r1, r2 = _normalize_rank_order(rank1, rank2)
    r1_idx = RANK_TO_INDEX[r1]
    r2_idx = RANK_TO_INDEX[r2]

    # Suited hands are indexed by counting combinations
    # For rank r1, suited combos are r1r2 where r2 > r1 (in index terms)
    # Offset = 13 (after pairs)
    # Count suited hands before this one
    offset = 13
    # Number of suited hands where first rank has index < r1_idx
    # For each rank i from 0 to r1_idx-1, there are (12-i) suited combos
    for i in range(r1_idx):
        offset += 12 - i
    # Add the position within rank r1's suited hands
    offset += r2_idx - r1_idx - 1
    return offset


def _offsuit_index(rank1: str, rank2: str) -> int:
    """Calculate index for an offsuit hand (91-168)."""
    # Same structure as suited but offset by 78 more
    suited_idx = _suited_index(rank1, rank2)
    return suited_idx + 78  # 91 - 13 = 78


def _parse_specific(hand: str) -> Tuple[int, bool, Tuple[Tuple[str, str], Tuple[str, str]]]:
    """Parse a specific card combo (e.g., AcKh, AsAd)."""
    if len(hand) != 4:
        raise ValueError(f"Invalid specific hand: {hand}")

    rank1 = hand[0].upper()
    suit1 = hand[1].lower()
    rank2 = hand[2].upper()
    suit2 = hand[3].lower()

    if rank1 not in RANK_TO_INDEX or rank2 not in RANK_TO_INDEX:
        raise ValueError(f"Invalid rank in hand: {hand}")
    if suit1 not in SUITS or suit2 not in SUITS:
        raise ValueError(f"Invalid suit in hand: {hand}")

    # Normalize order (higher rank first)
    if RANK_TO_INDEX[rank1] > RANK_TO_INDEX[rank2]:
        rank1, suit1, rank2, suit2 = rank2, suit2, rank1, suit1

    cards = ((rank1, suit1), (rank2, suit2))

    # Determine canonical index
    if rank1 == rank2:
        # Pair
        idx = RANK_TO_INDEX[rank1]
    elif suit1 == suit2:
        # Suited
        idx = _suited_index(rank1, rank2)
    else:
        # Offsuit
        idx = _offsuit_index(rank1, rank2)

    return idx, False, cards


def parse_hand(hand: str) -> Tuple[int, bool, Optional[Tuple[Tuple[str, str], Tuple[str, str]]]]:
    """
    Parse a hand string and return its canonical index.

    Args:
        hand: Hand string in one of these formats:
            - Canonical: "AA", "AKs", "AKo", "72o"
            - Specific: "AcKh", "AsAd", "7h2s"

    Returns:
        Tuple of (index, is_canonical, specific_cards):
            - index: 0-168 canonical hand index
            - is_canonical: True if input was canonical notation
            - specific_cards: None for canonical, or ((rank1, suit1), (rank2, suit2))

    Raises:
        ValueError: If hand string is invalid
    """
    if not hand:
        raise ValueError("Empty hand string")

    hand = hand.strip()

    # Check if it's a specific hand (4 chars: RsRs)
    if len(hand) == 4 and hand[1].lower() in SUITS and hand[3].lower() in SUITS:
        return _parse_specific(hand)

    # Otherwise parse as canonical
    return _parse_canonical(hand)


def hand_to_string(idx: int) -> str:
    """
    Convert a canonical hand index back to string representation.

    Args:
        idx: Hand index (0-168)

    Returns:
        Canonical hand string (e.g., "AA", "AKs", "AKo")

    Raises:
        ValueError: If index is out of range
    """
    if idx < 0 or idx >= HAND_COUNT:
        raise ValueError(f"Invalid hand index: {idx}")

    if idx < 13:
        # Pair
        rank = RANKS[idx]
        return rank + rank
    elif idx < 91:
        # Suited
        return _index_to_hand(idx - 13, "s")
    else:
        # Offsuit
        return _index_to_hand(idx - 91, "o")


def _index_to_hand(combo_idx: int, suffix: str) -> str:
    """Convert a suited/offsuit combo index (0-77) to hand string."""
    # Find which high rank this belongs to
    count = 0
    for r1_idx in range(13):
        combos_for_rank = 12 - r1_idx
        if count + combos_for_rank > combo_idx:
            # This is the rank
            r2_offset = combo_idx - count
            r2_idx = r1_idx + 1 + r2_offset
            return RANKS[r1_idx] + RANKS[r2_idx] + suffix
        count += combos_for_rank
    raise ValueError(f"Invalid combo index: {combo_idx}")


def canonical_index(hand: str) -> int:
    """
    Convenience function to get canonical index from any hand string.

    Args:
        hand: Hand string (canonical or specific)

    Returns:
        Canonical hand index (0-168)
    """
    idx, _, _ = parse_hand(hand)
    return idx


def get_matrix_layout() -> List[List[str]]:
    """
    Return a 13x13 matrix layout of canonical hands.

    The matrix is structured as:
    - Diagonal (i == j): Pairs (AA, KK, ..., 22)
    - Above diagonal (j > i): Suited hands
    - Below diagonal (i > j): Offsuit hands

    Rows and columns are indexed by rank: A=0, K=1, Q=2, ..., 2=12

    Returns:
        13x13 list of hand strings
    """
    matrix = [[None for _ in range(13)] for _ in range(13)]

    for row in range(13):
        for col in range(13):
            rank1 = RANKS[row]
            rank2 = RANKS[col]

            if row == col:
                # Diagonal - pairs
                matrix[row][col] = rank1 + rank2
            elif col > row:
                # Above diagonal - suited (higher rank in row)
                matrix[row][col] = rank1 + rank2 + "s"
            else:
                # Below diagonal - offsuit (higher rank in col)
                matrix[row][col] = rank2 + rank1 + "o"

    return matrix
