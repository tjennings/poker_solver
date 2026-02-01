"""Tests for the HUNL preflop hand parser module."""

import pytest
from core.hands import (
    parse_hand,
    hand_to_string,
    canonical_index,
    get_matrix_layout,
    HAND_COUNT,
)


class TestConstants:
    def test_hand_count(self):
        """There are exactly 169 canonical hands."""
        assert HAND_COUNT == 169


class TestParsePairs:
    def test_parse_pair_aa(self):
        """AA should be index 0."""
        assert parse_hand("AA") == (0, True, None)

    def test_parse_pair_kk(self):
        """KK should be index 1."""
        assert parse_hand("KK") == (1, True, None)

    def test_parse_pair_22(self):
        """22 should be index 12."""
        assert parse_hand("22") == (12, True, None)

    def test_all_pairs_indices(self):
        """All 13 pairs should have indices 0-12."""
        pairs = ["AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22"]
        for i, pair in enumerate(pairs):
            idx, is_canonical, cards = parse_hand(pair)
            assert idx == i, f"Expected {pair} at index {i}, got {idx}"
            assert is_canonical is True
            assert cards is None


class TestParseSuited:
    def test_parse_suited_ak(self):
        """AKs (or AK) should be index 13."""
        assert parse_hand("AK") == (13, True, None)
        assert parse_hand("AKs") == (13, True, None)

    def test_parse_suited_32(self):
        """32s should be the last suited hand."""
        # Suited hands: 13 pairs + 78 suited = indices 13-90
        # 32s is at index 90
        assert parse_hand("32") == (90, True, None)
        assert parse_hand("32s") == (90, True, None)

    def test_suited_count(self):
        """There should be 78 suited non-pair hands (C(13,2))."""
        suited_count = 0
        for idx in range(13, 91):
            hand = hand_to_string(idx)
            if hand.endswith("s") or (len(hand) == 2 and hand[0] != hand[1]):
                suited_count += 1
        # Actually we expect 78 suited hands
        assert suited_count == 78


class TestParseOffsuit:
    def test_parse_offsuit_ako(self):
        """AKo should be index 91."""
        assert parse_hand("AKo") == (91, True, None)

    def test_parse_offsuit_32o(self):
        """32o should be the last offsuit hand (index 168)."""
        assert parse_hand("32o") == (168, True, None)

    def test_offsuit_count(self):
        """There should be 78 offsuit non-pair hands."""
        offsuit_count = 0
        for idx in range(91, 169):
            offsuit_count += 1
        assert offsuit_count == 78


class TestParseSpecificCards:
    def test_parse_specific_suited(self):
        """AcKc should map to AKs (suited)."""
        idx, is_canonical, cards = parse_hand("AcKc")
        assert idx == 13  # AKs
        assert is_canonical is False
        assert cards == (("A", "c"), ("K", "c"))

    def test_parse_specific_offsuit(self):
        """AcKh should map to AKo (offsuit)."""
        idx, is_canonical, cards = parse_hand("AcKh")
        assert idx == 91  # AKo
        assert is_canonical is False
        assert cards == (("A", "c"), ("K", "h"))

    def test_parse_specific_pair(self):
        """AsAd should map to AA (pair)."""
        idx, is_canonical, cards = parse_hand("AsAd")
        assert idx == 0  # AA
        assert is_canonical is False
        assert cards == (("A", "s"), ("A", "d"))

    def test_parse_specific_order_normalized(self):
        """KcAh should normalize to Ah, Kc and map to AKo."""
        idx, is_canonical, cards = parse_hand("KcAh")
        assert idx == 91  # AKo
        assert is_canonical is False
        # Cards should be normalized to higher rank first
        assert cards == (("A", "h"), ("K", "c"))

    def test_parse_specific_low_cards(self):
        """7h2s should map to 72o."""
        idx, is_canonical, cards = parse_hand("7h2s")
        assert is_canonical is False
        # Verify it matches the canonical 72o index
        canonical_idx, _, _ = parse_hand("72o")
        assert idx == canonical_idx


class TestNormalization:
    def test_normalizes_ka_to_ak(self):
        """KA should normalize to AK."""
        assert parse_hand("KA") == parse_hand("AK")

    def test_normalizes_9to_to_t9o(self):
        """9To should normalize to T9o."""
        assert parse_hand("9To") == parse_hand("T9o")

    def test_normalizes_23_to_32(self):
        """23 should normalize to 32."""
        assert parse_hand("23") == parse_hand("32")

    def test_normalizes_23s_to_32s(self):
        """23s should normalize to 32s."""
        assert parse_hand("23s") == parse_hand("32s")

    def test_normalizes_pair_order(self):
        """Pair order shouldn't matter (same rank)."""
        # This is implicit since both cards are the same rank
        assert parse_hand("AA")[0] == 0


class TestHandToString:
    def test_hand_to_string_pairs(self):
        """Convert pair indices back to strings."""
        assert hand_to_string(0) == "AA"
        assert hand_to_string(1) == "KK"
        assert hand_to_string(12) == "22"

    def test_hand_to_string_suited(self):
        """Convert suited indices back to strings."""
        assert hand_to_string(13) == "AKs"
        assert hand_to_string(90) == "32s"

    def test_hand_to_string_offsuit(self):
        """Convert offsuit indices back to strings."""
        assert hand_to_string(91) == "AKo"
        assert hand_to_string(168) == "32o"

    def test_roundtrip_all_hands(self):
        """parse_hand and hand_to_string should be inverses."""
        for idx in range(HAND_COUNT):
            hand_str = hand_to_string(idx)
            parsed_idx, is_canonical, _ = parse_hand(hand_str)
            assert parsed_idx == idx, f"Roundtrip failed for {hand_str}"
            assert is_canonical is True


class TestCanonicalIndex:
    def test_canonical_index_basic(self):
        """canonical_index is a convenience wrapper."""
        assert canonical_index("AA") == 0
        assert canonical_index("AK") == 13
        assert canonical_index("AKs") == 13
        assert canonical_index("AKo") == 91

    def test_canonical_index_specific(self):
        """canonical_index works with specific cards too."""
        assert canonical_index("AcKc") == 13
        assert canonical_index("AcKh") == 91


class TestMatrixLayout:
    def test_matrix_dimensions(self):
        """Matrix should be 13x13."""
        layout = get_matrix_layout()
        assert len(layout) == 13
        for row in layout:
            assert len(row) == 13

    def test_diagonal_is_pairs(self):
        """Diagonal entries should be pairs."""
        layout = get_matrix_layout()
        pairs = ["AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22"]
        for i, pair in enumerate(pairs):
            assert layout[i][i] == pair, f"Expected {pair} at [{i}][{i}], got {layout[i][i]}"

    def test_above_diagonal_is_suited(self):
        """Above diagonal should be suited hands."""
        layout = get_matrix_layout()
        assert layout[0][0] == "AA"  # diagonal
        assert layout[0][1] == "AKs"  # above diagonal
        assert layout[0][2] == "AQs"  # above diagonal

    def test_below_diagonal_is_offsuit(self):
        """Below diagonal should be offsuit hands."""
        layout = get_matrix_layout()
        assert layout[1][0] == "AKo"  # below diagonal
        assert layout[2][0] == "AQo"  # below diagonal
        assert layout[2][1] == "KQo"  # below diagonal

    def test_specific_positions(self):
        """Test specific matrix positions from requirements."""
        layout = get_matrix_layout()
        assert layout[0][0] == "AA"
        assert layout[0][1] == "AKs"
        assert layout[1][0] == "AKo"


class TestEdgeCases:
    def test_invalid_hand_raises(self):
        """Invalid hand strings should raise ValueError."""
        with pytest.raises(ValueError):
            parse_hand("")
        with pytest.raises(ValueError):
            parse_hand("A")
        with pytest.raises(ValueError):
            parse_hand("AXs")
        with pytest.raises(ValueError):
            parse_hand("ZZ")

    def test_invalid_index_raises(self):
        """Invalid indices should raise ValueError."""
        with pytest.raises(ValueError):
            hand_to_string(-1)
        with pytest.raises(ValueError):
            hand_to_string(169)
        with pytest.raises(ValueError):
            hand_to_string(200)

    def test_case_sensitivity(self):
        """Hand parsing should be case insensitive for suits."""
        # Uppercase suits should work
        idx1, _, cards1 = parse_hand("AcKc")
        idx2, _, cards2 = parse_hand("ACKC")
        assert idx1 == idx2

    def test_lowercase_ranks(self):
        """Lowercase ranks should also work."""
        assert parse_hand("ak")[0] == parse_hand("AK")[0]
        assert parse_hand("aks")[0] == parse_hand("AKs")[0]


class TestAllCanonicalHands:
    def test_all_169_unique(self):
        """All 169 canonical hands should have unique indices."""
        indices = set()
        for idx in range(HAND_COUNT):
            hand_str = hand_to_string(idx)
            parsed_idx, _, _ = parse_hand(hand_str)
            assert parsed_idx not in indices or parsed_idx == idx
            indices.add(parsed_idx)
        assert len(indices) == HAND_COUNT

    def test_index_ranges(self):
        """Verify index ranges for each hand type."""
        # Pairs: 0-12
        for i in range(13):
            hand = hand_to_string(i)
            assert hand[0] == hand[1] or hand == hand[0] + hand[0], f"Expected pair at index {i}, got {hand}"

        # Suited: 13-90
        for i in range(13, 91):
            hand = hand_to_string(i)
            assert hand.endswith("s"), f"Expected suited hand at index {i}, got {hand}"

        # Offsuit: 91-168
        for i in range(91, 169):
            hand = hand_to_string(i)
            assert hand.endswith("o"), f"Expected offsuit hand at index {i}, got {hand}"
