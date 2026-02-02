"""Tests for hand evaluation module."""

import pytest
from core.cards import parse_card, card_to_string, expand_canonical_hand, get_blocked_cards
from core.hand_eval import (
    HandRank, evaluate_5cards, evaluate_hand, evaluate_specific_hand,
    compare_hands
)
from core.hands import canonical_index


class TestCardParsing:
    """Tests for card parsing utilities."""

    def test_parse_card_ace_hearts(self):
        card = parse_card("Ah")
        assert card == (12, 2)  # A=12, h=2

    def test_parse_card_two_clubs(self):
        card = parse_card("2c")
        assert card == (0, 0)  # 2=0, c=0

    def test_parse_card_ten_spades(self):
        card = parse_card("Ts")
        assert card == (8, 3)  # T=8, s=3

    def test_parse_card_lowercase(self):
        card = parse_card("ah")
        assert card == (12, 2)

    def test_parse_card_invalid(self):
        with pytest.raises(ValueError):
            parse_card("XX")

    def test_card_to_string(self):
        assert card_to_string((12, 2)) == "Ah"
        assert card_to_string((0, 0)) == "2c"
        assert card_to_string((8, 3)) == "Ts"


class TestExpandCanonicalHand:
    """Tests for expanding canonical hands to specific combos."""

    def test_expand_pair_no_blocks(self):
        # AA = index 0
        combos = expand_canonical_hand(0)
        assert len(combos) == 6  # C(4,2) = 6 combos for a pair

    def test_expand_suited_no_blocks(self):
        # AKs = index 13
        combos = expand_canonical_hand(13)
        assert len(combos) == 4  # 4 suits

    def test_expand_offsuit_no_blocks(self):
        # AKo = index 91
        combos = expand_canonical_hand(91)
        assert len(combos) == 12  # 4*3 = 12 combos

    def test_expand_pair_with_blocked(self):
        # AA with Ah blocked
        blocked = {parse_card("Ah")}
        combos = expand_canonical_hand(0, blocked)
        # Without Ah, we can only use Ac, Ad, As -> C(3,2) = 3 combos
        assert len(combos) == 3

    def test_expand_suited_with_blocked(self):
        # AKs with Ah blocked
        blocked = {parse_card("Ah")}
        combos = expand_canonical_hand(13, blocked)
        # AhKh blocked, only 3 suits left
        assert len(combos) == 3

    def test_expand_offsuit_with_blocked(self):
        # AKo with Ah blocked
        blocked = {parse_card("Ah")}
        combos = expand_canonical_hand(91, blocked)
        # Ah blocks 3 combos (AhKc, AhKd, AhKs)
        assert len(combos) == 9


class TestEvaluate5Cards:
    """Tests for 5-card hand evaluation."""

    def test_high_card(self):
        cards = [parse_card(c) for c in ["Ah", "Kd", "9c", "7s", "2h"]]
        rank, tiebreak = evaluate_5cards(cards)
        assert rank == HandRank.HIGH_CARD
        assert tiebreak == (12, 11, 7, 5, 0)  # A, K, 9, 7, 2

    def test_pair(self):
        cards = [parse_card(c) for c in ["Ah", "Ad", "Kc", "9s", "2h"]]
        rank, tiebreak = evaluate_5cards(cards)
        assert rank == HandRank.PAIR
        assert tiebreak[0] == 12  # Pair of Aces

    def test_two_pair(self):
        cards = [parse_card(c) for c in ["Ah", "Ad", "Kc", "Ks", "2h"]]
        rank, tiebreak = evaluate_5cards(cards)
        assert rank == HandRank.TWO_PAIR
        assert tiebreak[:2] == (12, 11)  # AA and KK

    def test_three_of_a_kind(self):
        cards = [parse_card(c) for c in ["Ah", "Ad", "Ac", "Ks", "2h"]]
        rank, tiebreak = evaluate_5cards(cards)
        assert rank == HandRank.THREE_KIND
        assert tiebreak[0] == 12  # Trip Aces

    def test_straight(self):
        cards = [parse_card(c) for c in ["Ah", "Kd", "Qc", "Js", "Th"]]
        rank, tiebreak = evaluate_5cards(cards)
        assert rank == HandRank.STRAIGHT
        assert tiebreak == (12,)  # Ace-high straight

    def test_wheel_straight(self):
        cards = [parse_card(c) for c in ["Ah", "2d", "3c", "4s", "5h"]]
        rank, tiebreak = evaluate_5cards(cards)
        assert rank == HandRank.STRAIGHT
        assert tiebreak == (3,)  # 5-high (wheel)

    def test_flush(self):
        cards = [parse_card(c) for c in ["Ah", "Kh", "9h", "7h", "2h"]]
        rank, tiebreak = evaluate_5cards(cards)
        assert rank == HandRank.FLUSH
        assert tiebreak[0] == 12  # Ace-high flush

    def test_full_house(self):
        cards = [parse_card(c) for c in ["Ah", "Ad", "Ac", "Ks", "Kh"]]
        rank, tiebreak = evaluate_5cards(cards)
        assert rank == HandRank.FULL_HOUSE
        assert tiebreak == (12, 11)  # Aces full of Kings

    def test_four_of_a_kind(self):
        cards = [parse_card(c) for c in ["Ah", "Ad", "Ac", "As", "Kh"]]
        rank, tiebreak = evaluate_5cards(cards)
        assert rank == HandRank.FOUR_KIND
        assert tiebreak[0] == 12  # Quad Aces

    def test_straight_flush(self):
        cards = [parse_card(c) for c in ["Ah", "Kh", "Qh", "Jh", "Th"]]
        rank, tiebreak = evaluate_5cards(cards)
        assert rank == HandRank.STRAIGHT_FLUSH
        assert tiebreak == (12,)  # Royal flush (Ace-high SF)

    def test_steel_wheel(self):
        """5-high straight flush."""
        cards = [parse_card(c) for c in ["Ah", "2h", "3h", "4h", "5h"]]
        rank, tiebreak = evaluate_5cards(cards)
        assert rank == HandRank.STRAIGHT_FLUSH
        assert tiebreak == (3,)  # 5-high


class TestEvaluateHand:
    """Tests for evaluating canonical hands with board."""

    def test_evaluate_pair_on_board(self):
        # AA on AhTh6c board = set of Aces
        board = ("Ah", "Th", "6c", "Jc", "8c")
        # AA = index 0, but Ah is blocked so we use other combos
        rank, _ = evaluate_hand(0, board)
        assert rank == HandRank.THREE_KIND  # Trip Aces

    def test_evaluate_flush_draw_completes(self):
        # AKs (clubs) on Ah Th 6c Jc 8c board
        # AcKc would be a club flush
        board = ("Ah", "Th", "6c", "Jc", "8c")
        # AKs = index 13, club combo = AcKc
        rank, _ = evaluate_hand(13, board)
        # AcKc with 6c Jc 8c makes a flush
        assert rank == HandRank.FLUSH


class TestEvaluateSpecificHand:
    """Tests for evaluating specific hole cards."""

    def test_specific_hand_full_house(self):
        hole = (parse_card("Ah"), parse_card("Ad"))
        board = ("Ac", "Kh", "Kd", "2s", "3c")
        rank, tiebreak = evaluate_specific_hand(hole, board)
        assert rank == HandRank.FULL_HOUSE
        assert tiebreak == (12, 11)  # Aces full of Kings

    def test_specific_hand_two_pair(self):
        hole = (parse_card("Ah"), parse_card("Kh"))
        board = ("As", "Kd", "7c", "2s", "3c")
        rank, tiebreak = evaluate_specific_hand(hole, board)
        assert rank == HandRank.TWO_PAIR
        assert tiebreak[:2] == (12, 11)  # AA and KK


class TestCompareHands:
    """Tests for hand comparison."""

    def test_higher_rank_wins(self):
        pair = (HandRank.PAIR, (12, 11, 9, 7))
        two_pair = (HandRank.TWO_PAIR, (5, 4, 2))
        assert compare_hands(two_pair, pair) == 1
        assert compare_hands(pair, two_pair) == -1

    def test_same_rank_tiebreak(self):
        high_pair = (HandRank.PAIR, (12, 11, 9, 7))
        low_pair = (HandRank.PAIR, (11, 12, 9, 7))
        assert compare_hands(high_pair, low_pair) == 1
        assert compare_hands(low_pair, high_pair) == -1

    def test_tie(self):
        hand1 = (HandRank.PAIR, (12, 11, 9, 7))
        hand2 = (HandRank.PAIR, (12, 11, 9, 7))
        assert compare_hands(hand1, hand2) == 0

    def test_straight_flush_beats_quads(self):
        sf = (HandRank.STRAIGHT_FLUSH, (12,))
        quads = (HandRank.FOUR_KIND, (12, 11))
        assert compare_hands(sf, quads) == 1


class TestBoardBlocking:
    """Tests for board card blocking."""

    def test_get_blocked_cards(self):
        board = ("Ah", "Th", "6c")
        blocked = get_blocked_cards(board)
        assert parse_card("Ah") in blocked
        assert parse_card("Th") in blocked
        assert parse_card("6c") in blocked
        assert len(blocked) == 3

    def test_blocked_cards_reduce_combos(self):
        # AKs with Ah and Kh blocked = only 2 valid combos
        blocked = {parse_card("Ah"), parse_card("Kh")}
        combos = expand_canonical_hand(13, blocked)  # AKs
        # AhKh blocked directly, AcKc, AdKd, AsKs remain = 3
        # Wait, Ah blocks AhKh, Kh doesn't affect other suits
        # So we have AcKc, AdKd, AsKs = 3 combos
        assert len(combos) == 3


class TestFixedBoardScenario:
    """Tests for the fixed board: Ah Th 6c | Jc | 8c."""

    def test_aa_vs_kk_on_fixed_board(self):
        board = ("Ah", "Th", "6c", "Jc", "8c")

        # AA: Has trips (Ah on board + two more aces possible)
        aa_value = evaluate_hand(0, board)  # AA

        # KK: Has a pair of kings
        kk_value = evaluate_hand(1, board)  # KK

        # AA should win (trips vs pair)
        assert compare_hands(aa_value, kk_value) == 1

    def test_aks_makes_flush_on_fixed_board(self):
        board = ("Ah", "Th", "6c", "Jc", "8c")

        # AKs (clubs) = AcKc makes a flush with 6c Jc 8c
        aks_value = evaluate_hand(13, board)

        # Should be at least a flush
        assert aks_value[0] >= HandRank.FLUSH

    def test_high_card_hand_on_fixed_board(self):
        board = ("Ah", "Th", "6c", "Jc", "8c")

        # 72o - worst hand, no pairs on this board
        # Index for 72o: need to calculate
        seventy_two_o = canonical_index("72o")
        value = evaluate_hand(seventy_two_o, board)

        # Should be at least high card (Ace from board)
        assert value[0] >= HandRank.HIGH_CARD
