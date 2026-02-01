"""Tests for action sequence parser."""

import pytest

from cli.parser import ParsedAction, ParsedSequence, parse_action_sequence


class TestParsedAction:
    """Tests for ParsedAction dataclass."""

    def test_parsed_action_creation(self):
        action = ParsedAction("SB", "raise", 2.5)
        assert action.position == "SB"
        assert action.action_type == "raise"
        assert action.amount == 2.5

    def test_parsed_action_no_amount(self):
        action = ParsedAction("BB", "call", None)
        assert action.position == "BB"
        assert action.action_type == "call"
        assert action.amount is None

    def test_to_history_string_fold(self):
        action = ParsedAction("SB", "fold", None)
        assert action.to_history_string() == "f"

    def test_to_history_string_call(self):
        action = ParsedAction("BB", "call", None)
        assert action.to_history_string() == "c"

    def test_to_history_string_check(self):
        action = ParsedAction("SB", "check", None)
        assert action.to_history_string() == "x"

    def test_to_history_string_raise(self):
        action = ParsedAction("SB", "raise", 2.5)
        assert action.to_history_string() == "r2.5"

    def test_to_history_string_raise_integer(self):
        action = ParsedAction("BB", "raise", 8.0)
        assert action.to_history_string() == "r8"

    def test_to_history_string_all_in(self):
        action = ParsedAction("SB", "all_in", None)
        assert action.to_history_string() == "a"


class TestParsedSequence:
    """Tests for ParsedSequence dataclass."""

    def test_parsed_sequence_creation(self):
        actions = [ParsedAction("SB", "raise", 2.5)]
        seq = ParsedSequence(actions=actions, stack_override=None)
        assert seq.actions == actions
        assert seq.stack_override is None

    def test_parsed_sequence_with_stack(self):
        actions = [ParsedAction("SB", "raise", 2.5)]
        seq = ParsedSequence(actions=actions, stack_override=50.0)
        assert seq.stack_override == 50.0

    def test_to_history_tuple_single(self):
        actions = [ParsedAction("SB", "raise", 2.5)]
        seq = ParsedSequence(actions=actions, stack_override=None)
        assert seq.to_history_tuple() == ("r2.5",)

    def test_to_history_tuple_multiple(self):
        actions = [
            ParsedAction("SB", "raise", 2.5),
            ParsedAction("BB", "raise", 8.0),
            ParsedAction("SB", "call", None),
        ]
        seq = ParsedSequence(actions=actions, stack_override=None)
        assert seq.to_history_tuple() == ("r2.5", "r8", "c")


class TestParseActionSequence:
    """Tests for parse_action_sequence function."""

    def test_parse_simple_open(self):
        result = parse_action_sequence("SBr2.5")
        assert result.stack_override is None
        assert len(result.actions) == 1
        assert result.actions[0] == ParsedAction("SB", "raise", 2.5)

    def test_parse_with_stack_override(self):
        result = parse_action_sequence("50bb SBr2.5")
        assert result.stack_override == 50

    def test_parse_full_sequence(self):
        result = parse_action_sequence("SBr2.5 BBr8 SBc")
        assert len(result.actions) == 3
        assert result.actions[2] == ParsedAction("SB", "call", None)

    def test_to_history_tuple(self):
        result = parse_action_sequence("SBr2.5 BBr8 SBc")
        assert result.to_history_tuple() == ("r2.5", "r8", "c")

    def test_invalid_position(self):
        with pytest.raises(ValueError, match="position"):
            parse_action_sequence("UTGr2.5")

    def test_case_insensitive(self):
        result = parse_action_sequence("sbR2.5 bbc")
        assert result.actions[0].position == "SB"
        assert result.actions[1].position == "BB"

    def test_parse_fold(self):
        result = parse_action_sequence("SBr2.5 BBf")
        assert len(result.actions) == 2
        assert result.actions[1] == ParsedAction("BB", "fold", None)

    def test_parse_check(self):
        result = parse_action_sequence("SBx BBx")
        assert len(result.actions) == 2
        assert result.actions[0] == ParsedAction("SB", "check", None)
        assert result.actions[1] == ParsedAction("BB", "check", None)

    def test_parse_all_in(self):
        result = parse_action_sequence("SBr2.5 BBa")
        assert len(result.actions) == 2
        assert result.actions[1] == ParsedAction("BB", "all_in", None)

    def test_parse_decimal_amounts(self):
        result = parse_action_sequence("SBr2.25 BBr10.5")
        assert result.actions[0].amount == 2.25
        assert result.actions[1].amount == 10.5

    def test_parse_integer_amounts(self):
        result = parse_action_sequence("SBr3 BBr12")
        assert result.actions[0].amount == 3.0
        assert result.actions[1].amount == 12.0

    def test_parse_fractional_bb_stack(self):
        result = parse_action_sequence("100.5bb SBr2.5")
        assert result.stack_override == 100.5

    def test_invalid_action_code(self):
        with pytest.raises(ValueError, match="action"):
            parse_action_sequence("SBz2.5")

    def test_empty_sequence(self):
        with pytest.raises(ValueError):
            parse_action_sequence("")

    def test_stack_only(self):
        with pytest.raises(ValueError):
            parse_action_sequence("50bb")

    def test_whitespace_handling(self):
        result = parse_action_sequence("  SBr2.5   BBc  ")
        assert len(result.actions) == 2

    def test_action_equality(self):
        action1 = ParsedAction("SB", "raise", 2.5)
        action2 = ParsedAction("SB", "raise", 2.5)
        assert action1 == action2

    def test_action_inequality(self):
        action1 = ParsedAction("SB", "raise", 2.5)
        action2 = ParsedAction("BB", "raise", 2.5)
        assert action1 != action2

    def test_raise_without_amount(self):
        with pytest.raises(ValueError, match="amount"):
            parse_action_sequence("SBr")
