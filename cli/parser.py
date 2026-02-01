"""Action sequence parser for HUNL preflop solver.

Parses action sequences like "50bb SBr2.5 BBr8 SBc" into structured data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# Valid positions in HUNL
VALID_POSITIONS = frozenset({"SB", "BB"})

# Action code to action type mapping
ACTION_CODES = {
    "r": "raise",
    "c": "call",
    "f": "fold",
    "a": "all_in",
    "x": "check",
}

# Action type to history string mapping
ACTION_TYPE_TO_STRING = {
    "fold": "f",
    "call": "c",
    "check": "x",
    "all_in": "a",
    "raise": "r",  # Raise will have amount appended
}


@dataclass(frozen=True)
class ParsedAction:
    """Represents a single parsed action.

    Attributes:
        position: Player position ("SB" or "BB")
        action_type: Type of action ("fold", "call", "check", "raise", "all_in")
        amount: Raise amount in big blinds (None for non-raise actions)
    """

    position: str
    action_type: str
    amount: Optional[float]

    def to_history_string(self) -> str:
        """Convert action to history string format.

        Returns:
            Action string: "f", "c", "x", "a", or "rX" where X is the raise amount
        """
        if self.action_type == "raise":
            # Format amount: show as integer if it's a whole number
            if self.amount is not None and self.amount == int(self.amount):
                return f"r{int(self.amount)}"
            return f"r{self.amount}"
        return ACTION_TYPE_TO_STRING[self.action_type]


@dataclass
class ParsedSequence:
    """Represents a parsed action sequence.

    Attributes:
        actions: List of parsed actions in order
        stack_override: Optional stack size in big blinds (from "Xbb" prefix)
    """

    actions: list[ParsedAction]
    stack_override: Optional[float]

    def to_history_tuple(self) -> tuple[str, ...]:
        """Convert sequence to history tuple format.

        Returns:
            Tuple of action strings like ("r2.5", "r8", "c")
        """
        return tuple(action.to_history_string() for action in self.actions)


# Regex patterns
STACK_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*bb$", re.IGNORECASE)
ACTION_PATTERN = re.compile(
    r"^(SB|BB)([rcfax])(\d+(?:\.\d+)?)?$",
    re.IGNORECASE,
)


def parse_action_sequence(sequence: str) -> ParsedSequence:
    """Parse an action sequence string into structured data.

    Args:
        sequence: Action sequence like "50bb SBr2.5 BBr8 SBc"

    Returns:
        ParsedSequence with actions and optional stack override

    Raises:
        ValueError: If sequence is invalid (bad position, action, or format)

    Examples:
        >>> result = parse_action_sequence("SBr2.5")
        >>> result.actions[0].position
        'SB'

        >>> result = parse_action_sequence("50bb SBr2.5")
        >>> result.stack_override
        50.0
    """
    # Strip and split by whitespace
    tokens = sequence.strip().split()

    if not tokens:
        raise ValueError("Empty action sequence")

    stack_override: Optional[float] = None
    actions: list[ParsedAction] = []

    for i, token in enumerate(tokens):
        # Check if this is a stack override (only valid as first token)
        stack_match = STACK_PATTERN.match(token)
        if stack_match:
            if i == 0:
                stack_override = float(stack_match.group(1))
                continue
            else:
                raise ValueError(f"Stack override must be first token, got: {token}")

        # Parse action
        action = _parse_action_token(token)
        actions.append(action)

    if not actions:
        raise ValueError("No actions in sequence")

    return ParsedSequence(actions=actions, stack_override=stack_override)


def _parse_action_token(token: str) -> ParsedAction:
    """Parse a single action token.

    Args:
        token: Action token like "SBr2.5" or "BBc"

    Returns:
        ParsedAction instance

    Raises:
        ValueError: If token format is invalid
    """
    action_match = ACTION_PATTERN.match(token)

    if not action_match:
        # Try to give a more specific error message
        upper_token = token.upper()
        if len(upper_token) >= 2:
            pos = upper_token[:2]
            if pos not in VALID_POSITIONS:
                raise ValueError(f"Invalid position in '{token}': expected SB or BB")
            if len(upper_token) > 2:
                action_code = upper_token[2].lower()
                if action_code not in ACTION_CODES:
                    raise ValueError(
                        f"Invalid action code in '{token}': "
                        f"expected one of r, c, f, a, x"
                    )
        raise ValueError(f"Invalid action format: {token}")

    position = action_match.group(1).upper()
    action_code = action_match.group(2).lower()
    amount_str = action_match.group(3)

    # Validate position
    if position not in VALID_POSITIONS:
        raise ValueError(f"Invalid position: {position}")

    # Get action type
    action_type = ACTION_CODES.get(action_code)
    if action_type is None:
        raise ValueError(f"Invalid action code: {action_code}")

    # Handle amount
    amount: Optional[float] = None
    if action_code == "r":
        if amount_str is None:
            raise ValueError(f"Raise action requires amount: {token}")
        amount = float(amount_str)
    elif amount_str is not None:
        # Non-raise action with amount - this is unusual but we'll accept it
        # and ignore the amount (could also raise an error)
        pass

    return ParsedAction(position=position, action_type=action_type, amount=amount)
