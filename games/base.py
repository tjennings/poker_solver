from abc import ABC, abstractmethod
from typing import List, Any, TypeVar

State = TypeVar('State')
Action = str


class Game(ABC):
    """
    Abstract base class for extensive-form games.

    Mirrors Definition 1 from Zinkevich et al.'s CFR paper.
    """

    @abstractmethod
    def initial_states(self) -> List[State]:
        """All possible starting states (after chance moves)."""
        pass

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """Check if state is terminal (z ∈ Z in paper)."""
        pass

    @abstractmethod
    def player(self, state: State) -> int:
        """Return player to act at state. P(h) in paper."""
        pass

    @abstractmethod
    def actions(self, state: State) -> List[Action]:
        """Available actions at state. A(h) in paper."""
        pass

    @abstractmethod
    def next_state(self, state: State, action: Action) -> State:
        """Return state after taking action."""
        pass

    @abstractmethod
    def utility(self, state: State, player: int) -> float:
        """Utility for player at terminal state. u_i(z) in paper."""
        pass

    @abstractmethod
    def info_set_key(self, state: State) -> str:
        """Map state to information set identifier. h → I in paper."""
        pass

    @abstractmethod
    def num_players(self) -> int:
        """Number of players in the game."""
        pass
