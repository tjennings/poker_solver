import pytest
from abc import ABC
from games.base import Game, State, Action


def test_game_is_abstract():
    """Game should be an abstract base class."""
    with pytest.raises(TypeError):
        Game()


def test_game_has_required_methods():
    """Game ABC should define all required abstract methods."""
    abstract_methods = {
        'initial_states',
        'is_terminal',
        'player',
        'actions',
        'next_state',
        'utility',
        'info_set_key',
        'num_players',
    }
    assert abstract_methods <= set(Game.__abstractmethods__)
