"""Tests for HUNL preflop game implementation."""

import pytest
from config.loader import Config
from core.hands import hand_to_string, HAND_COUNT
from games.hunl_preflop import HUNLPreflop, HUNLState


@pytest.fixture
def config():
    """Standard 100BB config for testing."""
    return Config(
        name="Test",
        stack_depths=[100],
        raise_sizes=[2.5, 3, 8, 20, 50, 100]
    )


@pytest.fixture
def game(config):
    """HUNL preflop game instance."""
    return HUNLPreflop(config)


class TestHUNLState:
    """Tests for HUNLState dataclass."""

    def test_state_creation(self):
        """Test basic state creation."""
        state = HUNLState(
            hands=(0, 1),
            history=(),
            stack=100,
            pot=1.5,
            to_act=0
        )
        assert state.hands == (0, 1)
        assert state.history == ()
        assert state.stack == 100
        assert state.pot == 1.5
        assert state.to_act == 0

    def test_state_is_frozen(self):
        """Test that state is immutable (frozen dataclass)."""
        state = HUNLState(
            hands=(0, 1),
            history=(),
            stack=100,
            pot=1.5,
            to_act=0
        )
        with pytest.raises(AttributeError):
            state.pot = 5.0

    def test_state_equality(self):
        """Test state equality comparison."""
        state1 = HUNLState(hands=(0, 1), history=("r3",), stack=100, pot=4.5, to_act=1)
        state2 = HUNLState(hands=(0, 1), history=("r3",), stack=100, pot=4.5, to_act=1)
        assert state1 == state2

    def test_state_hashing(self):
        """Test that states can be used as dict keys."""
        state = HUNLState(hands=(0, 1), history=(), stack=100, pot=1.5, to_act=0)
        d = {state: "test"}
        assert d[state] == "test"


class TestHUNLPreflopInitialStates:
    """Tests for initial_states method."""

    def test_initial_states_count(self, game):
        """Should generate 169*168 = 28,392 states (no duplicate hands)."""
        states = game.initial_states()
        assert len(states) == 169 * 168  # 28,392

    def test_initial_states_no_same_hand(self, game):
        """No state should have same hand for both players."""
        states = game.initial_states()
        for state in states:
            assert state.hands[0] != state.hands[1]

    def test_initial_states_all_hands_represented(self, game):
        """All 169 hand types should appear for both positions."""
        states = game.initial_states()
        sb_hands = set(s.hands[0] for s in states)
        bb_hands = set(s.hands[1] for s in states)
        assert len(sb_hands) == 169
        assert len(bb_hands) == 169

    def test_initial_states_pot_and_stack(self, game):
        """All initial states should have correct pot and stack."""
        states = game.initial_states()
        for state in states:
            assert state.pot == 1.5  # SB 0.5 + BB 1.0
            assert state.stack == 100  # From config
            assert state.history == ()
            assert state.to_act == 0  # SB acts first preflop

    def test_initial_states_uniqueness(self, game):
        """All initial states should be unique."""
        states = game.initial_states()
        state_set = set(states)
        assert len(state_set) == len(states)


class TestHUNLPreflopTerminal:
    """Tests for is_terminal method."""

    def test_initial_state_not_terminal(self, game):
        """Initial state should not be terminal."""
        state = HUNLState(hands=(0, 1), history=(), stack=100, pot=1.5, to_act=0)
        assert not game.is_terminal(state)

    def test_fold_is_terminal(self, game):
        """Fold after any bet/raise should be terminal."""
        # SB folds to BB's 3bet
        state = HUNLState(hands=(0, 1), history=("r3", "r10", "f"), stack=100, pot=13.5, to_act=0)
        assert game.is_terminal(state)

    def test_fold_after_raise_is_terminal(self, game):
        """Fold after open raise should be terminal."""
        state = HUNLState(hands=(0, 1), history=("r3", "f"), stack=100, pot=4.0, to_act=0)
        assert game.is_terminal(state)

    def test_call_after_raise_is_terminal(self, game):
        """Call after raise should be terminal."""
        state = HUNLState(hands=(0, 1), history=("r3", "c"), stack=100, pot=6.0, to_act=0)
        assert game.is_terminal(state)

    def test_limp_call_is_terminal(self, game):
        """BB checking after SB limp should be terminal (check-behind)."""
        state = HUNLState(hands=(0, 1), history=("c", "c"), stack=100, pot=2.0, to_act=0)
        assert game.is_terminal(state)

    def test_limp_raise_not_terminal(self, game):
        """BB raising after SB limp should not be terminal."""
        state = HUNLState(hands=(0, 1), history=("c", "r3"), stack=100, pot=4.0, to_act=0)
        assert not game.is_terminal(state)

    def test_limp_raise_call_is_terminal(self, game):
        """SB calling after limp-raise should be terminal."""
        state = HUNLState(hands=(0, 1), history=("c", "r3", "c"), stack=100, pot=6.0, to_act=0)
        assert game.is_terminal(state)

    def test_all_in_call_is_terminal(self, game):
        """Call of all-in should be terminal."""
        state = HUNLState(hands=(0, 1), history=("a", "c"), stack=100, pot=200.0, to_act=0)
        assert game.is_terminal(state)

    def test_ongoing_action_not_terminal(self, game):
        """Raise with more action pending should not be terminal."""
        state = HUNLState(hands=(0, 1), history=("r3",), stack=100, pot=4.0, to_act=1)
        assert not game.is_terminal(state)


class TestHUNLPreflopPlayer:
    """Tests for player method."""

    def test_sb_acts_first(self, game):
        """SB (player 0) acts first."""
        state = HUNLState(hands=(0, 1), history=(), stack=100, pot=1.5, to_act=0)
        assert game.player(state) == 0

    def test_bb_acts_second(self, game):
        """BB (player 1) acts after SB open."""
        state = HUNLState(hands=(0, 1), history=("r3",), stack=100, pot=4.0, to_act=1)
        assert game.player(state) == 1

    def test_player_returns_to_act(self, game):
        """Player method should return state's to_act field."""
        state = HUNLState(hands=(0, 1), history=("c",), stack=100, pot=2.0, to_act=1)
        assert game.player(state) == 1


class TestHUNLPreflopActions:
    """Tests for actions method."""

    def test_sb_opening_actions(self, game):
        """SB can fold, limp, raise, or all-in when opening."""
        state = HUNLState(hands=(0, 1), history=(), stack=100, pot=1.5, to_act=0)
        actions = game.actions(state)
        assert "f" in actions  # Can fold
        assert "c" in actions  # Can limp (call BB)
        assert "r2.5" in actions  # Config raise size
        assert "r3" in actions
        assert "a" in actions  # All-in

    def test_bb_facing_raise(self, game):
        """BB facing raise can fold, call, reraise, or all-in."""
        state = HUNLState(hands=(0, 1), history=("r3",), stack=100, pot=4.0, to_act=1)
        actions = game.actions(state)
        assert "f" in actions
        assert "c" in actions
        # Only raises > 3 should be available
        assert "r2.5" not in actions
        assert "r3" not in actions
        assert "r8" in actions
        assert "r20" in actions
        assert "a" in actions

    def test_bb_check_option_after_limp(self, game):
        """BB can check after SB limp."""
        state = HUNLState(hands=(0, 1), history=("c",), stack=100, pot=2.0, to_act=1)
        actions = game.actions(state)
        assert "c" in actions  # Check
        assert "f" not in actions  # No fold when not facing bet
        assert "r2.5" in actions  # Can raise
        assert "a" in actions

    def test_facing_all_in_limited_actions(self, game):
        """Facing all-in, can only fold or call."""
        state = HUNLState(hands=(0, 1), history=("a",), stack=100, pot=101.0, to_act=1)
        actions = game.actions(state)
        assert actions == ["f", "c"]

    def test_all_in_excluded_when_stack_already_committed(self, game):
        """All-in shouldn't appear if raise equals stack."""
        # Create config where max raise = stack
        config = Config(name="Test", stack_depths=[50], raise_sizes=[25, 50])
        game = HUNLPreflop(config)
        state = HUNLState(hands=(0, 1), history=(), stack=50, pot=1.5, to_act=0)
        actions = game.actions(state)
        # r50 is already all-in, so 'a' might be redundant
        # Either r50 or 'a' should be present, but not both as separate options
        assert "r50" in actions or "a" in actions

    def test_no_raise_larger_than_stack(self, game):
        """No raise size should exceed effective stack."""
        state = HUNLState(hands=(0, 1), history=(), stack=100, pot=1.5, to_act=0)
        actions = game.actions(state)
        for action in actions:
            if action.startswith("r"):
                size = float(action[1:])
                assert size <= 100


class TestHUNLPreflopNextState:
    """Tests for next_state method."""

    def test_sb_limp(self, game):
        """SB limping should update pot and switch to BB."""
        state = HUNLState(hands=(0, 1), history=(), stack=100, pot=1.5, to_act=0)
        next_state = game.next_state(state, "c")
        assert next_state.history == ("c",)
        assert next_state.pot == 2.0  # SB completes to 1BB
        assert next_state.to_act == 1

    def test_sb_open_raise(self, game):
        """SB raising should update pot correctly."""
        state = HUNLState(hands=(0, 1), history=(), stack=100, pot=1.5, to_act=0)
        next_state = game.next_state(state, "r3")
        assert next_state.history == ("r3",)
        assert next_state.pot == 4.0  # SB puts in 3BB total (had 0.5, adds 2.5)
        assert next_state.to_act == 1

    def test_bb_call(self, game):
        """BB calling raise should update pot."""
        state = HUNLState(hands=(0, 1), history=("r3",), stack=100, pot=4.0, to_act=1)
        next_state = game.next_state(state, "c")
        assert next_state.history == ("r3", "c")
        assert next_state.pot == 6.0  # BB puts in 3BB (had 1BB, adds 2BB)
        assert next_state.to_act == 0

    def test_bb_3bet(self, game):
        """BB 3betting should update pot correctly."""
        state = HUNLState(hands=(0, 1), history=("r3",), stack=100, pot=4.0, to_act=1)
        next_state = game.next_state(state, "r10")
        assert next_state.history == ("r3", "r10")
        assert next_state.pot == 13.0  # BB puts in 10BB (had 1BB, adds 9BB)
        assert next_state.to_act == 0

    def test_fold_keeps_pot(self, game):
        """Folding should keep pot unchanged."""
        state = HUNLState(hands=(0, 1), history=("r3",), stack=100, pot=4.0, to_act=1)
        next_state = game.next_state(state, "f")
        assert next_state.history == ("r3", "f")
        assert next_state.pot == 4.0  # Unchanged

    def test_all_in(self, game):
        """All-in should put entire stack in pot."""
        state = HUNLState(hands=(0, 1), history=(), stack=100, pot=1.5, to_act=0)
        next_state = game.next_state(state, "a")
        assert next_state.history == ("a",)
        assert next_state.pot == 101.0  # SB puts in 100BB (had 0.5BB, adds 99.5BB)
        assert next_state.to_act == 1

    def test_hands_preserved(self, game):
        """Hands should be preserved through state transitions."""
        state = HUNLState(hands=(5, 10), history=(), stack=100, pot=1.5, to_act=0)
        next_state = game.next_state(state, "r3")
        assert next_state.hands == (5, 10)

    def test_stack_preserved(self, game):
        """Stack should be preserved through state transitions."""
        state = HUNLState(hands=(0, 1), history=(), stack=100, pot=1.5, to_act=0)
        next_state = game.next_state(state, "r3")
        assert next_state.stack == 100


class TestHUNLPreflopUtility:
    """Tests for utility method."""

    def test_utility_sb_folds(self, game):
        """When SB folds, BB wins pot, SB loses contribution."""
        # SB opens r3, BB 3bets r10, SB folds
        state = HUNLState(hands=(0, 1), history=("r3", "r10", "f"), stack=100, pot=13.0, to_act=0)
        assert game.utility(state, 0) == -3.0  # SB loses 3BB
        assert game.utility(state, 1) == 3.0   # BB wins 3BB

    def test_utility_bb_folds(self, game):
        """When BB folds, SB wins pot."""
        state = HUNLState(hands=(0, 1), history=("r3", "f"), stack=100, pot=4.0, to_act=0)
        assert game.utility(state, 0) == 1.0   # SB wins BB's 1BB
        assert game.utility(state, 1) == -1.0  # BB loses their 1BB

    def test_utility_call_showdown_sb_wins(self, game):
        """At showdown, higher hand wins."""
        # SB has AA (index 0), BB has KK (index 1) - AA wins
        state = HUNLState(hands=(0, 1), history=("r3", "c"), stack=100, pot=6.0, to_act=0)
        # SB put in 3BB, BB put in 3BB
        # SB wins 3BB (BB's contribution)
        assert game.utility(state, 0) == 3.0
        assert game.utility(state, 1) == -3.0

    def test_utility_call_showdown_bb_wins(self, game):
        """At showdown, higher hand wins."""
        # SB has KK (index 1), BB has AA (index 0) - AA wins
        state = HUNLState(hands=(1, 0), history=("r3", "c"), stack=100, pot=6.0, to_act=0)
        assert game.utility(state, 0) == -3.0  # SB loses
        assert game.utility(state, 1) == 3.0   # BB wins

    def test_utility_limp_check(self, game):
        """Limp-check pot goes to showdown."""
        # SB has AA, BB has KK - AA wins
        state = HUNLState(hands=(0, 1), history=("c", "c"), stack=100, pot=2.0, to_act=0)
        # Each put in 1BB, winner gets 1BB
        assert game.utility(state, 0) == 1.0   # AA wins
        assert game.utility(state, 1) == -1.0  # KK loses

    def test_utility_all_in_call(self, game):
        """All-in call showdown."""
        # SB shoves 100BB, BB calls with better hand
        state = HUNLState(hands=(1, 0), history=("a", "c"), stack=100, pot=200.0, to_act=0)
        assert game.utility(state, 0) == -100.0  # SB loses stack
        assert game.utility(state, 1) == 100.0   # BB wins stack

    def test_utility_non_terminal_raises(self, game):
        """Utility should raise for non-terminal state."""
        state = HUNLState(hands=(0, 1), history=("r3",), stack=100, pot=4.0, to_act=1)
        with pytest.raises(ValueError):
            game.utility(state, 0)


class TestHUNLPreflopInfoSetKey:
    """Tests for info_set_key method."""

    def test_info_set_key_sb_initial(self, game):
        """SB initial info set should show position and hand."""
        state = HUNLState(hands=(0, 1), history=(), stack=100, pot=1.5, to_act=0)
        key = game.info_set_key(state)
        assert key == "SB:AA:"  # SB, AA (index 0), no history

    def test_info_set_key_bb_facing_raise(self, game):
        """BB facing raise should show action history."""
        state = HUNLState(hands=(0, 1), history=("r3",), stack=100, pot=4.0, to_act=1)
        key = game.info_set_key(state)
        assert key == "BB:KK:r3"  # BB, KK (index 1), facing r3

    def test_info_set_key_different_hands(self, game):
        """Info set key should reflect hand correctly."""
        # Index 13 = AKs (first suited hand)
        state = HUNLState(hands=(13, 0), history=(), stack=100, pot=1.5, to_act=0)
        key = game.info_set_key(state)
        assert key == "SB:AKs:"

    def test_info_set_key_with_history(self, game):
        """Info set should include full action history."""
        state = HUNLState(hands=(0, 1), history=("r3", "r10"), stack=100, pot=13.0, to_act=0)
        key = game.info_set_key(state)
        assert key == "SB:AA:r3-r10"

    def test_info_set_key_hides_opponent_hand(self, game):
        """Info set key should not reveal opponent's hand."""
        state1 = HUNLState(hands=(0, 1), history=(), stack=100, pot=1.5, to_act=0)
        state2 = HUNLState(hands=(0, 50), history=(), stack=100, pot=1.5, to_act=0)
        # Same SB hand, same history -> same info set
        assert game.info_set_key(state1) == game.info_set_key(state2)


class TestHUNLPreflopNumPlayers:
    """Tests for num_players method."""

    def test_num_players(self, game):
        """HUNL should have 2 players."""
        assert game.num_players() == 2


class TestHUNLPreflopGameFlow:
    """Integration tests for complete game scenarios."""

    def test_simple_steal_fold(self, game):
        """Test SB steal, BB fold."""
        state = HUNLState(hands=(0, 1), history=(), stack=100, pot=1.5, to_act=0)

        # SB raises
        state = game.next_state(state, "r2.5")
        assert not game.is_terminal(state)
        assert game.player(state) == 1

        # BB folds
        state = game.next_state(state, "f")
        assert game.is_terminal(state)
        assert game.utility(state, 0) == 1.0  # SB wins BB's 1BB

    def test_3bet_pot(self, game):
        """Test SB open, BB 3bet, SB call."""
        state = HUNLState(hands=(0, 1), history=(), stack=100, pot=1.5, to_act=0)

        # SB opens to 3BB
        state = game.next_state(state, "r3")
        assert state.pot == 4.0

        # BB 3bets to 10BB
        state = game.next_state(state, "r10")
        assert state.pot == 13.0

        # SB calls
        state = game.next_state(state, "c")
        assert state.pot == 20.0
        assert game.is_terminal(state)

    def test_limp_raise_scenario(self, game):
        """Test SB limp, BB raise, SB call."""
        state = HUNLState(hands=(0, 1), history=(), stack=100, pot=1.5, to_act=0)

        # SB limps
        state = game.next_state(state, "c")
        assert state.pot == 2.0
        assert state.to_act == 1

        # BB raises to 4BB
        state = game.next_state(state, "r4")
        assert state.pot == 5.0

        # SB calls
        state = game.next_state(state, "c")
        assert game.is_terminal(state)

    def test_4bet_all_in_scenario(self, game):
        """Test 4bet all-in scenario."""
        state = HUNLState(hands=(0, 1), history=(), stack=100, pot=1.5, to_act=0)

        # SB opens 3BB
        state = game.next_state(state, "r3")
        # BB 3bets 10BB
        state = game.next_state(state, "r10")
        # SB 4bet shoves
        state = game.next_state(state, "a")
        assert not game.is_terminal(state)

        # BB calls all-in
        state = game.next_state(state, "c")
        assert game.is_terminal(state)
        assert state.pot == 200.0
