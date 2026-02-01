import pytest
from itertools import permutations
from games.kuhn import KuhnState, KuhnPoker


class TestKuhnState:
    def test_state_is_immutable(self):
        """KuhnState should be immutable (frozen dataclass)."""
        state = KuhnState(cards=(0, 1), history=())
        with pytest.raises(AttributeError):
            state.cards = (1, 2)

    def test_state_equality(self):
        """States with same data should be equal."""
        s1 = KuhnState(cards=(0, 1), history=("p",))
        s2 = KuhnState(cards=(0, 1), history=("p",))
        assert s1 == s2

    def test_state_hashable(self):
        """States should be hashable for use as dict keys."""
        state = KuhnState(cards=(0, 1), history=())
        d = {state: "value"}
        assert d[state] == "value"


class TestKuhnPoker:
    @pytest.fixture
    def game(self):
        return KuhnPoker()

    def test_initial_states_count(self, game):
        """Should have 6 initial states (3P2 card permutations)."""
        states = game.initial_states()
        assert len(states) == 6

    def test_initial_states_are_all_deals(self, game):
        """Initial states should cover all card permutations."""
        states = game.initial_states()
        card_combos = {s.cards for s in states}
        expected = set(permutations([0, 1, 2], 2))
        assert card_combos == expected

    def test_player_alternates(self, game):
        """Player should alternate based on history length."""
        s0 = KuhnState(cards=(0, 1), history=())
        s1 = KuhnState(cards=(0, 1), history=("p",))
        s2 = KuhnState(cards=(0, 1), history=("p", "b"))

        assert game.player(s0) == 0
        assert game.player(s1) == 1
        assert game.player(s2) == 0

    def test_actions_always_two(self, game):
        """Should always have exactly 2 actions: pass and bet."""
        state = KuhnState(cards=(0, 1), history=())
        actions = game.actions(state)
        assert actions == ["p", "b"]

    def test_terminal_states(self, game):
        """Test all terminal conditions."""
        # pp = check-check (terminal)
        assert game.is_terminal(KuhnState((0, 1), ("p", "p"))) is True
        # bp = bet-fold (terminal)
        assert game.is_terminal(KuhnState((0, 1), ("b", "p"))) is True
        # bb = bet-call (terminal)
        assert game.is_terminal(KuhnState((0, 1), ("b", "b"))) is True
        # pbp = check-bet-fold (terminal)
        assert game.is_terminal(KuhnState((0, 1), ("p", "b", "p"))) is True
        # pbb = check-bet-call (terminal)
        assert game.is_terminal(KuhnState((0, 1), ("p", "b", "b"))) is True

        # Non-terminal
        assert game.is_terminal(KuhnState((0, 1), ())) is False
        assert game.is_terminal(KuhnState((0, 1), ("p",))) is False
        assert game.is_terminal(KuhnState((0, 1), ("b",))) is False
        assert game.is_terminal(KuhnState((0, 1), ("p", "b"))) is False

    def test_next_state(self, game):
        """next_state should append action to history."""
        s0 = KuhnState(cards=(0, 1), history=())
        s1 = game.next_state(s0, "p")
        assert s1 == KuhnState(cards=(0, 1), history=("p",))

        s2 = game.next_state(s1, "b")
        assert s2 == KuhnState(cards=(0, 1), history=("p", "b"))

    def test_num_players(self, game):
        """Kuhn Poker has 2 players."""
        assert game.num_players() == 2


class TestKuhnUtilities:
    @pytest.fixture
    def game(self):
        return KuhnPoker()

    def test_check_check_higher_wins(self, game):
        """pp: higher card wins 1 chip."""
        # P0 has King(2), P1 has Jack(0) -> P0 wins
        state = KuhnState(cards=(2, 0), history=("p", "p"))
        assert game.utility(state, 0) == 1.0
        assert game.utility(state, 1) == -1.0

        # P0 has Jack(0), P1 has Queen(1) -> P1 wins
        state = KuhnState(cards=(0, 1), history=("p", "p"))
        assert game.utility(state, 0) == -1.0
        assert game.utility(state, 1) == 1.0

    def test_bet_fold_bettor_wins(self, game):
        """bp: player 0 bet, player 1 folded -> P0 wins 1."""
        state = KuhnState(cards=(0, 2), history=("b", "p"))
        assert game.utility(state, 0) == 1.0
        assert game.utility(state, 1) == -1.0

    def test_bet_call_higher_wins(self, game):
        """bb: bet-call, higher card wins 2 chips."""
        # P0 has King, wins 2
        state = KuhnState(cards=(2, 1), history=("b", "b"))
        assert game.utility(state, 0) == 2.0
        assert game.utility(state, 1) == -2.0

        # P1 has King, wins 2
        state = KuhnState(cards=(1, 2), history=("b", "b"))
        assert game.utility(state, 0) == -2.0
        assert game.utility(state, 1) == 2.0

    def test_check_bet_fold(self, game):
        """pbp: check-bet-fold -> P1 wins 1."""
        state = KuhnState(cards=(2, 0), history=("p", "b", "p"))
        assert game.utility(state, 0) == -1.0
        assert game.utility(state, 1) == 1.0

    def test_check_bet_call(self, game):
        """pbb: check-bet-call, higher card wins 2."""
        state = KuhnState(cards=(2, 0), history=("p", "b", "b"))
        assert game.utility(state, 0) == 2.0
        assert game.utility(state, 1) == -2.0

    def test_zero_sum(self, game):
        """Utilities should always sum to zero."""
        for cards in permutations([0, 1, 2], 2):
            for terminal in KuhnPoker.TERMINALS:
                state = KuhnState(cards=cards, history=terminal)
                total = game.utility(state, 0) + game.utility(state, 1)
                assert total == 0.0, f"Non-zero sum at {state}"


class TestKuhnInfoSets:
    @pytest.fixture
    def game(self):
        return KuhnPoker()

    def test_info_set_includes_own_card(self, game):
        """Info set should encode player's card."""
        s1 = KuhnState(cards=(0, 1), history=())  # P0 has Jack
        s2 = KuhnState(cards=(2, 1), history=())  # P0 has King

        assert game.info_set_key(s1) != game.info_set_key(s2)
        assert "0" in game.info_set_key(s1)  # Jack = 0
        assert "2" in game.info_set_key(s2)  # King = 2

    def test_info_set_includes_history(self, game):
        """Info set should encode action history."""
        s1 = KuhnState(cards=(0, 1), history=())
        s2 = KuhnState(cards=(0, 1), history=("p",))
        s3 = KuhnState(cards=(0, 1), history=("p", "b"))

        k1 = game.info_set_key(s1)
        k2 = game.info_set_key(s2)
        k3 = game.info_set_key(s3)

        assert k1 != k2 != k3

    def test_info_set_hides_opponent_card(self, game):
        """Different opponent cards should map to same info set."""
        # P0 has Jack, history empty - can't distinguish opponent's card
        s1 = KuhnState(cards=(0, 1), history=())  # vs Queen
        s2 = KuhnState(cards=(0, 2), history=())  # vs King

        assert game.info_set_key(s1) == game.info_set_key(s2)

    def test_info_set_format(self, game):
        """Info set should be 'card:history' format."""
        state = KuhnState(cards=(1, 2), history=("p", "b"))
        # P0 to act, has Queen (1)
        key = game.info_set_key(state)
        assert key == "1:pb"

    def test_total_info_sets(self, game):
        """Kuhn poker has exactly 12 information sets."""
        info_sets = set()

        def traverse(state):
            if game.is_terminal(state):
                return
            info_sets.add(game.info_set_key(state))
            for action in game.actions(state):
                traverse(game.next_state(state, action))

        for initial in game.initial_states():
            traverse(initial)

        assert len(info_sets) == 12
