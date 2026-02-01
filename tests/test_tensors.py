import pytest
import torch
from core.tensors import CompiledGame, compile_game
from core.device import get_device
from games.kuhn import KuhnPoker


class TestCompiledGame:
    @pytest.fixture
    def compiled(self):
        game = KuhnPoker()
        device = get_device("cpu")
        return compile_game(game, device)

    def test_has_correct_num_info_sets(self, compiled):
        """Should have 12 info sets for Kuhn poker."""
        assert compiled.num_info_sets == 12

    def test_has_correct_num_players(self, compiled):
        """Should have 2 players."""
        assert compiled.num_players == 2

    def test_has_correct_max_actions(self, compiled):
        """Should have 2 actions (pass/bet)."""
        assert compiled.max_actions == 2

    def test_node_tensors_correct_shape(self, compiled):
        """Node tensors should have num_nodes elements."""
        n = compiled.num_nodes
        assert compiled.node_player.shape == (n,)
        assert compiled.node_info_set.shape == (n,)
        assert compiled.terminal_mask.shape == (n,)

    def test_action_tensors_correct_shape(self, compiled):
        """Action tensors should be [num_nodes, max_actions]."""
        n = compiled.num_nodes
        a = compiled.max_actions
        assert compiled.action_child.shape == (n, a)
        assert compiled.action_mask.shape == (n, a)

    def test_terminal_nodes_marked(self, compiled):
        """Terminal nodes should have player=-1 and terminal_mask=True."""
        terminal_indices = torch.where(compiled.terminal_mask)[0]
        assert len(terminal_indices) > 0

        for idx in terminal_indices:
            assert compiled.node_player[idx] == -1

    def test_info_set_mapping_bijective(self, compiled):
        """Info set mappings should be inverses."""
        for key, idx in compiled.info_set_to_idx.items():
            assert compiled.idx_to_info_set[idx] == key

    def test_all_tensors_on_correct_device(self, compiled):
        """All tensors should be on the specified device."""
        device = compiled.device
        assert compiled.node_player.device == device
        assert compiled.action_child.device == device
        assert compiled.terminal_utils.device == device
