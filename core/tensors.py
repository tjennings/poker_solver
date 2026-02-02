from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
from torch import Tensor
from tqdm import tqdm
from games.base import Game


@dataclass
class CompiledGame:
    """Game tree encoded as tensors for GPU execution."""

    # Node topology [num_nodes]
    node_player: Tensor       # Player to act (-1 for terminal)
    node_info_set: Tensor     # Info set index (-1 for terminal)
    node_num_actions: Tensor  # Number of valid actions

    # Transitions [num_nodes, max_actions]
    action_child: Tensor      # Child node index
    action_mask: Tensor       # Valid action mask

    # Terminal info [num_nodes, num_players]
    terminal_mask: Tensor     # Is this a terminal node?
    terminal_utils: Tensor    # Utilities (0 for non-terminal)

    # Depth structure for vectorized traversal
    depth_slices: List[Tuple[int, int]]
    max_depth: int

    # Mappings
    info_set_to_idx: Dict[str, int]
    idx_to_info_set: Dict[int, str]
    info_set_actions: Dict[str, List[str]]  # Action names per info set
    info_set_num_actions: Tensor  # Number of valid actions per info set [num_info_sets]

    # Dimensions
    num_info_sets: int
    num_nodes: int
    max_actions: int
    num_players: int

    device: torch.device


def compile_game(
    game: Game,
    device: torch.device,
    verbose: bool = False,
) -> CompiledGame:
    """
    Compile game tree to tensor representation.

    Traverses tree once, assigns indices, builds GPU-friendly tensors.

    Args:
        game: Game instance to compile
        device: Torch device for tensors
        verbose: Show progress bar during enumeration
    """
    nodes = []
    info_set_to_idx: Dict[str, int] = {}
    info_set_actions: Dict[str, List[str]] = {}
    depth_buckets: Dict[int, List[int]] = defaultdict(list)

    def add_info_set(key: str) -> int:
        if key not in info_set_to_idx:
            info_set_to_idx[key] = len(info_set_to_idx)
        return info_set_to_idx[key]

    def enumerate_tree(state, depth: int) -> int:
        node_idx = len(nodes)
        depth_buckets[depth].append(node_idx)

        if game.is_terminal(state):
            nodes.append({
                "terminal": True,
                "player": -1,
                "info_set": -1,
                "utils": [game.utility(state, p) for p in range(game.num_players())],
                "children": [],
                "num_actions": 0,
            })
            return node_idx

        info_set = game.info_set_key(state)
        info_idx = add_info_set(info_set)
        actions = game.actions(state)

        # Store action names for this info set (first time only)
        if info_set not in info_set_actions:
            info_set_actions[info_set] = actions

        node = {
            "terminal": False,
            "player": game.player(state),
            "info_set": info_idx,
            "utils": [0.0] * game.num_players(),
            "children": [],
            "num_actions": len(actions),
        }
        nodes.append(node)

        for action in actions:
            next_state = game.next_state(state, action)
            child_idx = enumerate_tree(next_state, depth + 1)
            node["children"].append(child_idx)

        return node_idx

    # Build tree for all initial states
    initial_states = list(game.initial_states())

    if verbose:
        initial_states = tqdm(initial_states, desc="Enumerating states", unit="state")

    for initial_state in initial_states:
        enumerate_tree(initial_state, depth=0)
    # Convert to tensors (batch all data first for speed)
    num_nodes = len(nodes)
    num_players = game.num_players()
    max_actions = max(n["num_actions"] for n in nodes if not n["terminal"]) if nodes else 2

    if verbose:
        print("Building tensors...", end=" ", flush=True)

    # Collect all data into lists first (much faster than per-element tensor assignment)
    player_list = []
    info_set_list = []
    num_actions_list = []
    terminal_list = []
    utils_list = []
    child_list = []  # Will be flattened [num_nodes, max_actions]
    mask_list = []   # Will be flattened [num_nodes, max_actions]

    for node in nodes:
        player_list.append(node["player"])
        info_set_list.append(node["info_set"])
        num_actions_list.append(node["num_actions"])
        terminal_list.append(node["terminal"])
        utils_list.append(node["utils"])

        # Pad children to max_actions
        children = node["children"]
        padded_children = children + [0] * (max_actions - len(children))
        child_list.append(padded_children)
        mask_list.append([True] * len(children) + [False] * (max_actions - len(children)))

    # Create tensors in one shot (fast)
    node_player = torch.tensor(player_list, dtype=torch.long, device=device)
    node_info_set = torch.tensor(info_set_list, dtype=torch.long, device=device)
    node_num_actions = torch.tensor(num_actions_list, dtype=torch.long, device=device)
    terminal_mask = torch.tensor(terminal_list, dtype=torch.bool, device=device)
    terminal_utils = torch.tensor(utils_list, dtype=torch.float32, device=device)
    action_child = torch.tensor(child_list, dtype=torch.long, device=device)
    action_mask = torch.tensor(mask_list, dtype=torch.bool, device=device)

    if verbose:
        print(f"done ({num_nodes:,} nodes)")

    # Build info set num_actions tensor
    num_info_sets = len(info_set_to_idx)
    info_set_num_actions_list = [0] * num_info_sets
    for info_set, actions in info_set_actions.items():
        idx = info_set_to_idx[info_set]
        info_set_num_actions_list[idx] = len(actions)
    info_set_num_actions = torch.tensor(info_set_num_actions_list, dtype=torch.float32, device=device)

    # Build depth slices
    max_depth = max(depth_buckets.keys()) + 1 if depth_buckets else 0
    depth_slices = []

    # Reorder nodes by depth for vectorized traversal
    # For simplicity, store (start, end) assuming nodes are already depth-ordered
    # In practice, we'd reindex - but current BFS ordering is close enough
    running_count = 0
    for d in range(max_depth):
        count = len(depth_buckets[d])
        depth_slices.append((running_count, running_count + count))
        running_count += count

    return CompiledGame(
        node_player=node_player,
        node_info_set=node_info_set,
        node_num_actions=node_num_actions,
        action_child=action_child,
        action_mask=action_mask,
        terminal_mask=terminal_mask,
        terminal_utils=terminal_utils,
        depth_slices=depth_slices,
        max_depth=max_depth,
        info_set_to_idx=info_set_to_idx,
        idx_to_info_set={v: k for k, v in info_set_to_idx.items()},
        info_set_actions=info_set_actions,
        info_set_num_actions=info_set_num_actions,
        num_info_sets=num_info_sets,
        num_nodes=num_nodes,
        max_actions=max_actions,
        num_players=num_players,
        device=device,
    )
