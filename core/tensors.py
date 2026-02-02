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

    # For multi-stack optimization: store terminal states for utility recomputation
    terminal_states: List = None  # List of (node_idx, state) for terminal nodes


def compile_game(
    game: Game,
    device: torch.device,
    verbose: bool = False,
    store_terminal_states: bool = False,
) -> CompiledGame:
    """
    Compile game tree to tensor representation.

    Traverses tree once, assigns indices, builds GPU-friendly tensors.

    Args:
        game: Game instance to compile
        device: Torch device for tensors
        verbose: Show progress bar during enumeration
        store_terminal_states: If True, store terminal states for utility recomputation
    """
    nodes = []
    info_set_to_idx: Dict[str, int] = {}
    info_set_actions: Dict[str, List[str]] = {}
    depth_buckets: Dict[int, List[int]] = defaultdict(list)
    terminal_states = [] if store_terminal_states else None

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
            if store_terminal_states:
                terminal_states.append((node_idx, state))
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
        terminal_states=terminal_states,
    )


def recompute_utilities(
    compiled: CompiledGame,
    game: Game,
    new_stack: float,
    verbose: bool = False,
) -> None:
    """
    Recompute terminal utilities for a different stack depth.

    This modifies the compiled game's terminal_utils tensor in-place.
    Requires the CompiledGame to have been compiled with store_terminal_states=True.

    Args:
        compiled: CompiledGame with terminal_states stored
        game: Game instance (used to access utility computation)
        new_stack: New stack depth to use for utility computation
        verbose: Show progress
    """
    if compiled.terminal_states is None:
        raise ValueError("CompiledGame was not compiled with store_terminal_states=True")

    if verbose:
        print(f"Recomputing utilities for {new_stack}BB stack...", end=" ", flush=True)

    # Build new utilities list
    num_nodes = compiled.num_nodes
    num_players = compiled.num_players
    new_utils = [[0.0] * num_players for _ in range(num_nodes)]

    for node_idx, state in compiled.terminal_states:
        # Create a new state with the updated stack
        # We need to reconstruct pot based on history with new stack
        new_state = _recompute_state_with_stack(state, new_stack)
        for p in range(num_players):
            new_utils[node_idx][p] = game.utility(new_state, p)

    # Update tensor
    compiled.terminal_utils = torch.tensor(
        new_utils, dtype=torch.float32, device=compiled.device
    )

    if verbose:
        print("done")


def _recompute_state_with_stack(state, new_stack: float):
    """
    Recompute a game state with a different stack depth.

    This recalculates the pot based on the action history with the new stack.
    """
    # Import here to avoid circular imports
    from games.hunl_preflop import HUNLState

    # Recalculate pot with new stack
    pot = 1.5  # Initial blinds
    sb_committed = 0.5
    bb_committed = 1.0

    for i, action in enumerate(state.history):
        acting_player = i % 2  # 0=SB, 1=BB

        if action == "f":
            pass  # Fold doesn't change pot
        elif action == "c":
            # Call - match current bet
            current_bet = _get_bet_at_action(state.history, i, new_stack)
            if acting_player == 0:
                amount_to_add = current_bet - sb_committed
                pot += amount_to_add
                sb_committed = current_bet
            else:
                amount_to_add = current_bet - bb_committed
                pot += amount_to_add
                bb_committed = current_bet
        elif action == "a":
            # All-in with new stack
            if acting_player == 0:
                amount_to_add = new_stack - sb_committed
                pot += amount_to_add
                sb_committed = new_stack
            else:
                amount_to_add = new_stack - bb_committed
                pot += amount_to_add
                bb_committed = new_stack
        elif action.startswith("r"):
            raise_to = float(action[1:])
            if acting_player == 0:
                amount_to_add = raise_to - sb_committed
                pot += amount_to_add
                sb_committed = raise_to
            else:
                amount_to_add = raise_to - bb_committed
                pot += amount_to_add
                bb_committed = raise_to

    return HUNLState(
        hands=state.hands,
        history=state.history,
        stack=new_stack,
        pot=pot,
        to_act=state.to_act,
    )


def _get_bet_at_action(history: tuple, action_idx: int, stack: float) -> float:
    """Get the bet amount at a specific action index."""
    for i in range(action_idx - 1, -1, -1):
        action = history[i]
        if action.startswith("r"):
            return float(action[1:])
        elif action == "a":
            return stack
    return 1.0  # Default is BB
