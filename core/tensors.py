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

    Optimized to enumerate tree structure ONCE, then compute info sets and
    utilities for all hand pairs in a batch.
    """
    from games.hunl_preflop import HUNLPreflop, HUNLState
    from core.hands import hand_to_string, HAND_COUNT
    from core.equity import get_preflop_equity

    # Check if this is an HUNL game we can optimize
    if isinstance(game, HUNLPreflop):
        return _compile_hunl_optimized(game, device, verbose, store_terminal_states)

    # Fall back to original implementation for other games
    return _compile_game_generic(game, device, verbose, store_terminal_states)


def _compile_hunl_optimized(
    game,
    device: torch.device,
    verbose: bool = False,
    store_terminal_states: bool = False,
) -> CompiledGame:
    """
    Optimized compilation for HUNL preflop.

    Enumerates tree structure once, then batches info set and utility computation.
    """
    from games.hunl_preflop import HUNLPreflop, HUNLState
    from core.hands import hand_to_string, HAND_COUNT
    from core.equity import get_preflop_equity

    # Force preflop-only mode for this optimization (the tree structure is the same)
    # Create a new game instance with preflop_only=True if needed
    if not game.preflop_only:
        game = HUNLPreflop(game.config, preflop_only=True)

    # Phase 1: Enumerate tree structure ONCE with template hands (using iteration)
    if verbose:
        print("Phase 1: Enumerating tree structure...", end=" ", flush=True)

    # Template state - hands don't affect tree structure
    template_state = HUNLState(
        hands=(0, 1),
        history=(),
        stack=game.stack_depth,
        pot=1.5,
        to_act=0,
    )

    # Tree structure: list of nodes with their properties
    tree_nodes = []  # [{history, player, is_terminal, actions, children}]
    depth_buckets: Dict[int, List[int]] = defaultdict(list)

    # Use iterative BFS - track parent info to fill children as we go
    history_to_idx = {}  # history tuple -> node index
    # Queue: (state, depth, parent_idx, action_idx_in_parent)
    pending = [(template_state, 0, None, None)]

    while pending:
        state, depth, parent_idx, action_idx = pending.pop(0)

        # Check if we've seen this history before (shouldn't happen in tree, but safety check)
        if state.history in history_to_idx:
            child_idx = history_to_idx[state.history]
            if parent_idx is not None:
                tree_nodes[parent_idx]["children"][action_idx] = child_idx
            continue

        node_idx = len(tree_nodes)
        history_to_idx[state.history] = node_idx
        depth_buckets[depth].append(node_idx)

        # Update parent's children list if this isn't the root
        if parent_idx is not None:
            tree_nodes[parent_idx]["children"][action_idx] = node_idx

        if game.is_terminal(state):
            tree_nodes.append({
                "history": state.history,
                "player": -1,
                "is_terminal": True,
                "actions": [],
                "children": [],
                "committed": state.committed,
            })
        else:
            actions = game.actions(state)
            # Pre-allocate children list with placeholders
            tree_nodes.append({
                "history": state.history,
                "player": game.player(state),
                "is_terminal": False,
                "actions": actions,
                "children": [-1] * len(actions),  # Will be filled as children are processed
                "committed": None,
            })
            for i, action in enumerate(actions):
                next_state = game.next_state(state, action)
                pending.append((next_state, depth + 1, node_idx, i))

    num_tree_nodes = len(tree_nodes)

    if verbose:
        print(f"done ({num_tree_nodes:,} unique nodes)")

    # Phase 2: Build info set mapping for all (hand, history) combinations
    if verbose:
        print("Phase 2: Building info set mappings...", end=" ", flush=True)

    info_set_to_idx: Dict[str, int] = {}
    info_set_actions: Dict[str, List[str]] = {}

    # For each non-terminal node in the tree, create info sets for all 169 hands
    for node in tree_nodes:
        if node["is_terminal"]:
            continue

        player = node["player"]
        position = "SB" if player == 0 else "BB"
        history_str = "-".join(node["history"])
        actions = node["actions"]

        for hand_idx in range(HAND_COUNT):
            hand_str = hand_to_string(hand_idx)
            info_set_key = f"{position}:{hand_str}:{history_str}"

            if info_set_key not in info_set_to_idx:
                info_set_to_idx[info_set_key] = len(info_set_to_idx)
                info_set_actions[info_set_key] = actions

    num_info_sets = len(info_set_to_idx)

    if verbose:
        print(f"done ({num_info_sets:,} info sets)")

    # Phase 3: Build tensors using vectorized operations
    if verbose:
        print("Phase 3: Building node tensors...", end=" ", flush=True)

    num_hand_pairs = HAND_COUNT * (HAND_COUNT - 1)  # 169 * 168
    total_nodes = num_tree_nodes * num_hand_pairs
    max_actions = max(len(n["actions"]) for n in tree_nodes if not n["is_terminal"]) if tree_nodes else 2

    # Pre-extract tree structure into arrays (just 2,302 elements)
    tree_players = []
    tree_terminal = []
    tree_num_actions = []
    tree_children = []
    tree_masks = []

    # For terminal nodes: precompute fold utilities (hand-independent)
    # Terminal utils format: (sb_util, bb_util) for fold, or None for showdown
    terminal_fold_utils = []  # For terminal nodes: (sb_util, bb_util) or None
    terminal_committed = []   # For terminal nodes: (sb_committed, bb_committed)
    terminal_node_indices = []  # Which tree indices are terminal

    for tree_idx, node in enumerate(tree_nodes):
        tree_players.append(node["player"])
        tree_terminal.append(node["is_terminal"])

        if node["is_terminal"]:
            tree_num_actions.append(0)
            tree_children.append([0] * max_actions)
            tree_masks.append([False] * max_actions)

            terminal_node_indices.append(tree_idx)
            committed = node["committed"]
            terminal_committed.append(committed)

            history = node["history"]
            last_action = history[-1] if history else "c"
            if last_action == "f":
                folder = (len(history) - 1) % 2
                if folder == 0:
                    terminal_fold_utils.append((-committed[0], committed[0]))
                else:
                    terminal_fold_utils.append((committed[1], -committed[1]))
            else:
                terminal_fold_utils.append(None)  # Needs equity calculation
        else:
            actions = node["actions"]
            children = node["children"]
            tree_num_actions.append(len(actions))
            padded_children = children + [0] * (max_actions - len(children))
            tree_children.append(padded_children)
            tree_masks.append([True] * len(children) + [False] * (max_actions - len(children)))

    # Convert tree structure to numpy arrays for fast operations
    import numpy as np
    tree_players_np = np.array(tree_players, dtype=np.int64)
    tree_terminal_np = np.array(tree_terminal, dtype=bool)
    tree_num_actions_np = np.array(tree_num_actions, dtype=np.int64)
    tree_children_np = np.array(tree_children, dtype=np.int64)
    tree_masks_np = np.array(tree_masks, dtype=bool)

    # Pre-compute info set indices for non-terminal nodes
    # info_set_lookup[tree_idx][hand_idx] = info_set_idx for that position
    non_terminal_indices = [i for i, t in enumerate(tree_terminal) if not t]
    info_set_lookup = {}  # tree_idx -> {hand_idx -> info_set_idx}

    for tree_idx in non_terminal_indices:
        node = tree_nodes[tree_idx]
        player = node["player"]
        position = "SB" if player == 0 else "BB"
        history_str = "-".join(node["history"])

        hand_to_info = {}
        for hand_idx in range(HAND_COUNT):
            hand_str = hand_to_string(hand_idx)
            info_set_key = f"{position}:{hand_str}:{history_str}"
            hand_to_info[hand_idx] = info_set_to_idx[info_set_key]
        info_set_lookup[tree_idx] = hand_to_info

    # Pre-allocate output arrays
    node_player_np = np.empty(total_nodes, dtype=np.int64)
    node_info_set_np = np.empty(total_nodes, dtype=np.int64)
    node_num_actions_np = np.empty(total_nodes, dtype=np.int64)
    terminal_mask_np = np.empty(total_nodes, dtype=bool)
    terminal_utils_np = np.zeros((total_nodes, 2), dtype=np.float32)
    action_child_np = np.empty((total_nodes, max_actions), dtype=np.int64)
    action_mask_np = np.empty((total_nodes, max_actions), dtype=bool)

    # Fill arrays by tiling tree structure for each hand pair
    hand_pairs = [(sb, bb) for sb in range(HAND_COUNT) for bb in range(HAND_COUNT) if sb != bb]

    if verbose:
        print(f"({total_nodes:,} nodes)...", end=" ", flush=True)

    terminal_states_list = [] if store_terminal_states else None

    for pair_idx, (sb_hand, bb_hand) in enumerate(hand_pairs):
        start_idx = pair_idx * num_tree_nodes
        end_idx = start_idx + num_tree_nodes

        # Copy tree structure (same for all hands)
        node_player_np[start_idx:end_idx] = tree_players_np
        node_num_actions_np[start_idx:end_idx] = tree_num_actions_np
        terminal_mask_np[start_idx:end_idx] = tree_terminal_np
        action_mask_np[start_idx:end_idx] = tree_masks_np

        # Adjust children indices for this hand pair
        action_child_np[start_idx:end_idx] = tree_children_np + start_idx

        # Fill info sets for non-terminal nodes
        for tree_idx in non_terminal_indices:
            global_idx = start_idx + tree_idx
            player = tree_players[tree_idx]
            hand_idx = sb_hand if player == 0 else bb_hand
            node_info_set_np[global_idx] = info_set_lookup[tree_idx][hand_idx]

        # Fill info sets for terminal nodes (always -1)
        for tree_idx in terminal_node_indices:
            node_info_set_np[start_idx + tree_idx] = -1

        # Fill utilities for terminal nodes
        sb_equity = get_preflop_equity(sb_hand, bb_hand)
        for i, tree_idx in enumerate(terminal_node_indices):
            global_idx = start_idx + tree_idx
            fold_utils = terminal_fold_utils[i]

            if fold_utils is not None:
                terminal_utils_np[global_idx, 0] = fold_utils[0]
                terminal_utils_np[global_idx, 1] = fold_utils[1]
            else:
                # Showdown - compute from equity
                committed = terminal_committed[i]
                pot = committed[0] + committed[1]
                terminal_utils_np[global_idx, 0] = sb_equity * pot - committed[0]
                terminal_utils_np[global_idx, 1] = (1 - sb_equity) * pot - committed[1]

            if store_terminal_states:
                history = tree_nodes[tree_idx]["history"]
                committed = terminal_committed[i]
                state = HUNLState(
                    hands=(sb_hand, bb_hand),
                    history=history,
                    stack=game.stack_depth,
                    pot=committed[0] + committed[1],
                    to_act=(len(history)) % 2,
                    committed=committed,
                )
                terminal_states_list.append((global_idx, state))

    if verbose:
        print("done")
        print("Phase 4: Converting to tensors...", end=" ", flush=True)

    # Convert numpy arrays to tensors
    node_player = torch.from_numpy(node_player_np).to(device)
    node_info_set = torch.from_numpy(node_info_set_np).to(device)
    node_num_actions = torch.from_numpy(node_num_actions_np).to(device)
    terminal_mask = torch.from_numpy(terminal_mask_np).to(device)
    terminal_utils = torch.from_numpy(terminal_utils_np).to(device)
    action_child = torch.from_numpy(action_child_np).to(device)
    action_mask = torch.from_numpy(action_mask_np).to(device)

    # Build info set num_actions tensor
    info_set_num_actions_list = [0] * num_info_sets
    for info_set, actions in info_set_actions.items():
        idx = info_set_to_idx[info_set]
        info_set_num_actions_list[idx] = len(actions)
    info_set_num_actions = torch.tensor(info_set_num_actions_list, dtype=torch.float32, device=device)

    # Build depth slices (simplified - just use single slice for all)
    max_depth = max(depth_buckets.keys()) + 1 if depth_buckets else 1
    depth_slices = [(0, total_nodes)]

    if verbose:
        print(f"done ({total_nodes:,} total nodes)")

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
        num_nodes=total_nodes,
        max_actions=max_actions,
        num_players=2,
        device=device,
        terminal_states=terminal_states_list,
    )


def _compile_game_generic(
    game: Game,
    device: torch.device,
    verbose: bool = False,
    store_terminal_states: bool = False,
) -> CompiledGame:
    """Original compile_game implementation for non-HUNL games."""
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

    initial_states = list(game.initial_states())
    if verbose:
        initial_states = tqdm(initial_states, desc="Enumerating states", unit="state")

    for initial_state in initial_states:
        enumerate_tree(initial_state, depth=0)

    num_nodes = len(nodes)
    num_players = game.num_players()
    max_actions = max(n["num_actions"] for n in nodes if not n["terminal"]) if nodes else 2

    if verbose:
        print("Building tensors...", end=" ", flush=True)

    player_list = []
    info_set_list = []
    num_actions_list = []
    terminal_list = []
    utils_list = []
    child_list = []
    mask_list = []

    for node in nodes:
        player_list.append(node["player"])
        info_set_list.append(node["info_set"])
        num_actions_list.append(node["num_actions"])
        terminal_list.append(node["terminal"])
        utils_list.append(node["utils"])

        children = node["children"]
        padded_children = children + [0] * (max_actions - len(children))
        child_list.append(padded_children)
        mask_list.append([True] * len(children) + [False] * (max_actions - len(children)))

    node_player = torch.tensor(player_list, dtype=torch.long, device=device)
    node_info_set = torch.tensor(info_set_list, dtype=torch.long, device=device)
    node_num_actions = torch.tensor(num_actions_list, dtype=torch.long, device=device)
    terminal_mask = torch.tensor(terminal_list, dtype=torch.bool, device=device)
    terminal_utils = torch.tensor(utils_list, dtype=torch.float32, device=device)
    action_child = torch.tensor(child_list, dtype=torch.long, device=device)
    action_mask = torch.tensor(mask_list, dtype=torch.bool, device=device)

    if verbose:
        print(f"done ({num_nodes:,} nodes)")

    num_info_sets = len(info_set_to_idx)
    info_set_num_actions_list = [0] * num_info_sets
    for info_set, actions in info_set_actions.items():
        idx = info_set_to_idx[info_set]
        info_set_num_actions_list[idx] = len(actions)
    info_set_num_actions = torch.tensor(info_set_num_actions_list, dtype=torch.float32, device=device)

    max_depth = max(depth_buckets.keys()) + 1 if depth_buckets else 0
    depth_slices = []
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
    """Recompute terminal utilities for a different stack depth."""
    if compiled.terminal_states is None:
        raise ValueError("CompiledGame was not compiled with store_terminal_states=True")

    if verbose:
        print(f"Recomputing utilities for {new_stack}BB stack...", end=" ", flush=True)

    from games.hunl_preflop import HUNLState

    num_nodes = compiled.num_nodes
    num_players = compiled.num_players
    new_utils = [[0.0] * num_players for _ in range(num_nodes)]

    for node_idx, state in compiled.terminal_states:
        new_state = _recompute_state_with_stack(state, new_stack)
        for p in range(num_players):
            new_utils[node_idx][p] = game.utility(new_state, p)

    compiled.terminal_utils = torch.tensor(
        new_utils, dtype=torch.float32, device=compiled.device
    )

    if verbose:
        print("done")


def _recompute_state_with_stack(state, new_stack: float):
    """Recompute a game state with a different stack depth."""
    from games.hunl_preflop import HUNLState

    pot = 1.5
    sb_committed = 0.5
    bb_committed = 1.0

    for i, action in enumerate(state.history):
        acting_player = i % 2

        if action == "f":
            pass
        elif action == "c":
            current_bet = _get_bet_at_action(state.history, i, new_stack)
            if acting_player == 0:
                pot += current_bet - sb_committed
                sb_committed = current_bet
            else:
                pot += current_bet - bb_committed
                bb_committed = current_bet
        elif action == "a":
            if acting_player == 0:
                pot += new_stack - sb_committed
                sb_committed = new_stack
            else:
                pot += new_stack - bb_committed
                bb_committed = new_stack
        elif action.startswith("r"):
            raise_to = float(action[1:])
            if acting_player == 0:
                pot += raise_to - sb_committed
                sb_committed = raise_to
            else:
                pot += raise_to - bb_committed
                bb_committed = raise_to

    return HUNLState(
        hands=state.hands,
        history=state.history,
        stack=new_stack,
        pot=pot,
        to_act=state.to_act,
        committed=(sb_committed, bb_committed),
    )


def _get_bet_at_action(history: tuple, action_idx: int, stack: float) -> float:
    """Get the bet amount at a specific action index."""
    for i in range(action_idx - 1, -1, -1):
        action = history[i]
        if action.startswith("r"):
            return float(action[1:])
        elif action == "a":
            return stack
    return 1.0
