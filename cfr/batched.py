from typing import Dict, List, Tuple
import torch
from torch import Tensor
from tqdm import tqdm

from core.tensors import CompiledGame


class BatchedCFR:
    """
    GPU-accelerated CFR using fully vectorized operations.

    Runs batch_size independent CFR iterations simultaneously.
    All node processing is done with tensor operations - no Python loops over nodes.
    """

    def __init__(
        self,
        compiled: CompiledGame,
        batch_size: int = 1024,
        verbose: bool = False,
        max_memory_gb: float = 4.0,
    ):
        self.compiled = compiled
        self.device = compiled.device
        self.verbose = verbose

        # Calculate memory-safe batch size
        self.batch_size = self._compute_safe_batch_size(batch_size, max_memory_gb)

        # Regrets and strategy sums: [num_info_sets, max_actions]
        self.regret_sum = torch.zeros(
            compiled.num_info_sets,
            compiled.max_actions,
            device=self.device,
        )
        self.strategy_sum = torch.zeros_like(self.regret_sum)
        self.iterations = 0

        # Pre-compute indices for vectorized operations
        self._precompute_indices()

    def _compute_safe_batch_size(self, requested_batch_size: int, max_memory_gb: float) -> int:
        """Compute a batch size that fits within memory limit."""
        num_nodes = self.compiled.num_nodes
        num_players = self.compiled.num_players

        # Memory per batch element (bytes):
        # - reach tensor: num_nodes * num_players * 4 (float32)
        # - utils tensor: num_nodes * num_players * 4 (float32)
        # - intermediate tensors: ~2x overhead
        bytes_per_batch = num_nodes * num_players * 4 * 2 * 2  # 2 tensors, 2x overhead

        max_memory_bytes = max_memory_gb * 1024 ** 3
        max_batch_size = max(1, int(max_memory_bytes / bytes_per_batch))

        if requested_batch_size > max_batch_size:
            if self.verbose:
                print(f"Reducing batch size from {requested_batch_size} to {max_batch_size} "
                      f"to fit in {max_memory_gb}GB memory limit")
            return max_batch_size

        return requested_batch_size

    def _precompute_indices(self):
        """Pre-compute all indices needed for vectorized tree traversal."""
        c = self.compiled

        # Non-terminal node mask and indices
        self.non_terminal_mask = ~c.terminal_mask
        self.non_terminal_indices = torch.where(self.non_terminal_mask)[0]
        self.num_non_terminal = len(self.non_terminal_indices)

        # Terminal node indices
        self.terminal_indices = torch.where(c.terminal_mask)[0]

        # For non-terminal nodes: info_set and player indices
        self.nt_info_sets = c.node_info_set[self.non_terminal_indices]  # [num_nt]
        self.nt_players = c.node_player[self.non_terminal_indices]  # [num_nt]
        self.nt_opponents = 1 - self.nt_players  # [num_nt]

        # Build edge list: (parent_idx, child_idx, action_idx, info_set, player)
        # grouped by depth for level-by-level processing
        self._build_depth_edges()

        # For regret updates: map non-terminal node index to position in nt arrays
        self.node_to_nt_idx = torch.zeros(c.num_nodes, dtype=torch.long, device=self.device)
        self.node_to_nt_idx[self.non_terminal_indices] = torch.arange(
            self.num_non_terminal, device=self.device
        )

        # Precompute batch indices for vectorized operations (avoids per-call tensor creation)
        self.batch_indices = torch.arange(self.batch_size, device=self.device)[:, None]

    def _build_depth_edges(self):
        """Build edges grouped by depth for level-parallel processing (vectorized)."""
        c = self.compiled

        if self.verbose:
            print("Building edge structure...", end=" ", flush=True)

        # Compute depth of each node using vectorized BFS
        depth = torch.full((c.num_nodes,), -1, dtype=torch.long, device=self.device)

        # Find root nodes (nodes that are not children of any other node)
        is_child = torch.zeros(c.num_nodes, dtype=torch.bool, device=self.device)
        valid_children = c.action_child[c.action_mask]
        is_child[valid_children] = True
        root_nodes = torch.where(~is_child)[0]
        depth[root_nodes] = 0

        # Vectorized BFS to compute depths
        current_depth = 0
        while True:
            # Get non-terminal nodes at current depth
            at_depth = (depth == current_depth) & self.non_terminal_mask
            if not at_depth.any():
                break

            # Get all valid (parent, child) pairs from nodes at this depth
            # action_mask[at_depth] gives [num_at_depth, max_actions]
            parent_mask = at_depth.unsqueeze(1) & c.action_mask  # [num_nodes, max_actions]

            # Get child indices for valid edges
            child_indices = c.action_child[parent_mask]  # [num_edges]

            # Set depth of children
            depth[child_indices] = current_depth + 1
            current_depth += 1

        max_depth = current_depth - 1 if current_depth > 0 else 0
        self.node_depth = depth
        self.max_depth = max_depth

        # Build all edges at once using vectorized operations
        # Create edge tensors for all non-terminal nodes
        nt_mask = self.non_terminal_mask  # [num_nodes]

        # Expand to [num_nodes, max_actions] for edge enumeration
        edge_valid = nt_mask.unsqueeze(1) & c.action_mask  # [num_nodes, max_actions]

        # Get indices of valid edges
        parent_nodes, action_indices = torch.where(edge_valid)
        child_nodes = c.action_child[parent_nodes, action_indices]
        info_sets = c.node_info_set[parent_nodes]
        players = c.node_player[parent_nodes]
        edge_depths = depth[parent_nodes]

        # Group edges by depth
        self.depth_edges: List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = []

        for d in range(max_depth + 1):
            mask = edge_depths == d
            if mask.any():
                self.depth_edges.append((
                    parent_nodes[mask],
                    child_nodes[mask],
                    action_indices[mask],
                    info_sets[mask],
                    players[mask],
                ))

        if self.verbose:
            num_edges = len(parent_nodes)
            print(f"done ({num_edges:,} edges, depth {max_depth})")

    def get_current_strategy(self) -> Tensor:
        """
        Equation (8): Strategy proportional to positive regret.

        Returns: [num_info_sets, max_actions]
        """
        positive_regrets = torch.clamp(self.regret_sum, min=0)
        totals = positive_regrets.sum(dim=1, keepdim=True)

        uniform = torch.ones_like(positive_regrets) / self.compiled.max_actions
        strategy = torch.where(
            totals > 0,
            positive_regrets / totals.clamp(min=1e-10),
            uniform,
        )
        return strategy

    def forward_reach(self, strategy: Tensor) -> Tensor:
        """
        Vectorized top-down pass: compute reach probabilities.

        Processes all edges at each depth level in parallel.

        Returns: [batch, num_nodes, num_players]
        """
        reach = torch.ones(
            self.batch_size,
            self.compiled.num_nodes,
            self.compiled.num_players,
            device=self.device,
        )

        # Process level by level (edges at same depth are independent)
        edge_iter = self.depth_edges
        if self.verbose:
            edge_iter = tqdm(edge_iter, desc="Forward pass", unit="depth", leave=False)

        for parents, children, actions, info_sets, players in edge_iter:
            # Get action probabilities for all edges at this level
            probs = strategy[info_sets, actions]  # [num_edges]

            # Get parent reach for all edges
            parent_reach = reach[:, parents, :]  # [batch, num_edges, num_players]

            # Child inherits parent reach, then multiply acting player's reach by action prob
            child_reach = parent_reach.clone()
            edge_indices = torch.arange(len(parents), device=self.device)
            child_reach[:, edge_indices, players] *= probs

            # Scatter to child nodes
            reach[:, children, :] = child_reach

        return reach

    def backward_utils(self, reach: Tensor, strategy: Tensor) -> Tensor:
        """
        Vectorized bottom-up pass: compute expected utilities.

        Processes all edges at each depth level in parallel (reverse order).

        Returns: [batch, num_nodes, num_players]
        """
        utils = torch.zeros(
            self.batch_size,
            self.compiled.num_nodes,
            self.compiled.num_players,
            device=self.device,
        )

        # Initialize terminal nodes (vectorized)
        utils[:, self.terminal_indices, :] = self.compiled.terminal_utils[self.terminal_indices]

        # Process levels in reverse order (children before parents)
        edge_iter = list(reversed(self.depth_edges))

        for parents, children, actions, info_sets, players in edge_iter:
            # Get action probabilities for all edges
            probs = strategy[info_sets, actions]  # [num_edges]

            # Get child utilities
            child_utils = utils[:, children, :]  # [batch, num_edges, num_players]

            # Weight by action probability
            weighted_utils = child_utils * probs[None, :, None]  # [batch, num_edges, num_players]

            # Accumulate to parent nodes using scatter_add
            # Need to handle multiple children per parent
            utils.scatter_add_(1, parents[None, :, None].expand(self.batch_size, -1, self.compiled.num_players), weighted_utils)

        return utils

    def train_step(self):
        """Fully vectorized batched CFR iteration - no Python loops over actions."""
        strategy = self.get_current_strategy()
        reach = self.forward_reach(strategy)
        utils = self.backward_utils(reach, strategy)

        # Vectorized regret update for all non-terminal nodes
        nt_idx = self.non_terminal_indices  # [num_nt]
        info_sets = self.nt_info_sets  # [num_nt]
        players = self.nt_players  # [num_nt]
        opponents = self.nt_opponents  # [num_nt]

        c = self.compiled
        num_nt = len(nt_idx)

        # Index reach and utils for non-terminal nodes
        nt_reach = reach[:, nt_idx, :]  # [batch, num_nt, num_players]
        nt_utils = utils[:, nt_idx, :]  # [batch, num_nt, num_players]

        # Get counterfactual reach (opponent's reach) and node utility (player's utility)
        cf_reach = nt_reach.gather(2, opponents[None, :, None].expand(self.batch_size, -1, 1)).squeeze(-1)  # [batch, num_nt]
        node_util = nt_utils.gather(2, players[None, :, None].expand(self.batch_size, -1, 1)).squeeze(-1)  # [batch, num_nt]

        # Get action mask and child indices for all actions at once
        action_mask = c.action_mask[nt_idx, :]  # [num_nt, max_actions]
        child_indices = c.action_child[nt_idx, :]  # [num_nt, max_actions]

        # Clamp child indices to valid range (invalid ones will be masked out)
        child_indices_clamped = child_indices.clamp(0, c.num_nodes - 1)

        # Get child utilities for ALL actions at once: [batch, num_nt, max_actions, num_players]
        # Use advanced indexing to gather all children
        all_child_utils = utils[:, child_indices_clamped, :]  # [batch, num_nt, max_actions, num_players]

        # Get action utilities for the acting player: [batch, num_nt, max_actions]
        players_expanded = players[None, :, None, None].expand(self.batch_size, -1, c.max_actions, 1)
        action_utils = all_child_utils.gather(3, players_expanded).squeeze(-1)  # [batch, num_nt, max_actions]

        # Compute regrets for all actions: cf_reach * (action_util - node_util)
        # node_util: [batch, num_nt] -> expand to [batch, num_nt, max_actions]
        regrets = cf_reach[:, :, None] * (action_utils - node_util[:, :, None])  # [batch, num_nt, max_actions]

        # Apply action mask and sum across batch
        regrets = regrets * action_mask[None, :, :].float()  # [batch, num_nt, max_actions]
        regret_sums = regrets.sum(dim=0)  # [num_nt, max_actions]

        # Scatter add regrets to info sets for all actions at once
        # Expand info_sets to [num_nt, max_actions] for scatter
        info_sets_expanded = info_sets[:, None].expand(-1, c.max_actions)  # [num_nt, max_actions]

        # Use scatter_add_ for each action dimension
        for a in range(c.max_actions):
            self.regret_sum[:, a].scatter_add_(0, info_sets_expanded[:, a], regret_sums[:, a])

        # Vectorized strategy sum update - all actions at once
        my_reach = nt_reach.gather(2, players[None, :, None].expand(self.batch_size, -1, 1)).squeeze(-1).sum(dim=0)  # [num_nt]

        # Get strategy values for all actions: [num_nt, max_actions]
        strat_at_nodes = strategy[info_sets, :]  # [num_nt, max_actions]

        # Compute contributions: my_reach * strategy * action_mask
        strat_contribs = my_reach[:, None] * strat_at_nodes * action_mask.float()  # [num_nt, max_actions]

        # Scatter add to strategy_sum for all actions
        for a in range(c.max_actions):
            self.strategy_sum[:, a].scatter_add_(0, info_sets_expanded[:, a], strat_contribs[:, a])

        # CFR+ modification: floor regrets at zero
        self.regret_sum.clamp_(min=0)

        self.iterations += self.batch_size

    def get_average_strategy(self) -> Dict[str, Dict[str, float]]:
        """Convert tensor strategy to dictionary format."""
        result = {}
        strategy_sum = self.strategy_sum.cpu().numpy()

        for info_set, idx in self.compiled.info_set_to_idx.items():
            action_names = self.compiled.info_set_actions.get(info_set, [])
            total = strategy_sum[idx].sum()

            if total > 0 and action_names:
                probs = {}
                for a, name in enumerate(action_names):
                    probs[name] = float(strategy_sum[idx, a] / total)
                result[info_set] = probs
            elif action_names:
                # Uniform distribution if no training yet
                num_actions = len(action_names)
                result[info_set] = {name: 1.0 / num_actions for name in action_names}

        return result
