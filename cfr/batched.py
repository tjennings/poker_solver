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

    def __init__(self, compiled: CompiledGame, batch_size: int = 1024, verbose: bool = False):
        self.compiled = compiled
        self.batch_size = batch_size
        self.device = compiled.device
        self.verbose = verbose

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

            # Child inherits parent reach
            child_reach = parent_reach.clone()

            # Multiply acting player's reach by action probability
            # probs is [num_edges], need to apply to correct player dimension
            batch_indices = torch.arange(self.batch_size, device=self.device)[:, None]
            edge_indices = torch.arange(len(parents), device=self.device)[None, :]

            # Update only the acting player's reach
            child_reach[batch_indices, edge_indices, players] *= probs

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
        if self.verbose:
            edge_iter = tqdm(edge_iter, desc="Backward pass", unit="depth", leave=False)

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
        """Vectorized batched CFR iteration."""
        strategy = self.get_current_strategy()
        reach = self.forward_reach(strategy)
        utils = self.backward_utils(reach, strategy)

        # Vectorized regret update for all non-terminal nodes
        nt_idx = self.non_terminal_indices  # [num_nt]
        info_sets = self.nt_info_sets  # [num_nt]
        players = self.nt_players  # [num_nt]
        opponents = self.nt_opponents  # [num_nt]

        # Get reach and utility values for non-terminal nodes
        # reach: [batch, num_nodes, num_players]
        # Use gather for efficient indexing without memory explosion
        batch_idx = torch.arange(self.batch_size, device=self.device)[:, None]

        # Index reach and utils for non-terminal nodes
        nt_reach = reach[:, nt_idx, :]  # [batch, num_nt, num_players]
        nt_utils = utils[:, nt_idx, :]  # [batch, num_nt, num_players]

        # Get counterfactual reach (opponent's reach) and node utility (player's utility)
        cf_reach = nt_reach.gather(2, opponents[None, :, None].expand(self.batch_size, -1, 1)).squeeze(-1)  # [batch, num_nt]
        node_util = nt_utils.gather(2, players[None, :, None].expand(self.batch_size, -1, 1)).squeeze(-1)  # [batch, num_nt]

        # For each action, compute regret and accumulate
        c = self.compiled
        for a in range(c.max_actions):
            # Mask for nodes that have this action
            action_valid = c.action_mask[nt_idx, a]  # [num_nt]
            if not action_valid.any():
                continue

            # Get child indices for this action
            child_idx = c.action_child[nt_idx, a]  # [num_nt]

            # Get action utility (child utility for acting player)
            child_utils = utils[:, child_idx, :]  # [batch, num_nt, num_players]
            action_util = child_utils.gather(2, players[None, :, None].expand(self.batch_size, -1, 1)).squeeze(-1)  # [batch, num_nt]

            # Compute regret: cf_reach * (action_util - node_util)
            regret = cf_reach * (action_util - node_util)  # [batch, num_nt]

            # Sum across batch and accumulate to info sets
            regret_sum = regret.sum(dim=0)  # [num_nt]

            # Scatter add to regret_sum tensor (only for valid actions)
            regret_sum = regret_sum * action_valid.float()
            self.regret_sum[:, a].scatter_add_(0, info_sets, regret_sum)

        # Vectorized strategy sum update
        my_reach = nt_reach.gather(2, players[None, :, None].expand(self.batch_size, -1, 1)).squeeze(-1).sum(dim=0)  # [num_nt]

        for a in range(c.max_actions):
            action_valid = c.action_mask[nt_idx, a]  # [num_nt]
            if not action_valid.any():
                continue

            strat_contrib = my_reach * strategy[info_sets, a] * action_valid.float()
            self.strategy_sum[:, a].scatter_add_(0, info_sets, strat_contrib)

        # CFR+ modification: floor regrets at zero
        self.regret_sum = torch.clamp(self.regret_sum, min=0)

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
