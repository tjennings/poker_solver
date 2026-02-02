from typing import Dict
import torch
from torch import Tensor
from tqdm import tqdm

from core.tensors import CompiledGame


class BatchedCFR:
    """
    GPU-accelerated CFR using batched parallel traversals.

    Runs batch_size independent CFR iterations simultaneously.
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
        Top-down pass: compute reach probabilities.

        Returns: [batch, num_nodes, num_players]
        """
        reach = torch.ones(
            self.batch_size,
            self.compiled.num_nodes,
            self.compiled.num_players,
            device=self.device,
        )

        # Process non-terminal nodes in topological order
        node_iter = range(self.compiled.num_nodes)
        if self.verbose:
            node_iter = tqdm(node_iter, desc="Forward pass", unit="node", leave=False)

        for node_idx in node_iter:
            if self.compiled.terminal_mask[node_idx]:
                continue

            player = self.compiled.node_player[node_idx].item()
            info_set = self.compiled.node_info_set[node_idx].item()

            for a in range(self.compiled.max_actions):
                if not self.compiled.action_mask[node_idx, a]:
                    continue

                child_idx = self.compiled.action_child[node_idx, a].item()
                prob = strategy[info_set, a]

                # Child inherits parent reach
                reach[:, child_idx, :] = reach[:, node_idx, :].clone()
                # Multiply acting player's reach by action probability
                reach[:, child_idx, player] *= prob

        return reach

    def backward_utils(self, reach: Tensor, strategy: Tensor) -> Tensor:
        """
        Bottom-up pass: compute expected utilities.

        Returns: [batch, num_nodes, num_players]
        """
        utils = torch.zeros(
            self.batch_size,
            self.compiled.num_nodes,
            self.compiled.num_players,
            device=self.device,
        )

        # Initialize terminal nodes
        terminal_indices = torch.where(self.compiled.terminal_mask)[0]
        for idx in terminal_indices:
            utils[:, idx, :] = self.compiled.terminal_utils[idx]

        # Process in reverse order (children before parents)
        node_iter = reversed(range(self.compiled.num_nodes))
        if self.verbose:
            node_iter = tqdm(list(node_iter), desc="Backward pass", unit="node", leave=False)

        for node_idx in node_iter:
            if self.compiled.terminal_mask[node_idx]:
                continue

            info_set = self.compiled.node_info_set[node_idx].item()

            # Weighted sum of child utilities
            for a in range(self.compiled.max_actions):
                if not self.compiled.action_mask[node_idx, a]:
                    continue

                child_idx = self.compiled.action_child[node_idx, a].item()
                prob = strategy[info_set, a]
                utils[:, node_idx, :] += prob * utils[:, child_idx, :]

        return utils

    def train_step(self):
        """One batched CFR iteration."""
        strategy = self.get_current_strategy()
        reach = self.forward_reach(strategy)
        utils = self.backward_utils(reach, strategy)

        # Update regrets for non-terminal nodes
        node_iter = range(self.compiled.num_nodes)
        if self.verbose:
            node_iter = tqdm(node_iter, desc="Updating regrets", unit="node", leave=False)

        for node_idx in node_iter:
            if self.compiled.terminal_mask[node_idx]:
                continue

            player = self.compiled.node_player[node_idx].item()
            opponent = 1 - player
            info_set = self.compiled.node_info_set[node_idx].item()

            # Counterfactual reach and node utility
            cf_reach = reach[:, node_idx, opponent]  # [batch]
            node_util = utils[:, node_idx, player]   # [batch]

            for a in range(self.compiled.max_actions):
                if not self.compiled.action_mask[node_idx, a]:
                    continue

                child_idx = self.compiled.action_child[node_idx, a].item()
                action_util = utils[:, child_idx, player]

                # Accumulate regret across batch
                regret = (cf_reach * (action_util - node_util)).sum()
                self.regret_sum[info_set, a] += regret

            # Accumulate strategy weighted by player reach
            my_reach = reach[:, node_idx, player].sum()
            for a in range(self.compiled.max_actions):
                if self.compiled.action_mask[node_idx, a]:
                    self.strategy_sum[info_set, a] += my_reach * strategy[info_set, a]

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
