from typing import Dict, Optional

from games.base import Game
from cfr.vanilla import VanillaCFR
from cfr.batched import BatchedCFR
from core.tensors import compile_game
from core.device import get_device
from core.exploitability import compute_exploitability


class Solver:
    """
    High-level API for solving extensive-form games with CFR.

    Example:
        solver = Solver(KuhnPoker(), device="auto")
        strategy = solver.solve(iterations=10000)
        print(f"Exploitability: {solver.exploitability()}")
    """

    def __init__(
        self,
        game: Game,
        device: str = "auto",
        batch_size: int = 1,
        verbose: bool = False,
    ):
        """
        Initialize solver.

        Args:
            game: Game instance to solve
            device: "auto", "cpu", "cuda", or "mps"
            batch_size: Number of parallel iterations (1 = vanilla CFR)
            verbose: Show progress during game tree compilation
        """
        self.game = game
        self.batch_size = batch_size

        if batch_size == 1:
            # Use vanilla CPU implementation
            self.engine = VanillaCFR(game)
            self.batched = False
        else:
            # Use batched GPU implementation
            self.device = get_device(device)
            self.compiled = compile_game(game, self.device, verbose=verbose)
            self.engine = BatchedCFR(self.compiled, batch_size)
            self.batched = True

    def solve(
        self,
        iterations: int,
        verbose: bool = True,
        log_interval: int = 1000,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run CFR for specified iterations.

        Args:
            iterations: Number of CFR iterations
            verbose: Print progress updates
            log_interval: How often to print (if verbose)

        Returns:
            Average strategy for all information sets
        """
        if self.batched:
            steps = max(1, iterations // self.batch_size)
            for step in range(steps):
                self.engine.train_step()

                if verbose and (step + 1) % max(1, log_interval // self.batch_size) == 0:
                    current_iter = (step + 1) * self.batch_size
                    exploit = self.exploitability()
                    print(f"Iteration {current_iter}: exploitability = {exploit:.6f}")
        else:
            if verbose:
                # Train in chunks for progress updates
                chunk_size = log_interval
                for i in range(0, iterations, chunk_size):
                    self.engine.train(min(chunk_size, iterations - i))
                    exploit = self.exploitability()
                    print(f"Iteration {i + chunk_size}: exploitability = {exploit:.6f}")
            else:
                self.engine.train(iterations)

        return self.get_strategy()

    def get_strategy(self) -> Dict[str, Dict[str, float]]:
        """Get current average strategy."""
        if self.batched:
            return self.engine.get_average_strategy()
        else:
            return self.engine.get_all_average_strategies()

    def exploitability(self) -> float:
        """Compute exploitability of current strategy."""
        return compute_exploitability(self.game, self.get_strategy())
