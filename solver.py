from typing import Dict, Optional
import gzip
import pickle

from tqdm import tqdm

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
        max_memory_gb: float = 4.0,
    ):
        """
        Initialize solver.

        Args:
            game: Game instance to solve
            device: "auto", "cpu", "cuda", or "mps"
            batch_size: Number of parallel iterations (1 = vanilla CFR)
            verbose: Show progress during game tree compilation
            max_memory_gb: Maximum GPU memory to use (will reduce batch_size if needed)
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
            self.engine = BatchedCFR(
                self.compiled,
                batch_size,
                verbose=verbose,
                max_memory_gb=max_memory_gb,
            )
            self.batched = True
            self.batch_size = self.engine.batch_size  # May have been reduced

    def solve(
        self,
        iterations: int,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run CFR for specified iterations.

        Args:
            iterations: Number of CFR iterations
            verbose: Print progress updates with progress bar

        Returns:
            Average strategy for all information sets
        """
        if self.batched:
            steps = max(1, iterations // self.batch_size)
            step_iter = range(steps)

            if verbose:
                step_iter = tqdm(
                    step_iter,
                    desc="Training CFR+",
                    unit="step",
                    total=steps,
                )

            for step in step_iter:
                self.engine.train_step()

            # Compute exploitability only at the end (requires expensive GPU sync)
            if verbose:
                exploit = self.exploitability()
                step_iter.set_postfix({"exploit": f"{exploit:.4f}"})
        else:
            iter_range = range(iterations)

            if verbose:
                iter_range = tqdm(
                    iter_range,
                    desc="Training CFR",
                    unit="iter",
                )

            for i in iter_range:
                self.engine.train(1)

                # Update progress bar with exploitability periodically
                if verbose and (i + 1) % max(1, iterations // 10) == 0:
                    exploit = self.exploitability()
                    iter_range.set_postfix({"exploit": f"{exploit:.4f}"})

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

    def save_strategy(self, path: str) -> None:
        """
        Save strategy to compressed file.

        Args:
            path: File path (recommended: .strategy.gz extension)
        """
        strategy = self.get_strategy()
        with gzip.open(path, "wb", compresslevel=6) as f:
            pickle.dump(strategy, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_strategy(path: str) -> Dict[str, Dict[str, float]]:
        """
        Load strategy from compressed file.

        Args:
            path: File path to load from

        Returns:
            Strategy dictionary
        """
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
