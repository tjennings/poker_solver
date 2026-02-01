# CFR Poker Solver

A Counterfactual Regret Minimization (CFR) solver for extensive-form games, implemented in Python with PyTorch for GPU acceleration.

Currently supports Kuhn Poker, with an architecture designed for extension to larger games.

## Features

- **Vanilla CFR** - CPU-based reference implementation
- **Batched CFR** - GPU-accelerated parallel execution
- **Multi-device support** - CPU, CUDA, and Apple Silicon (MPS)
- **Exploitability calculation** - Measure strategy quality
- **Extensible game interface** - Add new games easily

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+

### Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd poker_solver

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies (required before running)
pip install torch numpy pytest
```

Alternatively, install as a package:

```bash
pip install -e ".[dev]"
```

**Note:** PyTorch is ~200MB and may take a few minutes to download.

## Usage

**Important:** Make sure you've installed dependencies first (see Setup above).

### Command Line

```bash
# Solve Kuhn Poker with default settings (10,000 iterations)
python main.py

# More iterations for better convergence
python main.py --iterations 50000

# Use GPU acceleration with batching
python main.py --device auto --batch-size 1024

# Quiet mode (no progress output)
python main.py --iterations 10000 --quiet
```

### Python API

```python
from solver import Solver
from games.kuhn import KuhnPoker

# Create solver
solver = Solver(KuhnPoker())

# Train and get strategy
strategy = solver.solve(iterations=10000, verbose=False)

# Check exploitability (lower is better, 0 = Nash equilibrium)
print(f"Exploitability: {solver.exploitability():.6f}")

# View strategy for specific situations
print(f"Jack at root: {strategy['0:']}")  # Card 0 = Jack
print(f"King facing bet: {strategy['2:b']}")  # Card 2 = King
```

### GPU Acceleration

```python
# Use GPU with batched execution
solver = Solver(
    KuhnPoker(),
    device="auto",  # Automatically selects CUDA or MPS
    batch_size=1024
)
solver.solve(iterations=100000, verbose=True)
```

## Example Output

```
==================================================
CFR Poker Solver - Kuhn Poker
==================================================
Iterations: 10000
Device: auto
Batch size: 1

Training CFR...

==================================================
Results
==================================================
Time: 1.02 seconds
Iterations/sec: 9835
Final exploitability: 0.001486

Strategy:
----------------------------------------
  J (root)  : pass=0.784  bet=0.216
  J b       : pass=1.000  bet=0.000
  K (root)  : pass=0.332  bet=0.668
  K b       : pass=0.000  bet=1.000
  ...
```

## Project Structure

```
poker_solver/
├── main.py              # CLI entry point
├── solver.py            # High-level Solver API
├── games/
│   ├── base.py          # Abstract Game interface
│   └── kuhn.py          # Kuhn Poker implementation
├── cfr/
│   ├── vanilla.py       # CPU-based vanilla CFR
│   └── batched.py       # GPU-accelerated batched CFR
├── core/
│   ├── device.py        # Device selection utility
│   ├── tensors.py       # Game tree compilation
│   └── exploitability.py # Best response calculation
└── tests/               # Test suite
```

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_nash_convergence.py -v
```

## Algorithm

This implementation follows the CFR algorithm from:

> Zinkevich, M., Johanson, M., Bowling, M., & Piccione, C. (2007).
> *Regret Minimization in Games with Incomplete Information.*
> Advances in Neural Information Processing Systems.

Key equations implemented:
- **Regret Matching** (Eq. 8): Strategy proportional to positive regrets
- **Counterfactual Value** (Eq. 7): Expected utility given reaching an information set
- **Average Strategy**: Converges to Nash equilibrium

## License

MIT
