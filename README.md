# CFR Poker Solver

A Counterfactual Regret Minimization (CFR) solver for extensive-form games, implemented in Python with PyTorch for GPU acceleration.

Supports **Kuhn Poker** and **Heads-Up No-Limit Hold'em (HUNL)** with preflop through postflop solving and an interactive strategy explorer.

## Features

- **CFR+** - Faster convergence with regret flooring
- **Batched CFR** - GPU-accelerated parallel execution
- **Multi-device support** - CPU, CUDA, and Apple Silicon (MPS)
- **Exploitability calculation** - Measure strategy quality
- **HUNL Preflop Solver** - Full preflop equilibrium with 169 hand buckets
- **Per-Flop Solving** - Solve multiple flops independently and merge results
- **On-Demand Subgame Solving** - Solve postflop subtrees interactively
- **Interactive Strategy Explorer** - Color-coded 13x13 matrix display
- **Configurable Bet Sizes** - YAML presets for different stack depths
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

# 2. Create and activate virtual environment (use python3.14 if available)
python3.14 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies (required before running)
pip install torch numpy pytest pyyaml tqdm
```

Alternatively, install as a package:

```bash
pip install -e ".[dev]"
```

**Note:** PyTorch is ~200MB and may take a few minutes to download.

## Usage

**Important:** Make sure you've installed dependencies first (see Setup above).

### Command Line

#### Kuhn Poker

```bash
# Solve Kuhn Poker with default settings (10,000 iterations)
python main.py kuhn

# More iterations for better convergence
python main.py kuhn --iterations 50000

# Use GPU acceleration with batching
python main.py kuhn --device auto --batch-size 1024

# Quiet mode (no progress output)
python main.py kuhn --iterations 10000 --quiet
```

#### HUNL Preflop

```bash
# Solve HUNL Preflop with standard 100BB config (enters interactive mode)
python main.py hunl --config standard --iterations 100000

# Quick test with tiny config (5BB, faster)
python main.py hunl --config tiny --iterations 1000

# Train without entering interactive mode
python main.py hunl --config standard --iterations 50000 --no-interactive

# Start interactive mode at a specific action sequence
python main.py hunl --config standard --action "SBr2.5 BBr8"

# Override stack depth inline (50BB effective stack)
python main.py hunl --config standard --action "50bb SBr2.5"

# GPU acceleration
python main.py hunl --config standard --iterations 100000 --device cuda

# Save trained strategy to file (for instant loading later)
python main.py hunl --config standard --iterations 100000 --save strategy.gz

# Load pre-trained strategy (skips training entirely)
python main.py hunl --config standard --load strategy.gz

# Train full game (preflop through river) - requires simplified config
python main.py hunl --config perflop --iterations 1000 --postflop

# Train specific stack depth(s) only
python main.py hunl --config standard --stack 50 100 --iterations 50000
```

#### Per-Flop Solving

Solve multiple flop textures independently and merge results:

```bash
# Solve all 25 flops in category 25 (flop-only for speed)
python main.py hunl --config perflop --flops 25 --iterations 1000 --terminal-street flop

# Solve 49 flops through the turn
python main.py hunl --config perflop --flops 49 --iterations 500 --terminal-street turn

# Available flop categories: 25, 49, 85, 184 (from config/flops.yaml)
```

**Terminal Street Options:**
- `--terminal-street flop` - Showdown at flop (fastest, ~27s per flop)
- `--terminal-street turn` - Showdown at turn (slower, larger tree)
- `--terminal-street river` - Full game to river (slowest, may require simplified config)

#### Postflop with Custom Board

```bash
# Interactive mode with custom board (prompted when entering flop)
python main.py hunl --config standard --load strategy.gz

# Specify board via CLI (for non-interactive use)
python main.py hunl --config standard --load strategy.gz --board "AhKd2cJs8s"
```

**Action Notation:**
- `SBr2.5` - Small blind raises to 2.5 BB
- `BBc` - Big blind calls
- `SBa` - Small blind all-in
- `SBr2.5 BBr8 SBc` - SB opens 2.5, BB 3-bets to 8, SB calls

**Interactive Commands:**
- `f` - Fold
- `c` - Call/Check
- `r20` - Raise to 20 BB
- `a` - All-in
- `b` - Go back one step
- `q` - Quit

### Python API

#### Kuhn Poker

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

#### HUNL Preflop

```python
from solver import Solver
from games.hunl_preflop import HUNLPreflop
from config.loader import load_config

# Load config and create game
config = load_config("standard")
game = HUNLPreflop(config)

# Create solver with GPU
solver = Solver(game, device="auto", batch_size=1024)

# Train (may take several minutes for low exploitability)
strategy = solver.solve(iterations=100000, verbose=True)

# Check convergence
print(f"Exploitability: {solver.exploitability():.6f}")

# View strategy for specific situations
# Information set format: "position:hand:history"
print(f"SB opens with AA: {strategy['SB:AA:']}")
print(f"BB facing open with T9s: {strategy['BB:T9:r2.5']}")
print(f"SB facing 3-bet with 72o: {strategy['SB:72o:r2.5-r8']}")
```

**Hand Notation:**
- `AA`, `KK` - Pairs
- `AK`, `T9` - Suited hands (no suffix)
- `AKo`, `T9o` - Offsuit hands
- `AcKh`, `AsAd` - Specific suits (optional)

#### Per-Flop Solving

```python
from solver import Solver
from games.hunl_preflop import HUNLPreflop, make_board_config
from config import load_config, load_flops
from dataclasses import replace

# Load config and flops
config = load_config("config/presets/perflop.yaml")
flops = load_flops()[25]  # 25 flop textures

# Solve each flop independently
all_strategies = {}
for flop in flops[:5]:  # First 5 flops
    board = make_board_config(flop)  # Creates flop/turn/river board
    game = HUNLPreflop(
        replace(config, stack_depths=[25]),
        preflop_only=False,
        board=board,
        terminal_street='flop'  # Showdown at flop for speed
    )

    solver = Solver(game, device="cpu", batch_size=512)
    strategy = solver.solve(iterations=500, verbose=False)
    all_strategies["".join(flop)] = strategy

print(f"Solved {len(all_strategies)} flops")
```

#### Postflop Subgame Solving

```python
from solver import Solver
from games.postflop_subgame import PostflopSubgame
from config import load_config

config = load_config("config/presets/standard.yaml")

# Create postflop subgame starting from a specific pot/stack
subgame = PostflopSubgame(
    config=config,
    starting_pot=14.0,      # Pot after preflop action
    starting_stacks=(93.0, 93.0),  # Remaining stacks
    starting_street="flop",
    board={
        "flop": ("Ah", "Kd", "2c"),
        "turn": ("Ah", "Kd", "2c", "Js"),
        "river": ("Ah", "Kd", "2c", "Js", "8s"),
    }
)

solver = Solver(subgame, device="cpu", batch_size=512)
strategy = solver.solve(iterations=1000, verbose=True)
print(f"Postflop info sets: {len(strategy)}")
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

### Kuhn Poker

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

### HUNL Preflop Interactive Mode

```
HUNL Preflop | Stack: 100BB | Pot: 10.5BB
Action: SBr2.5 BBr8 | SB to act

AA  AK  AQ  AJ  AT  A9  A8  A7  A6  A5  A4  A3  A2
AKo KK  KQ  KJ  KT  K9  K8  K7  K6  K5  K4  K3  K2
AQo KQo QQ  QJ  QT  Q9  Q8  Q7  Q6  Q5  Q4  Q3  Q2
AJo KJo QJo JJ  JT  J9  J8  J7  J6  J5  J4  J3  J2
ATo KTo QTo JTo TT  T9  T8  T7  T6  T5  T4  T3  T2
A9o K9o Q9o J9o T9o 99  98  97  96  95  94  93  92
A8o K8o Q8o J8o T8o 98o 88  87  86  85  84  83  82
A7o K7o Q7o J7o T7o 97o 87o 77  76  75  74  73  72
A6o K6o Q6o J6o T6o 96o 86o 76o 66  65  64  63  62
A5o K5o Q5o J5o T5o 95o 85o 75o 65o 55  54  53  52
A4o K4o Q4o J4o T4o 94o 84o 74o 64o 54o 44  43  42
A3o K3o Q3o J3o T3o 93o 83o 73o 63o 53o 43o 33  32
A2o K2o Q2o J2o T2o 92o 82o 72o 62o 52o 42o 32o 22

Actions: f (fold) | c (call) | r20 (raise 20BB) | a (all-in) | b (back) | q (quit)
>
```

**Color Coding (in terminal):**
- Red = fold dominant
- Green = call dominant
- Blue = raise dominant
- Yellow = all-in dominant

Brightness indicates frequency (brighter = higher frequency).

## Project Structure

```
poker_solver/
├── main.py              # CLI entry point
├── solver.py            # High-level Solver API
├── games/
│   ├── base.py          # Abstract Game interface
│   ├── kuhn.py          # Kuhn Poker implementation
│   ├── hunl_preflop.py  # HUNL full game implementation
│   └── postflop_subgame.py # Postflop subgame for on-demand solving
├── cfr/
│   ├── vanilla.py       # CPU-based vanilla CFR
│   └── batched.py       # GPU-accelerated batched CFR+
├── core/
│   ├── device.py        # Device selection utility
│   ├── tensors.py       # Game tree compilation
│   ├── hands.py         # Hand parsing (169 buckets)
│   └── exploitability.py # Best response calculation
├── config/
│   ├── loader.py        # YAML config loading
│   ├── flops.yaml       # Flop configurations (25, 49, 85, 184 textures)
│   └── presets/
│       ├── standard.yaml # 100BB, standard sizes
│       ├── tiny.yaml     # 5BB, fast testing
│       └── perflop.yaml  # Simplified config for per-flop solving
├── cli/
│   ├── interactive.py   # Strategy explorer loop
│   ├── matrix.py        # 13x13 matrix rendering
│   └── parser.py        # Action notation parser
└── tests/               # Test suite
```

## Running Tests

```bash
# Run all tests (excludes slow HUNL tests by default)
pytest

# Run with verbose output
pytest -v

# Include slow HUNL integration tests
pytest -m "slow" -v

# Run all tests including slow ones
pytest -m "" -v

# Run specific test file
pytest tests/test_nash_convergence.py -v
```

## Configuration

### Config File Options

```yaml
name: "Config Name"
stack_depths: [25, 50, 100]    # Stack depths in BB
raise_sizes: [2.5, 8, 20]      # Preflop raise sizes in BB
max_bets_per_round: 4          # Max bets/raises per preflop round

# Postflop settings (for tractable tree sizes)
postflop_max_bets_per_round: 1 # Keep small (1-2) for tractability
postflop_raise_sizes: [5, 10]  # Optional; if omitted, uses pot-relative sizing
```

### Preset Configs

| Preset | Stack | Raise Sizes | Use Case |
|--------|-------|-------------|----------|
| `standard` | 100BB | 2.5, 8, 20, 100 | Full preflop solving |
| `tiny` | 5-25BB | 2, 3, 7 | Quick testing |
| `perflop` | 25BB | 2.5, 8 | Per-flop full game solving |

## Algorithm

This implementation uses **CFR+** (CFR with regret flooring), based on:

> Tammelin, O. (2014). *Solving Large Imperfect Information Games Using CFR+.*
> arXiv:1407.5042

Original CFR from:

> Zinkevich, M., Johanson, M., Bowling, M., & Piccione, C. (2007).
> *Regret Minimization in Games with Incomplete Information.*
> Advances in Neural Information Processing Systems.

Key equations implemented:
- **Regret Matching** (Eq. 8): Strategy proportional to positive regrets
- **CFR+ Modification**: Floor regrets at zero for faster convergence
- **Counterfactual Value** (Eq. 7): Expected utility given reaching an information set
- **Average Strategy**: Converges to Nash equilibrium

## License

MIT
