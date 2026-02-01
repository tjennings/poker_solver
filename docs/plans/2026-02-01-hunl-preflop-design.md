# HUNL Preflop Solver Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the CFR poker solver to support Heads-Up No-Limit Hold'em preflop with GPU-accelerated equilibrium computation and an interactive color-coded strategy matrix CLI.

**Architecture:** New `HUNLPreflop` game implementation using the existing `Game` interface, CFR+ for faster convergence, YAML config for bet sizing, and a terminal-based interactive explorer.

**Tech Stack:** PyTorch (GPU), PyYAML (config), ANSI escape codes (colors)

---

## 1. Action Encoding

Positional shorthand notation for action sequences:

| Example | Meaning |
|---------|---------|
| `SBr2.5` | SB raises to 2.5 BB |
| `BBc` | BB calls |
| `SBa` | SB all-in |
| `SBr2.5 BBr8 SBc` | SB opens 2.5, BB 3-bets to 8, SB calls |

Stack depth override inline: `50bb SBr2.5 BBr8`

Default stack: 100 BB (configurable in config file)

---

## 2. Hand Notation

**Canonical (169 buckets):**
- `AK` = suited (no suffix)
- `AKo` = offsuit
- `AA` = pair

**Specific suits (optional):**
- `AcKc` = Ace of clubs, King of clubs
- `AcKh` = Ace of clubs, King of hearts
- `AsAd` = Ace of spades, Ace of diamonds

Suit characters: `c` (clubs), `d` (diamonds), `h` (hearts), `s` (spades)

When user inputs specific combo, system maps to canonical hand for strategy lookup and applies card removal for opponent's matrix.

---

## 3. Config File Format

Location: `config/presets/standard.yaml` (defaults), or user-specified via `--config`

```yaml
name: "Standard 100BB"
stack_depth: 100

# All allowed raise sizes in BB
raise_sizes: [2.5, 3, 6, 8, 10, 15, 20, 25, 50, 100]
```

**Legal actions computed dynamically at each decision:**
- `fold` - if facing a bet
- `call` / `check` - match current bet or check
- `raise X` - for each size in raise_sizes where X > current bet and X ≤ stack
- `all_in` - always available

If no configured size exceeds current bet, min-raise = 2× current bet.

---

## 4. Game State

```python
@dataclass(frozen=True)
class HUNLState:
    hands: Tuple[int, int]      # (sb_hand, bb_hand) as 0-168 indices
    history: Tuple[str, ...]    # Action sequence, e.g., ("r2.5", "r8", "c")
    stack: float                # Effective stack in BB
    pot: float                  # Current pot in BB
    to_act: int                 # 0=SB, 1=BB
```

**Information Set Key Format:**
```
position:hand:action_history
```

Examples:
- `SB:AA:` → SB holds AA, opening decision
- `BB:T9:r2.5` → BB holds T9 suited, facing 2.5BB open
- `SB:72o:r2.5-r8` → SB holds 72 offsuit, facing 3-bet

**Initial States:**
- 169 × 168 = 28,392 starting hand combinations
- Weighted by card removal (combinatorics)

---

## 5. CLI Interface

**Launch:**
```bash
# Train and enter interactive mode
python main.py hunl --config standard.yaml --iterations 100000

# Jump to specific spot
python main.py hunl --config standard.yaml --action "SBr2.5 BBr8"

# Override stack inline
python main.py hunl --config standard.yaml --action "50bb SBr2.5"
```

**Matrix Display (13x13, left-aligned, colored by dominant action):**
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

**Color Coding (ANSI):**
- Red = fold dominant (dim 60%+, bright 85%+)
- Green = call dominant
- Blue = raise dominant
- Yellow = all-in dominant

**Navigation:**
- Type action to proceed (e.g., `c`, `r20`, `a`)
- `b` = go back one step
- `q` = quit

**Perspective:** Matrix shows current player to act.

---

## 6. CFR+ Implementation

**Modification to batched CFR:**
```python
# In cfr/batched.py, after regret update:
regrets = torch.clamp(regrets, min=0)  # CFR+ modification
```

Designed for future swap to Discounted CFR.

**Memory:**
- ~1.4M information sets × ~6 actions × 4 bytes = ~34MB regrets
- ~34MB cumulative strategy
- Sparse transition matrices in COO format

**Batched Traversal:**
- Process all 28,392 hand combinations in parallel
- GPU matrix operations on (num_hands × num_hands × num_actions) tensors

**Convergence:**
- Target: exploitability < 0.5% of pot after ~100K iterations
- Progress: display exploitability every 10K iterations
- Checkpoints for resuming training

---

## 7. Project Structure

**New Files:**
```
poker_solver/
├── games/
│   └── hunl_preflop.py      # HUNLPreflop game, HUNLState dataclass
├── config/
│   ├── loader.py            # YAML config loading & validation
│   └── presets/
│       ├── standard.yaml    # 100BB, standard sizes
│       └── aggressive.yaml  # 100BB, larger sizes
├── cli/
│   ├── interactive.py       # Action prompt loop, navigation
│   ├── matrix.py            # 13x13 matrix rendering, ANSI colors
│   └── parser.py            # Parse "50bb SBr2.5 BBr8" notation
└── core/
    └── hands.py             # Hand parsing (AK, AKo, AcKh), 169 buckets
```

**Modified Files:**
- `cfr/batched.py` - Add CFR+ regret flooring
- `main.py` - Add `hunl` subcommand
- `pyproject.toml` - Add `pyyaml` dependency

**Tests:**
- `tests/test_hunl_game.py` - State transitions, legality, payoffs
- `tests/test_hand_parser.py` - All input formats
- `tests/test_config.py` - Config loading, validation
- `tests/test_matrix_display.py` - Rendering correctness
- `tests/test_hunl_convergence.py` - CFR+ converges to low exploitability

**Dependencies:**
- `pyyaml` (new)
- `torch`, `numpy`, `pytest` (existing)
