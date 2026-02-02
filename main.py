#!/usr/bin/env python3
"""
CFR Poker Solver - Main Entry Point

Usage:
    python main.py                    # Solve Kuhn Poker with defaults
    python main.py kuhn               # Explicit kuhn subcommand
    python main.py hunl -c standard   # HUNL preflop with standard config
"""

import argparse
import os
import sys
import time

from games.kuhn import KuhnPoker
from solver import Solver


def run_kuhn(args):
    """Run Kuhn Poker solver."""
    print("=" * 50)
    print("CFR Poker Solver - Kuhn Poker")
    print("=" * 50)
    print(f"Iterations: {args.iterations}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Create solver
    solver = Solver(
        game=KuhnPoker(),
        device=args.device,
        batch_size=args.batch_size,
        verbose=not args.quiet,
    )

    # Solve
    print("Training CFR...")
    start_time = time.time()
    strategy = solver.solve(
        iterations=args.iterations,
        verbose=not args.quiet,
    )
    elapsed = time.time() - start_time

    # Results
    print()
    print("=" * 50)
    print("Results")
    print("=" * 50)
    print(f"Time: {elapsed:.2f} seconds")
    print(f"Iterations/sec: {args.iterations / elapsed:.0f}")
    print(f"Final exploitability: {solver.exploitability():.6f}")
    print()

    print("Strategy:")
    print("-" * 40)
    for info_set in sorted(strategy.keys()):
        probs = strategy[info_set]
        card = ["J", "Q", "K"][int(info_set[0])]
        history = info_set[2:] if len(info_set) > 2 else "(root)"
        print(f"  {card} {history:8s}: pass={probs['p']:.3f}  bet={probs['b']:.3f}")


def run_hunl(args):
    """Run HUNL Preflop solver."""
    from config import load_config, get_preset_path
    from games.hunl_preflop import HUNLPreflop
    from cli.parser import parse_action_sequence
    from cli.interactive import run_interactive

    # Load config (preset or file)
    config_path = args.config

    # Check if it's a preset name (no path separator and no extension)
    if os.path.sep not in config_path and not config_path.endswith('.yaml'):
        preset_path = get_preset_path(config_path)
        if os.path.exists(preset_path):
            config_path = preset_path
        else:
            print(f"Error: Config preset '{args.config}' not found at {preset_path}", file=sys.stderr)
            sys.exit(1)

    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse action sequence if provided (may override stack)
    initial_actions = ()
    if args.action:
        try:
            parsed = parse_action_sequence(args.action)
            initial_actions = parsed.to_history_tuple()
            # Apply stack override if present
            if parsed.stack_override is not None:
                # Create a new config with the overridden stack
                from dataclasses import replace
                config = replace(config, stack_depth=parsed.stack_override)
        except ValueError as e:
            print(f"Error parsing action sequence: {e}", file=sys.stderr)
            sys.exit(1)

    # Load pre-trained strategy or train new one
    if args.load:
        if not args.quiet:
            print("=" * 50)
            print("CFR Poker Solver - HUNL Preflop")
            print("=" * 50)
            print(f"Loading strategy from: {args.load}")

        try:
            strategy = Solver.load_strategy(args.load)
            if not args.quiet:
                print(f"Loaded {len(strategy):,} information sets")
                print()
        except FileNotFoundError:
            print(f"Error: Strategy file not found: {args.load}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error loading strategy: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        if not args.quiet:
            print("=" * 50)
            print("CFR Poker Solver - HUNL Preflop")
            print("=" * 50)
            print(f"Config: {config.name}")
            print(f"Stack depth: {config.stack_depth} BB")
            print(f"Raise sizes: {config.raise_sizes}")
            print(f"Iterations: {args.iterations}")
            print(f"Device: {args.device}")
            print(f"Batch size: {args.batch_size}")
            if initial_actions:
                print(f"Initial actions: {' -> '.join(initial_actions)}")
            print()

        # Create game and solver
        game = HUNLPreflop(config)
        solver = Solver(
            game=game,
            device=args.device,
            batch_size=args.batch_size,
            verbose=not args.quiet,
        )

        # Train
        start_time = time.time()
        strategy = solver.solve(
            iterations=args.iterations,
            verbose=not args.quiet,
        )
        elapsed = time.time() - start_time

        # Results
        if not args.quiet:
            print()
            print("=" * 50)
            print("Results")
            print("=" * 50)
            print(f"Time: {elapsed:.2f} seconds")
            print(f"Iterations/sec: {args.iterations / elapsed:.0f}")
            print(f"Final exploitability: {solver.exploitability():.6f}")
            print()

        # Save strategy if requested
        if args.save:
            solver.save_strategy(args.save)
            if not args.quiet:
                import os
                size_mb = os.path.getsize(args.save) / (1024 * 1024)
                print(f"Strategy saved to: {args.save} ({size_mb:.1f} MB)")
                print()

    # Build interactive-compatible strategy (grouped by hand)
    # The raw strategy is keyed by info_set_key "POSITION:HAND:HISTORY"
    # We need to extract the action distributions per hand at current node
    interactive_strategy = _build_interactive_strategy(strategy, initial_actions)

    # Enter interactive mode unless disabled
    if not args.no_interactive:
        run_interactive(config, interactive_strategy, initial_actions=initial_actions)
    else:
        if not args.quiet:
            print("Training complete. Use --no-interactive to skip interactive mode.")
        else:
            print(f"Training complete. Exploitability: {solver.exploitability():.6f}")


def _build_interactive_strategy(raw_strategy, initial_actions):
    """Build a strategy dictionary compatible with interactive mode.

    Args:
        raw_strategy: Dictionary from solver with keys like "SB:AA:r2.5-r8"
        initial_actions: Tuple of initial actions applied

    Returns:
        Dictionary mapping hand strings to ActionDistribution
    """
    from cli.matrix import ActionDistribution

    # Group strategy by hand for the current history
    history_str = "-".join(initial_actions) if initial_actions else ""

    # Collect all hands and their action distributions
    hands_strategy = {}

    for info_set_key, probs in raw_strategy.items():
        # Parse "POSITION:HAND:HISTORY"
        parts = info_set_key.split(":")
        if len(parts) != 3:
            continue
        position, hand, history = parts

        # Only include strategies matching current history prefix
        if history != history_str:
            continue

        # Build action distribution from probabilities
        if hand not in hands_strategy:
            hands_strategy[hand] = {}

        # Merge action probabilities
        for action, prob in probs.items():
            if action not in hands_strategy[hand]:
                hands_strategy[hand][action] = 0.0
            hands_strategy[hand][action] = prob

    # Convert to ActionDistribution objects
    result = {}
    for hand, actions in hands_strategy.items():
        # Parse action probabilities into ActionDistribution format
        fold = actions.get("f", 0.0)
        call = actions.get("c", 0.0)
        all_in = actions.get("a", 0.0)

        # Collect raise actions (anything starting with 'r')
        raises = {}
        for action_key, prob in actions.items():
            if action_key.startswith("r"):
                try:
                    size = float(action_key[1:])
                    raises[size] = prob
                except ValueError:
                    pass

        result[hand] = ActionDistribution(
            fold=fold,
            call=call,
            raises=raises,
            all_in=all_in,
        )

    return result


def add_kuhn_args(parser):
    """Add kuhn-specific arguments to a parser."""
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=10000,
        help="Number of CFR iterations (default: 10000)"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device (default: auto)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Batch size for parallel iterations (default: 1 = vanilla CFR)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )


def add_hunl_args(parser):
    """Add hunl-specific arguments to a parser."""
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Config file path or preset name (standard, aggressive)"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=100000,
        help="Number of CFR iterations (default: 100000)"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device (default: auto)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1024,
        help="Batch size for parallel iterations (default: 1024)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--action", "-a",
        type=str,
        help='Initial action sequence (e.g., "50bb SBr2.5 BBr8")'
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive mode after training"
    )
    parser.add_argument(
        "--save", "-s",
        type=str,
        metavar="FILE",
        help="Save trained strategy to file (recommended: .strategy.gz)"
    )
    parser.add_argument(
        "--load", "-l",
        type=str,
        metavar="FILE",
        help="Load pre-trained strategy from file (skips training)"
    )


def main():
    # Check if first positional arg is a subcommand
    subcommands = {"kuhn", "hunl"}
    has_subcommand = len(sys.argv) > 1 and sys.argv[1] in subcommands

    if has_subcommand:
        # Use subparser mode
        parser = argparse.ArgumentParser(
            description="CFR Poker Solver",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        subparsers = parser.add_subparsers(dest="command", help="Game to solve")

        # Kuhn subcommand
        kuhn_parser = subparsers.add_parser("kuhn", help="Solve Kuhn Poker")
        add_kuhn_args(kuhn_parser)

        # HUNL subcommand
        hunl_parser = subparsers.add_parser("hunl", help="Solve HUNL Preflop")
        add_hunl_args(hunl_parser)

        args = parser.parse_args()

        if args.command == "kuhn":
            run_kuhn(args)
        elif args.command == "hunl":
            run_hunl(args)
    else:
        # No subcommand - default to kuhn (backward compatibility)
        parser = argparse.ArgumentParser(description="CFR Poker Solver")
        add_kuhn_args(parser)
        args = parser.parse_args()
        run_kuhn(args)


if __name__ == "__main__":
    main()
