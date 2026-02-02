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
    from dataclasses import replace

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

    # Filter stack depths if --stack is provided
    if hasattr(args, 'stack') and args.stack:
        requested_stacks = [float(s) for s in args.stack]
        invalid_stacks = [s for s in requested_stacks if s not in config.stack_depths]
        if invalid_stacks:
            available = ", ".join(str(int(s) if s == int(s) else s) for s in config.stack_depths)
            print(f"Error: Stack(s) {invalid_stacks} not in config. Available: {available}", file=sys.stderr)
            sys.exit(1)
        config = replace(config, stack_depths=requested_stacks)

    # Parse action sequence if provided (may override stack for interactive)
    initial_actions = ()
    interactive_stack = None
    if args.action:
        try:
            parsed = parse_action_sequence(args.action)
            initial_actions = parsed.to_history_tuple()
            # Apply stack override if present (for interactive only)
            if parsed.stack_override is not None:
                interactive_stack = parsed.stack_override
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
                if Solver.is_multi_stack_strategy(strategy):
                    stacks = sorted(strategy.keys())
                    total_info_sets = sum(len(s) for s in strategy.values())
                    stacks_str = ", ".join(str(int(s) if s == int(s) else s) for s in stacks)
                    print(f"Loaded {total_info_sets:,} information sets across {len(stacks)} stacks: {stacks_str}BB")
                else:
                    print(f"Loaded {len(strategy):,} information sets")
                print()
        except FileNotFoundError:
            print(f"Error: Strategy file not found: {args.load}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error loading strategy: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Train for all configured stack depths
        stack_depths = config.stack_depths
        multi_stack = len(stack_depths) > 1

        if not args.quiet:
            print("=" * 50)
            print("CFR Poker Solver - HUNL Preflop")
            print("=" * 50)
            print(f"Config: {config.name}")
            if multi_stack:
                stacks_str = ", ".join(str(int(s) if s == int(s) else s) for s in stack_depths)
                print(f"Stack depths: {stacks_str} BB")
            else:
                print(f"Stack depth: {stack_depths[0]} BB")
            print(f"Raise sizes: {config.raise_sizes}")
            print(f"Iterations: {args.iterations}")
            print(f"Device: {args.device}")
            print(f"Batch size: {args.batch_size}")
            if initial_actions:
                print(f"Initial actions: {' -> '.join(initial_actions)}")
            print()

        # Train each stack sequentially
        all_strategies = {}
        total_start_time = time.time()

        # Check if we can reuse compiled game across stacks
        # Trees are identical if all raises fit within all stacks
        can_share_compilation = (
            multi_stack and
            args.batch_size > 1 and
            max(config.raise_sizes) <= min(stack_depths)
        )

        shared_solver = None
        if can_share_compilation and not args.quiet:
            print("(Sharing game tree compilation across stacks)")
            print()

        for i, stack_depth in enumerate(stack_depths):
            if multi_stack and not args.quiet:
                print(f"[{i+1}/{len(stack_depths)}] Training {int(stack_depth) if stack_depth == int(stack_depth) else stack_depth}BB stack...")
                print("-" * 40)

            # Create config for this stack depth
            single_stack_config = replace(config, stack_depths=[stack_depth])

            # Create game and solver (preflop_only=True for optimized compilation)
            game = HUNLPreflop(single_stack_config, preflop_only=True)

            if can_share_compilation:
                if shared_solver is None:
                    # First stack: compile with terminal state storage
                    solver = Solver(
                        game=game,
                        device=args.device,
                        batch_size=args.batch_size,
                        verbose=not args.quiet,
                        max_memory_gb=args.max_memory,
                        store_terminal_states=True,
                    )
                    shared_solver = solver
                else:
                    # Subsequent stacks: reuse compiled game, update utilities
                    solver = shared_solver
                    solver.game = game  # Update game reference
                    solver.update_utilities_for_stack(stack_depth, verbose=not args.quiet)
                    solver.reset()
            else:
                solver = Solver(
                    game=game,
                    device=args.device,
                    batch_size=args.batch_size,
                    verbose=not args.quiet,
                    max_memory_gb=args.max_memory,
                )

            # Train
            start_time = time.time()
            stack_strategy = solver.solve(
                iterations=args.iterations,
                verbose=not args.quiet,
            )
            elapsed = time.time() - start_time

            all_strategies[stack_depth] = stack_strategy

            # Results for this stack
            if not args.quiet:
                print()
                if multi_stack:
                    print(f"[{int(stack_depth) if stack_depth == int(stack_depth) else stack_depth}BB] ", end="")
                print(f"Time: {elapsed:.2f}s | Exploitability: {solver.exploitability():.6f}")
                print()

        total_elapsed = time.time() - total_start_time

        # Final summary for multi-stack
        if multi_stack and not args.quiet:
            print("=" * 50)
            print("Training Complete")
            print("=" * 50)
            print(f"Total time: {total_elapsed:.2f} seconds")
            print()

        # Use nested format for multi-stack, flat for single
        if multi_stack:
            strategy = all_strategies
        else:
            strategy = all_strategies[stack_depths[0]]

        # Save strategy if requested
        if args.save:
            if multi_stack:
                Solver.save_multi_stack_strategy(args.save, all_strategies)
            else:
                solver.save_strategy(args.save)
            if not args.quiet:
                size_mb = os.path.getsize(args.save) / (1024 * 1024)
                print(f"Strategy saved to: {args.save} ({size_mb:.1f} MB)")
                print()

    # Enter interactive mode unless disabled
    # For loaded strategies, check if we need to filter to a specific stack
    if interactive_stack is not None and Solver.is_multi_stack_strategy(strategy):
        if interactive_stack in strategy:
            # Start with the requested stack selected
            pass  # InteractiveSession will handle this
        else:
            available = sorted(strategy.keys())
            available_str = ", ".join(str(int(s) if s == int(s) else s) for s in available)
            print(f"Warning: Requested stack {interactive_stack}BB not in strategy. Available: {available_str}")

    if not args.no_interactive:
        run_interactive(config, strategy, initial_actions=initial_actions)
    else:
        if not args.quiet:
            print("Training complete. Use --no-interactive to skip interactive mode.")


def add_common_args(parser, iterations_default=10000, batch_default=1):
    """Add common arguments shared by all game solvers."""
    parser.add_argument(
        "--iterations", "-i", type=int, default=iterations_default,
        help=f"Number of CFR iterations (default: {iterations_default})"
    )
    parser.add_argument(
        "--device", "-d", type=str, default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device (default: auto)"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=batch_default,
        help=f"Batch size for parallel iterations (default: {batch_default})"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output"
    )


def add_kuhn_args(parser):
    """Add kuhn-specific arguments to a parser."""
    add_common_args(parser, iterations_default=10000, batch_default=1)


def add_hunl_args(parser):
    """Add hunl-specific arguments to a parser."""
    parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Config file path or preset name (standard, aggressive)"
    )
    add_common_args(parser, iterations_default=100000, batch_default=1024)
    parser.add_argument(
        "--max-memory", "-m", type=float, default=4.0,
        help="Maximum GPU memory in GB (default: 4.0)"
    )
    parser.add_argument(
        "--action", "-a", type=str,
        help='Initial action sequence (e.g., "50bb SBr2.5 BBr8")'
    )
    parser.add_argument("--no-interactive", action="store_true",
        help="Skip interactive mode after training")
    parser.add_argument(
        "--save", "-s", type=str, metavar="FILE",
        help="Save trained strategy to file (recommended: .strategy.gz)"
    )
    parser.add_argument(
        "--load", "-l", type=str, metavar="FILE",
        help="Load pre-trained strategy from file (skips training)"
    )
    parser.add_argument(
        "--stack", type=float, nargs="+", metavar="DEPTH",
        help="Train only specific stack depth(s) from config (e.g., --stack 25 50)"
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
