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
from config.loader import load_flops


def parse_board(board_str: str) -> tuple:
    """Parse a board string like 'AhKd2c' into tuple of cards ('Ah', 'Kd', '2c').

    Args:
        board_str: Board string with 2-char cards (e.g., "AhKd2cJs8d")

    Returns:
        Tuple of card strings

    Raises:
        ValueError: If board string is invalid
    """
    if not board_str:
        return None

    # Remove spaces
    board_str = board_str.replace(" ", "")

    # Each card is 2 characters (rank + suit)
    if len(board_str) % 2 != 0:
        raise ValueError(f"Invalid board string: {board_str} (must be pairs of rank+suit)")

    cards = []
    valid_ranks = "23456789TJQKA"
    valid_suits = "cdhs"

    for i in range(0, len(board_str), 2):
        rank = board_str[i].upper()
        suit = board_str[i + 1].lower()

        if rank not in valid_ranks:
            raise ValueError(f"Invalid rank '{rank}' in board string")
        if suit not in valid_suits:
            raise ValueError(f"Invalid suit '{suit}' in board string")

        cards.append(rank + suit)

    if len(cards) < 3:
        raise ValueError(f"Board must have at least 3 cards (flop), got {len(cards)}")
    if len(cards) > 5:
        raise ValueError(f"Board cannot have more than 5 cards, got {len(cards)}")

    return tuple(cards)


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


def run_hunl_per_flop(args, config, flops):
    """Run HUNL solver for multiple flops, solving each independently.

    Args:
        args: Parsed command line arguments
        config: Config object
        flops: List of flop tuples to solve

    Returns:
        Combined strategy dict
    """
    from games.hunl_preflop import HUNLPreflop, make_board_config
    from dataclasses import replace

    total_flops = len(flops)
    all_strategies = {}
    total_start_time = time.time()

    # Get terminal street setting
    terminal_street = getattr(args, 'terminal_street', 'river')

    if not args.quiet:
        print(f"Solving {total_flops} flops independently...")
        if terminal_street != 'river':
            print(f"(Terminal street: {terminal_street})")
        print()

    for i, flop in enumerate(flops):
        flop_str = "".join(flop)

        if not args.quiet:
            print(f"[{i+1}/{total_flops}] Flop: {flop[0]} {flop[1]} {flop[2]}")
            print("-" * 40)

        # Create board config for this flop
        board_config = make_board_config(flop)

        # Create config for single stack depth
        single_stack_config = replace(config, stack_depths=[config.stack_depths[0]])

        # Create game with this specific flop
        game = HUNLPreflop(single_stack_config, preflop_only=False, board=board_config, terminal_street=terminal_street)

        # Create solver
        solver = Solver(
            game=game,
            device=args.device,
            batch_size=args.batch_size,
            verbose=not args.quiet,
        )

        # Solve
        start_time = time.time()
        flop_strategy = solver.solve(
            iterations=args.iterations,
            verbose=not args.quiet,
        )
        elapsed = time.time() - start_time

        # Store strategy keyed by flop
        all_strategies[flop_str] = flop_strategy

        if not args.quiet:
            print(f"Time: {elapsed:.2f}s | Info sets: {len(flop_strategy):,}")
            print()

    total_elapsed = time.time() - total_start_time

    if not args.quiet:
        print("=" * 50)
        print("Per-Flop Solving Complete")
        print("=" * 50)
        print(f"Total flops: {total_flops}")
        print(f"Total time: {total_elapsed:.2f}s")
        total_info_sets = sum(len(s) for s in all_strategies.values())
        print(f"Total info sets: {total_info_sets:,}")
        print()

    return all_strategies


def run_hunl(args):
    """Run HUNL Preflop solver."""
    from config import load_config, get_preset_path
    from games.hunl_preflop import HUNLPreflop, make_board_config
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

    # Handle --flops mode (per-flop solving)
    if hasattr(args, 'flops') and args.flops is not None:
        try:
            all_flops = load_flops()
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if args.flops not in all_flops:
            available = ", ".join(str(k) for k in sorted(all_flops.keys()))
            print(f"Error: Flop category {args.flops} not found. Available: {available}", file=sys.stderr)
            sys.exit(1)

        flops = all_flops[args.flops]

        if not args.quiet:
            print("=" * 50)
            print("CFR Poker Solver - HUNL Per-Flop")
            print("=" * 50)
            print(f"Config: {config.name}")
            print(f"Stack depth: {config.stack_depths[0]} BB")
            print(f"Raise sizes: {config.raise_sizes}")
            print(f"Flop category: {args.flops} ({len(flops)} flops)")
            print(f"Iterations: {args.iterations}")
            print(f"Device: {args.device}")
            print(f"Batch size: {args.batch_size}")
            print()

        strategy = run_hunl_per_flop(args, config, flops)

        # Save strategy if requested
        if args.save:
            Solver.save_multi_stack_strategy(args.save, strategy)
            if not args.quiet:
                size_mb = os.path.getsize(args.save) / (1024 * 1024)
                print(f"Strategy saved to: {args.save} ({size_mb:.1f} MB)")
                print()

        # Skip interactive mode for per-flop solving (too many strategies to navigate)
        if not args.quiet:
            print("Per-flop solving complete. Use --load to explore individual flop strategies.")
        return

    # Determine if we're training preflop-only or full game
    preflop_only = not args.postflop

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
        # When loading a strategy, enable postflop (subgame solving handles it on-demand)
        preflop_only = False

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
            game_mode = "HUNL Full Game" if args.postflop else "HUNL Preflop"
            print(f"CFR Poker Solver - {game_mode}")
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

            # Create game and solver
            game = HUNLPreflop(single_stack_config, preflop_only=preflop_only)

            if can_share_compilation:
                if shared_solver is None:
                    # First stack: compile with terminal state storage
                    solver = Solver(
                        game=game,
                        device=args.device,
                        batch_size=args.batch_size,
                        verbose=not args.quiet,
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
                if preflop_only:
                    print(f"Time: {elapsed:.2f}s | Exploitability: {solver.exploitability():.6f}")
                else:
                    # Skip exploitability for postflop - tree too deep for recursive calculation
                    print(f"Time: {elapsed:.2f}s")
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
        # Parse board if provided
        board = None
        if hasattr(args, 'board') and args.board:
            board = parse_board(args.board)

        # Always enable postflop in interactive mode - subgame solving handles it on-demand
        run_interactive(config, strategy, initial_actions=initial_actions, preflop_only=False, board=board)
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
    parser.add_argument(
        "--postflop", action="store_true",
        help="Train full game through river (default: preflop only)"
    )
    parser.add_argument(
        "--board", type=str,
        help='Board cards for postflop (e.g., "AhKd2c" for flop, "AhKd2cJs" through turn)'
    )
    parser.add_argument(
        "--flops", type=int, metavar="CATEGORY",
        help='Solve for configured flops (25, 49, 85, or 184 from config/flops.yaml)'
    )
    parser.add_argument(
        "--terminal-street", type=str, choices=["flop", "turn", "river"], default="river",
        help='Street at which showdown occurs (default: river). Use "flop" for faster per-flop solving.'
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
