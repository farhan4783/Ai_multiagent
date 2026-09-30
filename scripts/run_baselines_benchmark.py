"""
CLI Runner for RCPSP Baseline Benchmarks.

Runs OR-Tools exact solver and classical dispatching heuristics (CPM, SPT, LPT, MSF, Random)
across benchmark project instances in the PettingZoo ProjectSchedulingEnv.
"""

import argparse
import sys

from project_marl.baselines.evaluator import BenchmarkSuite
from project_marl.core.logging import setup_logger

logger = setup_logger("benchmark_cli")


def main() -> None:
    parser = argparse.ArgumentParser(description="STGAT-MARL Phase 3 Baselines Benchmark Suite")
    parser.add_argument("--episodes", type=int, default=5, help="Number of benchmark episodes")
    parser.add_argument("--developers", type=int, default=3, help="Number of developer agents")
    parser.add_argument("--max-tasks", type=int, default=12, help="Max tasks per project")
    parser.add_argument("--no-exact", action="store_true", help="Skip OR-Tools CP-SAT exact solve")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("      STGAT-MARL Phase 3: Classical Baseline Benchmark Suite      ")
    print("=" * 70)
    print(f"Episodes:    {args.episodes}")
    print(f"Developers:  {args.developers}")
    print(f"Max Tasks:   {args.max_tasks}")
    print(f"Seed:        {args.seed}")
    print(f"Exact Solve: {not args.no_exact}")
    print("-" * 70)

    suite = BenchmarkSuite(
        num_episodes=args.episodes,
        num_developers=args.developers,
        max_tasks=args.max_tasks,
        seed=args.seed,
    )

    results = suite.run_all(solve_exact=not args.no_exact)
    markdown_table = suite.format_table(results)

    print("\n" + "=" * 70)
    print("                        BENCHMARK RESULTS                        ")
    print("=" * 70)
    print(markdown_table)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
