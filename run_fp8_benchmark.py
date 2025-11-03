#!/usr/bin/env python3
"""
Run FP8 GEMM Benchmark Suite

This script runs comprehensive benchmarks comparing Low-Rank GEMM with
cuBLAS FP8 and TensorRT FP8 implementations, scaling to GPU memory limits.
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_fp8 import LowRankGEMMBenchmarkSuite


def main():
    parser = argparse.ArgumentParser(description='Low-Rank GEMM Performance Benchmark Suite')
    parser.add_argument('--max-size', type=int, default=None,
                       help='Maximum matrix size to test (default: auto-detect from GPU memory)')
    parser.add_argument('--sizes', type=int, nargs='+', default=None,
                       help='Specific sizes to test (overrides max-size)')
    parser.add_argument('--output', type=str, default='fp8_benchmark_results',
                       help='Output filename prefix (default: fp8_benchmark_results)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with small matrices only')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    print("🚀 Low-Rank GEMM Performance Benchmark Suite")
    print("=" * 60)

    # Create benchmark suite
    benchmark = LowRankGEMMBenchmarkSuite()

    # Determine test sizes
    if args.quick:
        sizes = [256, 512, 1024]
        print("⚡ Running quick benchmark with small matrices")
    elif args.sizes:
        sizes = args.sizes
        print(f"📏 Testing specific sizes: {sizes}")
    else:
        max_size = args.max_size or benchmark.memory_manager.get_max_matrix_size()
        # Generate size progression up to memory limit
        sizes = []
        current = 256
        while current <= max_size:
            sizes.append(current)
            if current < 1024:
                current *= 2  # Double for small sizes
            else:
                current = int(current * 1.5)  # 1.5x for larger sizes
        sizes = list(set(sizes))  # Remove duplicates
        sizes.sort()
        print(f"🔬 Testing sizes up to GPU memory limit: {sizes}")

    # Run benchmark
    print(f"\n🏃 Running benchmark with {len(sizes)} matrix sizes...")
    results_df = benchmark.run_scaling_benchmark(sizes=sizes)

    if results_df.empty:
        print("❌ Benchmark failed - no results generated")
        return 1

    # Save results
    csv_file = f"{args.output}.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"💾 Results saved to {csv_file}")

    # Generate plots
    if not args.no_plots:
        plot_file = f"{args.output}.png"
        benchmark.create_performance_plots(results_df, plot_file)
        print(f"📊 Performance plots saved to {plot_file}")

    # Print key insights
    print("\n🎯 Key Insights:")
    print("-" * 30)

    # Find best method for different size ranges
    small_results = results_df[results_df['size'] <= 1024]
    large_results = results_df[results_df['size'] > 1024]

    if not small_results.empty:
        best_small = small_results.loc[small_results['time_ms'].idxmin()]
        print(f"Small matrices (≤1024): {best_small['method']} "
              f"({best_small['time_ms']:.2f}ms avg)")

    if not large_results.empty:
        best_large = large_results.loc[large_results['time_ms'].idxmin()]
        print(f"Large matrices (>1024): {best_large['method']} "
              f"({best_large['time_ms']:.2f}ms avg)")

    # Check FP8 error bounds
    lowrank_results = results_df[results_df['method'].str.contains('LowRank')]
    if not lowrank_results.empty and 'within_fp8_bounds' in lowrank_results.columns:
        within_bounds = lowrank_results['within_fp8_bounds'].mean()
        print(".1%")

    print("\n✅ Benchmark completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
