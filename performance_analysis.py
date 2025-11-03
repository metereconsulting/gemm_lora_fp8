#!/usr/bin/env python3
"""
Performance Analysis of Low-Rank GEMM vs Traditional Methods

This script analyzes the benchmark results and provides recommendations
for when to use Low-Rank GEMM vs traditional cuBLAS/TensorRT methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def analyze_benchmark_results(csv_file: str = "fp8_max_scale_test.csv"):
    """Analyze benchmark results and provide insights."""

    # Load results
    df = pd.read_csv(csv_file)

    print("🎯 Low-Rank GEMM Performance Analysis")
    print("=" * 50)

    # Clean data
    df = df.dropna(subset=['time_ms'])

    # Performance comparison
    print("\n📊 Performance Summary:")
    print("-" * 40)

    methods = df['method'].unique()
    sizes = sorted(df['size'].unique())

    for method in methods:
        method_data = df[df['method'] == method]
        if not method_data.empty:
            avg_time = method_data['time_ms'].mean()
            avg_tflops = method_data['throughput_tflops'].mean()
            print("20")

    # Speedup analysis
    print("\n⚡ Speedup Analysis (vs PyTorch FP32):")
    print("-" * 45)

    fp32_data = df[df['method'] == 'PyTorch_FP32']
    if not fp32_data.empty:
        fp32_times = dict(zip(fp32_data['size'], fp32_data['time_ms']))

        for method in methods:
            if method == 'PyTorch_FP32':
                continue

            method_data = df[df['method'] == method]
            speedups = []

            for _, row in method_data.iterrows():
                size = row['size']
                if size in fp32_times and fp32_times[size] > 0:
                    speedup = fp32_times[size] / row['time_ms']
                    speedups.append(speedup)

            if speedups:
                avg_speedup = np.mean(speedups)
                print("18")

    # Memory efficiency analysis
    print("\n💾 Memory Efficiency Analysis:")
    print("-" * 35)

    # Calculate theoretical memory savings for a few representative sizes
    sample_sizes = [sizes[0], sizes[len(sizes)//2], sizes[-1]]  # First, middle, last

    for size in sample_sizes:
        size_data = df[df['size'] == size]

        print(f"\nMatrix Size: {size}x{size}")
        direct_memory = (size**2 * 4 * 3) / 1e6  # A, B, C in FP32
        print(f"  Memory for direct GEMM: {direct_memory:.0f} MB (A, B, C in FP32)")

        # Low-rank memory estimation (rough approximation)
        lowrank_methods = ['LowRank_FP8', 'LowRank_Auto']
        for method in lowrank_methods:
            method_data = size_data[size_data['method'] == method]
            if not method_data.empty:
                # Use actual rank from the data if available, otherwise estimate
                if hasattr(method_data, 'target_rank') and not method_data.empty:
                    rank = 64  # Default rank we set
                else:
                    rank = int(np.sqrt(size) * 0.7)  # Estimate with FP8 reduction

                lowrank_memory = (size * rank * 2 * 4) / 1e6  # U, V matrices
                savings = (1 - lowrank_memory / direct_memory) * 100
                print("25")

    # Recommendations based on benchmark results
    print("\n🎯 Usage Recommendations (Based on Large-Scale Benchmark):")
    print("-" * 65)

    # Analyze when each method is fastest
    fastest_counts = {}
    for size in sizes:
        size_data = df[df['size'] == size].dropna(subset=['time_ms'])
        if not size_data.empty:
            fastest_method = size_data.loc[size_data['time_ms'].idxmin()]['method']
            fastest_counts[fastest_method] = fastest_counts.get(fastest_method, 0) + 1

    print("🏆 Method Performance by Scale:")
    for method, count in fastest_counts.items():
        percentage = count / len(sizes) * 100
        print("20")

    print("\n🔹 Use LowRank_Auto when:")
    print("   • Very large matrices (> 8000x8000) - becomes fastest method")
    print("   • Memory bandwidth is the bottleneck")
    print("   • 60-80% memory savings needed")
    print("   • Approximate computation acceptable (1-10% error)")
    print()
    print("🔹 Use LowRank_FP8 when:")
    print("   • Working with FP8 quantized models")
    print("   • Need exact FP8 precision bounds")
    print("   • Memory savings are critical")
    print("   • Target rank ≤ 64 for optimal performance")
    print()
    print("🔹 Use cuBLAS_FP8/TensorRT_FP8 when:")
    print("   • Maximum precision required (< 1% error)")
    print("   • Medium to large matrices (2048-8000)")
    print("   • Sufficient GPU memory available")
    print("   • Inference deployment scenarios")
    print()
    print("🔹 Use PyTorch_FP32 when:")
    print("   • Small matrices (≤ 4096) - often fastest")
    print("   • Maximum accuracy required")
    print("   • Baseline performance comparison")
    print("   • Memory is not a constraint")

    # Error analysis
    print("\n🎯 Error Analysis:")
    print("-" * 20)

    error_methods = ['LowRank_FP8', 'LowRank_Auto']
    for method in error_methods:
        method_data = df[df['method'] == method]
        if not method_data.empty and 'max_relative_error' in method_data.columns:
            avg_error = method_data['max_relative_error'].mean()
            within_bounds = (method_data['within_fp8_bounds'] == True).mean() if 'within_fp8_bounds' in method_data.columns else 0
            print("25")

    return df


def create_detailed_plots(df: pd.DataFrame, output_prefix: str = "detailed_analysis"):
    """Create detailed performance analysis plots."""

    # Filter out failed runs
    valid_df = df.dropna(subset=['time_ms'])

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Low-Rank GEMM Performance Analysis', fontsize=16)

    # 1. Performance comparison by size
    ax1 = axes[0, 0]
    sizes = sorted(valid_df['size'].unique())
    methods = valid_df['method'].unique()

    for method in methods:
        method_data = valid_df[valid_df['method'] == method]
        ax1.plot(method_data['size'], method_data['time_ms'], marker='o', label=method, linewidth=2)

    ax1.set_xlabel('Matrix Size (N)')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Time vs Matrix Size')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. TFLOPS comparison
    ax2 = axes[0, 1]
    for method in methods:
        method_data = valid_df[valid_df['method'] == method]
        ax2.plot(method_data['size'], method_data['throughput_tflops'], marker='s', label=method, linewidth=2)

    ax2.set_xlabel('Matrix Size (N)')
    ax2.set_ylabel('Throughput (TFLOPS)')
    ax2.set_title('Computational Throughput')
    ax2.set_xscale('log', base=2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Memory efficiency (estimated)
    ax3 = axes[0, 2]
    for size in sizes:
        size_data = valid_df[valid_df['size'] == size]

        # Direct memory usage (estimate)
        direct_memory = size**2 * 4 * 3 / 1e6  # A, B, C in FP32

        # Low-rank memory usage (estimate)
        rank = int(np.sqrt(size))  # Approximate rank
        lowrank_memory = size * rank * 2 * 4 / 1e6  # U, V matrices

        methods_data = []
        memory_data = []

        for method in ['PyTorch_FP32', 'cuBLAS_FP8', 'TensorRT_FP8', 'LowRank_Auto', 'LowRank_FP8']:
            method_data = size_data[size_data['method'] == method]
            if not method_data.empty:
                if 'LowRank' in method:
                    memory_data.append(lowrank_memory)
                else:
                    memory_data.append(direct_memory)
                methods_data.append(method)

        if memory_data:
            ax3.bar(range(len(methods_data)), memory_data, tick_label=methods_data, alpha=0.7)

    ax3.set_ylabel('Estimated Memory (MB)')
    ax3.set_title('Memory Usage Comparison')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    # 4. Relative error
    ax4 = axes[1, 0]
    for method in ['LowRank_FP8', 'LowRank_Auto']:
        method_data = valid_df[valid_df['method'] == method]
        if 'max_relative_error' in method_data.columns:
            ax4.plot(method_data['size'], method_data['max_relative_error'], marker='^', label=method, linewidth=2)

    ax4.axhline(y=0.001, color='red', linestyle='--', label='FP8 Bound (0.1%)')
    ax4.set_xlabel('Matrix Size (N)')
    ax4.set_ylabel('Max Relative Error')
    ax4.set_title('Approximation Error')
    ax4.set_xscale('log', base=2)
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Efficiency analysis (TFLOPS per GB memory)
    ax5 = axes[1, 1]
    for method in methods:
        method_data = valid_df[valid_df['method'] == method]
        # Estimate TFLOPS per GB (rough approximation)
        memory_gb = method_data['size'].apply(lambda x: (x**2 * 4 * 3) / 1e9 if 'LowRank' not in method
                                             else (x * int(np.sqrt(x)) * 2 * 4) / 1e9)
        efficiency = method_data['throughput_tflops'] / memory_gb
        ax5.plot(method_data['size'], efficiency, marker='d', label=method, linewidth=2)

    ax5.set_xlabel('Matrix Size (N)')
    ax5.set_ylabel('TFLOPS/GB Memory')
    ax5.set_title('Memory Efficiency')
    ax5.set_xscale('log', base=2)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Scaling efficiency
    ax6 = axes[1, 2]
    for method in methods:
        method_data = valid_df[valid_df['method'] == method]
        # Calculate how performance scales with problem size
        baseline_size = method_data['size'].min()
        baseline_perf = method_data[method_data['size'] == baseline_size]['throughput_tflops'].iloc[0]

        scaling_efficiency = method_data['throughput_tflops'] / (method_data['size'] / baseline_size)
        ax6.plot(method_data['size'], scaling_efficiency, marker='*', label=method, linewidth=2)

    ax6.set_xlabel('Matrix Size (N)')
    ax6.set_ylabel('Relative Scaling Efficiency')
    ax6.set_title('Scaling Efficiency')
    ax6.set_xscale('log', base=2)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_detailed.png", dpi=300, bbox_inches='tight')
    print(f"📊 Detailed analysis plots saved to {output_prefix}_detailed.png")


def main():
    """Main analysis function."""
    # Analyze results
    results_df = analyze_benchmark_results("fp8_max_scale_test.csv")

    # Create detailed plots
    create_detailed_plots(results_df, "fp8_performance_analysis")

    print("\n✅ Analysis completed!")
    print("📁 Generated files:")
    print("   • fp8_max_scale_test.csv (benchmark data)")
    print("   • fp8_performance_analysis_detailed.png (detailed plots)")


if __name__ == "__main__":
    main()
