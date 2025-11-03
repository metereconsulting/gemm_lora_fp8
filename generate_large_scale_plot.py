#!/usr/bin/env python3
"""
Generate Large Scale Performance Plot for arXiv Paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_large_scale_plot(csv_file: str, output_file: str):
    """Create the large scale performance plot for the paper."""

    # Load data
    df = pd.read_csv(csv_file)

    # Clean data - remove failed runs
    df = df.dropna(subset=['time_ms'])

    # Create the main performance plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

    # Color scheme
    colors = {
        'LowRank_Auto': '#2E86C1',      # Blue
        'TorchCompile_FP16': '#28B463', # Green
        'cuBLAS_OptimizedFP8': '#E74C3C', # Red
        'LowRank_FP8': '#9B59B6',       # Purple
        'PyTorch_FP32': '#95A5A6'       # Gray
    }

    # Plot 1: Time vs Size (log scale)
    for method in df['method'].unique():
        method_data = df[df['method'] == method].sort_values('size')
        ax1.plot(method_data['size'], method_data['time_ms'],
                marker='o', linewidth=2, markersize=6,
                color=colors.get(method, 'black'),
                label=method.replace('_', ' '))

    ax1.set_xlabel('Matrix Size (N)')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Time to Solution vs Matrix Size')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    # Set custom x-ticks to show actual numbers
    ax1.set_xticks([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 20480])
    ax1.set_xticklabels(['128', '256', '512', '1024', '2048', '4096', '8192', '16384', '20480'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Throughput (GFLOPS)
    for method in df['method'].unique():
        method_data = df[df['method'] == method].sort_values('size')
        ax2.plot(method_data['size'], method_data['throughput_gflops'],
                marker='s', linewidth=2, markersize=6,
                color=colors.get(method, 'black'),
                label=method.replace('_', ' '))

    ax2.set_xlabel('Matrix Size (N)')
    ax2.set_ylabel('Throughput (GFLOPS)')
    ax2.set_title('Throughput vs Matrix Size')
    ax2.set_xscale('log', base=2)
    # Set custom x-ticks to show actual numbers
    ax2.set_xticks([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 20480])
    ax2.set_xticklabels(['128', '256', '512', '1024', '2048', '4096', '8192', '16384', '20480'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Relative Error
    for method in df['method'].unique():
        if method != 'PyTorch_FP32':  # Skip FP32 (no error)
            method_data = df[df['method'] == method].sort_values('size')
            ax3.plot(method_data['size'], method_data['mean_relative_error'],
                    marker='^', linewidth=2, markersize=6,
                    color=colors.get(method, 'black'),
                    label=method.replace('_', ' '))

    ax3.set_xlabel('Matrix Size (N)')
    ax3.set_ylabel('Relative Error (%)')
    ax3.set_title('Approximation Error vs Matrix Size')
    ax3.set_xscale('log', base=2)
    ax3.set_yscale('log')
    # Set custom x-ticks to show actual numbers
    ax3.set_xticks([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 20480])
    ax3.set_xticklabels(['128', '256', '512', '1024', '2048', '4096', '8192', '16384', '20480'], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Speedup vs PyTorch FP32
    fp32_data = df[df['method'] == 'PyTorch_FP32'].set_index('size')['time_ms']

    for method in df['method'].unique():
        if method != 'PyTorch_FP32':
            method_data = df[df['method'] == method].sort_values('size')
            speedup = []
            sizes = []
            for _, row in method_data.iterrows():
                if row['size'] in fp32_data.index:
                    speedup.append(fp32_data[row['size']] / row['time_ms'])
                    sizes.append(row['size'])

            ax4.plot(sizes, speedup, marker='d', linewidth=2, markersize=6,
                    color=colors.get(method, 'black'),
                    label=method.replace('_', ' '))

    ax4.set_xlabel('Matrix Size (N)')
    ax4.set_ylabel('Speedup vs PyTorch FP32')
    ax4.set_title('Speedup vs PyTorch FP32')
    ax4.set_xscale('log', base=2)
    # Set custom x-ticks to show actual numbers
    ax4.set_xticks([128, 256, 512, 1024, 2048, 4096, 8192, 16384, 20480])
    ax4.set_xticklabels(['128', '256', '512', '1024', '2048', '4096', '8192', '16384', '20480'], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Add more space for rotated x-axis labels
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Large scale performance plot saved to {output_file}")

    # Print key insights
    print("\n🎯 Key Insights from RTX 4090 Large Scale Benchmark:")
    print("=" * 60)

    # Find best method for each size
    size_stats = []
    for size in sorted(df['size'].unique()):
        size_data = df[df['size'] == size]
        if not size_data.empty:
            best_method = size_data.loc[size_data['time_ms'].idxmin(), 'method']
            best_time = size_data['time_ms'].min()
            best_gflops = size_data['throughput_gflops'].max()
            size_stats.append((size, best_method, best_time, best_gflops))

    for size, method, time_ms, gflops in size_stats:
        print("10")

    # Overall statistics
    print("\n🏆 Overall Performance Summary (N=1024 to 20480):")
    print("-" * 50)
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        if not method_data.empty:
            avg_time = method_data['time_ms'].mean()
            avg_gflops = method_data['throughput_gflops'].mean()
            print("15")

if __name__ == "__main__":
    create_large_scale_plot("complete_scaling_data.csv", "rtx4090_large_scale_performance.png")
