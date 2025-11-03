"""
Comprehensive Low-Rank GEMM Benchmark Suite

Compares Low-Rank GEMM performance against actual hardware-accelerated methods:
- cuBLAS FP16 (TensorCore accelerated)
- TorchCompile FP16 (compiled optimization)
- PyTorch FP32 (baseline)

Note: True FP8 GEMM is not supported in current PyTorch, so we compare against
the best available hardware-accelerated alternatives.

Features:
- Runtime GPU memory capacity measurement
- Hardware-accelerated FP16 GEMM implementations
- Torch.compile optimized GEMM
- N vs Time-to-solution plots (log2 scale)
- Error bound verification
- Comprehensive performance analysis
"""

import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import gc
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from contextlib import contextmanager
import warnings

# Optional imports
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None

try:
    import torch_tensorrt
    TORCH_TENSORRT_AVAILABLE = True
except ImportError:
    TORCH_TENSORRT_AVAILABLE = False
    torch_tensorrt = None

# Import our low-rank GEMM
from low_rank_gemm import LowRankGEMM

# FP8 constants and error bounds
FP8_E4M3_MAX = 448.0  # Maximum value for FP8 E4M3
FP8_E5M2_MAX = 57344.0  # Maximum value for FP8 E5M2
FP8_PRECISION_LOSS_FACTOR = 1e-3  # Expected relative precision loss in FP8

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class GPUMemoryManager:
    """Manages GPU memory for large matrix operations."""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_cuda = torch.cuda.is_available()
        self.memory_info = self._get_memory_info()

    def _get_memory_info(self) -> Dict[str, float]:
        """Get GPU memory information."""
        if not self.is_cuda:
            return {"total": 8e9, "free": 8e9, "used": 0}  # 8GB simulation

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        allocated_memory = torch.cuda.memory_allocated(0)

        # Estimate free memory (conservative estimate)
        free_memory = total_memory - reserved_memory - allocated_memory
        free_memory = max(free_memory * 0.8, 0)  # Use 80% to be safe

        return {
            "total": total_memory,
            "free": free_memory,
            "used": allocated_memory
        }

    def get_max_matrix_size(self, dtype: torch.dtype = torch.float32) -> int:
        """Calculate maximum square matrix size that fits in GPU memory."""
        if not self.is_cuda:
            return 4096  # Conservative CPU limit

        bytes_per_element = torch.tensor([], dtype=dtype).element_size()
        # For GEMM: A(m,k) @ B(k,n) = C(m,n), with k≈m≈n for square matrices
        # Memory needed: ~3 * size^2 * bytes_per_element (A, B, C)
        # Plus workspace memory

        max_elements = self.memory_info["free"] // (bytes_per_element * 4)  # Conservative factor
        max_size = int(np.sqrt(max_elements))

        # Round down to nearest power of 2 for cleaner benchmarking
        max_size = 2 ** int(np.log2(max_size))

        return min(max_size, 16384)  # Cap at 16K for practicality


class CublasOptimizedFP8GEMM:
    """Optimized cuBLAS FP8 GEMM using advanced PyTorch operations.

    This implementation uses the most efficient available PyTorch operations
    to achieve FP8-like performance, potentially beating regular cuBLAS.
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Use tighter scaling for better precision vs performance trade-off
        self.fp8_scale = 224.0  # Half of max FP8 range for better accuracy

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Perform optimized FP8-like GEMM operations."""
        # Method 1: Try PyTorch's _scaled_mm if available (most efficient)
        if hasattr(torch, '_scaled_mm'):
            try:
                # Use _scaled_mm with automatic scaling
                a_fp16 = a.half()
                b_fp16 = b.half()
                scale_a = torch.tensor(1.0, device=self.device, dtype=torch.float32)
                scale_b = torch.tensor(1.0, device=self.device, dtype=torch.float32)

                result = torch._scaled_mm(a_fp16, b_fp16, scale_a, scale_b)
                return result.float()
            except:
                pass  # Fall back to optimized scaling

        # Method 2: Optimized scaling with TensorCores
        # Use more efficient scaling approach
        with torch.cuda.amp.autocast(dtype=torch.float16):
            # Scale and convert in one operation for better memory access
            a_scaled = torch.div(a, self.fp8_scale, out=torch.empty_like(a, dtype=torch.float16))
            b_scaled = torch.div(b, self.fp8_scale, out=torch.empty_like(b, dtype=torch.float16))

            # FP16 GEMM with TensorCores
            c_fp16 = torch.matmul(a_scaled, b_scaled)

            # Efficient scaling back (avoid large intermediate values)
            scale_sq = self.fp8_scale * self.fp8_scale
            c_scaled = torch.mul(c_fp16.float(), scale_sq)

            return c_scaled


class TorchCompileFP16GEMM:
    """Torch.compile FP16 GEMM implementation for comparison (simulates TensorRT-like optimization)."""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.compiled_fn = None

    def _ensure_compiled(self):
        """Ensure the GEMM function is compiled."""
        if self.compiled_fn is None:
            @torch.compile(mode='reduce-overhead')
            def fp16_gemm(a, b):
                # Convert to FP16 and perform GEMM
                a_fp16 = a.half()
                b_fp16 = b.half()
                c_fp16 = torch.matmul(a_fp16, b_fp16)
                return c_fp16.float()  # Convert back to FP32

            self.compiled_fn = fp16_gemm

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Perform FP16 GEMM using torch.compile (simulates TensorRT optimization)."""
        self._ensure_compiled()
        return self.compiled_fn(a, b)


class LowRankGEMMBenchmarkSuite:
    """Comprehensive Low-Rank GEMM benchmark suite comparing against hardware-accelerated baselines."""

    def __init__(self):
        self.memory_manager = GPUMemoryManager()
        self.device = self.memory_manager.device
        self.is_cuda = self.memory_manager.is_cuda

        # Initialize GEMM implementations
        self.methods = {
            'LowRank_FP8': LowRankGEMM(use_fp8=True, auto_kernel=False),
            'LowRank_Auto': LowRankGEMM(auto_kernel=True),
            'cuBLAS_OptimizedFP8': CublasOptimizedFP8GEMM(),  # Optimized FP8-like operations
            'TorchCompile_FP16': TorchCompileFP16GEMM(),  # torch.compile optimized FP16
            'PyTorch_FP32': lambda a, b: torch.matmul(a, b),
        }

        # Benchmark results
        self.results = {}

        # Optional verbose initialization (can be disabled)
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
            memory_gb = self.memory_manager.memory_info['total'] / 1e9
            print(f"Benchmark Suite initialized on {device_name} ({memory_gb:.1f} GB GPU)")

    @contextmanager
    def memory_context(self):
        """Context manager for memory management."""
        if self.is_cuda:
            torch.cuda.empty_cache()
        try:
            yield
        finally:
            if self.is_cuda:
                torch.cuda.empty_cache()

    def benchmark_single_operation(self, method_name: str, a: torch.Tensor, b: torch.Tensor, num_runs: int = 5) -> Dict[str, float]:
        """Benchmark a single GEMM operation."""
        method = self.methods[method_name]

        # Warmup
        with self.memory_context():
            for _ in range(2):
                _ = method(a, b)
                if self.is_cuda:
                    torch.cuda.synchronize()

        # Benchmark
        times = []
        with self.memory_context():
            for _ in range(num_runs):
                start_time = time.time()
                c = method(a, b)
                if self.is_cuda:
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)

        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'throughput': (2 * a.numel() * b.shape[-1]) / np.mean(times),  # GFLOPS approximation
        }

    def verify_fp8_error_bounds(self, reference: torch.Tensor, result: torch.Tensor, method_name: str) -> Dict[str, float]:
        """Verify that errors stay within FP8 bounds."""
        diff = torch.abs(reference - result)
        relative_error = diff / (torch.abs(reference) + 1e-8)

        # FP8 expected error bounds
        fp8_relative_error_bound = FP8_PRECISION_LOSS_FACTOR
        fp8_absolute_error_bound = FP8_E4M3_MAX * 2**(-3)  # ~3-4 bits precision

        return {
            'max_absolute_error': diff.max().item(),
            'max_relative_error': relative_error.max().item(),
            'mean_absolute_error': diff.mean().item(),
            'mean_relative_error': relative_error.mean().item(),
            'within_fp8_bounds': relative_error.max().item() <= fp8_relative_error_bound,
            'frobenius_error': torch.norm(diff, p='fro').item(),
            'frobenius_relative_error': torch.norm(relative_error, p='fro').item(),
        }

    def run_scaling_benchmark(self, max_size: Optional[int] = None, sizes: Optional[List[int]] = None, verbose: bool = False) -> pd.DataFrame:
        """Run scaling benchmark with different matrix sizes."""
        if max_size is None:
            max_size = self.memory_manager.get_max_matrix_size()

        if sizes is None:
            # Generate size range
            min_size = 256
            sizes = []
            current = min_size
            while current <= max_size:
                sizes.append(current)
                current = int(current * 1.5)  # Geometric progression
            sizes = list(set(sizes))  # Remove duplicates
            sizes.sort()

        if verbose:
            print(f"Running benchmark with {len(sizes)} matrix sizes...")

        results = []

        for size in sizes:
            if verbose:
                print(f"Testing {size}x{size}...")
            try:
                # Create test matrices
                a = torch.randn(size, size, device=self.device, dtype=torch.float32)
                b = torch.randn(size, size, device=self.device, dtype=torch.float32)

                # Compute reference (FP32)
                with self.memory_context():
                    reference = torch.matmul(a, b)

                # Benchmark each method
                for method_name in self.methods.keys():
                    try:
                        perf_results = self.benchmark_single_operation(method_name, a, b)
                        error_results = self.verify_fp8_error_bounds(reference, self.methods[method_name](a, b), method_name)

                        result = {
                            'size': size,
                            'method': method_name,
                            'time_ms': perf_results['mean_time'] * 1000,
                            'time_std_ms': perf_results['std_time'] * 1000,
                            'throughput_gflops': perf_results['throughput'] / 1e9,
                            **error_results
                        }
                        results.append(result)

                    except Exception as e:
                        print(f"      {method_name}: Failed - {e}")
                        results.append({
                            'size': size,
                            'method': method_name,
                            'time_ms': np.nan,
                            'error': str(e)
                        })

                # Memory cleanup
                del a, b, reference
                if self.is_cuda:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"   Size {size}: Failed - {e}")
                continue

        return pd.DataFrame(results)

    def create_performance_plots(self, results_df: pd.DataFrame, save_path: str = "fp8_benchmark_results.png"):
        """Create performance plots."""
        # Filter out failed runs
        valid_results = results_df.dropna(subset=['time_ms'])

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('FP8 GEMM Performance Benchmark', fontsize=16)

        # 1. Time vs Size (log-log)
        ax1 = axes[0, 0]
        for method in valid_results['method'].unique():
            method_data = valid_results[valid_results['method'] == method]
            ax1.plot(method_data['size'], method_data['time_ms'],
                    marker='o', label=method, linewidth=2, markersize=4)
        ax1.set_xlabel('Matrix Size (N)')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Time to Solution vs Matrix Size')
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Throughput vs Size
        ax2 = axes[0, 1]
        for method in valid_results['method'].unique():
            method_data = valid_results[valid_results['method'] == method]
            ax2.plot(method_data['size'], method_data['throughput_gflops'],
                    marker='s', label=method, linewidth=2, markersize=4)
        ax2.set_xlabel('Matrix Size (N)')
        ax2.set_ylabel('Throughput (GFLOPS)')
        ax2.set_title('Throughput vs Matrix Size')
        ax2.set_xscale('log', base=2)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Relative Error vs Size
        ax3 = axes[1, 0]
        for method in valid_results['method'].unique():
            method_data = valid_results[valid_results['method'] == method]
            if 'max_relative_error' in method_data.columns:
                ax3.plot(method_data['size'], method_data['max_relative_error'],
                        marker='^', label=method, linewidth=2, markersize=4)
        ax3.axhline(y=FP8_PRECISION_LOSS_FACTOR, color='red', linestyle='--',
                   label=f'FP8 Bound ({FP8_PRECISION_LOSS_FACTOR})')
        ax3.set_xlabel('Matrix Size (N)')
        ax3.set_ylabel('Max Relative Error')
        ax3.set_title('Error Bounds vs Matrix Size')
        ax3.set_xscale('log', base=2)
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Speedup relative to PyTorch FP32
        ax4 = axes[1, 1]
        fp32_data = valid_results[valid_results['method'] == 'PyTorch_FP32']
        if not fp32_data.empty:
            fp32_times = dict(zip(fp32_data['size'], fp32_data['time_ms']))

            for method in valid_results['method'].unique():
                if method == 'PyTorch_FP32':
                    continue
                method_data = valid_results[valid_results['method'] == method]
                speedups = []
                sizes = []
                for _, row in method_data.iterrows():
                    size = row['size']
                    if size in fp32_times and fp32_times[size] > 0:
                        speedup = fp32_times[size] / row['time_ms']
                        speedups.append(speedup)
                        sizes.append(size)

                if speedups:
                    ax4.plot(sizes, speedups, marker='d', label=method,
                            linewidth=2, markersize=4)

        ax4.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, label='Baseline')
        ax4.set_xlabel('Matrix Size (N)')
        ax4.set_ylabel('Speedup vs PyTorch FP32')
        ax4.set_title('Speedup Analysis')
        ax4.set_xscale('log', base=2)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Performance plots saved to {save_path}")

        # Print summary statistics
        self._print_summary_statistics(valid_results)

    def _print_summary_statistics(self, results_df: pd.DataFrame):
        """Print summary statistics."""
        print("\n📈 Benchmark Summary:")
        print("=" * 50)

        # Best method for each size
        sizes = sorted(results_df['size'].unique())
        for size in sizes:
            size_data = results_df[results_df['size'] == size].dropna(subset=['time_ms'])
            if not size_data.empty:
                fastest = size_data.loc[size_data['time_ms'].idxmin()]
                print(f"Size {size:4d}: Fastest = {fastest['method']:<15s} "
                      f"({fastest['time_ms']:6.2f} ms)")

        # Overall statistics
        print("\n🏆 Overall Statistics:")
        for method in results_df['method'].unique():
            method_data = results_df[results_df['method'] == method].dropna(subset=['time_ms'])
            if not method_data.empty:
                avg_time = method_data['time_ms'].mean()
                avg_throughput = method_data['throughput_gflops'].mean()
                print(f"  {method:<15s}: {avg_time:6.2f} ms avg, "
                      f"{avg_throughput:6.1f} GFLOPS avg")

    def run_full_benchmark(self, save_plots: bool = True, verbose: bool = False) -> pd.DataFrame:
        """Run complete benchmark suite."""
        if verbose:
            print("Starting Low-Rank GEMM Benchmark Suite")

        # Run scaling benchmark
        results_df = self.run_scaling_benchmark(verbose=verbose)

        if save_plots and not results_df.empty:
            self.create_performance_plots(results_df)

        if verbose:
            print("Benchmark completed successfully")
        return results_df


def main():
    """Main benchmark function."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Enable TF32 for fair comparison
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Parse command line arguments for verbose mode
    import sys
    verbose = '--verbose' in sys.argv

    # Create and run benchmark
    benchmark = LowRankGEMMBenchmarkSuite()
    results = benchmark.run_full_benchmark(verbose=verbose)

    # Save results to CSV
    results.to_csv('fp8_benchmark_results.csv', index=False)
    if verbose:
        print("Results saved to fp8_benchmark_results.csv")

    return results


if __name__ == "__main__":
    results = main()
