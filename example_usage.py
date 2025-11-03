"""
Example usage of the Low-Rank GEMM module with FP8 and TensorRT support.

This script demonstrates how to use the LowRankGEMM and AdaptiveLowRankGEMM
modules for efficient matrix multiplication with low-rank approximations,
FP8 precision, and TensorRT optimization.
"""

import torch
import time
from low_rank_gemm import LowRankGEMM, AdaptiveLowRankGEMM


def basic_usage_example():
    """Basic usage example with fixed rank."""
    print("Basic Low-Rank GEMM Example")
    print("=" * 30)

    # Create test matrices
    m, k, n = 1000, 800, 1200
    a = torch.randn(m, k)
    b = torch.randn(k, n)

    print(f"Matrix A: {a.shape}")
    print(f"Matrix B: {b.shape}")

    # Exact computation
    start_time = time.time()
    exact_result = torch.matmul(a, b)
    exact_time = time.time() - start_time
    print(".4f")

    # Low-rank approximation with different ranks
    ranks = [50, 100, 200]

    for rank in ranks:
        low_rank_gemm = LowRankGEMM(target_rank=rank)

        start_time = time.time()
        approx_result = low_rank_gemm(a, b)
        approx_time = time.time() - start_time

        error = LowRankGEMM.compute_error(exact_result, approx_result)
        speedup = exact_time / approx_time

        print(f"Rank {rank:3d}: Time={approx_time:.4f}s, Error={error:.6f}, Speedup={speedup:.2f}x")


def adaptive_example():
    """Example using adaptive rank selection based on error tolerance."""
    print("\nAdaptive Low-Rank GEMM Example")
    print("=" * 35)

    # Create test matrices
    size = 500
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    print(f"Matrix size: {size}x{size}")

    # Test different error tolerances
    tolerances = [0.1, 0.01, 0.001]

    for tol in tolerances:
        adaptive_gemm = AdaptiveLowRankGEMM(error_tolerance=tol)

        start_time = time.time()
        result, final_rank = adaptive_gemm(a, b)
        adaptive_time = time.time() - start_time

        # Compute exact result for comparison
        exact_result = torch.matmul(a, b)
        error = LowRankGEMM.compute_error(exact_result, result)

        print(".1e")


def batch_processing_example():
    """Example of batch processing multiple matrix multiplications."""
    print("\nBatch Processing Example")
    print("=" * 25)

    # Create batch of matrices
    batch_size = 10
    m, k, n = 200, 150, 250

    a_batch = torch.randn(batch_size, m, k)
    b_batch = torch.randn(batch_size, k, n)

    print(f"Batch size: {batch_size}")
    print(f"Matrix A batch: {a_batch.shape}")
    print(f"Matrix B batch: {b_batch.shape}")

    # Low-rank GEMM on batch
    low_rank_gemm = LowRankGEMM(target_rank=50)

    start_time = time.time()
    batch_result = low_rank_gemm(a_batch, b_batch)
    batch_time = time.time() - start_time

    print(f"Batch processing time: {batch_time:.4f}s")
    print(f"Result shape: {batch_result.shape}")

    # Compare with exact batch computation
    start_time = time.time()
    exact_batch_result = torch.matmul(a_batch, b_batch)
    exact_batch_time = time.time() - start_time

    print(".4f")
    print(".2f")


def decomposition_method_comparison():
    """Compare different decomposition methods."""
    print("\nDecomposition Method Comparison")
    print("=" * 35)

    size = 800
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    rank = 100

    methods = ['svd', 'randomized_svd']

    for method in methods:
        low_rank_gemm = LowRankGEMM(
            target_rank=rank,
            decomposition_method=method
        )

        start_time = time.time()
        result = low_rank_gemm(a, b)
        comp_time = time.time() - start_time

        # Compute error vs exact
        exact_result = torch.matmul(a, b)
        error = LowRankGEMM.compute_error(exact_result, result)

        print(f"{method:15s}: Time={comp_time:.4f}s, Error={error:.6f}")


def memory_efficiency_demo():
    """Demonstrate memory efficiency for large matrices."""
    print("\nMemory Efficiency Demo")
    print("=" * 23)

    # Create a large matrix
    size = 2000
    rank = 200

    print(f"Creating {size}x{size} matrices (rank {rank} approximation)")

    # Measure memory usage (approximate)
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    # Low-rank approximation memory
    low_rank_gemm = LowRankGEMM(target_rank=rank, decomposition_method='randomized_svd')

    # Store the decomposed matrices
    u_a, s_a, v_a = low_rank_gemm._approximate_matrix(a)
    u_b, s_b, v_b = low_rank_gemm._approximate_matrix(b)

    # Calculate memory requirements
    full_memory = a.numel() + b.numel()  # Original matrices
    low_rank_memory = (u_a.numel() + s_a.numel() + v_a.numel() +
                      u_b.numel() + s_b.numel() + v_b.numel())  # Decomposed matrices

    compression_ratio = full_memory / low_rank_memory

    print(f"Full matrices memory: {full_memory} elements")
    print(f"Low-rank memory: {low_rank_memory} elements")
    print(".2f")

    # Time comparison
    start_time = time.time()
    exact_result = torch.matmul(a, b)
    exact_time = time.time() - start_time

    start_time = time.time()
    approx_result = low_rank_gemm(a, b)
    approx_time = time.time() - start_time

    error = LowRankGEMM.compute_error(exact_result, approx_result)

    print(".4f")
    print(".6f")


def fp8_precision_example():
    """Example using FP8 precision for memory efficiency."""
    print("\nFP8 Precision Example")
    print("=" * 22)

    # Create large matrices where FP8 benefits are significant
    size = 1000
    a = torch.randn(size, size // 2)
    b = torch.randn(size // 2, size)

    print(f"Matrix sizes: {a.shape} @ {b.shape}")

    # Regular precision
    gemm_fp32 = LowRankGEMM(target_rank=50, use_fp8=False, auto_kernel=False)
    start_time = time.time()
    result_fp32 = gemm_fp32(a, b)
    fp32_time = time.time() - start_time

    # FP8 precision with fallback
    gemm_fp8 = LowRankGEMM(target_rank=50, use_fp8=True, auto_kernel=False)
    start_time = time.time()
    result_fp8 = gemm_fp8(a, b)
    fp8_time = time.time() - start_time

    # Compare results
    error = LowRankGEMM.compute_error(result_fp32, result_fp8)

    print(".4f")
    print(".4f")
    print(".6f")
    print(".2f")


def auto_kernel_example():
    """Example demonstrating automatic kernel selection."""
    print("\nAuto-Kernel Selection Example")
    print("=" * 29)

    # Test different matrix sizes to trigger different kernel decisions
    test_cases = [
        (100, 80, 120, "Small matrices"),
        (1000, 800, 1200, "Large matrices"),
        (2000, 1500, 2500, "Very large matrices")
    ]

    for m, k, n, desc in test_cases:
        print(f"\n{desc}: {m}x{k} @ {k}x{n}")

        a = torch.randn(m, k)
        b = torch.randn(k, n)

        # Auto kernel selection
        gemm_auto = LowRankGEMM(auto_kernel=True)
        start_time = time.time()
        result = gemm_auto(a, b)
        auto_time = time.time() - start_time

        print(".4f")
        print(f"  Result shape: {result.shape}")


def tensorrt_optimization_example():
    """Example using TensorRT optimization (with torch.compile fallback)."""
    print("\nTensorRT Optimization Example")
    print("=" * 30)

    # Create matrices suitable for TensorRT optimization
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)

    print("Matrix size: 512x512 (suitable for TensorRT)")

    # Regular computation
    gemm_regular = LowRankGEMM(target_rank=64, use_tensorrt=False, auto_kernel=False)
    start_time = time.time()
    result_regular = gemm_regular(a, b)
    regular_time = time.time() - start_time

    # TensorRT optimized (falls back to torch.compile)
    try:
        gemm_trt = LowRankGEMM(target_rank=64, use_tensorrt=True, auto_kernel=False)
        start_time = time.time()
        result_trt = gemm_trt(a, b)
        trt_time = time.time() - start_time

        error = LowRankGEMM.compute_error(result_regular, result_trt)

        print(".4f")
        print(".4f")
        print(".6f")
        print(".2f")
    except Exception as e:
        print(f"  TensorRT not available: {e}")
        print("  Install torch-tensorrt for full TensorRT support")


def comprehensive_performance_comparison():
    """Comprehensive comparison of all optimization features."""
    print("\nComprehensive Performance Comparison")
    print("=" * 39)

    # Test matrix
    a = torch.randn(800, 600)
    b = torch.randn(600, 1000)

    configurations = [
        ("FP32 Baseline", LowRankGEMM(target_rank=50, use_fp8=False, auto_kernel=False)),
        ("FP8 Optimized", LowRankGEMM(target_rank=50, use_fp8=True, auto_kernel=False)),
        ("Auto Kernel", LowRankGEMM(target_rank=50, auto_kernel=True)),
        ("SVD Method", LowRankGEMM(target_rank=50, decomposition_method='svd', auto_kernel=False)),
        ("Randomized SVD", LowRankGEMM(target_rank=50, decomposition_method='randomized_svd', auto_kernel=False)),
    ]

    results = []
    baseline_time = None

    for name, gemm in configurations:
        try:
            start_time = time.time()
            result = gemm(a, b)
            comp_time = time.time() - start_time

            if baseline_time is None:
                baseline_time = comp_time

            speedup = baseline_time / comp_time if comp_time > 0 else 1.0
            results.append((name, comp_time, speedup, result.shape))

        except Exception as e:
            results.append((name, float('inf'), 0.0, f"Error: {e}"))

    # Print results table
    print("Configuration          | Time (s) | Speedup | Status")
    print("-" * 55)
    for name, comp_time, speedup, status in results:
        if comp_time == float('inf'):
            print("25")
        else:
            print("25")


if __name__ == "__main__":
    # Set random seed for reproducible results
    torch.manual_seed(42)

    # Run all examples
    basic_usage_example()
    adaptive_example()
    batch_processing_example()
    decomposition_method_comparison()
    memory_efficiency_demo()
    fp8_precision_example()
    auto_kernel_example()
    tensorrt_optimization_example()
    comprehensive_performance_comparison()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("Features demonstrated:")
    print("  ✓ Low-rank matrix approximations")
    print("  ✓ FP8 precision support with fallback")
    print("  ✓ Auto-kernel selection based on hardware/tensors")
    print("  ✓ TensorRT optimization (torch.compile fallback)")
    print("  ✓ Multiple decomposition methods (SVD, Randomized SVD)")
    print("  ✓ Batch processing support")
    print("  ✓ Adaptive rank selection")
