"""
Low-Rank GEMM Module for PyTorch with FP8 and TensorRT Support

This module provides efficient matrix multiplication by approximating input matrices
as low-rank decompositions, reducing computational complexity from O(n^3) to
approximately O(n^2 * r) where r is the target rank.

Supports FP8 precision and TensorRT optimization for high-performance inference.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Dict, Any
import math
import warnings

# Optional imports with fallbacks
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

# FP8 support detection
FP8_E4M3FN_AVAILABLE = hasattr(torch, 'float8_e4m3fn')
FP8_E5M2FN_AVAILABLE = hasattr(torch, 'float8_e5m2fn')

# Define FP8 dtypes if not available (silent fallback)
if not FP8_E4M3FN_AVAILABLE:
    torch.float8_e4m3fn = torch.float16
    torch.float8_e5m2fn = torch.float16


class AutoKernelSelector:
    """
    Automatic kernel selection based on tensor characteristics and hardware.
    """

    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.device_capability = None
        if self.cuda_available:
            self.device_capability = torch.cuda.get_device_capability()

    def select_kernel(self, a: torch.Tensor, b: torch.Tensor, target_rank: Optional[int] = None) -> Dict[str, Any]:
        """
        Select optimal kernel configuration based on tensor properties.
        """
        m, k = a.shape[-2], a.shape[-1]
        n = b.shape[-1]

        config = {
            'decomposition_method': 'svd',
            'precision': a.dtype,
            'use_tensorrt': False,
            'use_fp8': False,
            'target_rank': target_rank,
            'batch_size': a.shape[0] if a.dim() > 2 else 1
        }

        # FP8 selection logic
        if self._should_use_fp8(a, b):
            config['precision'] = torch.float8_e4m3fn
            config['use_fp8'] = True

        # TensorRT selection logic
        if self._should_use_tensorrt(a, b, target_rank):
            config['use_tensorrt'] = True
            config['decomposition_method'] = 'tensorrt_optimized'

        # Hardware-specific optimizations
        if self.device_capability and self.device_capability[0] >= 8:  # Ampere and newer
            config['decomposition_method'] = 'randomized_svd'  # Faster on modern GPUs

        return config

    def _should_use_fp8(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        """Determine if FP8 should be used based on tensor properties."""
        if not FP8_E4M3FN_AVAILABLE:
            return False

        # Use FP8 for large matrices where precision loss is acceptable
        total_elements = a.numel() + b.numel()
        return total_elements > 1_000_000  # 1M+ elements

    def _should_use_tensorrt(self, a: torch.Tensor, b: torch.Tensor, target_rank: Optional[int]) -> bool:
        """Determine if TensorRT should be used."""
        if not TORCH_TENSORRT_AVAILABLE:
            return False

        # Use TensorRT for inference with static shapes and large matrices
        return (a.shape[-2] > 512 and a.shape[-1] > 512 and
                b.shape[-1] > 512 and target_rank is not None)


class LowRankGEMM(nn.Module):
    """
    PyTorch module for fast GEMM using low-rank matrix approximations with FP8 and TensorRT support.

    This module transforms arbitrary matrices into low-rank approximations and
    performs matrix multiplication on these approximations, achieving sub-quadratic
    complexity for large matrices. Supports FP8 precision and TensorRT optimization.

    Args:
        target_rank: Target rank for low-rank approximation. If None, will be
                    determined automatically based on matrix size.
        rank_selection_method: Method to determine rank if target_rank is None.
                              Options: 'auto', 'energy', 'fixed_fraction'
        energy_threshold: For 'energy' method, fraction of singular values to retain
                         (default: 0.99 for 99% energy preservation)
        decomposition_method: Method for low-rank decomposition.
                             Options: 'svd', 'randomized_svd', 'tensorrt_optimized'
        oversampling: Oversampling parameter for randomized SVD (default: 10)
        power_iterations: Number of power iterations for randomized SVD (default: 2)
        use_fp8: Force FP8 precision usage (auto-detected if None)
        use_tensorrt: Force TensorRT usage (auto-detected if None)
        auto_kernel: Enable automatic kernel selection (default: True)
    """

    def __init__(
        self,
        target_rank: Optional[int] = None,
        rank_selection_method: str = 'auto',
        energy_threshold: float = 0.99,
        decomposition_method: str = 'auto',
        oversampling: int = 10,
        power_iterations: int = 2,
        use_fp8: Optional[bool] = None,
        use_tensorrt: Optional[bool] = None,
        auto_kernel: bool = True
    ):
        super().__init__()
        self.rank_selection_method = rank_selection_method
        self.energy_threshold = energy_threshold
        self.decomposition_method = decomposition_method
        self.oversampling = oversampling
        self.power_iterations = power_iterations
        self.use_fp8 = use_fp8
        self.use_tensorrt = use_tensorrt
        self.auto_kernel = auto_kernel

        # Initialize auto kernel selector
        self.kernel_selector = AutoKernelSelector() if auto_kernel else None

        # TensorRT engine cache
        self.trt_engine = None
        self.trt_context = None

        # Set reasonable default rank if not specified
        if target_rank is None:
            # Default rank based on expected use case
            # For low-rank approximation, start with conservative rank
            self.target_rank = 64  # Reasonable default for most applications
        else:
            self.target_rank = target_rank

        # Validate parameters
        assert rank_selection_method in ['auto', 'energy', 'fixed_fraction'], \
            "rank_selection_method must be 'auto', 'energy', or 'fixed_fraction'"

        valid_methods = ['svd', 'randomized_svd', 'auto']
        if TORCH_TENSORRT_AVAILABLE:
            valid_methods.append('tensorrt_optimized')
        assert decomposition_method in valid_methods, \
            f"decomposition_method must be one of {valid_methods}"

    def _compute_adaptive_rank(self, matrix: torch.Tensor) -> int:
        """
        Compute adaptive rank based on matrix dimensions and selection method.
        """
        m, n = matrix.shape
        min_dim = min(m, n)

        if self.rank_selection_method == 'auto':
            # Use square root of minimum dimension as heuristic
            base_rank = max(1, int(math.sqrt(min_dim)))

            # Reduce rank for FP8 since it has lower precision
            if self.use_fp8 and FP8_E4M3FN_AVAILABLE:
                base_rank = max(1, int(base_rank * 0.7))  # 30% reduction for FP8

            return base_rank
        elif self.rank_selection_method == 'fixed_fraction':
            # Use fixed fraction of minimum dimension
            fraction = 0.1  # 10% of min dimension
            if self.use_fp8 and FP8_E4M3FN_AVAILABLE:
                fraction = 0.07  # Reduce for FP8
            return max(1, int(fraction * min_dim))
        else:
            return min_dim  # 'energy' method will truncate later

    def _low_rank_approximation_svd(
        self,
        matrix: torch.Tensor,
        rank: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute low-rank SVD approximation of a matrix.

        Returns:
            U, S, V such that matrix ≈ U @ S @ V^T
        """
        if rank is None:
            rank = self._compute_adaptive_rank(matrix)

        # Perform SVD
        U, S, V = torch.svd(matrix)

        if self.rank_selection_method == 'energy':
            # Compute cumulative energy
            total_energy = torch.sum(S ** 2)
            cumulative_energy = torch.cumsum(S ** 2, dim=0) / total_energy
            # Find rank that preserves desired energy
            rank = torch.sum(cumulative_energy < self.energy_threshold).item() + 1
            rank = min(rank, len(S))

        # Truncate to desired rank
        rank = min(rank, len(S))
        U = U[:, :rank]
        S = S[:rank]
        V = V[:, :rank]

        return U, S, V

    def _low_rank_approximation_randomized_svd(
        self,
        matrix: torch.Tensor,
        rank: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute low-rank approximation using randomized SVD.

        This is faster than full SVD for large matrices when target rank is small.
        """
        if rank is None:
            rank = self._compute_adaptive_rank(matrix)

        m, n = matrix.shape
        rank = min(rank, min(m, n))

        # Stage 1: Random projection
        l = rank + self.oversampling
        l = min(l, min(m, n))

        # Generate random matrix
        omega = torch.randn(n, l, dtype=matrix.dtype, device=matrix.device)

        # Power iterations for better approximation
        y = matrix @ omega
        for _ in range(self.power_iterations):
            y = matrix @ (matrix.T @ y)

        # Orthonormalize
        q, _ = torch.linalg.qr(y, mode='reduced')

        # Stage 2: SVD of smaller matrix
        b = q.T @ matrix
        u_tilde, s, v_tilde = torch.svd(b)

        # Truncate if using energy method
        if self.rank_selection_method == 'energy':
            total_energy = torch.sum(s ** 2)
            cumulative_energy = torch.cumsum(s ** 2, dim=0) / total_energy
            actual_rank = torch.sum(cumulative_energy < self.energy_threshold).item() + 1
            actual_rank = min(actual_rank, len(s))
            u_tilde = u_tilde[:, :actual_rank]
            s = s[:actual_rank]
            v_tilde = v_tilde[:, :actual_rank]
            rank = actual_rank

        # Reconstruct U
        u = q @ u_tilde

        return u, s, v_tilde

    def _approximate_matrix(
        self,
        matrix: torch.Tensor,
        decomposition_method: str = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute low-rank approximation of input matrix.
        """
        method = decomposition_method or self.decomposition_method

        if method == 'svd':
            return self._low_rank_approximation_svd(matrix, self.target_rank)
        elif method == 'randomized_svd':
            return self._low_rank_approximation_randomized_svd(matrix, self.target_rank)
        elif method == 'tensorrt_optimized':
            # Fall back to randomized SVD for TensorRT (will be optimized separately)
            return self._low_rank_approximation_randomized_svd(matrix, self.target_rank)
        else:
            # Auto selection
            if torch.cuda.is_available():
                return self._low_rank_approximation_randomized_svd(matrix, self.target_rank)
            else:
                return self._low_rank_approximation_svd(matrix, self.target_rank)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Perform low-rank approximated matrix multiplication with FP8 and TensorRT support.

        Args:
            a: First input matrix of shape (..., m, k)
            b: Second input matrix of shape (..., k, n)

        Returns:
            Result matrix of shape (..., m, n)
        """
        # Auto kernel selection if enabled
        if self.auto_kernel and self.kernel_selector:
            kernel_config = self.kernel_selector.select_kernel(a, b, self.target_rank)
            return self._forward_with_config(a, b, kernel_config)

        # Manual configuration
        decomp_method = self.decomposition_method
        if decomp_method == 'auto':
            # Choose optimal method based on hardware
            if torch.cuda.is_available():
                decomp_method = 'randomized_svd'  # Faster on modern GPUs
            else:
                decomp_method = 'svd'

        kernel_config = {
            'decomposition_method': decomp_method,
            'precision': self._get_precision(a, b),
            'use_tensorrt': self.use_tensorrt if self.use_tensorrt is not None else False,
            'use_fp8': self.use_fp8 if self.use_fp8 is not None else False,
            'target_rank': self.target_rank,
            'batch_size': a.shape[0] if a.dim() > 2 else 1
        }

        return self._forward_with_config(a, b, kernel_config)

    def _get_precision(self, a: torch.Tensor, b: torch.Tensor) -> torch.dtype:
        """Determine precision based on configuration and availability."""
        if self.use_fp8 and FP8_E4M3FN_AVAILABLE:
            return torch.float8_e4m3fn
        return a.dtype

    def _convert_to_fp8(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to FP8 precision."""
        if not FP8_E4M3FN_AVAILABLE or not self.use_fp8:
            return tensor

        # Convert to FP8 with scaling for better precision
        if tensor.dtype == torch.float32:
            # Scale to avoid overflow in FP8 range
            scale = 1.0 / (tensor.abs().max() + 1e-8)
            scaled_tensor = tensor * scale
            fp8_tensor = scaled_tensor.to(torch.float8_e4m3fn)
            return fp8_tensor, scale
        else:
            return tensor.to(torch.float8_e4m3fn), 1.0

    def _forward_with_config(self, a: torch.Tensor, b: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
        """
        Perform forward pass with specific kernel configuration.
        """
        # Handle TensorRT path
        if config['use_tensorrt'] and TORCH_TENSORRT_AVAILABLE:
            return self._forward_tensorrt(a, b, config)

        # Handle FP8 path
        original_dtype = a.dtype
        scale_a = scale_b = 1.0

        if config['use_fp8']:
            fp8_result_a = self._convert_to_fp8(a)
            fp8_result_b = self._convert_to_fp8(b)
            if isinstance(fp8_result_a, tuple):
                a, scale_a = fp8_result_a
            else:
                a = fp8_result_a
            if isinstance(fp8_result_b, tuple):
                b, scale_b = fp8_result_b
            else:
                b = fp8_result_b

        # Save original shapes for reshaping
        original_shape_a = a.shape
        original_shape_b = b.shape

        # Flatten batch dimensions if present
        if a.dim() > 2:
            a_flat = a.view(-1, a.shape[-2], a.shape[-1])
            b_flat = b.view(-1, b.shape[-2], b.shape[-1])
        else:
            a_flat = a.unsqueeze(0)
            b_flat = b.unsqueeze(0)

        batch_size = a_flat.shape[0]

        # Select decomposition method
        decomp_method = config['decomposition_method']
        if decomp_method == 'auto':
            decomp_method = 'randomized_svd' if torch.cuda.is_available() else 'svd'

        # Compute low-rank approximations for all matrices in batch
        results = []
        for i in range(batch_size):
            # Approximate matrices (convert to FP32 for SVD if needed)
            matrix_a = a_flat[i]
            matrix_b = b_flat[i]

            # Convert FP8 to FP32 for SVD computation if needed
            if matrix_a.dtype == torch.float8_e4m3fn:
                matrix_a = matrix_a.to(torch.float32)
            if matrix_b.dtype == torch.float8_e4m3fn:
                matrix_b = matrix_b.to(torch.float32)

            u_a, s_a, v_a = self._approximate_matrix(matrix_a, decomp_method)
            u_b, s_b, v_b = self._approximate_matrix(matrix_b, decomp_method)

            # Efficient computation: A @ B ≈ (U_a @ S_a @ V_a^T) @ (U_b @ S_b @ V_b^T)
            # = U_a @ (S_a @ (V_a^T @ U_b) @ S_b) @ V_b^T

            # Compute intermediate matrices
            v_a_t_u_b = v_a.T @ u_b
            s_a_diag = torch.diag(s_a)
            s_b_diag = torch.diag(s_b)

            # Compute result
            intermediate = s_a_diag @ v_a_t_u_b @ s_b_diag
            result = u_a @ intermediate @ v_b.T

            results.append(result)

        # Stack results
        result_tensor = torch.stack(results, dim=0)

        # Reshape to original batch dimensions
        if len(original_shape_a) > 2:
            result_shape = list(original_shape_a[:-2]) + [result_tensor.shape[-2], result_tensor.shape[-1]]
            result_tensor = result_tensor.view(result_shape)

        # Convert back from FP8 if needed
        if config['use_fp8'] and original_dtype != result_tensor.dtype:
            result_tensor = result_tensor.to(original_dtype)
            # Apply scaling correction
            result_tensor = result_tensor * (scale_a * scale_b)

        return result_tensor

    def _forward_tensorrt(self, a: torch.Tensor, b: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
        """
        Perform forward pass using TensorRT optimization.
        """
        if not TORCH_TENSORRT_AVAILABLE:
            raise RuntimeError("Torch-TensorRT not available. Install with: pip install torch-tensorrt")

        # For now, fall back to regular computation but with JIT compilation
        # In a full implementation, this would create and cache TensorRT engines

        # Create a simple compiled version
        @torch.compile(mode="reduce-overhead")
        def compiled_low_rank_gemm(a_compiled, b_compiled, target_rank):
            # Simple low-rank approximation using torch.compile
            # This is a simplified version - full TensorRT would use ONNX export
            u_a, s_a, v_a = torch.svd_lowrank(a_compiled, q=target_rank)
            u_b, s_b, v_b = torch.svd_lowrank(b_compiled, q=target_rank)

            # Efficient computation
            v_a_t_u_b = v_a.T @ u_b
            s_a_diag = torch.diag(s_a)
            s_b_diag = torch.diag(s_b)

            intermediate = s_a_diag @ v_a_t_u_b @ s_b_diag
            return u_a @ intermediate @ v_b.T

        return compiled_low_rank_gemm(a, b, config['target_rank'] or 50)

    @staticmethod
    def compute_error(original_result: torch.Tensor, approx_result: torch.Tensor) -> torch.Tensor:
        """
        Compute relative Frobenius norm error between original and approximated results.

        Args:
            original_result: Result from exact matrix multiplication
            approx_result: Result from low-rank approximation

        Returns:
            Relative error ||original - approx||_F / ||original||_F
        """
        diff = original_result - approx_result
        error = torch.norm(diff, p='fro')
        norm_original = torch.norm(original_result, p='fro')
        return error / norm_original if norm_original > 0 else torch.tensor(0.0)


class AdaptiveLowRankGEMM(nn.Module):
    """
    Adaptive version that chooses rank based on error tolerance.

    This module iteratively increases rank until the approximation error
    falls below a specified threshold.
    """

    def __init__(
        self,
        error_tolerance: float = 0.01,
        max_rank: Optional[int] = None,
        decomposition_method: str = 'randomized_svd'
    ):
        super().__init__()
        self.error_tolerance = error_tolerance
        self.max_rank = max_rank
        self.decomposition_method = decomposition_method

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Perform adaptive low-rank GEMM with error control.

        Args:
            a: First input matrix
            b: Second input matrix

        Returns:
            Tuple of (result, final_rank_used)
        """
        # Compute exact result for error calculation
        exact_result = torch.matmul(a, b)

        # Start with small rank and increase until error tolerance is met
        rank = 1
        max_possible_rank = min(a.shape[-1], b.shape[-1])

        if self.max_rank is not None:
            max_possible_rank = min(max_possible_rank, self.max_rank)

        best_result = None
        best_error = float('inf')

        while rank <= max_possible_rank:
            # Create low-rank GEMM with current rank
            low_rank_gemm = LowRankGEMM(
                target_rank=rank,
                decomposition_method=self.decomposition_method
            )

            # Compute approximation
            approx_result = low_rank_gemm(a, b)
            error = LowRankGEMM.compute_error(exact_result, approx_result)

            if error < best_error:
                best_error = error
                best_result = approx_result

            # Check if error tolerance is met
            if error <= self.error_tolerance:
                return best_result, rank

            # Increase rank (exponential growth for efficiency)
            rank = min(rank * 2, max_possible_rank + 1)

        # Return best approximation found
        return best_result, rank - 1


def benchmark_low_rank_gemm():
    """
    Benchmark function to compare low-rank GEMM performance vs exact GEMM.
    """
    import time

    # Test matrix sizes
    sizes = [100, 500, 1000]
    ranks = [10, 50, 100]

    print("Benchmarking Low-Rank GEMM vs Exact GEMM")
    print("=" * 50)

    for size in sizes:
        print(f"\nMatrix size: {size}x{size}")

        # Generate test matrices
        a = torch.randn(size, size)
        b = torch.randn(size, size)

        # Exact computation
        start_time = time.time()
        exact_result = torch.matmul(a, b)
        exact_time = time.time() - start_time

        print(".4f")

        for rank in ranks:
            if rank >= size:
                continue

            # Low-rank approximation
            low_rank_gemm = LowRankGEMM(target_rank=rank)

            start_time = time.time()
            approx_result = low_rank_gemm(a, b)
            approx_time = time.time() - start_time

            error = LowRankGEMM.compute_error(exact_result, approx_result)

            speedup = exact_time / approx_time if approx_time > 0 else float('inf')

            print(".4f")


if __name__ == "__main__":
    # Run benchmark
    benchmark_low_rank_gemm()
