#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp8.h>
#include <iostream>

// Error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(status) << std::endl; \
            return; \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cublasGetStatusString(status) << std::endl; \
            return; \
        } \
    } while(0)

// cuBLAS handle
static cublasHandle_t cublas_handle = nullptr;

// Initialize cuBLAS
void init_cublas() {
    if (cublas_handle == nullptr) {
        CHECK_CUBLAS(cublasCreate(&cublas_handle));
    }
}

// FP8 GEMM using cuBLAS
torch::Tensor cublas_fp8_gemm(
    torch::Tensor a,
    torch::Tensor b,
    float scale_a = 1.0f,
    float scale_b = 1.0f,
    float scale_result = 1.0f
) {
    TORCH_CHECK(a.device().is_cuda(), "Input tensor A must be on CUDA device");
    TORCH_CHECK(b.device().is_cuda(), "Input tensor B must be on CUDA device");
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn, "Input tensor A must be FP8 E4M3");
    TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn, "Input tensor B must be FP8 E4M3");

    init_cublas();

    int m = a.size(0);
    int k = a.size(1);
    int n = b.size(1);

    TORCH_CHECK(b.size(0) == k, "Matrix dimensions don't match for GEMM");

    // Create output tensor (FP32 for accumulation)
    auto result = torch::empty({m, n}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    // Get raw pointers
    const __nv_fp8_e4m3* a_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(a.data_ptr());
    const __nv_fp8_e4m3* b_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(b.data_ptr());
    float* c_ptr = result.data_ptr();

    // cuBLAS GEMM with FP8
    // Note: This uses the basic GEMM, but for true FP8 performance we might need
    // cublasLtMatmul with proper scaling and compute type configuration

    CHECK_CUBLAS(cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,  // Note: cuBLAS uses column-major, so dimensions are swapped
        &scale_result,
        b_ptr, CUDA_R_8F_E4M3, n,
        a_ptr, CUDA_R_8F_E4M3, k,
        &scale_result,
        c_ptr, CUDA_R_32F, n,
        CUDA_R_32F,  // Compute type
        CUBLAS_GEMM_DEFAULT
    ));

    return result;
}

// Cleanup
void cleanup_cublas() {
    if (cublas_handle != nullptr) {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
    }
}

// Register the extension
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cublas_fp8_gemm", &cublas_fp8_gemm, "cuBLAS FP8 GEMM",
          py::arg("a"), py::arg("b"),
          py::arg("scale_a") = 1.0f,
          py::arg("scale_b") = 1.0f,
          py::arg("scale_result") = 1.0f);
    m.def("init_cublas", &init_cublas, "Initialize cuBLAS handle");
    m.def("cleanup_cublas", &cleanup_cublas, "Cleanup cuBLAS handle");
}
