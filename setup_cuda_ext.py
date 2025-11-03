from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='cublas_fp8_extension',
    ext_modules=[
        CUDAExtension(
            'cublas_fp8_extension',
            ['cublas_fp8_extension.cpp'],
            libraries=['cublas'],
            extra_compile_args={
                'cxx': [],
                'nvcc': [
                    '-arch=sm_89',  # RTX 4090 compute capability
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda'
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
