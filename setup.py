"""
Setup script for Low-Rank GEMM package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="low-rank-gemm",
    version="1.0.0",
    author="Low-Rank GEMM Team",
    author_email="",
    description="High-performance low-rank matrix multiplication with FP8 and TensorRT support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/low-rank-gemm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "psutil>=5.8.0",
    ],
    extras_require={
        "tensorrt": ["torch-tensorrt"],
        "cutlass": ["cutlass-python"],
        "dev": [
            "pytest>=6.0",
            "black",
            "flake8",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "low-rank-benchmark=run_fp8_benchmark:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
