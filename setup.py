"""setup.py — builds C++/CUDA extension for my-vllm.

Aligns with vllm/setup.py.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="my_vllm_csrc",
    ext_modules=[
        CUDAExtension(
            name="my_vllm._C",
            sources=["csrc/cache_kernels.cu"],
            extra_compile_args={
                "cxx": ["-O2", "-std=c++17"],
                "nvcc": [
                    "-O2", "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-arch=sm_89",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
