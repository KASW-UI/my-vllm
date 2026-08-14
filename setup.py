"""setup.py — builds C++/CUDA extensions for my-vllm.

Aligns with vllm/setup.py.  Two CUDA extensions are built:

  * ``my_vllm._C`` — cache kernels (reshape_and_cache_flash /
    swap_blocks / copy_blocks / cp_gather_cache / convert_fp8 /
    concat_and_cache_mla) + custom all-reduce.  Loaded by
    ``my_vllm._custom_ops``.

  * ``my_vllm.cumem_allocator`` — vendored from
    ``vllm/csrc/cumem_allocator.cpp``.  Implements the CUDA
    Virtual Memory Management (VMM) API
    (``cuMemCreate`` / ``cuMemAddressReserve`` / ``cuMemMap`` /
    ``cuMemUnmap`` / ``cuMemRelease``).  Exposes two C entry
    points (``my_malloc`` / ``my_free``) that PyTorch's
    ``CUDAPluggableAllocator`` calls via dlsym, plus three
    Python-callable functions (``init_module``,
    ``python_create_and_map``, ``python_unmap_and_release``)
    that vLLM's ``CuMemAllocator`` uses for the sleep / wake_up
    cycle.  Loaded by ``my_vllm.device_allocator.cumem``.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="my_vllm_csrc",
    ext_modules=[
        CUDAExtension(
            name="my_vllm._C",
            sources=[
                "csrc/cache_kernels.cu",
                "csrc/custom_all_reduce.cu",
            ],
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
        CUDAExtension(
            name="my_vllm.cumem_allocator",
            sources=[
                "csrc/cumem_allocator.cpp",
            ],
            # cumem_allocator.cpp is plain C++ with CUDA Driver
            # API calls (cuda.h, not CUDA kernels) — no nvcc
            # extension flags needed.
            extra_compile_args={
                "cxx": [
                    "-O2",
                    "-std=c++17",
                ],
            },
            # Link against CUDA driver library (libcuda.so) for
            # cuMemCreate / cuMemMap / cuMemUnmap / cuMemRelease.
            libraries=["cuda"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
