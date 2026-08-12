/* Minimal cache kernels — single dtype (half) for fast compilation.
   Aligns with vllm/csrc/libtorch_stable/cache_kernels.cu.

   Full dispatch (AT_DISPATCH_FLOATING_TYPES_AND2) in real vLLM source at
   csrc/libtorch_stable/cache_kernels.cu:701
*/

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK_ERROR() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
        } \
    } while (0)


__global__ void reshape_and_cache_kernel_f16(
    const at::Half* __restrict__ key,
    const at::Half* __restrict__ value,
    at::Half* __restrict__ key_cache,
    at::Half* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    int num_tokens,
    int num_heads,
    int head_size,
    int block_size) {

    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    if (token_idx >= num_tokens) return;

    int64_t slot = slot_mapping[token_idx];
    if (slot < 0) return;

    int block_id = slot / block_size;
    int block_off = slot % block_size;

    // Layouts:
    //   key:        [num_tokens, num_heads, head_size]
    //   key_cache:  [num_blocks, num_heads, block_size, head_size]
    int src_off = token_idx * num_heads * head_size + head_idx * head_size;
    int dst_base = block_id * num_heads * block_size * head_size
                 + head_idx * block_size * head_size
                 + block_off * head_size;

    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        key_cache[dst_base + i] = key[src_off + i];
        value_cache[dst_base + i] = value[src_off + i];
    }
}


torch::Tensor reshape_and_cache(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor slot_mapping) {

    int num_tokens = key.size(0);
    int num_heads = key.size(1);
    int head_size = key.size(2);
    int block_size = key_cache.size(2);

    TORCH_CHECK(key.dtype() == torch::kFloat16, "Only float16 supported");
    TORCH_CHECK(key.is_cuda());

    dim3 grid(num_tokens, num_heads);
    dim3 block(256);

    reshape_and_cache_kernel_f16<<<grid, block, 0, 0>>>(
        reinterpret_cast<const at::Half*>(key.data_ptr()),
        reinterpret_cast<const at::Half*>(value.data_ptr()),
        reinterpret_cast<at::Half*>(key_cache.data_ptr()),
        reinterpret_cast<at::Half*>(value_cache.data_ptr()),
        slot_mapping.data_ptr<int64_t>(),
        num_tokens, num_heads, head_size, block_size);

    CUDA_CHECK_ERROR();
    return torch::Tensor();  // void return
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reshape_and_cache", &reshape_and_cache,
          "Write K/V into paged KV cache (float16 only)");
}
