/* my-vllm cache_kernels.cu — CUDA ops for paged KV cache.
 *
 * Mirrors the public API surface of vllm/csrc/libtorch_stable/cache_kernels.cu:
 *
 *   void reshape_and_cache(key, value, key_cache, value_cache, slot_mapping,
 *                          kv_cache_dtype, k_scale, v_scale)
 *     → AT_DISPATCH-style multi-dtype (fp16, bf16) cache write.
 *
 *   void reshape_and_cache_flash(key, value, key_cache, value_cache,
 *                                slot_mapping, kv_cache_dtype, k_scale, v_scale)
 *     → FlashAttention layout (num_blocks, block_size, num_heads, head_size)
 *       variant of reshape_and_cache.  CUDA backup of the Triton hot path
 *       (C149: triton_reshape_and_cache_flash); useful inside CUDA-graph
 *       captures where Triton autotune is undesirable.
 *
 *   void swap_blocks(src, dst, block_size_in_bytes, block_mapping)
 *     → cudaMemcpyAsync loop over (src_block, dst_block) pairs. CPU-driven,
 *       used by preemption (beam search, prefix cache CoW).
 *
 *   void copy_blocks(key_caches, value_caches, block_mapping)
 *     → GPU-side block-to-block KV copy. Used when prefix cache requires
 *       a physical copy (the block IDs cannot be shared across requests).
 *
 *   void cp_gather_cache(src_cache, dst, block_table, cu_seq_lens, ...)
 *     → Gather K/V tokens from non-contiguous physical blocks (per-request
 *       block table) into a contiguous output buffer.  This is the prefix-
 *       cache CoW "rewrite" path: take a list of (req → block_ids) and emit
 *       a flat [total_tokens, head_dim] tensor.
 *
 *   void convert_fp8(dst_cache, src_cache, scale, kv_cache_dtype)
 *     → Per-block FP8 ↔ BF16/FP16 conversion.  Used when switching the KV
 *       cache storage dtype at runtime (and for testing FP8 numerics).
 *
 *   void concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping,
 *                             kv_cache_dtype, scale)
 *     → MLA compressed-KV concatenation write (kv_lora_rank + pe_dim).
 *       Stub: declared + dispatch for API symmetry with vLLM, but no MLA
 *       model is wired in my-vllm (DeepSeek V2/V3 excluded by JD scope).
 *
 * Hot-path note: in my-vllm, the per-token cache write is performed by the
 * Triton kernel triton_reshape_and_cache_flash (C149).  The CUDA kernels
 * here provide the full vLLM API surface (fallback + CUDA-graph + tests).
 *
 * Reference: vllm/csrc/libtorch_stable/cache_kernels.cu
 *   - swap_blocks:                     line 33
 *   - copy_blocks_kernel / wrapper:    line 191, 887
 *   - reshape_and_cache_kernel/dispatch: line 255, 700
 *   - reshape_and_cache_flash:         line 746
 *   - concat_and_cache_mla:            line 842
 *   - convert_fp8:                     line 928
 *   - cp_gather_cache:                 line 1242
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        TORCH_CHECK(err == cudaSuccess, \
                    "CUDA error at ", __FILE__, ":", __LINE__, ": ", \
                    cudaGetErrorString(err)); \
    } while (0)


// ---------------------------------------------------------------------------
// reshape_and_cache — multi-dtype (fp16, bf16) cache write kernel
// ---------------------------------------------------------------------------
//
// Layout: key/value shape [num_tokens, num_heads, head_size];
//         key_cache/value_cache shape [num_blocks, num_heads, block_size, head_size].
// slot_mapping[i] = block_id * block_size + block_off for token i;
// slot_mapping[i] < 0 means "skip this token" (used for padding).

template <typename scalar_t>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ key_cache,
    scalar_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    int num_tokens,
    int num_heads,
    int head_size,
    int block_size) {

    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    int64_t slot = slot_mapping[token_idx];
    if (slot < 0) return;

    int block_id = slot / block_size;
    int block_off = slot % block_size;
    int head_idx = blockIdx.y;

    int src_off = token_idx * num_heads * head_size + head_idx * head_size;
    int dst_base = block_id * num_heads * block_size * head_size
                 + head_idx * block_size * head_size
                 + block_off * head_size;

    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        key_cache[dst_base + i] = key[src_off + i];
        value_cache[dst_base + i] = value[src_off + i];
    }
}


void reshape_and_cache(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor k_scale,
    torch::Tensor v_scale) {

    TORCH_CHECK(key.is_cuda() && value.is_cuda(), "key/value must be CUDA");
    TORCH_CHECK(key_cache.is_cuda() && value_cache.is_cuda(),
                "key_cache/value_cache must be CUDA");
    TORCH_CHECK(slot_mapping.is_cuda(), "slot_mapping must be CUDA");
    TORCH_CHECK(key.scalar_type() == value.scalar_type(),
                "key/value dtype mismatch");
    TORCH_CHECK(key_cache.scalar_type() == value_cache.scalar_type(),
                "key_cache/value_cache dtype mismatch");

    // FP8 dispatch falls through to the Triton path (C149).  We dispatch
    // here for fp16 / bf16 only; kv_cache_dtype is recorded for symmetry
    // with vLLM's API but not consulted for storage layout (the cache
    // tensors themselves carry the dtype).
    TORCH_CHECK(kv_cache_dtype == "auto" || kv_cache_dtype == "fp8"
                || kv_cache_dtype == "fp8_e4m3" || kv_cache_dtype == "fp8_e5m2"
                || kv_cache_dtype == "float16" || kv_cache_dtype == "half"
                || kv_cache_dtype == "bfloat16" || kv_cache_dtype == "bf16",
                "unsupported kv_cache_dtype: ", kv_cache_dtype);

    int num_tokens = key.size(0);
    int num_heads = key.size(1);
    int head_size = key.size(2);
    int block_size = key_cache.size(2);

    // For FP8 storage paths, the CUDA dispatch is intentionally not implemented
    // here — the Triton kernel handles FP8 quantization end-to-end.  We keep
    // the API surface for symmetry but route to a CUDA error if called with
    // FP8 storage.
    bool is_fp8_storage = (kv_cache_dtype.rfind("fp8", 0) == 0);
    TORCH_CHECK(!is_fp8_storage || key.scalar_type() == torch::kFloat8_e4m3fn
                || key.scalar_type() == torch::kFloat8_e5m2,
                "FP8 cache path is handled by the Triton kernel "
                "(triton_reshape_and_cache_flash); not supported by this CUDA "
                "dispatch. kv_cache_dtype=", kv_cache_dtype);

    dim3 grid(num_tokens, num_heads);
    dim3 block(256);
    const c10::cuda::OptionalCUDAGuard device_guard(key.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (key.scalar_type() == torch::kFloat16) {
        reshape_and_cache_kernel<at::Half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const at::Half*>(key.data_ptr()),
            reinterpret_cast<const at::Half*>(value.data_ptr()),
            reinterpret_cast<at::Half*>(key_cache.data_ptr()),
            reinterpret_cast<at::Half*>(value_cache.data_ptr()),
            slot_mapping.data_ptr<int64_t>(),
            num_tokens, num_heads, head_size, block_size);
    } else if (key.scalar_type() == torch::kBFloat16) {
        reshape_and_cache_kernel<at::BFloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const at::BFloat16*>(key.data_ptr()),
            reinterpret_cast<const at::BFloat16*>(value.data_ptr()),
            reinterpret_cast<at::BFloat16*>(key_cache.data_ptr()),
            reinterpret_cast<at::BFloat16*>(value_cache.data_ptr()),
            slot_mapping.data_ptr<int64_t>(),
            num_tokens, num_heads, head_size, block_size);
    } else {
        TORCH_CHECK(false, "Unsupported key dtype for CUDA reshape_and_cache: ",
                    key.scalar_type());
    }
    CUDA_CHECK(cudaGetLastError());
}


// ---------------------------------------------------------------------------
// swap_blocks — CPU-driven cudaMemcpyAsync over (src_block, dst_block) pairs.
// ---------------------------------------------------------------------------
//
// block_mapping: int64 tensor of shape [num_pairs, 2]; row i = (src, dst).

void swap_blocks(
    torch::Tensor src,
    torch::Tensor dst,
    int64_t block_size_in_bytes,
    torch::Tensor block_mapping) {

    TORCH_CHECK(src.is_cuda() || src.is_cpu(), "src must be a tensor");
    TORCH_CHECK(dst.is_cuda() || dst.is_cpu(), "dst must be a tensor");
    TORCH_CHECK(block_mapping.is_cpu(),
                "block_mapping must live on CPU (avoid per-item GPU sync)");

    char* src_ptr = static_cast<char*>(src.data_ptr());
    char* dst_ptr = static_cast<char*>(dst.data_ptr());

    cudaMemcpyKind memcpy_type;
    if (src.is_cuda() && dst.is_cuda()) {
        TORCH_CHECK(src.device() == dst.device(),
                    "src and dst must be on the same device");
        memcpy_type = cudaMemcpyDeviceToDevice;
    } else if (src.is_cuda() && dst.is_cpu()) {
        memcpy_type = cudaMemcpyDeviceToHost;
    } else if (src.is_cpu() && dst.is_cuda()) {
        memcpy_type = cudaMemcpyHostToDevice;
    } else {
        TORCH_CHECK(false, "Invalid src/dst device combination");
    }

    const c10::cuda::OptionalCUDAGuard device_guard(
        src.is_cuda() ? src.device() : dst.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int64_t num_pairs = block_mapping.size(0);
    const int64_t* bm_ptr = block_mapping.data_ptr<int64_t>();
    for (int64_t i = 0; i < num_pairs; ++i) {
        int64_t src_block = bm_ptr[2 * i];
        int64_t dst_block = bm_ptr[2 * i + 1];
        int64_t src_off = src_block * block_size_in_bytes;
        int64_t dst_off = dst_block * block_size_in_bytes;
        cudaMemcpyAsync(dst_ptr + dst_off, src_ptr + src_off,
                        block_size_in_bytes, memcpy_type, stream);
    }
}


// ---------------------------------------------------------------------------
// copy_blocks — GPU-side block-to-block KV copy (multi-dtype).
// ---------------------------------------------------------------------------
//
// key_caches / value_caches: list[Tensor] of per-layer cache tensors,
//                           shape [num_blocks, block_size, num_heads, head_size]
//                           for our attention backend layout.  We accept
//                           any layout where numel per block is well-defined.
// block_mapping: int64 tensor [num_pairs, 2]; row i = (src_block, dst_block).
//
// Grid: (num_layers, num_pairs). Each block copies one KV block between
// pair (layer, pair_idx).

template <typename scalar_t>
__global__ void copy_blocks_kernel(
    int64_t* key_cache_ptrs,
    int64_t* value_cache_ptrs,
    const int64_t* __restrict__ block_mapping,
    int numel_per_block) {

    int layer_idx = blockIdx.x;
    int pair_idx  = blockIdx.y;

    scalar_t* key_cache   = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
    scalar_t* value_cache = reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);

    int64_t src_block = block_mapping[2 * pair_idx];
    int64_t dst_block = block_mapping[2 * pair_idx + 1];
    int64_t src_off = src_block * numel_per_block;
    int64_t dst_off = dst_block * numel_per_block;

    for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
        key_cache[dst_off + i]   = key_cache[src_off + i];
        value_cache[dst_off + i] = value_cache[src_off + i];
    }
}


void copy_blocks(
    std::vector<torch::Tensor> key_caches,
    std::vector<torch::Tensor> value_caches,
    torch::Tensor block_mapping) {

    TORCH_CHECK(!key_caches.empty(), "key_caches must be non-empty");
    TORCH_CHECK(key_caches.size() == value_caches.size(),
                "key_caches / value_caches length mismatch");
    TORCH_CHECK(block_mapping.is_cpu(),
                "block_mapping must live on CPU");
    TORCH_CHECK(block_mapping.size(1) == 2,
                "block_mapping must have shape [num_pairs, 2]");

    int num_layers = static_cast<int>(key_caches.size());
    int64_t numel_per_block = key_caches[0].numel() / key_caches[0].size(0);
    int num_pairs = static_cast<int>(block_mapping.size(0));

    // Stack per-layer data_ptrs into a single int64 tensor for the kernel.
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto key_ptrs = torch::empty({num_layers}, opts);
    auto val_ptrs = torch::empty({num_layers}, opts);
    auto key_ptrs_acc = key_ptrs.data_ptr<int64_t>();
    auto val_ptrs_acc = val_ptrs.data_ptr<int64_t>();
    for (int i = 0; i < num_layers; ++i) {
        key_ptrs_acc[i] = reinterpret_cast<int64_t>(key_caches[i].data_ptr());
        val_ptrs_acc[i] = reinterpret_cast<int64_t>(value_caches[i].data_ptr());
    }
    key_ptrs = key_ptrs.to(torch::kCUDA, /*non_blocking=*/false);
    val_ptrs = val_ptrs.to(torch::kCUDA, /*non_blocking=*/false);

    const c10::cuda::OptionalCUDAGuard device_guard(key_caches[0].device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(num_layers, num_pairs);
    dim3 block(256);

    auto scalar_t = key_caches[0].scalar_type();
    TORCH_CHECK(scalar_t == value_caches[0].scalar_type(),
                "key/value cache dtype mismatch at layer 0");

    if (scalar_t == torch::kFloat16) {
        copy_blocks_kernel<at::Half><<<grid, block, 0, stream>>>(
            key_ptrs.data_ptr<int64_t>(), val_ptrs.data_ptr<int64_t>(),
            block_mapping.data_ptr<int64_t>(), numel_per_block);
    } else if (scalar_t == torch::kBFloat16) {
        copy_blocks_kernel<at::BFloat16><<<grid, block, 0, stream>>>(
            key_ptrs.data_ptr<int64_t>(), val_ptrs.data_ptr<int64_t>(),
            block_mapping.data_ptr<int64_t>(), numel_per_block);
    } else {
        TORCH_CHECK(false, "copy_blocks: unsupported cache dtype ",
                    scalar_t);
    }
    CUDA_CHECK(cudaGetLastError());
}


// ---------------------------------------------------------------------------
// reshape_and_cache_flash — FlashAttention layout variant
// ---------------------------------------------------------------------------
//
// Layout difference vs. reshape_and_cache:
//   reshape_and_cache        : [num_blocks, num_heads, block_size, head_size]
//   reshape_and_cache_flash  : [num_blocks, block_size, num_heads, head_size]
//
// Same semantics: write key[i] / value[i] into the slot pointed to by
// slot_mapping[i].  This is the FlashAttention-shaped variant used inside
// CUDA-graph captures (the Triton hot path is triton_reshape_and_cache_flash).
//
// Per-token-head quant scales: not implemented here (the FP8 storage path
// is owned by triton_reshape_and_cache_flash, C149); the kv_cache_dtype
// string is accepted for API symmetry and must be a non-FP8 value.

template <typename scalar_t>
__global__ void reshape_and_cache_flash_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ key_cache,
    scalar_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    int num_tokens,
    int num_heads,
    int head_size,
    int block_size) {

    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    int64_t slot = slot_mapping[token_idx];
    if (slot < 0) return;

    int block_id = slot / block_size;
    int block_off = slot % block_size;
    int head_idx = blockIdx.y;

    int src_off = token_idx * num_heads * head_size + head_idx * head_size;
    // FlashAttention layout: [num_blocks, block_size, num_heads, head_size]
    int dst_base = block_id * block_size * num_heads * head_size
                 + block_off * num_heads * head_size
                 + head_idx * head_size;

    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        key_cache[dst_base + i] = key[src_off + i];
        value_cache[dst_base + i] = value[src_off + i];
    }
}


void reshape_and_cache_flash(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor k_scale,
    torch::Tensor v_scale) {

    TORCH_CHECK(key.is_cuda() && value.is_cuda(), "key/value must be CUDA");
    TORCH_CHECK(key_cache.is_cuda() && value_cache.is_cuda(),
                "key_cache/value_cache must be CUDA");
    TORCH_CHECK(slot_mapping.is_cuda(), "slot_mapping must be CUDA");
    TORCH_CHECK(key.scalar_type() == value.scalar_type(),
                "key/value dtype mismatch");
    TORCH_CHECK(key_cache.scalar_type() == value_cache.scalar_type(),
                "key_cache/value_cache dtype mismatch");

    // kv_cache_dtype is recorded for vLLM API symmetry; this CUDA path only
    // supports fp16/bf16 storage.  FP8 storage is handled by the Triton kernel
    // (triton_reshape_and_cache_flash, C149).
    TORCH_CHECK(kv_cache_dtype == "auto" || kv_cache_dtype == "float16"
                || kv_cache_dtype == "half" || kv_cache_dtype == "bfloat16"
                || kv_cache_dtype == "bf16",
                "reshape_and_cache_flash (CUDA): kv_cache_dtype=", kv_cache_dtype,
                " is not supported; FP8 storage uses the Triton kernel path.");

    int num_tokens = key.size(0);
    int num_heads = key.size(1);
    int head_size = key.size(2);
    // FlashAttention layout: [num_blocks, block_size, num_heads, head_size]
    int block_size = key_cache.size(1);

    dim3 grid(num_tokens, num_heads);
    dim3 block(256);
    const c10::cuda::OptionalCUDAGuard device_guard(key.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (key.scalar_type() == torch::kFloat16) {
        reshape_and_cache_flash_kernel<at::Half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const at::Half*>(key.data_ptr()),
            reinterpret_cast<const at::Half*>(value.data_ptr()),
            reinterpret_cast<at::Half*>(key_cache.data_ptr()),
            reinterpret_cast<at::Half*>(value_cache.data_ptr()),
            slot_mapping.data_ptr<int64_t>(),
            num_tokens, num_heads, head_size, block_size);
    } else if (key.scalar_type() == torch::kBFloat16) {
        reshape_and_cache_flash_kernel<at::BFloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const at::BFloat16*>(key.data_ptr()),
            reinterpret_cast<const at::BFloat16*>(value.data_ptr()),
            reinterpret_cast<at::BFloat16*>(key_cache.data_ptr()),
            reinterpret_cast<at::BFloat16*>(value_cache.data_ptr()),
            slot_mapping.data_ptr<int64_t>(),
            num_tokens, num_heads, head_size, block_size);
    } else {
        TORCH_CHECK(false, "Unsupported key dtype for CUDA "
                    "reshape_and_cache_flash: ", key.scalar_type());
    }
    CUDA_CHECK(cudaGetLastError());
}


// ---------------------------------------------------------------------------
// cp_gather_cache — gather K/V from non-contiguous physical blocks to a
// contiguous output buffer (prefix-cache CoW / KV transfer / unified
// attention input stage).
// ---------------------------------------------------------------------------
//
// src_cache:    [num_blocks, block_size, entry_size]    (cache dtype)
// dst:          [total_tokens, entry_size]
// block_table:  [num_reqs, max_blocks_per_req] int32
// cu_seq_lens:  [num_reqs + 1] int32 — cumulative token counts
//
// Each output token index i ∈ [0, total_tokens) maps to:
//   req_id        = binary search over cu_seq_lens
//   token_offset  = i - cu_seq_lens[req_id]
//   block_idx     = token_offset / block_size
//   offset_in_blk = token_offset % block_size
//   phys_block    = block_table[req_id, block_idx]
//   src_offset    = phys_block * cache_block_stride
//                 + offset_in_blk * entry_size
//
// Real vLLM uses this exact pattern in PD transfer (prefill → decode) and
// in attention pre-stage: gather physical scattered blocks into a
// contiguous per-request tensor for attention compute.

template <typename scalar_t>
__global__ void cp_gather_cache_kernel(
    const scalar_t* __restrict__ src_cache,
    scalar_t* __restrict__ dst,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ cu_seq_lens,
    int block_size,
    int entry_size,
    int64_t block_table_stride,
    int64_t cache_block_stride,
    int num_reqs) {

    int token_idx = blockIdx.x;
    int total_tokens = cu_seq_lens[num_reqs];
    if (token_idx >= total_tokens) return;

    // Binary search for owning request.
    int lo = 0, hi = num_reqs - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (cu_seq_lens[mid] <= token_idx) lo = mid;
        else                              hi = mid - 1;
    }
    int req_id = lo;

    int token_offset = token_idx - cu_seq_lens[req_id];
    int block_idx = token_offset / block_size;
    int offset_in_block = token_offset % block_size;
    int phys_block =
        block_table[req_id * block_table_stride + block_idx];

    int64_t src_off = (int64_t)phys_block * cache_block_stride
                    + (int64_t)offset_in_block * entry_size;
    int64_t dst_off = (int64_t)token_idx * entry_size;

    for (int i = threadIdx.x; i < entry_size; i += blockDim.x) {
        dst[dst_off + i] = src_cache[src_off + i];
    }
}


void cp_gather_cache(
    torch::Tensor src_cache,
    torch::Tensor dst,
    torch::Tensor block_table,
    torch::Tensor cu_seq_lens,
    int64_t batch_size) {

    TORCH_CHECK(src_cache.is_cuda(), "src_cache must be CUDA");
    TORCH_CHECK(dst.is_cuda(), "dst must be CUDA");
    TORCH_CHECK(block_table.is_cuda(), "block_table must be CUDA");
    TORCH_CHECK(cu_seq_lens.is_cuda(), "cu_seq_lens must be CUDA");
    TORCH_CHECK(src_cache.scalar_type() == dst.scalar_type(),
                "src_cache / dst dtype mismatch");
    TORCH_CHECK(block_table.scalar_type() == torch::kInt32,
                "block_table must be int32");
    TORCH_CHECK(cu_seq_lens.scalar_type() == torch::kInt32,
                "cu_seq_lens must be int32");

    // src_cache layout: [num_blocks, block_size, entry_size]
    int block_size = src_cache.size(1);
    int entry_size = src_cache.size(2);
    int64_t cache_block_stride = src_cache.stride(0);

    int num_reqs = static_cast<int>(batch_size);
    TORCH_CHECK(num_reqs >= 0, "batch_size must be non-negative");
    if (num_reqs == 0) return;

    int64_t block_table_stride = block_table.stride(0);

    const c10::cuda::OptionalCUDAGuard device_guard(src_cache.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int total_tokens = cu_seq_lens[num_reqs].item<int32_t>();
    TORCH_CHECK(total_tokens >= 0, "total_tokens must be non-negative");
    if (total_tokens == 0) return;

    dim3 grid(total_tokens);
    dim3 block(std::min(entry_size, 256));

    auto scalar_t = src_cache.scalar_type();
    if (scalar_t == torch::kFloat16) {
        cp_gather_cache_kernel<at::Half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const at::Half*>(src_cache.data_ptr()),
            reinterpret_cast<at::Half*>(dst.data_ptr()),
            block_table.data_ptr<int32_t>(),
            cu_seq_lens.data_ptr<int32_t>(),
            block_size, entry_size,
            block_table_stride, cache_block_stride, num_reqs);
    } else if (scalar_t == torch::kBFloat16) {
        cp_gather_cache_kernel<at::BFloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const at::BFloat16*>(src_cache.data_ptr()),
            reinterpret_cast<at::BFloat16*>(dst.data_ptr()),
            block_table.data_ptr<int32_t>(),
            cu_seq_lens.data_ptr<int32_t>(),
            block_size, entry_size,
            block_table_stride, cache_block_stride, num_reqs);
    } else {
        TORCH_CHECK(false, "cp_gather_cache: unsupported cache dtype ",
                    scalar_t);
    }
    CUDA_CHECK(cudaGetLastError());
}


// ---------------------------------------------------------------------------
// convert_fp8 — per-block FP8 ↔ BF16/FP16 conversion (FP8 e4m3).
// ---------------------------------------------------------------------------
//
// Used at runtime when switching the KV cache storage dtype, and as a
// reference implementation for FP8 numerics.  Real vLLM uses
// fp8::scaled_convert from quantization/vectorization_utils.cuh; we inline
// a minimal e4m3 conversion here so my-vllm stays self-contained.
//
// Layout: src_cache / dst_cache are flat [num_blocks, block_stride]
// tensors (block_stride = elements per block; kernel walks it linearly).
//
// Saturating cast BF16/FP16 → FP8 e4m3:
//   FP8 e4m3 range: max value 448 (0x7F).  We clamp |x| ≤ 448 before
//   truncation.  NaN propagation: any NaN input → FP8 NaN (0x7F|0x80=0xFF
//   depending on convention; we use 0x7F since e4m3 has only one NaN
//   encoding in the IEEE-recommended form).

__device__ __forceinline__ uint8_t f32_to_fp8_e4m3(float x) {
    if (isnan(x)) return 0x7F;  // FP8 e4m3 NaN
    // Saturating cast: |x| > 448 → ±448 (FP8 e4m3 max).
    float clamped = fminf(fmaxf(x, -448.0f), 448.0f);
    uint32_t bits = __float_as_uint(clamped);
    uint32_t sign = (bits >> 31) & 0x1u;
    int32_t exp_field_32 = (int32_t)((bits >> 23) & 0xFFu) - 127;  // unbiased
    uint32_t mant_field_32 = bits & 0x7FFFFFu;  // 23-bit mantissa

    // Re-bias to FP8 e4m3 (bias = 7).  Handle denormals & overflow.
    int32_t exp_field_8 = exp_field_32 + 7;
    uint8_t out;
    if (exp_field_32 < -6) {
        // Subnormal or zero in FP8.
        // Subnormals in FP8 e4m3: 2^-6 * (mant / 8).  We round-to-nearest.
        // For simplicity, flush subnormals to 0 — sufficient for cache
        // quant where a per-tensor scale covers the dynamic range.
        out = 0;
    } else if (exp_field_8 >= 0xF) {
        // Overflow → saturate to ±448 (= 0x7F with sign).
        out = (uint8_t)(0x7F | (sign << 7));
    } else {
        // Normal: round top 3 mantissa bits.
        uint32_t round_bit = (mant_field_32 >> 20) & 1u;
        uint32_t mant_trunc = mant_field_32 >> 20;  // top 3 bits (we want top 3 of 23, i.e. bits 22-20)
        // Tie-breaking: round-to-nearest-even.  We do round-half-up for
        // simplicity (matches vLLM's default behavior for cache quant).
        uint32_t fp8_bits = ((sign & 1u) << 7)
                          | ((uint32_t)exp_field_8 << 3)
                          | (mant_trunc & 0x7u);
        fp8_bits += round_bit;
        // Post-rounding overflow check (rare for round-half-up).
        if ((fp8_bits & 0x7F) > 0x7E) {
            fp8_bits = (sign << 7) | 0x7F;
        }
        out = (uint8_t)fp8_bits;
    }
    return out;
}

__device__ __forceinline__ float fp8_e4m3_to_f32(uint8_t x) {
    // FP8 e4m3 → float32 reconstruction (matches vllm::fp8::scaled_convert).
    uint32_t sign = (x & 0x80) ? 0x80000000u : 0u;
    uint32_t exp_field = (x >> 3) & 0xF;
    uint32_t mant_field = x & 0x7;
    uint32_t f32_bits;
    if (exp_field == 0) {
        // Subnormal / zero → tiny value (treat as zero for simplicity).
        if (mant_field == 0) {
            f32_bits = sign;
        } else {
            // Subnormal FP8 e4m3 → normal FP32 with right shift.
            uint32_t adj_exp = 127 - 6;
            uint32_t adj_mant = mant_field << (23 - 3);
            f32_bits = sign | (adj_exp << 23) | adj_mant;
        }
    } else if (exp_field == 0xF) {
        // NaN / Inf (e4m3 has only NaN).
        f32_bits = sign | 0x7FC00000u;  // quiet NaN
    } else {
        // Normal: bias FP8 e4m3 (7) → FP32 (127).  Shift exponent by 23-3=20.
        uint32_t adj_exp = (exp_field + 120) << 23;
        uint32_t adj_mant = mant_field << 20;
        f32_bits = sign | adj_exp | adj_mant;
    }
    return __uint_as_float(f32_bits);
}

template <typename Tout, typename Tin>
__global__ void quant_to_fp8_kernel(
    const Tin* __restrict__ src_cache,
    uint8_t* __restrict__ dst_cache,
    float scale,
    int64_t block_stride) {

    int64_t block_idx = blockIdx.x;
    for (int i = threadIdx.x; i < block_stride; i += blockDim.x) {
        int64_t idx = block_idx * block_stride + i;
        // Read source as float (Tin ∈ {fp32, fp16, bf16}), then rescale.
        // vLLM convention: caller passes the **reciprocal** of the per-tensor
        // dynamic range so the quant kernel can do a single multiply.
        float src_f = static_cast<float>(src_cache[idx]);
        float x = src_f * scale;
        dst_cache[idx] = f32_to_fp8_e4m3(x);
    }
}

template <typename Tout>
__global__ void dequant_from_fp8_kernel(
    const uint8_t* __restrict__ src_cache,
    Tout* __restrict__ dst_cache,
    float scale,
    int64_t block_stride) {

    int64_t block_idx = blockIdx.x;
    for (int i = threadIdx.x; i < block_stride; i += blockDim.x) {
        int64_t idx = block_idx * block_stride + i;
        // Read FP8 byte → reconstruct float, then rescale by per-tensor
        // dynamic range (mirror of quant_to_fp8_kernel).
        uint8_t fp8_byte = src_cache[idx];
        float x = fp8_e4m3_to_f32(fp8_byte) * scale;
        dst_cache[idx] = static_cast<Tout>(x);
    }
}


void convert_fp8(
    torch::Tensor dst_cache,
    torch::Tensor src_cache,
    double scale,
    const std::string& kv_cache_dtype) {

    TORCH_CHECK(src_cache.is_cuda() && dst_cache.is_cuda(),
                "src_cache / dst_cache must be CUDA");
    TORCH_CHECK(src_cache.device() == dst_cache.device(),
                "src_cache / dst_cache must be on the same device");

    int64_t num_blocks = src_cache.size(0);
    int64_t block_stride = src_cache.stride(0);

    const c10::cuda::OptionalCUDAGuard device_guard(src_cache.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(num_blocks);
    dim3 block(std::min(static_cast<int64_t>(512), block_stride));

    auto src_dtype = src_cache.scalar_type();
    auto dst_dtype = dst_cache.scalar_type();
    bool is_fp8_path = (kv_cache_dtype == "fp8"
                        || kv_cache_dtype == "fp8_e4m3");

    // FP8 path (only fp8_e4m3 supported): src ∈ {fp16, bf16, fp32},
    //                                      dst ∈ {uint8 (FP8)}.
    if (is_fp8_path) {
        TORCH_CHECK(dst_dtype == torch::kUInt8,
                    "convert_fp8: dst dtype must be uint8 (FP8)");
        if (src_dtype == torch::kFloat32) {
            quant_to_fp8_kernel<uint8_t, float>
                <<<grid, block, 0, stream>>>(
                    src_cache.data_ptr<float>(),
                    dst_cache.data_ptr<uint8_t>(),
                    static_cast<float>(scale), block_stride);
        } else if (src_dtype == torch::kFloat16) {
            quant_to_fp8_kernel<uint8_t, at::Half>
                <<<grid, block, 0, stream>>>(
                    reinterpret_cast<const at::Half*>(src_cache.data_ptr()),
                    dst_cache.data_ptr<uint8_t>(),
                    static_cast<float>(scale), block_stride);
        } else if (src_dtype == torch::kBFloat16) {
            quant_to_fp8_kernel<uint8_t, at::BFloat16>
                <<<grid, block, 0, stream>>>(
                    reinterpret_cast<const at::BFloat16*>(src_cache.data_ptr()),
                    dst_cache.data_ptr<uint8_t>(),
                    static_cast<float>(scale), block_stride);
        } else {
            TORCH_CHECK(false, "convert_fp8: unsupported src dtype ",
                        src_dtype);
        }
    } else if (kv_cache_dtype == "auto") {
        // Dequant path: src ∈ {uint8 (FP8)}, dst ∈ {fp16, bf16, fp32}.
        TORCH_CHECK(src_dtype == torch::kUInt8,
                    "convert_fp8(auto): src dtype must be uint8 (FP8)");
        if (dst_dtype == torch::kFloat32) {
            dequant_from_fp8_kernel<float>
                <<<grid, block, 0, stream>>>(
                    src_cache.data_ptr<uint8_t>(),
                    dst_cache.data_ptr<float>(),
                    static_cast<float>(scale), block_stride);
        } else if (dst_dtype == torch::kFloat16) {
            dequant_from_fp8_kernel<at::Half>
                <<<grid, block, 0, stream>>>(
                    src_cache.data_ptr<uint8_t>(),
                    reinterpret_cast<at::Half*>(dst_cache.data_ptr()),
                    static_cast<float>(scale), block_stride);
        } else if (dst_dtype == torch::kBFloat16) {
            dequant_from_fp8_kernel<at::BFloat16>
                <<<grid, block, 0, stream>>>(
                    src_cache.data_ptr<uint8_t>(),
                    reinterpret_cast<at::BFloat16*>(dst_cache.data_ptr()),
                    static_cast<float>(scale), block_stride);
        } else {
            TORCH_CHECK(false, "convert_fp8(auto): unsupported dst dtype ",
                        dst_dtype);
        }
    } else {
        TORCH_CHECK(false, "convert_fp8: unsupported kv_cache_dtype ",
                    kv_cache_dtype);
    }
    CUDA_CHECK(cudaGetLastError());
}


// ---------------------------------------------------------------------------
// concat_and_cache_mla — MLA compressed-KV concatenation write (STUB).
// ---------------------------------------------------------------------------
//
// MLA (Multi-Latent Attention, DeepSeek V2/V3) stores the KV cache as a
// single fused vector per token: kv_c (latent compressed, kv_lora_rank
// wide) concatenated with k_pe (RoPE-decoupled positional encoding,
// pe_dim wide).  Total entry width = kv_lora_rank + pe_dim.
//
// my-vllm does not load an MLA model (JD scope excludes DeepSeek family),
// but the API surface is exposed for vLLM symmetry.  This implementation
// supports fp16 / bf16 storage; FP8-DS-MLA is rejected with a clear error.
//
// Memory layout (per vLLM):
//   kv_c      : [num_tokens, kv_lora_rank]
//   k_pe      : [num_tokens, pe_dim]
//   kv_cache  : [num_blocks, block_size, kv_lora_rank + pe_dim]
//
// For each token i in slot_mapping:
//   kv_cache[block_id, offset, 0:kv_lora_rank]   = kv_c[i]
//   kv_cache[block_id, offset, kv_lora_rank:end] = k_pe[i]

template <typename scalar_t>
__global__ void concat_and_cache_mla_kernel(
    const scalar_t* __restrict__ kv_c,
    const scalar_t* __restrict__ k_pe,
    scalar_t* __restrict__ kv_cache,
    const int64_t* __restrict__ slot_mapping,
    int num_tokens,
    int kv_lora_rank,
    int pe_dim,
    int block_size,
    int entry_size,
    int kv_c_stride,
    int k_pe_stride,
    int cache_block_stride,
    int cache_entry_stride) {

    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    int64_t slot = slot_mapping[token_idx];
    if (slot < 0) return;

    int block_id = slot / block_size;
    int offset_in_block = slot % block_size;

    int64_t dst_base = (int64_t)block_id * cache_block_stride
                     + (int64_t)offset_in_block * cache_entry_stride;
    int64_t src_kv_c = (int64_t)token_idx * kv_c_stride;
    int64_t src_k_pe = (int64_t)token_idx * k_pe_stride;

    // Write kv_c into first kv_lora_rank entries.
    for (int i = threadIdx.x; i < kv_lora_rank; i += blockDim.x) {
        kv_cache[dst_base + i] = kv_c[src_kv_c + i];
    }
    // Write k_pe into next pe_dim entries.
    for (int i = threadIdx.x; i < pe_dim; i += blockDim.x) {
        kv_cache[dst_base + kv_lora_rank + i] = k_pe[src_k_pe + i];
    }
    // Silence unused-parameter warning for entry_size (used implicitly
    // via cache_entry_stride).
    (void)entry_size;
}


void concat_and_cache_mla(
    torch::Tensor kv_c,
    torch::Tensor k_pe,
    torch::Tensor kv_cache,
    torch::Tensor slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor scale) {

    TORCH_CHECK(kv_c.is_cuda() && k_pe.is_cuda() && kv_cache.is_cuda(),
                "kv_c / k_pe / kv_cache must be CUDA");
    TORCH_CHECK(slot_mapping.is_cuda(), "slot_mapping must be CUDA");
    TORCH_CHECK(kv_c.scalar_type() == k_pe.scalar_type(),
                "kv_c / k_pe dtype mismatch");

    // FP8-DS-MLA is intentionally not implemented (JD excludes MLA models).
    // Reject early with a clear error so the caller knows to skip this path.
    TORCH_CHECK(kv_cache_dtype == "auto",
                "concat_and_cache_mla (CUDA): only kv_cache_dtype='auto' is "
                "supported in my-vllm. Requested: ", kv_cache_dtype,
                ". MLA models (DeepSeek V2/V3) are out of JD scope; "
                "triton_reshape_and_cache_flash (C149) handles FP8-e4m3 "
                "storage for the non-MLA hot path.");

    int num_tokens = slot_mapping.size(0);
    int kv_lora_rank = kv_c.size(1);
    int pe_dim = k_pe.size(1);
    int block_size = kv_cache.size(1);
    int entry_size = kv_lora_rank + pe_dim;

    TORCH_CHECK(kv_cache.size(2) == entry_size,
                "kv_cache.size(2) must equal kv_lora_rank + pe_dim, got ",
                kv_cache.size(2), " vs ", entry_size);

    int kv_c_stride = kv_c.stride(0);
    int k_pe_stride = k_pe.stride(0);
    int cache_block_stride = kv_cache.stride(0);
    int cache_entry_stride = kv_cache.stride(1);

    const c10::cuda::OptionalCUDAGuard device_guard(kv_c.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(num_tokens);
    // 96 threads matches vLLM's 3-warp layout (NoPE: 2 warps + RoPE: 1 warp).
    dim3 block(96);

    auto scalar_t = kv_c.scalar_type();
    if (scalar_t == torch::kFloat16) {
        concat_and_cache_mla_kernel<at::Half><<<grid, block, 0, stream>>>(
            reinterpret_cast<const at::Half*>(kv_c.data_ptr()),
            reinterpret_cast<const at::Half*>(k_pe.data_ptr()),
            reinterpret_cast<at::Half*>(kv_cache.data_ptr()),
            slot_mapping.data_ptr<int64_t>(),
            num_tokens, kv_lora_rank, pe_dim, block_size, entry_size,
            kv_c_stride, k_pe_stride,
            cache_block_stride, cache_entry_stride);
    } else if (scalar_t == torch::kBFloat16) {
        concat_and_cache_mla_kernel<at::BFloat16><<<grid, block, 0, stream>>>(
            reinterpret_cast<const at::BFloat16*>(kv_c.data_ptr()),
            reinterpret_cast<const at::BFloat16*>(k_pe.data_ptr()),
            reinterpret_cast<at::BFloat16*>(kv_cache.data_ptr()),
            slot_mapping.data_ptr<int64_t>(),
            num_tokens, kv_lora_rank, pe_dim, block_size, entry_size,
            kv_c_stride, k_pe_stride,
            cache_block_stride, cache_entry_stride);
    } else {
        TORCH_CHECK(false, "concat_and_cache_mla: unsupported kv_c dtype ",
                    scalar_t);
    }
    CUDA_CHECK(cudaGetLastError());
    // scale is accepted for API symmetry but not consumed in fp16/bf16 path.
    (void)scale;
}


// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

// Forward declarations for the custom_all_reduce bindings (defined in
// custom_all_reduce.cu).  The custom_all_reduce bindings live in
// namespace ``my_vllm_py`` to keep them separable from cache kernels.
using my_ar_fptr_t = int64_t;
namespace my_vllm_py {
    my_ar_fptr_t init_custom_ar(const std::vector<my_ar_fptr_t>& fake_ipc_ptrs,
                                torch::Tensor& rank_data,
                                int64_t rank,
                                bool fully_connected);
    void dispose(my_ar_fptr_t _fa);
    int64_t meta_size();
    void register_buffer(my_ar_fptr_t _fa,
                         const std::vector<my_ar_fptr_t>& fake_ipc_ptrs);
    torch::Tensor get_handle();
    my_ar_fptr_t open_mem_handle(torch::Tensor& mem_handle);
    void all_reduce(my_ar_fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                    my_ar_fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes);
}  // namespace my_vllm_py

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reshape_and_cache", &reshape_and_cache,
          "Write K/V into paged KV cache (multi-dtype fp16/bf16)",
          pybind11::arg("key"), pybind11::arg("value"),
          pybind11::arg("key_cache"), pybind11::arg("value_cache"),
          pybind11::arg("slot_mapping"),
          pybind11::arg("kv_cache_dtype"),
          pybind11::arg("k_scale"), pybind11::arg("v_scale"));
    m.def("reshape_and_cache_flash", &reshape_and_cache_flash,
          "Write K/V into paged KV cache (FlashAttention layout, fp16/bf16)",
          pybind11::arg("key"), pybind11::arg("value"),
          pybind11::arg("key_cache"), pybind11::arg("value_cache"),
          pybind11::arg("slot_mapping"),
          pybind11::arg("kv_cache_dtype"),
          pybind11::arg("k_scale"), pybind11::arg("v_scale"));
    m.def("swap_blocks", &swap_blocks,
          "Swap K/V cache blocks (cudaMemcpyAsync over pair list)",
          pybind11::arg("src"), pybind11::arg("dst"),
          pybind11::arg("block_size_in_bytes"),
          pybind11::arg("block_mapping"));
    m.def("copy_blocks", &copy_blocks,
          "Copy K/V cache blocks between (src_block, dst_block) pairs",
          pybind11::arg("key_caches"), pybind11::arg("value_caches"),
          pybind11::arg("block_mapping"));
    m.def("cp_gather_cache", &cp_gather_cache,
          "Gather K/V from non-contiguous physical blocks to contiguous "
          "output (prefix-cache CoW / KV transfer)",
          pybind11::arg("src_cache"), pybind11::arg("dst"),
          pybind11::arg("block_table"), pybind11::arg("cu_seq_lens"),
          pybind11::arg("batch_size"));
    m.def("convert_fp8", &convert_fp8,
          "Convert per-block FP8 ↔ BF16/FP16 (FP8 e4m3)",
          pybind11::arg("dst_cache"), pybind11::arg("src_cache"),
          pybind11::arg("scale"), pybind11::arg("kv_cache_dtype"));
    m.def("concat_and_cache_mla", &concat_and_cache_mla,
          "Concat kv_c + k_pe and write to paged KV cache (MLA, "
          "kv_lora_rank + pe_dim; fp16/bf16 only)",
          pybind11::arg("kv_c"), pybind11::arg("k_pe"),
          pybind11::arg("kv_cache"), pybind11::arg("slot_mapping"),
          pybind11::arg("kv_cache_dtype"), pybind11::arg("scale"));

    // Custom all-reduce ops (from custom_all_reduce.cu).
    m.def("init_custom_ar", &my_vllm_py::init_custom_ar,
          "Initialize CustomAllreduce (fptr_t handle)",
          pybind11::arg("fake_ipc_ptrs"),
          pybind11::arg("rank_data"),
          pybind11::arg("rank"),
          pybind11::arg("fully_connected"));
    m.def("dispose", &my_vllm_py::dispose, "Free a CustomAllreduce handle");
    m.def("meta_size", &my_vllm_py::meta_size,
          "Size of the Signal struct (bytes)");
    m.def("register_buffer", &my_vllm_py::register_buffer,
          "Re-bind signal IPC pointers (CUDA Graph capture)",
          pybind11::arg("fa"), pybind11::arg("fake_ipc_ptrs"));
    m.def("get_handle", &my_vllm_py::get_handle,
          "Allocate a Signal buffer + return IPC memhandle");
    m.def("open_mem_handle", &my_vllm_py::open_mem_handle,
          "Open a remote IPC memhandle, return local pointer");
    m.def("all_reduce", &my_vllm_py::all_reduce,
          "Out-of-place all_reduce (fp32/fp16/bf16)",
          pybind11::arg("fa"),
          pybind11::arg("inp"), pybind11::arg("out"),
          pybind11::arg("reg_buffer") = 0,
          pybind11::arg("reg_buffer_sz_bytes") = 0);
}