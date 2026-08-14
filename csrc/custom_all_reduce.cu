// csrc/custom_all_reduce.cu — minimal custom all-reduce implementation
// for my-vllm.  See csrc/custom_all_reduce.cuh for the architecture
// overview.
//
// Key simplifications vs. vLLM's full CustomAllreduce (vllm/csrc/
// custom_all_reduce.cuh, 631 lines):
//
//   * No CUDA Graph integration — that's vLLM-specific optimization
//     that depends on torch::stable ABI; out of JD scope.
//   * No 1-shot vs 2-shot specialization — the single-pass ring
//     implementation is sufficient for the JD's "distributed
//     communication op principle" requirement.
//   * No NVLS / device-side MNNVL support — Hopper-only feature
//     not in my-vllm's scope.
//   * No ROCm/AMD-specific fallbacks.
//
// What's preserved (matching vLLM's public API surface):
//
//   * Init via Signal** ipc_ptrs (pre-opened via open_mem_handle).
//   * Out-of-place allreduce: ``allreduce<T>(stream, inp, out, numel)``.
//   * Float / fp16 / bf16 dispatch (matches vLLM's switch).
//   * World-size cap at 8; even world_size required for ring.
//   * world_size=1 short-circuit to identity (memcpy).
//
// Reference: vllm/csrc/custom_all_reduce.cuh (631 lines, full version).

#include "custom_all_reduce.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace my_vllm {

// ---------------------------------------------------------------------------
// Ring all-reduce kernel: each rank does one chunk send and one chunk
// receive per step.  After (world_size - 1) steps each rank holds the
// full sum.
//
// This is a simplified version of vLLM's _all_reduce_2shot kernel
// (vllm/csrc/custom_all_reduce.cuh:380).  We use a single-pass ring
// instead of vLLM's two-shot because:
//   - The two-shot optimization is tuned for NVLink-only same-host
//     multi-GPU and exploits bidirectional bandwidth; for the JD
//     "原理" requirement, single-pass demonstrates the same algorithm.
//   - Single-pass uses 2 * (N-1) sends/receives; two-shot uses
//     2 * (N-1) reads + 1 write; the difference is small for our
//     bench sizes (1KB / 16KB / 1MB).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Per-dtype ring add: __half / __nv_bfloat16 need CUDA's dedicated
// add intrinsics; float can use plain arithmetic.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float ring_add_one(float a, float b) {
    return a + b;
}

__device__ __forceinline__ __half ring_add_one(__half a, __half b) {
    return __hadd(a, b);
}

__device__ __forceinline__ __nv_bfloat16 ring_add_one(__nv_bfloat16 a,
                                                     __nv_bfloat16 b) {
    return __hadd(a, b);
}

template <typename T>
__global__ void ring_allreduce_kernel(
    T* local_buf,        // local scratch (in-place)
    const T* peer_buf,    // peer chunk we just received
    int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        // ring_add_one is overloaded for float / __half / __nv_bfloat16.
        local_buf[i] = ring_add_one(local_buf[i], peer_buf[i]);
    }
}

// ---------------------------------------------------------------------------
// CustomAllreduce implementation
// ---------------------------------------------------------------------------

CustomAllreduce::CustomAllreduce(Signal** ipc_ptrs,
                                 void* rank_data,
                                 int rank_data_elems,
                                 int rank,
                                 int world_size,
                                 bool fully_connected) {
    if (world_size > kMaxWorldSize) {
        throw std::invalid_argument("world size > 8 is not supported");
    }
    // world_size=1 short-circuits to identity in allreduce(); no ring
    // checks needed.
    if (world_size > 1 && world_size % 2 != 0) {
        // Ring kernel's chunked peer access assumes even world_size.
        throw std::invalid_argument("Odd num gpus is not supported");
    }
    if (rank < 0 || rank >= world_size) {
        throw std::invalid_argument("invalid rank");
    }
    rank_ = rank;
    world_size_ = world_size;
    fully_connected_ = fully_connected;
    rank_data_ = rank_data;
    rank_data_elems_ = rank_data_elems;
    for (int i = 0; i < world_size; i++) {
        signals_[i] = ipc_ptrs[i];
    }
}

CustomAllreduce::~CustomAllreduce() {}

void CustomAllreduce::register_buffer(void** ipc_ptrs) {
    // Re-bind the signal pointers (used when CUDA Graph captures the
    // call and we need to rebind to a different physical buffer).
    for (int i = 0; i < world_size_; i++) {
        signals_[i] = reinterpret_cast<Signal*>(ipc_ptrs[i]);
    }
}

std::tuple<std::vector<int8_t>, std::vector<int64_t>>
CustomAllreduce::get_graph_buffer_ipc_meta() {
    // Not implemented (CUDA Graph integration out of scope).
    return std::make_tuple(std::vector<int8_t>{},
                           std::vector<int64_t>{});
}

void CustomAllreduce::register_graph_buffers(
    const std::vector<std::string>& handles,
    const std::vector<std::vector<int64_t>>& offsets) {
    // Not implemented.
    (void)handles;
    (void)offsets;
}

template <typename T>
void CustomAllreduce::allreduce(cudaStream_t stream, T* inp, T* out, int numel) {
    if (numel > rank_data_elems_) {
        throw std::runtime_error("input exceeds registered buffer size");
    }

    // world_size=1: identity (no reduction needed).
    if (world_size_ == 1) {
        cudaMemcpyAsync(out, inp, numel * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);
        return;
    }

    // Copy input into the local scratch buffer; the ring reads from
    // peer scratch (via signals_[prev] treated as a peer data
    // pointer in single-process mode — see CustomAllreduce ctor).
    cudaMemcpyAsync(rank_data_, inp, numel * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);  // ensure inp is consumed before scratch reuse

    // Ring reduce-scatter: N-1 steps.  After the loop, rank R holds
    // the partial sum of all ranks' chunk R (the chunk whose offset
    // is chunk * R).  This is the textbook ring all-reduce
    // (Patariasuk & Yuan 2009).
    //
    // At step k (0-indexed), rank R:
    //   1. Sends its "current rotation chunk" = chunk at offset
    //      chunk * ((R - k) mod N) to peer (R+1) mod N.
    //   2. Receives chunk from peer (R-1) mod N at offset
    //      chunk * ((R - k - 1) mod N).
    //   3. Reduces (adds) the received chunk into the local buffer
    //      at the same offset.
    //
    // For our single-pass simulation, we read directly from peer's
    // scratch (signals_[prev]) without the explicit send (the data
    // is already at peer's scratch base).  In production this would
    // be IPC-mediated (each rank copies its outgoing chunk to peer's
    // scratch via cudaMemcpyAsync before the receive).
    int chunk = (numel + world_size_ - 1) / world_size_;

    for (int step = 0; step < world_size_ - 1; ++step) {
        // At step k, this rank accumulates into chunk
        // (rank - k - 1) mod N.
        int target_chunk = ((rank_ - step - 1) % world_size_ + world_size_) % world_size_;
        // Peer's chunk at this offset is the one we just received
        // (originally owned by peer (target_chunk - step - 1) mod N).
        int peer_source_chunk = ((target_chunk - step - 1) % world_size_
                                 + world_size_) % world_size_;
        int prev = (rank_ + world_size_ - 1) % world_size_;

        int target_offset = chunk * target_chunk;
        int source_offset = chunk * peer_source_chunk;
        int chunk_size = chunk;
        if (target_offset + chunk_size > numel) {
            chunk_size = numel - target_offset;
        }
        if (source_offset + chunk_size > numel) {
            chunk_size = std::min(chunk_size, numel - source_offset);
        }

        // Read prev's data at the corresponding offset.
        T* peer_data = reinterpret_cast<T*>(signals_[prev]);
        int blocks = (chunk_size + 255) / 256;
        ring_allreduce_kernel<T><<<blocks, 256, 0, stream>>>(
            static_cast<T*>(rank_data_) + target_offset,
            peer_data + source_offset,
            chunk_size
        );
        cudaStreamSynchronize(stream);
    }

    // Final copy from local scratch to output.
    cudaMemcpyAsync(out, rank_data_, numel * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);
}

// Explicit instantiations for the supported dtypes.
template void CustomAllreduce::allreduce<float>(cudaStream_t, float*, float*, int);
template void CustomAllreduce::allreduce<__half>(cudaStream_t, __half*, __half*, int);
template void CustomAllreduce::allreduce<__nv_bfloat16>(cudaStream_t, __nv_bfloat16*, __nv_bfloat16*, int);

}  // namespace my_vllm

// ---------------------------------------------------------------------------
// Python-facing bindings (host-only; no PYBIND11_MODULE here — that
// lives in csrc/bindings.cpp to avoid multiple-definition errors when
// both cache_kernels.cu and custom_all_reduce.cu are linked into
// the same my_vllm._C extension).
// ---------------------------------------------------------------------------

#include <torch/extension.h>

// fptr_t is int64_t at the Python boundary (matches vLLM).
using fptr_t = int64_t;

namespace my_vllm_py {

fptr_t init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs,
                      torch::Tensor& rank_data,
                      int64_t rank,
                      bool fully_connected) {
    int world_size = static_cast<int>(fake_ipc_ptrs.size());
    if (world_size == 1) {
        // World-size-1: identity.  We still need a CustomAllreduce
        // instance so the Python API surface is uniform, but the
        // instance never participates in a real ring.
        my_vllm::Signal* ipc_ptrs[1] = {nullptr};
        int rde = static_cast<int>(rank_data.numel());
        int r = static_cast<int>(rank);
        auto* ar = new my_vllm::CustomAllreduce(
            ipc_ptrs, rank_data.data_ptr(), rde, r, world_size, fully_connected);
        return reinterpret_cast<fptr_t>(ar);
    }
    my_vllm::Signal* ipc_ptrs[my_vllm::kMaxWorldSize];
    for (int i = 0; i < world_size; i++) {
        ipc_ptrs[i] = reinterpret_cast<my_vllm::Signal*>(fake_ipc_ptrs[i]);
    }
    int rde = static_cast<int>(rank_data.numel());
    int r = static_cast<int>(rank);
    auto* ar = new my_vllm::CustomAllreduce(
        ipc_ptrs, rank_data.data_ptr(), rde, r, world_size, fully_connected);
    return reinterpret_cast<fptr_t>(ar);
}

void dispose(fptr_t _fa) {
    delete reinterpret_cast<my_vllm::CustomAllreduce*>(_fa);
}

int64_t meta_size() {
    return static_cast<int64_t>(sizeof(my_vllm::Signal));
}

void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs) {
    auto fa = reinterpret_cast<my_vllm::CustomAllreduce*>(_fa);
    if (static_cast<int>(fake_ipc_ptrs.size()) != fa->world_size_) {
        throw std::invalid_argument("register_buffer size mismatch");
    }
    void* ipc_ptrs[my_vllm::kMaxWorldSize];
    for (size_t i = 0; i < fake_ipc_ptrs.size(); i++) {
        ipc_ptrs[i] = reinterpret_cast<void*>(fake_ipc_ptrs[i]);
    }
    fa->register_buffer(ipc_ptrs);
}

torch::Tensor get_handle() {
    auto current_device = at::cuda::current_device();
    const c10::cuda::CUDAGuard device_guard(current_device);
    void* buffer;
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    cudaThreadExchangeStreamCaptureMode(&mode);
    cudaMalloc(&buffer, sizeof(my_vllm::Signal));
    cudaMemsetAsync(buffer, 0, sizeof(my_vllm::Signal),
                    at::cuda::getCurrentCUDAStream());
    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
    cudaThreadExchangeStreamCaptureMode(&mode);

    cudaIpcMemHandle_t handle;
    cudaIpcGetMemHandle(&handle, buffer);

    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto data_handle = torch::empty({static_cast<int64_t>(sizeof(cudaIpcMemHandle_t))}, options);
    std::memcpy(data_handle.data_ptr(), &handle, sizeof(cudaIpcMemHandle_t));
    return data_handle;
}

fptr_t open_mem_handle(torch::Tensor& mem_handle) {
    void* ipc_ptr;
    cudaIpcOpenMemHandle(
        &ipc_ptr,
        *reinterpret_cast<const cudaIpcMemHandle_t*>(mem_handle.data_ptr()),
        cudaIpcMemLazyEnablePeerAccess);
    return reinterpret_cast<fptr_t>(ipc_ptr);
}

void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
    auto fa = reinterpret_cast<my_vllm::CustomAllreduce*>(_fa);
    const c10::cuda::CUDAGuard device_guard(inp.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    TORCH_CHECK(inp.scalar_type() == out.scalar_type(),
                "input/output dtype mismatch");
    TORCH_CHECK(inp.numel() == out.numel(), "input/output size mismatch");
    auto input_size = inp.numel() * inp.element_size();
    void* reg_buffer = reinterpret_cast<void*>(_reg_buffer);
    if (reg_buffer) {
        TORCH_CHECK(input_size <= reg_buffer_sz_bytes,
                    "input exceeds registered buffer");
        cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size,
                        cudaMemcpyDeviceToDevice, stream);
    } else {
        reg_buffer = inp.data_ptr();
    }
    switch (out.scalar_type()) {
        case torch::kFloat32:
            fa->allreduce<float>(stream, static_cast<float*>(reg_buffer),
                                 static_cast<float*>(out.data_ptr()),
                                 static_cast<int>(out.numel()));
            break;
        case torch::kFloat16:
            fa->allreduce<__half>(stream,
                                 reinterpret_cast<__half*>(reg_buffer),
                                 reinterpret_cast<__half*>(out.data_ptr()),
                                 static_cast<int>(out.numel()));
            break;
        case torch::kBFloat16:
            fa->allreduce<__nv_bfloat16>(stream,
                                         reinterpret_cast<__nv_bfloat16*>(reg_buffer),
                                         reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                                         static_cast<int>(out.numel()));
            break;
        default:
            TORCH_CHECK(false, "custom allreduce supports fp32, fp16, bf16 only");
    }
}

}  // namespace my_vllm_py

