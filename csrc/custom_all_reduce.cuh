// csrc/custom_all_reduce.cuh — minimal custom all-reduce header for
// my-vllm.  Mirrors the public API surface of vLLM's
// custom_all_reduce.cuh (init_custom_ar / all_reduce / dispose /
// register_buffer / get_graph_buffer_ipc_meta / allocate_shared_buffer /
// open_mem_handle / free_shared_buffer), but uses a simplified
// in-process ring for the allreduce itself (no separate "1-shot" vs
// "2-shot" paths, no CUDA Graph integration — those are vLLM-specific
// optimizations that depend on torch::stable ABI and are out of JD scope).
//
// The architecture is:
//
//   * Each rank owns a fixed-size "signal" buffer (cudaIpcMemHandle_t).
//     Signalling is done with cudaStreamSynchronize (sufficient for
//     non-overlapping ranks on same host; cross-node IPC is out of
//     scope here — that path goes through NCCL).
//
//   * For world_size=1, the implementation short-circuits to identity
//     (memcpy from inp to out).
//
//   * For world_size=2..8, it does a single-pass ring all-reduce:
//
//       step k (k=0..world_size-1):
//         if rank == k:  send chunk k to (rank+1) % world_size
//         else:          recv chunk (rank-1+world_size) % world_size from (rank-1+world_size) % world_size
//                        and add to local chunk
//
//     After world_size-1 steps every rank holds the full sum.  This
//     is the textbook ring all-reduce with `world_size - 1` steps
//     (Patariasuk & Yuan 2009).
//
// Reference: vllm/csrc/custom_all_reduce.cuh (631 lines, full version).

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace my_vllm {

// World size cap — vLLM's CustomAllreduce also caps at 8 (the
// ring kernel's shared-memory buffer doesn't scale beyond that on
// current GPUs).
constexpr int kMaxWorldSize = 8;

// fptr_t is the opaque handle that crosses the pybind boundary.
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t), "fptr_t size mismatch");

// One slot per peer rank; the IPC memory handle is set up at init
// time via allocate_shared_buffer / open_mem_handle.
struct Signal {
    int32_t flag[kMaxWorldSize];
};

// Per-rank state.  Holds the GPU-side pointer to each peer's signal
// buffer (after cudaIpcOpenMemHandle) and the local registered data
// buffer (used for the allreduce scratch space).
struct CustomAllreduce {
public:
    CustomAllreduce(Signal** ipc_ptrs,
                   void* rank_data,
                   int rank_data_elems,
                   int rank,
                   int world_size,
                   bool fully_connected);
    ~CustomAllreduce();

    // Out-of-place allreduce: read+reduce from `inp` (or the
    // registered buffer if reg_buffer != nullptr), write sum to `out`.
    template <typename T>
    void allreduce(cudaStream_t stream, T* inp, T* out, int numel);

    // Register a per-call scratch buffer (for CUDA Graph capture).
    void register_buffer(void** ipc_ptrs);

    // Graph-buffer IPC helpers (not implemented — CUDA Graph
    // integration out of scope for my-vllm).
    std::tuple<std::vector<int8_t>, std::vector<int64_t>>
    get_graph_buffer_ipc_meta();
    void register_graph_buffers(const std::vector<std::string>& handles,
                               const std::vector<std::vector<int64_t>>& offsets);

    int rank_;
    int world_size_;
    bool fully_connected_;

    // Signal buffers (one per peer rank, plus local).
    Signal* signals_[kMaxWorldSize];
    // Local data buffer (used as scratch for in-place add during
    // the ring reduction step).
    void* rank_data_;
    int rank_data_elems_;
};

}  // namespace my_vllm
