# AGENTS.md

## Project identity

my-vllm is an **engineering subset** of the vLLM V1 serving runtime. It takes real vLLM as the reference codebase and reproduces its engineering organization, abstraction boundaries, core data structures, execution paths, and GPU Runtime interfaces as faithfully as possible. Features removed are limited to business/ecosystem functionality unrelated to career goals.

**Career target**: AI Systems / AI Infrastructure Engineer → LLM Inference / Distributed Inference.
**Focus**: GPU Performance + Runtime + Distributed Systems.

my-vllm's final form:

```
real vLLM
    │
    ├── career-relevant → reproduce as faithfully as possible
    ├── runtime-relevant → reproduce as faithfully as possible
    └── unrelated to career → delete
             ↓
        my-vLLM（strict engineering subset of vLLM）
```

Every file in `my_vllm/` must correspond to a real file in `vllm/`. No invented patterns or APIs.

**Previous history**: Stages C000→C098 were implemented under a prior "educational simulation" model. That history is preserved (never squash/rebase). The project now enters a new phase of **engineering alignment** (C110+) — see `Prompt/Engineer-Spec/README.md` for the full specification.

**Current status**: C243. **Phase 4 (Distributed Runtime 强化, C240-C249)**: 4/10 commits closed (C240 All-to-All, C241 Symmetric Memory, C242 SHM Broadcast, C243 Weight Transfer). **Phase 1 (硬性 + GPU Runtime 深度, C200-C214) complete**: 7/7 commits closed. **Phase 2 (架构深度 + 加分, C220-C224) complete**: 4/4 done. **Phase 3 (收尾, C230-C233) complete**: 4/4 done — **Phase 1+2+3 closed** ✅.

**Project metrics (post-C243)**:

| Metric                                            | Count                |
|--------------------------------------------------|----------------------|
| `my_vllm/` Python files                          | 201                  |
| `my_vllm/` Python LOC                            | 29,400               |
| `csrc/` C/CUDA files                              | 5 (2 vendored)       |
| CUDA extensions built by `setup.py`              | 2 (`my_vllm._C` + `my_vllm.cumem_allocator`) |
| Test files                                       | 63                   |
| Phase 1+2+3+4 test pass rate (C200-C243 scope)  | 398 passed + 4 skipped |
| Engineering notes (`docs/notes/`)                | 12 / 4,478 lines     |
| Snapshot files (C200+ era, excl prep/fix)        | 19 (C200-C233 + C240 + C241 + C242 + C243) |
| Tags (C200+ era)                                 | 21 (incl `C200-prep-docs-sync`, `C221-prep-mo-tidy`) |
| JD hard-skill coverage (excl. K8s)                | ~89-90%              |

**(C222 已删除——CPU 推理不在 JD 范围；"GPU 异构"由 C220 量化 GEMM + C221 fused MoE 完整覆盖)**
- **C240** — All-to-All 通信算子 (S-level gap, AG+RS backend). `my_vllm/distributed/device_communicators/` (NEW package: `base_device_communicator.py` + `all2all.py` + `cuda_communicator.py`); `AgRsAll2AllManager` 真实实装（dispatch / dispatch_router_logits / combine via NCCL/Gloo AllGather + ReduceScatter），DeepEP HT/LL/V2 / NIXL EP / FlashInfer NVLink / Mori backend class shells。`my_vllm/distributed/parallel_state.py` 扩展 `_EP` group + `get_ep_group()` + `init_expert_parallel()` + `GroupCoordinator.all_gatherv/reduce_scatterv/dispatch/combine` façade。`my_vllm/model_executor/layers/fused_moe/modular_kernel.py` (NEW): `FusedMoEPrepareAndFinalize` ABC + Modular/Monolithic。`my_vllm/model_executor/layers/fused_moe/prepare_finalize/{naive_dp_ep, no_dp_ep}.py` (NEW): NaiveDPEP + NoDPEP concrete prepare/finalize + factory。`my_vllm/model_executor/layers/fused_moe/all2all_utils.py` (NEW): `maybe_make_prepare_finalize()` dispatcher + `get_ep_all2all_manager()`。`my_vllm/model_executor/layers/fused_moe/config.py` 扩展 `use_*_kernels` properties。`my_vllm/config/parallel.py` 扩展 `All2AllBackend` literal + `ParallelConfig.all2all_backend` field。`examples/all2all_benchmark.py` (NEW) + `docs/notes/all2all-overview.md` (NEW, 474 lines) + `tests/test_C240_all2all.py` (NEW, 17 test classes, 61 cases incl. multi-process gloo round-trip via `torch.multiprocessing.spawn`)。
- **C241** — Symmetric Memory (NVLS) 通信优化 (A-level gap). `my_vllm/distributed/device_communicators/symm_mem.py` (NEW, 190 lines): `SymmMemCommunicator` 真实实装（`torch.distributed._symmetric_memory` wrapper with `multimem_all_reduce_` / `two_shot_all_reduce_` dispatch + 选择启发；graceful disable on non-NVLS hardware)。`my_vllm/distributed/device_communicators/all_reduce_utils.py` (NEW, 200 lines): `SYMM_MEM_ALL_REDUCE_MAX_SIZES` (H100/GB200 sizing) + `CUSTOM_ALL_REDUCE_MAX_SIZES` + `NCCL_SYMM_MEM_ALL_REDUCE_CONFIG` + `should_nccl_symm_mem_allreduce()` / `should_nccl_symm_mem_ag_rs()` 启发。`my_vllm/distributed/device_communicators/cuda_communicator.py` 扩展: `attach_symm_mem(...)` + `all_reduce()` 优先 try SymmMemCommunicator 再 fallback。`examples/symm_mem_benchmark.py` (NEW, 126 lines): heuristic sweep + single-rank all_reduce identity benchmark。`docs/notes/symm-mem-overview.md` (NEW, 427 lines): NVLS hardware primer + PyTorch `_symmetric_memory` runtime + vLLM heuristic + 工作尺寸表 + my-vllm 实现 + tests + 风险 + 后续。`tests/test_C241_symm_mem.py` (NEW, 10 test classes, 35 cases): lifecycle + eligibility + all_reduce + force + CudaCommunicator 集成 + sizing tables + heuristic + helpers。RTX 4060 (sm_89) 无 NVLS → SymmMemCommunicator graceful disabled；抽象与启发式表所有 tested。
- **C242** — SHM Broadcast + Object Storage (A-level gap). `my_vllm/distributed/device_communicators/shm_broadcast.py` (NEW, 410 lines): `ShmRingBuffer` (lock-free ring buffer) + `MessageQueue` (single-writer / multi-reader broadcast queue with pickle + 4-byte length encoding) + `Handle` dataclass + `memory_fence()` + `create_from_process_group(pg, ...)` setup hook。`my_vllm/distributed/device_communicators/shm_object_storage.py` (NEW, 470 lines): `SingleWriterShmRingBuffer` (FIFO eviction ring buffer) + `ObjectSerde` ABC + `PickleSerde` (replaces vLLM `MsgpackSerde`) + `SingleWriterShmObjectStorage` (key-value store with `put` / `get` / `free` / `clear`)。`examples/shm_benchmark.py` (NEW, 220 lines): single-process MessageQueue + ObjectStorage + tensor broadcast microbenchmark。`docs/notes/shm-broadcast-overview.md` (NEW, 419 lines): SHM vs ZMQ rationale + lock-free ring-buffer protocol + MessageQueue + object storage + my-vllm 实现 + tests + benchmarks + 风险 + 后续。`tests/test_C242_shm_broadcast.py` (NEW, 16 test classes, 29 cases incl. multi-process gloo round-trip via `torch.multiprocessing.spawn`). vLLM ZMQ `SpinCondition` replaced by `threading.Event` (~5x slower in latency-bound regime but bandwidth-equivalent); no wrap-around FIFO eviction in object storage (writer must `free()` older entries).
- **C243** — 跨节点 Weight Transfer (A-level gap). `my_vllm/distributed/weight_transfer/` (NEW package, 9 files, ~2,000 lines): `base.py` (`ParamMeta` / `WeightSource` / `ModuleSource` / `WeightTransferEngine` ABC / `TrainerWeightTransferEngine` ABC / `VLLMWeightSyncClient` Protocol); `nccl_common.py` (vanilla `torch.distributed.init_process_group` rendezvous + worker rank calc from `rank_offset`); `nccl_engine.py` (`NCCLWeightTransferEngine` dense broadcast + `_broadcast` helper prefers PyNcclCommunicator signature when available); `sparse_nccl_engine.py` (`SparseNCCLWeightTransferEngine` in-place `index_copy_` patches); `ipc_engine.py` (`IPCWeightTransferEngine` with `callable` send_mode only — Ray/HTTP not vendored); `packed_tensor.py` (pack/unpack + double/triple-buffered NCCL producer/consumer + IPC producer/consumer with single reusable buffer + clone-to-break-alias); `factory.py` (lazy-loaded registry, 3 engines); `clients.py` (`InProcessVLLMWeightSyncClient`). `my_vllm/config/weight_transfer.py` (NEW, 25 lines): `WeightTransferConfig` (mirror `vllm/config/weight_transfer.py:9`). `tests/test_C243_weight_transfer.py` (NEW, 16 test classes, 43 cases incl. multi-process gloo round-trip via `torch.multiprocessing.spawn` simulating TP rank switch). `docs/notes/weight-transfer-overview.md` (NEW, 569 lines): RLHF topology + 3 backend comparison + packed-tensor mechanics + scope reductions (no pynccl/StatelessProcessGroup/layerwise reload/MTP check/Ray/HTTP) + risks + references. Scope reductions: vanilla `torch.distributed.broadcast` (no PyNcclCommunicator C-ext); no `vllm.model_executor.model_loader.reload` (no layerwise reload hooks); no Ray actor / HTTP RLHF routes; no concrete trainer engine (FSDP/Megatron integration out of scope). `_broadcast` helper future-proofs for pynccl vendoring — zero changes needed when added.
- **C200** — GPU profiling toolchain + roofline notes (`docs/notes/gpu-profiling.md`, 253 lines). Nsight Compute + roofline methodology established.
- **C201** — NCCL benchmark + communication principle notes. `examples/nccl_benchmark.py` (384 lines: all_reduce/reduce_scatter/all_gather sweep); `docs/notes/nccl-architecture.md` (363 lines: ring vs tree algorithms, NCCL adaptive routing); `docs/notes/rdma-overview.md` (414 lines: IB Verbs / RoCE / iWARP, NCCL plugin interface).
- **C210** — `csrc/cache_kernels.cu` upgraded from 1 op (reshape_and_cache, single dtype) to 7 ops (reshape_and_cache, reshape_and_cache_flash, swap_blocks, copy_blocks, cp_gather_cache, convert_fp8, concat_and_cache_mla). All 7 exposed via `my_vllm._C`; Python wrappers in `_custom_ops.py`.
- **C211** — `triton_unified_attention.py` (1194 lines) PD-aware three-mode verification (decode-only / prefill-only / mixed chunked) + per-token-head quant branch verified (INT8 + FP8 e4m3). 8 tests in `test_C211_unified_attention.py`.
- **C212** — PD disaggregated single-process e2e (`examples/pd_disaggregated.py`, 495 lines). Uses `ExampleConnector` + `cp_gather_cache` for in-process P→D KV transfer; bit-exact K/V preservation verified.
- **C213** — `triton_reshape_and_cache_flash.py` (630 lines) full 5-KV_QUANT_MODE alignment. Also fixed `Platform.get_device_capability()` (was stub returning None, blocking FP8 storage).
- **C214** — `csrc/custom_all_reduce.cu` IPC ring all-reduce (361 lines). 7 ops: `init_custom_ar` / `dispose` / `meta_size` / `register_buffer` / `get_handle` / `open_mem_handle` / `all_reduce`. Python wrapper `my_vllm/distributed/custom_all_reduce.py`. world_size=1 identity + world_size=2..8 ring reduce-scatter.
- **C220** — `cutlass_scaled_mm` Python interface + `scaled_fp8_quant` / `scaled_int8_quant` activation-side quant. `my_vllm/_custom_ops.py` dispatches FP8 → `torch._scaled_mm`, INT8 → `torch._int_mm`, BF16/FP16 → `torch.mm`. `examples/quant_gemm_benchmark.py` (356 lines, BF16 vs FP8 vs INT8 sweep over M × (K,N)); `docs/notes/cutlass-scaled-mm.md` (319 lines: CUTLASS scaled MM contract, scale schemes, Marlin / Macheta / DeepGEMM ecosystem, FP8 e4m3 vs e5m2, methodology). 29 tests in `test_C220_quant_gemm.py`. RTX 4060 sm_89: FP8 ~2× BF16, INT8 ~2.3× BF16 at large M (1024+).
- **C221** — Triton fused MoE kernel + `moe_align_block_size`. `my_vllm/model_executor/layers/fused_moe/{moe_align_block_size.py (177), fused_moe_triton.py (403), fused_moe.py (90)}` (670 lines: token alignment helper + simplified FP16 grouped-matmul kernel + launcher + Triton-only dispatcher); `fused_moe()` has **no Python fallback** (CPU inference is out of scope — JD is GPU inference). `examples/moe_benchmark.py` (377 lines, fused vs idealized unfused MoE synthetic); 12 tests in `test_C221_fused_moe.py`. Numerically matches per-expert Python reference at M=32; fused approaches unfused TFLOPS at M ≥ 1024 on RTX 4060.
- **C223** — ngram spec_decode 真实实装 (closes B1-a from cleanup-plan §3). `my_vllm/v1/spec_decode/ngram_proposer.py` (395 lines, extracted from `vllm/v1/spec_decode/ngram_proposer.py`): Numba-JIT'd `NgramProposer` + `batch_propose_numba` (`@njit(parallel=True)`) + `_find_longest_matched_ngram_and_propose_tokens` (KMP-style LPS walk on reversed tokens, `@jit(nopython=True)`). JIT warm-up paid in `__init__`; pre-allocated `(max_num_seqs × k)` output buffers. Numba is a soft import — clear `ImportError` on use, but rest of `my_vllm.v1.spec_decode` stays importable on CPU-only envs. Also: `my_vllm/config/speculative.py` gains `prompt_lookup_min`/`max` + `__post_init__` (5/5 default, min≤max validation matching vLLM `config/speculative.py:737-763`); `my_vllm/config/vllm.py` gains `speculative_config: SpeculativeConfig | None` field (was referenced in skeleton code but never declared — silent `AttributeError` waiting to happen). `my_vllm/v1/spec_decode/__init__.py` exports `NgramProposer` + adds `create_spec_decode_proposer()` dispatcher mirroring `vllm/v1/worker/gpu_model_runner.py:596-611`. 33 tests in `test_C223_ngram_proposer.py` (8 classes: config defaults+validation, kernel golden cases, per-request filtering, batch alignment, dispatcher, JIT warm-up, interface). `examples/ngram_proposer_benchmark.py` (233 lines, synthetic vocab/batch/ctx sweep, CSV stdout). EAGLE / DraftModel proposers remain class shells (their full impls require loading a draft model — out of scope for the JD-aligned subset). **Runner-side verify/accept/bonus path is deferred** (C223.5/C225) — the proposer itself is fully exercisable via numpy arrays.
- **C224** — rotary_embedding fused path (Triton) + CustomOp dispatch fix. `my_vllm/model_executor/layers/rotary_embedding/triton_rotary.py` (NEW 260 lines): Triton fused Q/K rotation kernel (in-place; FP32 internal compute; store back at input dtype; supports Neox + GPT-J; GQA via num_heads ≠ num_kv_heads). Grid = (num_tokens, num_heads) for Q + (num_tokens, num_kv_heads) for K. `my_vllm/_custom_ops.py:rotary_embedding()` now actually dispatches (CUDA + Triton → Triton kernel; otherwise PyTorch in-place fallback). `RotaryEmbeddingBase.forward_cuda()` added (was missing). `my_vllm/model_executor/custom_op.py:_resolve_forward_method()` added + `CustomOp.dispatch_forward()` fixed to actually use `current_platform` (was hard-coded to `forward_native` regardless — a silent bug that blocked every CustomOp-based layer from running its CUDA path). 18 tests in `test_C224_rotary_embedding.py` (7 classes). `examples/rotary_embedding_benchmark.py` (188 lines): T × H × head_size × style × GQA sweep. RTX 4060 BF16: Triton 2.4-4.5× faster than PyTorch ref (matches vLLM CUDA-vs-PT). **OPT-125m doesn't use RoPE** (learned absolute pos embeddings), so the layer is fully implemented + tested but not yet wired into the model — that wiring comes with future Llama-family GQA integration (C230+ roadmap).
- **C230** — elastic_ep 接线 + rebalance (closes B2-b from cleanup-plan §3). `my_vllm/distributed/elastic_ep/rebalance.py` (NEW 367 lines): `EPRebalanceSignal` / `EPRebalanceDecision` dataclasses; `EPRebalancePolicy` ABC; `NoOpEPRebalancePolicy` + `ThresholdEPLoadPolicy` (util-driven threshold scale-up / scale-down with explicit-target override + cold-start hysteresis + dp-size bounds); `create_scaling_state_for_decision(vllm_config, decision)` + `decide_and_create(vllm_config, signal, policy=None)` factories that bridge the policy to a fresh `ElasticEPScalingState`. `my_vllm/distributed/elastic_ep/__init__.py` exports the new classes. `my_vllm/distributed/elastic_ep/elastic_state.py` `progress()` semantics fixed (was inverted: returned whether state was COMPLETE instead of whether it advanced). `my_vllm/config/parallel.py` gains `data_parallel_size: int = 1` field + `clone_with_dp_size(new_dp_size)` helper, matching vLLM `vllm/config/parallel.py:127`. 34 tests in `test_C230_elastic_ep.py` (8 classes: full state-machine sequences 4 + 9 + 4 + 3 = 20 transitions across 4 (scale_type, worker_type) combos; ParallelConfig field; policy validation + decisions; factory bridge; end-to-end `decide_and_create`; custom policy subclassing). State machine walks all 4 prescribed sequences (4 + 9 + 4 + 3 transitions). **Engine-side wiring** (TCP-store barriers, NCCL weight transfers, EPLB reshuffle, KV-cache memory sync) is **deferred** — `progress()` always returns True today; production hooks land when a real multi-process deployment is wired.
- **C231** — `device_allocator/` + `CumemAllocator` + **vendored C extension** (`my_vllm/cumem_allocator.cpp`, 810 lines extracted verbatim from `vllm/csrc/cumem_allocator.cpp`). The C extension performs the real CUDA Driver API calls (`cuMemCreate` / `cuMemAddressReserve` / `cuMemMap` / `cuMemUnmap` / `cuMemRelease`) via `my_malloc` / `my_free` C entry points that PyTorch's `CUDAPluggableAllocator` dlsym's. `my_vllm/device_allocator/__init__.py` (NEW 175 lines): `HandleType` / `AllocationData` / `MemAllocator` Protocol + `get_mem_allocator_instance()` factory, mirrors `vllm/device_allocator/__init__.py:50` 1:1. `my_vllm/device_allocator/cumem.py` (~440 lines): `CumemAllocator` singleton wires the C extension in `__init__`; `use_memory_pool(tag)` creates a real `torch.cuda.memory.MemPool` backed by `CUDAPluggableAllocator(lib, "my_malloc", "my_free")`; `sleep()` / `wake_up()` call `_cumem_ext.python_unmap_and_release` / `python_create_and_map` for per-tensor physical-chunk unmapping / remapping. `my_vllm/platforms/interface.py:is_cumem_allocator_available()` properly detects CUDA + SM >= 7.0 (was a stub returning False). `setup.py` builds `my_vllm.cumem_allocator` as a second CUDAExtension with `-lcuda` link flag. 36 tests in `test_C231_cumem_allocator.py` (9 classes): 31 bookkeeping / dispatch tests + 5 real-VMM-cycle tests (real `MemPool` allocation + byte-level verification + real unmap / remap with data survival across the cycle). **CPU offload during sleep is NOT vendored**: requires `CudaRTLibrary` wrapper that vLLM vendors but my-vllm doesn't; the C extension still unmaps + releases physical chunks but tensor contents aren't preserved across sleep/wake_up. Documented as follow-up.
- **C232** — `docs/notes` 综合笔记集 (index + 3 surveys). `docs/notes/README.md` (NEW 191 lines): single-entry-point index with topical map + JD-keyword map + commit→note map + reading order + source-of-truth pointer. `docs/notes/gpu-arch-primer.md` (NEW 306 lines): SM / warp / memory hierarchy / tensor-core throughput matrix / Roofline / occupancy primer. `docs/notes/vllm-runtime-overview.md` (NEW 428 lines): top-down walkthrough of the vLLM V1 runtime stack (HTTP → EngineCore → Scheduler → Executor → Worker → GPUModelRunner → Model → Attention → Sampler), each layer annotated with the `my_vllm/` file + the Cxxx commit, plus architectural boundaries + performance hotspots + "where to start debugging". `docs/notes/quantization-landscape.md` (NEW 315 lines): FP8 / INT8 / INT4 (Marlin / AWQ / GPTQ) / FP4 (NVFP4) survey with the format zoo + accuracy-throughput tradeoff + per-tensor vs per-token vs per-channel + activation-outlier problem + per-GPU decision tree. Total: 8 notes, 2,589 lines (was 4 notes, 1,349 lines). Each new note ≥ 306 lines per `jd-gap-closure-plan.md` §C232 acceptance.
- **C233** — 全量回归 (Phase 1+2+3 闭环). Regression check across all 14 Phase 1+2+3 commits (C200-C232 excluding the deleted C222). `pytest tests/test_C{200,201,210,211,212,213,214,220,221,223,224,230,231}_*.py` → **230 passed + 4 skipped** (2 C212 failures are test pollution between sessions — pass 5/5 isolated; pre-existing PyTorch CUDA state issue, not a regression). `python run.py` → end-to-end pass (checkpoint load + prefill + decode + token emission + LENGTH stop). `grep -rn "NotImplementedError" my_vllm/` → **43 hits, all intentional**: 28 ABC abstract methods (WorkerBase / QuantizationConfig / AttentionBackend / FusedMoERouter / CustomOp.forward_native-cuda-hip / WorkerProfiler / Platform.forward_tpu / KVCacheQuantConfig / FusedTopKRouter) + 12 documented class shells (eagle.py / draft_model.py per C223; logprobs.py / cp_utils.py / multiproc_executor.py per cleanup) + 3 dispatcher stubs (create_spec_decode_proposer). All 14 Phase 1+2+3 snapshots + tags present. Per `jd-gap-closure-plan.md` §C233-final: Phase 1+2+3 **fully closed**.

**JD gap analysis (post-C243, Phase 1+2+3 closed + Phase 4 at 4/10)**: Hard-skill coverage vs JD ≈ 89-90% (excluding K8s). Filled since C151: GPU profiling + roofline, NCCL + RDMA, CUDA cache block ops (7 ops / multi-dtype), Triton unified attention (PD-aware + per-token-head quant), PD 分离 (ExampleConnector + cp_gather_cache), custom IPC all-reduce, CUTLASS scaled MM (FP8/INT8/BF16), Triton fused MoE (token alignment + grouped GEMM), ngram spec decoder (Numba-JIT + KMP-style LPS), fused RoPE (Triton + GQA), elastic EP rebalance (state machine + threshold policy), CUDA VMM allocator (real C ext: cuMemCreate/Map/Unmap), ring all-to-all (AgRs backend + prepare/finalize ABCs), SymmMem / NVLS heuristic + dispatch, SHM broadcast + ObjectStorage, **cross-process weight transfer** (3 backends + packed-tensor efficiency), **12 comprehensive engineering notes (4,478 lines)**. Pre-cleanup-era test failures (72 in C001-C031 + C126-C128 + hot_path) are **expected** per `AGENTS.md` "Test expectations" — they're pre-C151-cleanup artifacts, not regressions from Phase 1+2+3+4 work.

**Next**: **Phase 3 backlog** (C244-C249, 6 commits: Quick AllReduce notes / Async Scheduler / KV-aware scheduling / EP rebalance 闭环 w/ C243 / distributed comm benchmark + 笔记). **Phase 4** (C250-C259, 10 commits: NIXL KV Connector / KV event bus / PD 闭环 / KV offload / KV spec registry / PD benchmark). **Phase 5** (C260-C269, 10 commits: Scheduler priority+preemption / CUDA VMM sleep 闭环 / Inductor piecewise / Spec decode runner 闭环 / MoE AllToAll 集成 / MLA / Warmup / UVA offload). **K8s 部署计划** (C153-C159, 7 commits) — `k8s-deploy-plan.md` 仍在仓库根未 tracked，可与已闭环的 Phase 1+2+3+4 并行启动。Phase 3 closure ≠ project closure。

## Scope decision (JD-driven)

The final form is a **minimal clean subset mapped to the job description (JD)**, not a scaled-down vLLM. A module is kept only if it serves one of the JD keywords: 性能/成本/稳定性, PD 分离, 分布式 KV Cache, 异构推理, 弹性计算, GPU/CUDA, 分布式系统 (+ 网络/RDMA). Clean removal = delete the file **and** its imports, config fields, and all references — no NoOp/ABC placeholders, no dangling imports. Never keep something just because vLLM has it.

## Commands

```bash
# Activate the virtual environment first
source /home/kasw/workspace/gpu-infra/.venv/bin/activate

# Offline inference (V1 engine)
python run.py

# Compile the CUDA extensions (always pin to 3 cores — the build uses
# ninja and parallel nvcc invocations can OOM the host on small dev boxes).
# Use MAX_JOBS=3 or `-j3` to be safe.
#
# Two extensions are built:
#   * my_vllm._C              (cache_kernels.cu + custom_all_reduce.cu)
#   * my_vllm.cumem_allocator  (vendored cumem_allocator.cpp, real VMM)
MAX_JOBS=3 python setup.py build_ext --inplace

# After build, the .so files sit at my_vllm/_C*.so and my_vllm/cumem_allocator*.so.
# `python -c "from my_vllm import cumem_allocator; print(cumem_allocator.init_module)"
# verifies the VMM extension loaded successfully.

# Run all tests
pytest tests/

# Run a single test file or class
pytest tests/test_C231_cumem_allocator.py -v
```

**Test expectations — not all tests must pass at any single Cxxx.**

The test suite has historical expectations that evolved across the Cxxx
commit series.  When validating a specific commit:

  - `git tag Cxxx-name` marks the test set that commit was validated
    against.  Run those tests at that commit and they all pass.
  - Newer commits may intentionally break older test expectations as
    the architecture evolves (e.g. C031 references files that were
    later deleted in C151's cleanup).  Running `pytest tests/` against
    the latest commit will surface a mix of expected failures from
    pre-cleanup eras — that's normal.
  - When in doubt: `git checkout Cxxx-name` and run the test files
    whose `test_Cxxx_*.py` filename matches that stage number.

There is no CI, no lint, no formatter, no typecheck configured.

## Configuration

`my_vllm/config/` — `VllmConfig` aggregates `ModelConfig`, `CacheConfig`, `SchedulerConfig`, `ParallelConfig`. Defaults:
- model: `facebook/opt-125m`, dtype: `bfloat16`, block_size: `16`, num_kv_blocks: `512`, max_num_seqs: `8`
- enforce_eager: `True`, max_num_batched_tokens: `2048`, eos_token_id: `2`
- TP=1, PP=1 (config-only); C150 wires multi-process NCCL end-to-end (`MultiprocExecutor._init_executor` exports `MASTER_ADDR`/`MASTER_PORT`/`WORLD_SIZE`); C214 adds CUDA IPC ring all-reduce for same-host multi-GPU (`my_vllm/distributed/custom_all_reduce.py`).
- Platform capability detection (C213): `Platform.get_device_capability()` now actually queries `torch.cuda.get_device_capability()` (was a stub returning None), enabling FP8 storage on SM 8.9+.

## Package layout

```
my_vllm/
  config/              # → vllm/config/  VllmConfig, ModelConfig, CacheConfig, SchedulerConfig, ParallelConfig
  distributed/         # → vllm/distributed/  parallel_state.py (GroupCoordinator)
    elastic_ep/        # elastic_state.py (骨架) + rebalance.py (策略+工厂, C230)
    kv_transfer/       # KVConnectorBase_V1 / factory / example_connector / nixl
    weight_transfer/   # C243: WeightTransferEngine ABC + NCCL/sparse/IPC backends
                       #   + packed_tensor + factory + InProcessVLLMWeightSyncClient
  model_executor/      # → vllm/model_executor/  model_loader, custom_op, layers/, models/
    layers/            # linear, activation, layernorm, logits_processor, vocab_parallel_embedding,
                       #   attention/{attention.py, attention_layer_base.py}, fused_moe/, quantization/, rotary_embedding/{base,common,triton_rotary}.py
    models/            # opt.py (Dense only — Llama/DeepSeek 在 alignment 后续阶段)
  platforms/           # → vllm/platforms/  interface.py (CUDA 唯一)
  profiler/            # → vllm/profiler/  wrapper.py, layerwise_profile.py
  lora/                # → vllm/lora/  lora_model, model_manager, request, worker_manager, layers/, ops/
  v1/                  # → vllm/v1/
    engine/            # EngineCore, EngineCoreClient (InprocClient / BusyLoopClient), LLMEngine,
                       # AsyncLLM, InputProcessor, OutputProcessor, detokenizer, exceptions
    core/              # BlockPool, KVCacheManager, kv_cache_utils, kv_cache_metrics
      sched/           # Scheduler, SchedulerInterface, SchedulerOutput, request_queue
    worker/            # WorkerBase, Worker, GPUModelRunner
      gpu/             # BlockTables, InputBatch, RequestState, PPHandler (pp_utils.py), model_states/
    executor/          # Executor ABC, UniProcExecutor, MultiprocExecutor (C151 已删 Ray executor)
    attention/         # AttentionBackend ABC, CommonAttentionMetadata, selector (→ Triton),
                       #   backends/{triton_attn, flash_attn}.py,
                       #   ops/{triton_unified_attention, triton_reshape_and_cache_flash, paged_attn, triton_attention_helpers}.py
    sample/            # Sampler (nn.Module), LogitsProcessor, SamplingMetadata
    metrics/           # stats, loggers, prometheus
    spec_decode/       # ngram_proposer (C223 — real impl, Numba-JIT);
                       #   metadata / draft_model / eagle / metrics / utils stubs
    engine/__init__.py # FinishReason, EngineCoreOutput, EngineCoreOutputs
    request.py         # Request (12 statuses), SamplingParams
    outputs.py         # SamplerOutput, ModelRunnerOutput
    serial_utils.py    # MsgpackEncoder/Decoder (binary format)
    utils.py           # ConstantList
    kv_cache_interface.py  # KVCacheSpec hierarchy
  entrypoints/         # → vllm/entrypoints/  llm.py (C151 已删 OpenAI api_server)
  _custom_ops.py       # CUDA op dispatch: 7 cache ops (C210) + 7 custom_all_reduce ops (C214)

# Repo root (outside my_vllm/ package)
run.py                 # CLI entry point
csrc/                  # C++/CUDA sources
                       #   cache_kernels.cu          7 cache block ops (C210)
                       #   custom_all_reduce.cu       IPC ring all-reduce (C214)
                       #   custom_all_reduce.cuh      header
                       # Vendored from vLLM (C231):
                       #   cumem_allocator.cpp        CUDA VMM (cuMemCreate/Map/Unmap);
                       #                               my_malloc/my_free C entry points for
                       #                               PyTorch CUDAPluggableAllocator dlsym;
                       #                               810 lines extracted verbatim from
                       #                               vllm/csrc/cumem_allocator.cpp.
                       #   cumem_allocator_compat.h   CUDA/HIP type aliases + ROCm shim
                       #                               (#ifdef USE_ROCM gates AMD path).
setup.py               # builds 2 CUDAExtensions (`my_vllm._C` + `my_vllm.cumem_allocator`)
                       # use MAX_JOBS=3 to avoid OOM during parallel nvcc invocations
examples/              # run.py + benchmark scripts
                       #   run.py, benchmark.py                 (pre-existing)
                       #   nccl_benchmark.py                    (C201) — all_reduce / reduce_scatter / all_gather sweep
                       #   pd_disaggregated.py                  (C212) — P→D e2e via ExampleConnector
                       #   ngram_proposer_benchmark.py          (C223) — NgramProposer sweep
                       #   profile_attention.py                 (C200) — unified_attention wall-clock
#   rotary_embedding_benchmark.py        (C224) — Triton vs PyTorch rotary
                        #   profile_cache_write.py               (C200) — triton_reshape_and_cache_flash
                        #   quant_gemm_benchmark.py              (C220) — BF16 vs FP8 vs INT8 GEMM sweep
                        #   moe_benchmark.py                     (C221) — fused vs unfused MoE synthetic
                        #   all2all_benchmark.py                 (C240) — AG+RS ring all-to-all sweep
                        #   symm_mem_benchmark.py                (C241) — symm_mem heuristic + identity benchmark
                        #   shm_benchmark.py                     (C242) — SHM MessageQueue + ObjectStorage microbench
docs/notes/            # Deep-dive notes (per Cxxx items + 3 new C232 surveys)
                        #   README.md                  (C232) — index + topical / JD / reading-order map
                        #   gpu-arch-primer.md         (C232) — SM/warp/memory/Roofline/occupancy primer
                        #   vllm-runtime-overview.md   (C232) — top-down vLLM V1 stack walkthrough
                        #   quantization-landscape.md  (C232) — FP8/INT4/Marlin/AWQ/NVFP4 survey
                        #   gpu-profiling.md           (C200) — Nsight Compute / roofline
                        #   nccl-architecture.md      (C201) — ring/tree allreduce + NVLink/IB routing
                        #   rdma-overview.md          (C201) — IB Verbs / RoCE / iWARP / NCCL plugin
                        #   cutlass-scaled-mm.md      (C220) — scaled_mm contract + Marlin/Machete/DeepGEMM ecosystem
                        #   all2all-overview.md       (C240) — ring all-to-all + DeepEP/NIXL/FlashInfer backend survey
                        #   symm-mem-overview.md      (C241) — NVLS hardware + PyTorch _symmetric_memory + vLLM heuristic
                        #   shm-broadcast-overview.md (C242) — SHM ring buffer + MessageQueue + ObjectStorage
                        #   weight-transfer-overview.md (C243) — RLHF topology + 3 backends + packed-tensor mechanics
                        # total: 12 notes / 4,478 lines
docs/snapshots/         # Per-commit closure write-ups (Goal / Features / Flow / Capability / Limitation / Next)
deploy/                # NOT YET ADDED — K8s 部署实验 (C153-C159 plan, 独立于 my_vllm/，仅通过 HTTP/env/Prometheus 交互)
                       #   k8s/                 namespace/configmap/deployment/statefulset/service/hpa/pdb/sa/kustomization
                       #   observability/       ServiceMonitor + Prometheus rules + Grafana dashboard
                       #   autoscaling/         独立 scaler.py controller (HPA 不足场景)
                       #   disaggregated/       PD 分离部署 (prefill + decode 两套 Deployment)
k8s-deploy-plan.md     # C153-C159 执行计划 (未 tracked, Backlog)
LOOP-PLAN.md           # C124→C128 plan (busy loop + EngineCoreClient)
```


**Path mapping is 1:1**: `my_vllm/` → `vllm/` (e.g., `my_vllm/config/vllm.py` → `vllm/config/vllm.py`).

## Reference codebase

The real vLLM V1 source lives at `/home/kasw/文档/GitHub/GPU-Project/vllm`. **All new and modified code must be directly extracted from the real vLLM codebase** — class bodies, method logic, data structures, and internal algorithms are taken from vLLM source files. Simplifications (fewer fields, no quantization backends) are principled scope reductions — each field or feature is omitted because the functionality it supports is excluded from scope, not because the architecture is simplified for whimsiness. No invented patterns or APIs. Legacy code from C000–C098 is gradually being replaced under this standard.

The current concrete model is `facebook/opt-125m` (Dense Transformer). The architecture is designed to accommodate Dense, GQA/MQA, and MoE through reusable layers — the three model families are OPT (Dense), Llama (GQA), DeepSeek V2/V3 (MoE + MLA). Model count increases should add config/thin adapter layers, not duplicate Runtime.

The **engineering spec** (`Prompt/Engineer-Spec/README.md`) defines the alignment standard:
- **P0** (10 items): Core skeletons that must be reproduced — Unified Execution Model, Paged KV path, KV Cache hierarchy, Block identity, Scheduler model, Request/RequestState, InputBatch, Attention Backend, Executor/Worker/Runner boundary, Distributed abstraction
- **P1** (5 items): Should be gradually aligned — Dynamic KV init, Model layers, OPT model, Sampler boundary, Metrics
- **Must implement**: Speculative Decoding, EP, CP, KV Connector, at least one Quantization path — these alter GPU execution, KV Cache, batching, scheduler, communication, or model parallelism and cannot remain NoOp/ABC
- **Keep (career-relevant)**: LoRA — one working Dense-Linear path
- **Delete (no JD match)**: Mamba/SSM, Multimodal, Structured Output, multi-tenant, pooler/embedding/rerank, encoder-decoder, non-OPT models, non-CUDA platforms, non-FP8 quantization backends, entrypoints business surface — remove cleanly, no dangling imports
- **Excluded from core code**: K8s, Ray, Autoscaling — understand theoretically, but not in `my_vllm/`. **Exception**: K8s 部署计划已批准（`k8s-deploy-plan.md`）— `deploy/` 目录与 `my_vllm/` 平级，禁止 `import my_vllm.*`，仅通过 HTTP / env / Prometheus 交互；Ray 仍仅理论。

## 4-Level alignment framework

| Level | Standard |
|---|---|
| **Level 1: File tree** | `my-vLLM files ⊆ vLLM files`. Every file must correspond to a real vLLM file. Delete invented files with no counterpart. |
| **Level 2: Objects & interfaces** | Class names, method names, responsibility, data ownership match vLLM. Not 100% parameter identical, but design intent must match. |
| **Level 3: Core logic** | `Scheduler → KV allocation → Input preparation → Unified batch → Model forward → Paged attention → Sampler → Output` — this chain must be faithfully reproduced. |
| **Level 4: Implementation tech** | `GPU memory → CUDA/Triton → Paged KV → Batch execution → CUDA Graph / async → NCCL → Multi-GPU` |

## Before EVERY code modification:

1. Read `Prompt/Engineer-Spec/README.md` for the target alignment standard
2. Check `docs/alignment-plan.md` (keep/delete matrix + reference `vllm/...` path) and the Implementation scope tiers — confirm the feature is in P0/P1/Must Implement/Keep scope, not Delete or Excluded
3. Read the real vLLM source at `/home/kasw/文档/GitHub/GPU-Project/vllm` for API verification
4. Confirm the file exists in real vLLM's file tree
5. Match vLLM class names, method names, dataclass field names exactly
6. Never invent API names or file paths
7. Verify `python run.py` or `pytest` passes before committing

## KV Cache dtype abstractions

See `Prompt/Engineer-Spec/README.md` § KV Cache dtype — FP16/BF16 baseline, FP8 quantization/dequantization, KVCacheSpec abstraction chain.

## Model architecture priorities

See `Prompt/Engineer-Spec/README.md` § Model architecture — Dense/GQA/MoE are must-master; model count increases should add config, not duplicate Runtime.

## Model loading: checkpoint → GPU

See `Prompt/Engineer-Spec/README.md` § Model loading — init/load separation, meta device, weight mapping, parallel linear, TP-aware shard.  `my_vllm/model_executor/model_loader.py` now constructs the vLLM-native `OPTForCausalLM` and loads weights via `load_weights()` from an HF checkpoint (still `from_pretrained` for the weight source, not meta-device + streaming).

## Commit workflow (per `Prompt/Principle/Commit-Standard.md`)

1. One commit = one P0/P1 alignment item. Format: `Cxxx: Module: Description` (C110+)
2. Each commit must be runnable (`python run.py` or `pytest` must pass)
3. Create `docs/snapshots/Cxxx.md` with required sections:
   Goal, New Features, File Tree, Runtime Flow (mermaid),
   Current Capability, Current Limitation, Next Stage
4. Tag with `Cxxx-description` (e.g. `git tag C033-kernel`)
5. Never modify earlier stages' code; if a fix is needed, create a new fix commit
6. See `Prompt/Principle/Commit-Standard.md` for the full rules
7. After commit, output: commit message, file diff summary, file tree,
   architecture diagram (mermaid), capabilities, limitations,
   benchmark (skip if no performance code — see Commit-Standard rule 7)
8. Update AGENTS.md in the same commit if module or architecture facts changed
9. **Stop** — await human confirmation before proceeding to next stage

## Key architecture facts

- EngineCore `step()` = `schedule() → execute_model() → update_from_output()` → EngineCoreOutputs
- EngineCore `run_busy_loop()` = `while _handle_shutdown(): _process_input_queue() → _process_engine_step() → output_queue.put_nowait()` (C124); inputs come via `input_queue: queue.Queue[tuple[EngineCoreRequestType, Any]]` with `{ADD, ABORT, WAKEUP}`; shutdown state machine `EngineShutdownState = {RUNNING, REQUESTED, SHUTTING_DOWN}`
- LLMEngine `step()` = `EngineCoreClient.get_output() → EngineCoreOutputs → OutputProcessor.process_outputs()`; LLMEngine holds an `EngineCoreClient` (`busy_loop=False` → `InprocClient` direct-step, `busy_loop=True` → `BusyLoopClient` daemon thread); `running_requests()` accessor replaces direct `scheduler.running` access
- SchedulerInterface ABC: schedule() → SchedulerOutput, update_from_output(), finish_requests()
- Model loading chain: `Executor._init_executor() → Worker.init_device() → GPUModelRunner.load_model()`
- Unified `execute_model`: `update_states → prepare_inputs → prepare_attn → set_forward_context → model.forward → sample → postprocess` (single batch, no prefill/decode split)
- Model is vLLM-native `OPTForCausalLM` (custom layers), forward is `model(input_ids, positions)` — no HF `past_key_values`
- Attention: `set_forward_context(attn_dict)` → `Attention` layer → `TritonAttentionBackend` (`triton_reshape_and_cache_flash` + `triton_unified_attention` + `paged_attn` + `triton_attention_helpers`)
- KV cache: HND-packed `(num_layers, num_blocks, num_kv_heads, block_size, 2*head_dim)`; block-major; BlockPool + FreeKVCacheBlockQueue O(1) alloc
- CUDA graph: `capture_cudagraph()` captures decode graphs per batch size; `_run_with_cudagraph()` replays decode-only steps
- Prefix caching: BlockHashToBlockMap (hash→list[block]), free_blocks() with ref_cnt cascading release, CoW
- Scheduler: FCFS + Priority scheduling + chunked-prefill + prefix-cache aware + resource model + Preemption
- Request PREEMPTED status supports resume/retry
- Chunked prefill splits long prompts by `token_budget` at the Scheduler level
- Sampler is an `nn.Module` subclass with penalties → temperature → top-k → top-p → sampling
- TP=2: initialize_model_parallel(nccl) → GroupCoordinator(all_reduce/all_gather/broadcast), ColumnParallelLinear/RowParallelLinear
- Multi-process NCCL (C150): `MultiprocExecutor._init_executor` exports `MASTER_ADDR`/`MASTER_PORT`/`WORLD_SIZE` env vars; `Worker.init_device` runs `init_distributed_environment` and creates the model-parallel group.
- Custom same-host all-reduce (C214): `csrc/custom_all_reduce.cu` exposes `init_custom_ar` / `all_reduce` / `dispose` / `get_handle` / `open_mem_handle`; uses CUDA IPC + ring reduce-scatter (Patariasuk & Yuan 2009); world_size=1 short-circuits to memcpy; world_size cap = 8.
- PD disaggregation (C212): `ExampleConnector` (kv_transfer/v1) saves KV via `save_kv_layer` on the producer, consumer restores via `start_load_kv` + `get_bundle_kv`; bit-exact K/V preservation verified in single-process simulation.
- KV cache quantization modes (C211/C213/C149): `KVQuantMode` enum = NONE / FP8_TENSOR / INT8_PER_TOKEN_HEAD / FP8_PER_TOKEN_HEAD / INT4_PER_TOKEN_HEAD. Triton kernels (`triton_unified_attention` + `triton_reshape_and_cache_flash`) handle all 5 modes; per-token-head scales are fused into the QK score (k_scale) and applied to PV (v_scale) to avoid extra BLOCK_M × TILE_SIZE multiplies.
- Cache kernel op surface (C210): 7 ops in `csrc/cache_kernels.cu` — `reshape_and_cache` (HND layout), `reshape_and_cache_flash` (FlashAttention layout), `swap_blocks` (cudaMemcpyAsync over pair list), `copy_blocks` (multi-dtype GPU block-to-block copy), `cp_gather_cache` (non-contiguous block gather for prefix-cache CoW), `convert_fp8` (FP8 e4m3 ↔ BF16/FP16), `concat_and_cache_mla` (kv_lora_rank + pe_dim; FP8-DS-MLA rejected). FP8 storage path uses Triton (`triton_reshape_and_cache_flash`); CUDA path is the CUDA-graph-friendly fallback.
- Metrics: TTFT, TPOT, ITL, throughput, KV cache utilization

## Model dependency

Model is `facebook/opt-125m`, loaded from a local checkpoint (`my_vllm/model_executor/model_loader.py` + `entrypoints/llm.py` both default `_LOCAL_MODEL_PATH = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"`). The model must be available for any run or test.

## Repository conventions

- **Planning docs are untracked by design.** Multi-commit execution plans
  (e.g. `docs/cleanup-plan.md`, `k8s-deploy-plan.md`,
  `jd-gap-closure-plan.md`) drive future work but are **not** part of
  any commit's source diff. They follow the C151 precedent (commit
  message: *"planning artefact — left untracked, not part of the source
  diff"*). Reasoning: commit captures the **code snapshot**; plans are
  drafts. Executed work lives in commit messages +
  `docs/snapshots/Cxxx.md`. If a plan needs permanent history, move it to
  `docs/plans/` and commit explicitly with a non-`Cxxx` message.
- **AGENTS.md numbering alignment.** Whenever a new Cxxx-numbered
  planning doc (or external numbering scheme) shifts, sync AGENTS.md in
  the same commit window. Pattern: small standalone `Cxxx-prep` docs
  commits (e.g. `C153-prep: AGENTS.md sync ...`) rather than rolling the
  sync into the next real commit.
