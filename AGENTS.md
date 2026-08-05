# AGENTS.md

## Project identity

This is an **educational simulation** of the vLLM V1 serving runtime. Each Git commit represents one architectural evolution stage (C001→C032), in sequence. The Git history itself is the curriculum — **never squash or rebase**. Every commit has a matching `docs/snapshots/Cxxx.md` and a Git tag.

Architecture reference: `C014-Final.md` at repo root.

## Commands

```bash
# Activate the virtual environment first
source /home/kasw/workspace/gpu-infra/.venv/bin/activate

# Offline inference (V1 engine)
python run.py

# Only the V1 attention backend works — PyTorch SDPA (torch_attn.py)
# FlashAttention and CUDA graphs are stubs. enforce_eager=True always.

# Run all tests
pytest tests/

# Run a single test file or class
pytest tests/test_C031_runtime.py -v
pytest tests/test_C031_runtime.py::TestC031FileCompleteness -v
```

There is no CI, no lint, no formatter, no typecheck configured.

## Package layout — real code vs. stale stubs

**Active code lives in `my_vllm/v1/`.** This is the real engine.

```
my_vllm/v1/
  engine/          # EngineCore, LLMEngine, AsyncLLM, I/O processors, detokenizer
  core/sched/      # Scheduler (FCFS, chunked-prefill, prefix-cache aware)
  core/            # BlockPool, KVCacheManager, kv_cache_utils
  worker/          # Worker, GPUModelRunner, physical KV cache pool
  executor/        # ABC + UniProcExecutor
  attention/       # BackendABC, CommonAttentionMetadata, backends/ops/paged_attn.py
  sample/          # Sampler (nn.Module), LogitsProcessor
  metrics/         # collector
  request.py       # Request dataclass (11 statuses)
```

Legacy stub directories (mostly empty `__init__.py` after C032 cleanup):

```
my_vllm/engine/        # ← abandoned, bridge imports removed in C032
my_vllm/cache/         # ← abandoned
my_vllm/scheduler/     # ← abandoned
my_vllm/worker/        # ← abandoned
my_vllm/executor/      # ← abandoned
my_vllm/server/        # ← abandoned (examples/server.py uses this — outdated)
my_vllm/generation/    # ← abandoned
my_vllm/batch/         # ← abandoned
```

**Do not write new code into the legacy directories.** All new code goes under `my_vllm/v1/`.

## Configuration

`my_vllm/config.py` — `EngineConfig` dataclass. Defaults:
- model: `facebook/opt-125m` (local at `/home/kasw/文档/GitHub/cpu-vllm/opt-125m`)
- dtype: `bfloat16`, block_size: `16`, num_kv_blocks: `512`, max_running: `8`
- enforce_eager: `True`, max_new_tokens: `50`, eos_token_id: `2`

## Design documents (Principle/)

When implementing a new stage CXXX, **first read the corresponding Principle design doc**:

| Stage range | Design doc |
|---|---|
| C000–C014 | `Principle/C000-C014/C000-C014-Do-to-my-vllm.md`, `C014-Final.md` |
| C015–C031 | `Principle/C015-C031/C015-C031-Do-to-my-vllm.md`, `C031-Final.md` |
| C032 | `Principle/C032/C032.md` |
| C033–C050 | `Principle/C033-C050/C033-C050.md`, `C050-Final.md` |
| C051–C070 | `Principle/C051-C070/C051-C070.md`, `C070-Final.md` |
| C071–C100 | `Principle/C071-C100/` |
| C101–C120 | `Principle/C101-C120/` |
| C121–C140 | `Principle/C121-C140/` |

Each design doc specifies: industrial motivation, existing code baseline, exact file changes, acceptance criteria, and what NOT to change. This is the authoritative spec for that stage.

## Reference codebase

The real vLLM V1 source lives at `/home/kasw/文档/GitHub/GPU-Project/vllm`.
When implementing a stage, **reference it for production patterns**:
- API signatures, class hierarchies, module boundaries
- KV cache shapes, attention metadata structures, scheduler output formats

Do NOT copy-paste blindly — extract the minimal interface contract needed. This project simulates the evolution, not a port.

## Before EVERY code modification:

- Re-read `Principle/Commit-Standard.md`
- Read the matching Principle design doc for the target stage
- Verify the planned change complies with both
- Only then modify code

## Commit workflow (when adding new stages)

1. One commit = one stage. Format: `Cxxx: Module: Description`
2. Each commit must be runnable (`python run.py` or `pytest` must pass)
3. Create `docs/snapshots/Cxxx.md` for the new stage
4. Tag with `Cxxx-description` (e.g. `git tag C033-kernel`)
5. Never modify earlier stages' code; if a fix is needed, create a new fix commit
6. See `Principle/Commit-Standard.md` for the full rules
7. Read the matching `Principle/CXXX-CYYY/` design doc for stage-specific requirements

## Key architecture facts

- EngineCore `step()` = `schedule() → execute_model() → update_from_output()`
- KV cache is physical-block-based (BlockPool + FreeKVCacheBlockQueue, doubly-linked, O(1) alloc)
- Prefix caching uses `cached_block_hash_to_block` dict on BlockPool
- Chunked prefill splits long prompts by `token_budget` at the Scheduler level
- PagedAttention ops are pure PyTorch (`reshape_and_cache_pytorch`, `gather_cached_kv_pytorch`)
- Sampler is an `nn.Module` subclass with penalties → temperature → top-k → top-p → sampling
- Tensor parallel stubs degrade to identity when `world_size=1`

## Model dependency

Hardcoded to `facebook/opt-125m`. Tests load from the HuggingFace cache or the local path above. The model must be available for any run or test.
