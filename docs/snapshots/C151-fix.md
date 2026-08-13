# C151-fix: tests — remove dead OpenAI api_server references from test_C127

## Goal

修复 C151 阶段遗留：删除 `my_vllm/entrypoints/openai/api_server.py`
时未同步清理 `tests/test_C127_async_llm_busy_loop.py` 中的引用，导致
C151 之后 `pytest tests/` 在 collection 阶段失败：

```
tests/test_C127_async_llm_busy_loop.py:18: in <module>
    from my_vllm.entrypoints.openai.api_server import create_server
E   ModuleNotFoundError: No module named 'my_vllm.entrypoints.openai'
```

Per AGENTS.md § Commit workflow rule 5（不允许修改历史阶段；如需修复
另起新 commit），本 commit 是 C151 的 follow-up fix，与 C200 解耦。

## What this commit did

1. `tests/test_C127_async_llm_busy_loop.py:18` — 删除
   `from my_vllm.entrypoints.openai.api_server import create_server`。
2. 删除 `TestApiServerUsesBusyLoop` 整个 class（含
   `test_create_server_accepts_busy_loop_engine`），因为它测试的
   `create_server` / `CompletionHandler` 在 C151 已被删。
3. 更新模块 docstring，反映测试范围（AsyncLLM BusyLoopClient 直
   接验证，不再涉及 OpenAI server 路径）。

## Verification

```
$ pytest tests/test_C127_async_llm_busy_loop.py -v
1 failed (TP state leak from prior test in same session — pre-existing
test isolation issue, not introduced by this fix)
6 passed

$ pytest tests/test_C127_async_llm_busy_loop.py::TestLLMEngineBusyLoopConstruction::test_busy_loop_true_yields_busy_loop_client -v
1 passed   # isolated run works
```

## Current Limitation

`TestLLMEngineBusyLoopConstruction::test_busy_loop_true_yields_busy_loop_client`
in the full session fails due to test isolation（前面某个 test 调用
`init_distributed_environment` 后没清理 `_TP` 全局变量）。这是
pre-existing 问题，不在本 commit 修复范围。后续 C232 笔记集阶段可
统一处理。

## Next Stage

`C200-profile: GPU profiling 工具链 + roofline 实验`（与本 fix 解耦）。