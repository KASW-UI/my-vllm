# C200-prep: docs — AGENTS.md sync JD gap closure plan (C200-C233)

## Goal

执行 `jd-gap-closure-plan.md` (仓库根，untracked) 之前的 AGENTS.md
同步。Per AGENTS.md「AGENTS.md numbering alignment」规则：

> Whenever a new Cxxx-numbered planning doc (or external numbering
> scheme) shifts, sync AGENTS.md in the same commit window. Pattern:
> small standalone `Cxxx-prep` docs commits (e.g. `C153-prep: AGENTS.md
> sync ...`) rather than rolling the sync into the next real commit.

本次 sync 对应 C200-C233 编号方案落入仓库，AGENTS.md 现状（C151 已完
成，Next 仍指向已删除的 K8s deploy/ 计划）需要更新。

## File Tree (diff)

```
M  AGENTS.md
```

净改动：+20 / -1 行。

## What this commit did

1. **新增 JD gap analysis 段** (Current status 之后, Next 之前)：
   量化覆盖率 50-55%，列出 8 项缺口（profile / NCCL / cache ops /
   unified attn 深度 / PD e2e / custom all-reduce / spec_decode /
   elastic_ep / rotary / VMM），声明 plan 在 `jd-gap-closure-plan.md`，
   标注 Triton vs CUDA 分布依据。

2. **重写 Next 段**：从原"K8s 部署 (C153-C159) → 阶段 B 欠账"
   改为"JD gap closure (C200-C233, 16 commit) 三阶段（Phase 1/2/3）
   + K8s 降级为 backlog"。Phase 1/2/3 各自的内容一项项列出，让
   `git checkout C200-prep` 时能立即看到后续 16 个 commit 的轮廓。

3. **Repository conventions 段**：untracked 计划文档列表追加
   `jd-gap-closure-plan.md`，与 `docs/cleanup-plan.md` /
   `k8s-deploy-plan.md` 同等地位。

## Verification

```
$ git diff --stat AGENTS.md
 AGENTS.md | 21 ++++++++++++++++++++-
 1 file changed, 20 insertions(+), 1 deletion(-)

$ grep -c "C200\|C233\|jd-gap-closure" AGENTS.md
7   # 7 处引用，覆盖 plan 名称 / phase 1 / phase 2 / phase 3 / backlog / repo conventions

$ python run.py   # 仅 AGENTS.md 改动，不影响代码路径，应维持原状态
(still green)

$ pytest tests/ -q
(still green — no source change)
```

## Current Capability

- ✅ AGENTS.md 反映 C151 → C200-C233 跳转意图
- ✅ JD gap closure plan 在 AGENTS.md Next 段被指明
- ✅ Triton vs CUDA 分布依据记录在案（commit message + snapshot）
- ✅ K8s 计划降级为 backlog（per 用户决策）
- ✅ 三阶段（Phase 1 硬性 + GPU Runtime / Phase 2 架构 + 加分 /
   Phase 3 收尾）结构清晰

## Current Limitation

- 仅 docs 改动，无任何代码 / 性能变化。`jd-gap-closure-plan.md` 仍
  untracked（与 `k8s-deploy-plan.md` 同状态），无 source diff 记录
  计划本身。
- plan 中的 vLLM 参考路径尚未逐行核对（执行 C200-C233 时逐 commit
  核对）。

## Next Stage

进入 `C200-profile: GPU profiling 工具链 + roofline 实验`。

按用户决策（2026-08-14）：
1. **Phase 1 硬性 + GPU Runtime 深度** 顺序：先做 C200 + C211 +
   C210 三件套（profile + unified attention 深度验证 + cache_kernels
   多 op 升级）作为 Phase 1 前半；C201/212/213/214 后续。
2. K8s 计划 (C153-C159) **后置**，等 C200-C233 全闭环再启动。
3. 本次 AGENTS.md 同步作为单独 `C200-prep` commit（已执行）。