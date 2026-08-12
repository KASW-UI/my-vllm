"""Multi-tenant isolation config — aligns with vLLM scheduling layer tenant isolation.

JD alignment:
  - 多租户混布推理: 同一集群上多个租户共享 GPU 资源
  - 优化大模型推理成本: 通过混布提高 GPU 利用率

vLLM approach (per user intel):
  Multi-tenant is NOT a kernel-level concern — no dedicated CUDA/Triton kernels.
  It is handled at the scheduling layer (Ray/K8s resource orchestration):
    - Each tenant gets a resource quota (GPU slots, KV cache budget)
    - Priority-based scheduling between tenants
    - LoRA adapter isolation: each tenant loads different LoRA adapters
    - KV cache namespace isolation for security

Design:
  TenantConfig describes per-tenant resource limits.
  MultiTenantConfig aggregates all tenants and their policies.
  Integrated with Scheduler at schedule() time via resource budget checks.
"""

from dataclasses import dataclass, field


@dataclass
class TenantConfig:
    """Per-tenant resource limits and isolation parameters.

    Aligns with vLLM's per-user quota concepts in Ray/K8s deployment layer.
    Each tenant gets guaranteed minimums and optional burst capacity.
    """

    tenant_id: str
    name: str = ""

    # Resource quotas
    max_gpu_slots: int = 2          # max concurrent requests on GPU
    max_kv_cache_blocks: int = 256   # dedicated KV cache blocks
    max_tokens_per_request: int = 2048

    # Priority — higher = scheduled first
    priority: int = 0

    # LoRA adapter isolation
    lora_adapters: list[str] = field(default_factory=list)
    """LoRA adapters loaded for this tenant. Each adapter modifies a subset
    of the base model weights. Multiple tenants can share the same base model
    while using different LoRA adapters — this is the primary mechanism for
    multi-tenant model customization without duplicating the full model."""

    # KV cache namespace isolation
    kv_cache_namespace: str = ""
    """Prefix added to KV cache block keys to prevent cross-tenant cache
    pollution. Defaults to tenant_id if empty."""

    def __post_init__(self):
        if not self.kv_cache_namespace:
            self.kv_cache_namespace = f"tenant:{self.tenant_id}"


@dataclass
class MultiTenantConfig:
    """Aggregates all tenant configurations with scheduling policy.

    Aligns with vLLM deployment-level multi-tenancy configuration.
    The Scheduler queries this config at schedule() time to enforce
    per-tenant resource limits and implement fair-sharing between tenants.
    """

    enabled: bool = False
    tenants: list[TenantConfig] = field(default_factory=list)

    # Scheduling policy for multi-tenant
    policy: str = "fair_share"        # "fair_share" | "strict_priority" | "weighted"

    # Preemption policy
    preempt_lower_priority: bool = True
    """If True, a high-priority tenant can preempt a lower-priority tenant's
    request when resources are scarce. If False, all tenants are equal."""

    # Burst capacity
    allow_burst: bool = True
    """If True, tenants can temporarily exceed their quota when the cluster
    has idle resources. Prevents GPU underutilization."""

    def get_tenant(self, tenant_id: str) -> TenantConfig | None:
        for t in self.tenants:
            if t.tenant_id == tenant_id:
                return t
        return None

    @property
    def total_gpu_slots(self) -> int:
        return sum(t.max_gpu_slots for t in self.tenants)

    @property
    def total_kv_cache_blocks(self) -> int:
        return sum(t.max_kv_cache_blocks for t in self.tenants)


def schedule_order(
    tenants: list[TenantConfig],
    policy: str = "fair_share",
) -> list[TenantConfig]:
    """Return tenants in the order they should be scheduled.

    policy="fair_share": round-robin ordering, interleaved at request level.
    policy="strict_priority": highest priority first, preempt lower if needed.
    policy="weighted": probability of scheduling ∝ priority.

    Aligns with vLLM RequestQueue scheduling policy dispatch:
      v1/core/sched/request_queue.py SchedulingPolicy enum.
    """
    if policy == "strict_priority":
        return sorted(tenants, key=lambda t: -t.priority)

    if policy == "weighted":
        import random
        total = sum(t.priority + 1 for t in tenants)
        result: list[TenantConfig] = []
        remaining = list(tenants)
        while remaining:
            r = random.random() * total
            cumulative = 0.0
            for t in remaining:
                cumulative += (t.priority + 1) / total
                if r <= cumulative:
                    result.append(t)
                    remaining.remove(t)
                    total = sum(tt.priority + 1 for tt in remaining)
                    break
        return result

    # fair_share default: preserve insertion order, interleaved
    return list(tenants)
