"""tests/test_C255_kv_offload_base.py — KV offload base + config + factory + file_mapper.

C255: KV Offload base + config.

Verifies the plug-in surface for KV cache offloading backends:

* **OffloadKey encoding** — `make_offload_key` packs hash + group
  index into bytes; `get_offload_block_hash` / `get_offload_group_idx`
  round-trip.
* **Enum / dataclass hierarchy** — `LookupResult` (MISS / HIT /
  HIT_PENDING / RETRY), `OffloadPolicy` (BLOCK_LEVEL /
  REQUEST_LEVEL), `Locality` (LOCAL / REMOTE), `ReqContext`,
  `RequestOffloadingContext`, `ScheduleEndContext`,
  `PrepareStoreOutput`, `OffloadingEvent`.
* **LoadStoreSpec** — base class + `BlockIDsLoadStoreSpec`
  intermediate + `GPULoadStoreSpec` concrete (asserts
  `sum(group_sizes) == len(block_ids)` and
  `len(block_indices) == len(group_sizes)`).
* **Metric metadata** — `OffloadingMetricMetadata` +
  `Counter` / `Gauge` / `Histogram` subclasses.
* **Canonical KV cache types** — `CanonicalKVCacheTensor` /
  `CanonicalKVCacheRef` / `CanonicalKVCaches`.
* **`OffloadingConfig`** — frozen dataclass aggregation of
  groups / model / cache / parallel sub-configs.
* **`OffloadingSpecFactory`** — lazy-loading registry:
  - `register_spec` (idempotent guard: re-registering same name
    raises ValueError).
  - `get_spec_cls` resolves by name; falls back to
    `spec_module_path` for ad-hoc specs.
  - `create_spec` instantiates with config.
  - `reset` / `known_specs` test helpers.
* **`FileMapper`** — disk layout primitive:
  - `get_file_name` produces `<base>_r<rank>/<hhh>/<hh>_g<g>/<hex>.bin`.
  - `get_config_file_path` returns `<base>/config.json`.
  - `parallel_agnostic=True` (with `is_parallelism_agnostic=True`)
    collapses TP/PP/PCP/DCP to 1 + rank to 0 so multiple
    layouts share the same base path.
  - `base_path` SHA-256 prefix is deterministic + model-name safe
    (HF ID `/` → `_`).
* **OffloadingSpec construction** — reads `extra_config` for
  `self_describing_kv_events` + `offload_prompt_only` flags;
  aggregates `tokens_per_block` from groups; sets
  `tokens_per_hash` / `blocks_per_chunk` from cache config.
* **OffloadingManager ABC** — cannot instantiate without
  subclassing all 4 abstract methods (lookup / prepare_load /
  prepare_store / on_new_request).
* **OffloadingWorker ABC** — cannot instantiate without
  subclassing all 4 abstract methods (submit_store / submit_load /
  get_finished / wait).
* **OffloadingSpec ABC** — cannot instantiate without subclassing
  `get_manager` + `get_worker`.

C255 is the plug-in surface; concrete backends land with C256
(CPU/NVMe tiering) and C257 (Simple KV Offload).
"""

from __future__ import annotations

from typing import Type

import pytest

from my_vllm.v1.kv_offload import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCacheTensor,
    CanonicalKVCaches,
    FileMapper,
    GPULoadStoreSpec,
    Locality,
    LookupResult,
    LoadStoreSpec,
    OffloadKey,
    OffloadPolicy,
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingCounterMetadata,
    OffloadingEvent,
    OffloadingGaugeMetadata,
    OffloadingGroupConfig,
    OffloadingHistogramMetadata,
    OffloadingKVEventsConfig,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingModelConfig,
    OffloadingParallelConfig,
    OffloadingSpec,
    OffloadingSpecFactory,
    OffloadingWorker,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
    TransferResult,
    get_offload_block_hash,
    get_offload_group_idx,
    make_offload_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_offloading_config(
    *,
    extra_config: dict | None = None,
    groups: tuple[OffloadingGroupConfig, ...] | None = None,
    rank: int = 0,
    world_size: int = 1,
    tp_size: int = 1,
    pp_size: int = 1,
    pcp_size: int = 1,
    dcp_size: int = 1,
    dp_index: int = 0,
    is_parallelism_agnostic: bool = False,
    enable_kv_cache_events: bool = False,
    tokens_per_hash: int = 16,
    blocks_per_chunk: int = 1,
    worker_kv_bytes_per_block: int = 4096,
) -> OffloadingConfig:
    """Build a sane OffloadingConfig with the given overrides."""
    return OffloadingConfig(
        groups=groups or (
            OffloadingGroupConfig(
                tokens_per_block=16,
                layer_names=("layer.0", "layer.1"),
            ),
        ),
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        enable_kv_cache_events=enable_kv_cache_events,
        extra_config=extra_config or {},
        engine_id="engine-0",
        model=OffloadingModelConfig(
            name="facebook/opt-125m", dtype="float16",
        ),
        cache=OffloadingCacheConfig(
            tokens_per_hash=tokens_per_hash,
            blocks_per_chunk=blocks_per_chunk,
        ),
        parallel=OffloadingParallelConfig(
            rank=rank,
            world_size=world_size,
            tp_size=tp_size,
            pp_size=pp_size,
            pcp_size=pcp_size,
            dcp_size=dcp_size,
            data_parallel_index=dp_index,
            is_parallelism_agnostic=is_parallelism_agnostic,
        ),
    )


# ===========================================================================
# OffloadKey encoding tests
# ===========================================================================


class TestOffloadKeyEncoding:
    """Pack/unpack of OffloadKey (block hash + group index)."""

    def test_make_offload_key_round_trip(self):
        h = bytes.fromhex("deadbeef" * 4)
        key = make_offload_key(h, group_idx=0)
        assert get_offload_block_hash(key) == h
        assert get_offload_group_idx(key) == 0

    def test_make_offload_key_group_index_high(self):
        h = b"\x01\x02\x03\x04"
        # 32-bit max: 2^32 - 1 = 0xFFFFFFFF
        key = make_offload_key(h, group_idx=0xFFFFFFFF)
        assert get_offload_group_idx(key) == 0xFFFFFFFF

    def test_make_offload_key_multiple_groups(self):
        h = b"\xab" * 16
        keys = [make_offload_key(h, g) for g in range(8)]
        # Same hash + different group → different keys.
        assert len(set(keys)) == 8
        for g, k in enumerate(keys):
            assert get_offload_block_hash(k) == h
            assert get_offload_group_idx(k) == g

    def test_make_offload_key_empty_hash(self):
        # Edge case: empty hash + group index 0 → 4 bytes only.
        key = make_offload_key(b"", group_idx=0)
        assert get_offload_block_hash(key) == b""
        assert get_offload_group_idx(key) == 0

    def test_make_offload_key_appends_be_u32(self):
        # Verify the on-disk encoding is 4-byte big-endian unsigned int.
        h = b"\x00" * 4
        key = make_offload_key(h, group_idx=1)
        # Bytes after the 4-byte hash: 0x00 0x00 0x00 0x01.
        assert key[-4:] == b"\x00\x00\x00\x01"

    def test_make_offload_key_big_endian(self):
        # 0x01020304 → bytes 0x01 0x02 0x03 0x04 (BE).
        key = make_offload_key(b"", group_idx=0x01020304)
        assert key[-4:] == b"\x01\x02\x03\x04"


# ===========================================================================
# Enum + dataclass tests
# ===========================================================================


class TestEnumsAndDataclasses:
    """The non-ABC types in `base.py`."""

    def test_lookup_result_values(self):
        assert LookupResult.MISS.value == 1
        assert LookupResult.HIT.value == 2
        assert LookupResult.HIT_PENDING.value == 3
        assert LookupResult.RETRY.value == 4

    def test_lookup_result_distinct(self):
        assert len(set(LookupResult)) == 4

    def test_offload_policy_values(self):
        assert OffloadPolicy.BLOCK_LEVEL.value == "block_level"
        assert OffloadPolicy.REQUEST_LEVEL.value == "request_level"

    def test_locality_values(self):
        assert Locality.LOCAL.value == "LOCAL"
        assert Locality.REMOTE.value == "REMOTE"

    def test_req_context(self):
        ctx = ReqContext(req_id="r1")
        assert ctx.req_id == "r1"
        assert ctx.kv_transfer_params is None

        ctx2 = ReqContext(req_id="r2", kv_transfer_params={"x": 1})
        assert ctx2.kv_transfer_params == {"x": 1}

    def test_request_offloading_context_default(self):
        ctx = RequestOffloadingContext()
        assert ctx.policy == OffloadPolicy.BLOCK_LEVEL

    def test_request_offloading_context_request_level(self):
        ctx = RequestOffloadingContext(policy=OffloadPolicy.REQUEST_LEVEL)
        assert ctx.policy == OffloadPolicy.REQUEST_LEVEL

    def test_schedule_end_context(self):
        ctx = ScheduleEndContext(
            new_req_ids={"r1", "r2"},
            preempted_req_ids={"r3"},
        )
        assert ctx.new_req_ids == {"r1", "r2"}
        assert ctx.preempted_req_ids == {"r3"}

    def test_prepare_store_output(self):
        h = b"\x01\x02\x03\x04"
        key = make_offload_key(h, group_idx=0)
        out = PrepareStoreOutput(
            keys_to_store=[key],
            store_spec=LoadStoreSpec(),
            evicted_keys=[],
        )
        assert out.keys_to_store == [key]
        assert isinstance(out.store_spec, LoadStoreSpec)
        assert out.evicted_keys == []

    def test_offloading_event(self):
        h = b"\x01\x02\x03\x04"
        key = make_offload_key(h, group_idx=0)
        ev = OffloadingEvent(
            keys=[key], medium="cpu", removed=False,
            locality=Locality.LOCAL,
        )
        assert ev.medium == "cpu"
        assert not ev.removed
        assert ev.locality == Locality.LOCAL

    def test_transfer_result(self):
        r = TransferResult(job_id=1, success=True)
        assert r.job_id == 1
        assert r.success
        assert r.transfer_size is None
        assert r.transfer_time is None

    def test_offloading_kv_events_config(self):
        cfg = OffloadingKVEventsConfig(
            enable_kv_cache_events=True,
            self_describing_kv_events=False,
        )
        assert cfg.enable_kv_cache_events
        assert not cfg.self_describing_kv_events


# ===========================================================================
# LoadStoreSpec hierarchy tests
# ===========================================================================


class TestLoadStoreSpecHierarchy:
    """The LoadStoreSpec / BlockIDsLoadStoreSpec / GPULoadStoreSpec hierarchy."""

    def test_base_load_store_spec_repr(self):
        s = LoadStoreSpec()
        # Subclasses customize __repr__; base just shows class name.
        assert "LoadStoreSpec" in repr(s)

    def test_block_ids_load_store_spec_constructs(self):
        s = BlockIDsLoadStoreSpec([0, 1, 2, 3])
        assert list(s.block_ids) == [0, 1, 2, 3]

    def test_block_ids_load_store_spec_stores_int64(self):
        s = BlockIDsLoadStoreSpec([5, 6, 7])
        assert s.block_ids.dtype == "int64"

    def test_block_ids_load_store_spec_repr(self):
        s = BlockIDsLoadStoreSpec([10, 20, 30])
        # repr delegated to numpy array.
        assert "10" in repr(s)

    def test_gpu_load_store_spec_basic(self):
        # Single group, single block.
        s = GPULoadStoreSpec(
            block_ids=[0],
            group_sizes=[1],
            block_indices=[0],
        )
        assert s.group_sizes == [1]
        assert s.block_indices == [0]

    def test_gpu_load_store_spec_multi_group(self):
        # 3 groups × 2 blocks = 6 blocks.
        s = GPULoadStoreSpec(
            block_ids=[0, 1, 2, 3, 4, 5],
            group_sizes=[2, 2, 2],
            block_indices=[0, 2, 4],
        )
        assert s.group_sizes == [2, 2, 2]
        assert s.block_indices == [0, 2, 4]

    def test_gpu_load_store_spec_asserts_sum_group_sizes(self):
        with pytest.raises(AssertionError, match="sum\\(group_sizes\\)"):
            GPULoadStoreSpec(
                block_ids=[0, 1, 2],
                group_sizes=[1, 1],  # sum=2, but len=3
                block_indices=[0, 1],
            )

    def test_gpu_load_store_spec_asserts_block_indices_len(self):
        with pytest.raises(AssertionError, match="len\\(block_indices\\)"):
            GPULoadStoreSpec(
                block_ids=[0, 1, 2, 3],
                group_sizes=[2, 2],
                block_indices=[0],  # len=1 vs group_sizes len=2
            )


# ===========================================================================
# Metric metadata tests
# ===========================================================================


class TestOffloadingMetricMetadata:
    """The metric metadata hierarchy."""

    def test_base_metadata_fields(self):
        m = OffloadingMetricMetadata(
            documentation="Test metric",
            labelnames=("label_a",),
        )
        assert m.documentation == "Test metric"
        assert m.labelnames == ("label_a",)

    def test_counter_metadata_is_metadata(self):
        m = OffloadingCounterMetadata(documentation="Counter")
        assert isinstance(m, OffloadingMetricMetadata)

    def test_gauge_metadata_is_metadata(self):
        m = OffloadingGaugeMetadata(documentation="Gauge")
        assert isinstance(m, OffloadingMetricMetadata)

    def test_histogram_metadata_buckets(self):
        m = OffloadingHistogramMetadata(
            documentation="Histogram",
            buckets=(0.1, 0.5, 1.0, 5.0),
        )
        assert m.buckets == (0.1, 0.5, 1.0, 5.0)
        assert isinstance(m, OffloadingMetricMetadata)

    def test_histogram_metadata_buckets_default_none(self):
        m = OffloadingHistogramMetadata(documentation="Histogram")
        assert m.buckets is None


# ===========================================================================
# Canonical KV cache tests
# ===========================================================================


class TestCanonicalKVCacheTypes:
    """The CanonicalKVCacheTensor/Ref/Caches dataclasses."""

    def test_tensor_construction(self, tmp_path):
        import torch
        t = torch.empty(10, 4096, dtype=torch.int8)
        c = CanonicalKVCacheTensor(tensor=t, page_size_bytes=4096)
        assert c.tensor is t
        assert c.page_size_bytes == 4096

    def test_ref_construction(self):
        ref = CanonicalKVCacheRef(
            tensor_idx=0, page_size_bytes=4096,
        )
        assert ref.tensor_idx == 0
        assert ref.page_size_bytes == 4096

    def test_caches_construction(self, tmp_path):
        import torch
        t = torch.empty(10, 4096, dtype=torch.int8)
        c = CanonicalKVCaches(
            tensors=[CanonicalKVCacheTensor(t, 4096)],
            group_data_refs=[[CanonicalKVCacheRef(0, 4096)]],
        )
        assert len(c.tensors) == 1
        assert len(c.group_data_refs) == 1


# ===========================================================================
# OffloadingConfig tests
# ===========================================================================


class TestOffloadingConfig:
    """The OffloadingConfig + sub-configs."""

    def test_default_construction(self):
        cfg = _make_offloading_config()
        assert cfg.engine_id == "engine-0"
        assert cfg.model.name == "facebook/opt-125m"
        assert cfg.model.dtype == "float16"
        assert cfg.cache.tokens_per_hash == 16
        assert cfg.parallel.rank == 0
        assert cfg.parallel.world_size == 1

    def test_groups_is_tuple(self):
        cfg = _make_offloading_config()
        assert isinstance(cfg.groups, tuple)
        assert cfg.groups[0].tokens_per_block == 16

    def test_extra_config_passes_through(self):
        cfg = _make_offloading_config(
            extra_config={"spec_name": "X", "offload_prompt_only": False},
        )
        assert cfg.extra_config["spec_name"] == "X"
        assert cfg.extra_config["offload_prompt_only"] is False

    def test_parallel_config_fields(self):
        cfg = _make_offloading_config(
            rank=3, world_size=8, tp_size=4, pp_size=2,
            pcp_size=1, dcp_size=2, dp_index=5,
            is_parallelism_agnostic=True,
        )
        assert cfg.parallel.rank == 3
        assert cfg.parallel.world_size == 8
        assert cfg.parallel.tp_size == 4
        assert cfg.parallel.pp_size == 2
        assert cfg.parallel.dcp_size == 2
        assert cfg.parallel.data_parallel_index == 5
        assert cfg.parallel.is_parallelism_agnostic

    def test_config_is_frozen(self):
        cfg = _make_offloading_config()
        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.engine_id = "other"  # type: ignore[misc]


# ===========================================================================
# OffloadingSpecFactory tests
# ===========================================================================


class TestOffloadingSpecFactory:
    """The lazy-loading spec registry + factory."""

    def setup_method(self):
        OffloadingSpecFactory.reset()

    def teardown_method(self):
        OffloadingSpecFactory.reset()

    def test_register_spec_rejects_duplicate(self):
        OffloadingSpecFactory.register_spec(
            "TestSpecA",
            "tests.test_C255_kv_offload_base",
            "_DummySpec",
        )
        with pytest.raises(ValueError, match="already registered"):
            OffloadingSpecFactory.register_spec(
                "TestSpecA",
                "tests.test_C255_kv_offload_base",
                "_DummySpec",
            )

    def test_get_spec_cls_raises_for_unknown(self):
        # Unknown name + no spec_module_path → ValueError.
        with pytest.raises(ValueError, match="Unsupported spec type"):
            OffloadingSpecFactory.get_spec_cls(
                {"spec_name": "_UnknownSpec_MnoPqr"}
            )

    def test_get_spec_cls_default_name(self):
        # No "spec_name" in extra_config → defaults to "CPUOffloadingSpec".
        # CPUOffloadingSpec IS pre-registered as a built-in (C256), so
        # the lookup succeeds.
        spec_cls = OffloadingSpecFactory.get_spec_cls({})
        assert spec_cls.__name__ == "CPUOffloadingSpec"

    def test_get_spec_cls_via_spec_module_path(self):
        # Ad-hoc registration via extra_config["spec_module_path"].
        spec_cls = OffloadingSpecFactory.get_spec_cls({
            "spec_name": "_DummySpec",
            "spec_module_path": "tests.test_C255_kv_offload_base",
        })
        # Compare by qualified name + module identity — pytest may
        # import the test module twice (once via pytest's discovery,
        # once via importlib) producing two module objects each
        # with their own _DummySpec class. `is` fails across these.
        assert spec_cls.__name__ == _DummySpec.__name__
        assert issubclass(spec_cls, OffloadingSpec)

    def test_get_spec_cls_via_register(self):
        OffloadingSpecFactory.register_spec(
            "_DummySpec",
            "tests.test_C255_kv_offload_base",
            "_DummySpec",
        )
        spec_cls = OffloadingSpecFactory.get_spec_cls({"spec_name": "_DummySpec"})
        assert spec_cls.__name__ == _DummySpec.__name__
        assert issubclass(spec_cls, OffloadingSpec)

    def test_create_spec_via_register(self):
        OffloadingSpecFactory.register_spec(
            "_DummySpec",
            "tests.test_C255_kv_offload_base",
            "_DummySpec",
        )
        cfg = _make_offloading_config(
            extra_config={"spec_name": "_DummySpec"},
        )
        spec = OffloadingSpecFactory.create_spec(cfg)
        assert isinstance(spec, OffloadingSpec)
        # Spec's config round-trips through.
        assert spec.config is cfg

    def test_create_spec_via_spec_module_path(self):
        cfg = _make_offloading_config(
            extra_config={
                "spec_name": "_DummySpec",
                "spec_module_path": "tests.test_C255_kv_offload_base",
            },
        )
        spec = OffloadingSpecFactory.create_spec(cfg)
        assert isinstance(spec, OffloadingSpec)

    def test_create_spec_default_name(self):
        # Default name → "CPUOffloadingSpec" → pre-registered as a
        # built-in (C256), so lookup succeeds. The instantiation
        # itself fails because cpu_bytes_to_use is required.
        cfg = _make_offloading_config()
        with pytest.raises(Exception, match="cpu_bytes_to_use"):
            OffloadingSpecFactory.create_spec(cfg)

    def test_known_specs(self):
        OffloadingSpecFactory.register_spec(
            "_DummySpec",
            "tests.test_C255_kv_offload_base",
            "_DummySpec",
        )
        assert "_DummySpec" in OffloadingSpecFactory.known_specs()

    def test_lazy_loading(self):
        # Spec is registered but the module is NOT imported until lookup.
        # We verify by registering with a bogus module path; the lookup
        # is what triggers the import.
        OffloadingSpecFactory.register_spec(
            "_LazySpec", "nonexistent.module.path", "SomeClass",
        )
        # Registering alone must not raise (lazy).
        assert "_LazySpec" in OffloadingSpecFactory.known_specs()
        # Lookup triggers the import → ModuleNotFoundError.
        with pytest.raises(ModuleNotFoundError):
            OffloadingSpecFactory.get_spec_cls({"spec_name": "_LazySpec"})


# ===========================================================================
# FileMapper tests
# ===========================================================================


class TestFileMapper:
    """The disk layout primitive."""

    def test_basic_file_name(self):
        m = FileMapper(
            root_dir="/tmp",
            model_name="facebook/opt-125m",
            tokens_per_hash=16,
            blocks_per_file=1,
            tp_size=1, pp_size=1, pcp_size=1, dcp_size=1,
            rank=0, dtype="float16",
        )
        h = bytes.fromhex("deadbeef" * 4)
        key = make_offload_key(h, group_idx=0)
        fname = m.get_file_name(key)
        # Layout: <base>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash>.bin
        assert fname.endswith("_r0/dea/db_g0/deadbeefdeadbeefdeadbeefdeadbeef.bin")

    def test_file_name_with_different_groups(self):
        m = FileMapper(
            root_dir="/tmp",
            model_name="m",
            tokens_per_hash=16,
            blocks_per_file=1,
            tp_size=1, pp_size=1, pcp_size=1, dcp_size=1,
            rank=0, dtype="float16",
        )
        h = b"\xab" * 16
        for g in range(3):
            fname = m.get_file_name(make_offload_key(h, g))
            assert f"_g{g}/" in fname

    def test_rank_appears_in_path(self):
        m = FileMapper(
            root_dir="/tmp",
            model_name="m",
            tokens_per_hash=16,
            blocks_per_file=1,
            tp_size=4, pp_size=1, pcp_size=1, dcp_size=1,
            rank=2, dtype="float16",
        )
        h = b"\x01" * 16
        fname = m.get_file_name(make_offload_key(h, 0))
        assert "_r2/" in fname

    def test_config_file_path(self):
        m = FileMapper(
            root_dir="/tmp/kv",
            model_name="m",
            tokens_per_hash=16,
            blocks_per_file=1,
            tp_size=1, pp_size=1, pcp_size=1, dcp_size=1,
            rank=0, dtype="float16",
        )
        path = m.get_config_file_path()
        assert path.endswith("/config.json")
        # base_path lives directly under root_dir + safe_model_name + digest.
        assert path.startswith("/tmp/kv/m_")

    def test_base_path_safe_for_hf_id(self):
        # HuggingFace IDs like "org/model" → "org_model" (no nesting).
        m = FileMapper(
            root_dir="/tmp",
            model_name="org/model",
            tokens_per_hash=16,
            blocks_per_file=1,
            tp_size=1, pp_size=1, pcp_size=1, dcp_size=1,
            rank=0, dtype="float16",
        )
        # base_path = "/tmp/org_model_<digest>".
        assert m.base_path.startswith("/tmp/org_model_")

    def test_base_path_digest_deterministic(self):
        # Same fields → same digest.
        kwargs = dict(
            root_dir="/tmp",
            model_name="m",
            tokens_per_hash=16,
            blocks_per_file=1,
            tp_size=1, pp_size=1, pcp_size=1, dcp_size=1,
            rank=0, dtype="float16",
        )
        m1 = FileMapper(**kwargs)
        m2 = FileMapper(**kwargs)
        assert m1.base_path == m2.base_path

    def test_base_path_changes_with_fields(self):
        kwargs = dict(
            root_dir="/tmp",
            model_name="m",
            tokens_per_hash=16,
            blocks_per_file=1,
            tp_size=1, pp_size=1, pcp_size=1, dcp_size=1,
            rank=0, dtype="float16",
        )
        m1 = FileMapper(**kwargs)
        m2 = FileMapper(**{**kwargs, "dtype": "float32"})
        assert m1.base_path != m2.base_path

    def test_parallel_agnostic_collapses_layouts(self):
        # When parallel_agnostic=True, TP/PP/PCP/DCP all → 1 and rank → 0.
        m = FileMapper(
            root_dir="/tmp",
            model_name="m",
            tokens_per_hash=16,
            blocks_per_file=1,
            tp_size=4, pp_size=2, pcp_size=2, dcp_size=2,
            rank=3, dtype="float16",
            parallel_agnostic=True,
        )
        assert m.rank == 0
        # tp/pp/pcp/dcp collapsed to 1 → base_path identical to a TP=1 deployment.
        m_tp1 = FileMapper(
            root_dir="/tmp",
            model_name="m",
            tokens_per_hash=16,
            blocks_per_file=1,
            tp_size=1, pp_size=1, pcp_size=1, dcp_size=1,
            rank=0, dtype="float16",
        )
        assert m.base_path == m_tp1.base_path

    def test_get_run_config_includes_all_fields(self):
        m = FileMapper(
            root_dir="/tmp",
            model_name="m",
            tokens_per_hash=16,
            blocks_per_file=1,
            tp_size=1, pp_size=1, pcp_size=1, dcp_size=1,
            rank=0, dtype="float16",
            kv_cache_groups=[{"tokens_per_block": 16, "layer_names": ["a"]}],
        )
        cfg = m.get_run_config()
        assert cfg["model_name"] == "m"
        assert cfg["tp_size"] == 1
        assert cfg["kv_cache_groups"] == [{"tokens_per_block": 16, "layer_names": ["a"]}]
        assert cfg["inference_engine"] == "my-vllm"

    def test_from_offloading_spec_constructs(self):
        cfg = _make_offloading_config()
        spec = _DummySpec(cfg)
        m = FileMapper.from_offloading_spec("/tmp", spec)
        assert m.rank == 0
        assert "facebook/opt-125m" in m.base_path or "facebook_opt-125m" in m.base_path


# ===========================================================================
# OffloadingSpec tests
# ===========================================================================


class TestOffloadingSpec:
    """The OffloadingSpec base class surface."""

    def test_construction_basic(self):
        cfg = _make_offloading_config(
            enable_kv_cache_events=True,
            extra_config={
                "self_describing_kv_events": True,
                "offload_prompt_only": False,
            },
        )
        spec = _DummySpec(cfg)
        assert spec.config is cfg
        assert spec.kv_events_config.enable_kv_cache_events
        assert spec.kv_events_config.self_describing_kv_events
        assert spec.offload_prompt_only is False
        assert spec.tokens_per_block == (16,)
        assert spec.tokens_per_hash == 16
        assert spec.blocks_per_chunk == 1

    def test_construction_default_flags(self):
        cfg = _make_offloading_config()
        spec = _DummySpec(cfg)
        # Default offload_prompt_only = True (matches vLLM).
        assert spec.offload_prompt_only is True
        assert not spec.kv_events_config.enable_kv_cache_events
        assert not spec.kv_events_config.self_describing_kv_events

    def test_tokens_per_block_aggregates_groups(self):
        groups = (
            OffloadingGroupConfig(tokens_per_block=16, layer_names=("a",)),
            OffloadingGroupConfig(tokens_per_block=32, layer_names=("b",)),
        )
        cfg = _make_offloading_config(groups=groups)
        spec = _DummySpec(cfg)
        assert spec.tokens_per_block == (16, 32)

    def test_build_metric_definitions_default_empty(self):
        cfg = _make_offloading_config()
        spec = _DummySpec(cfg)
        assert spec.build_metric_definitions({}) == {}


# ===========================================================================
# ABC enforcement tests
# ===========================================================================


class TestABCEnforcement:
    """ABC instantiation requires subclassing all abstract methods."""

    def test_offloading_manager_cannot_instantiate(self):
        with pytest.raises(TypeError):
            OffloadingManager()  # type: ignore[abstract]

    def test_offloading_worker_cannot_instantiate(self):
        with pytest.raises(TypeError):
            OffloadingWorker()  # type: ignore[abstract]

    def test_offloading_spec_cannot_instantiate(self):
        cfg = _make_offloading_config()
        with pytest.raises(TypeError):
            OffloadingSpec(cfg)  # type: ignore[abstract]

    def test_partial_offloading_manager_still_abstract(self):
        # Only 3 of 4 abstract methods implemented.
        class PartialManager(OffloadingManager):
            def lookup(self, key, req_context):
                return LookupResult.MISS

            def prepare_load(self, keys, req_context):
                return LoadStoreSpec()

            def prepare_store(self, keys, req_context):
                return None

            # Missing: on_new_request

        ctx = ReqContext(req_id="r1")
        with pytest.raises(TypeError):
            PartialManager()  # type: ignore[abstract]
        # Once we add the last one, instantiation succeeds.
        class CompleteManager(PartialManager):
            def on_new_request(self, req_context):
                return RequestOffloadingContext()

        m = CompleteManager()
        assert m.lookup(make_offload_key(b"x", 0), ctx) == LookupResult.MISS

    def test_partial_offloading_worker_still_abstract(self):
        class PartialWorker(OffloadingWorker):
            def submit_store(self, job_id, src_spec, dst_spec):
                return True

            def submit_load(self, job_id, src_spec, dst_spec):
                return True

            def get_finished(self):
                return []

            # Missing: wait

        with pytest.raises(TypeError):
            PartialWorker()  # type: ignore[abstract]

    def test_default_hooks_are_no_ops(self):
        # All "default" hooks return None / ().
        spec = _DummySpec(_make_offloading_config())
        # Spec's __init__ ran successfully — no exception.

        # OffloadingManager default hooks:
        m = _DummyManager()
        ctx = ReqContext(req_id="r1")
        assert m.touch([], ctx) is None
        assert m.complete_load([], ctx) is None
        assert m.complete_store([], ctx) is None
        assert m.on_request_finished(ctx) is None
        assert list(m.take_events()) == []
        m.on_schedule_end(
            ScheduleEndContext(new_req_ids=set(), preempted_req_ids=set())
        )
        assert m.has_pending_work() is False
        assert m.reset_cache() is None
        assert m.get_stats() is None
        assert m.shutdown() is None

        # OffloadingWorker default hooks:
        w = _DummyWorker()
        assert w.shutdown() is None


# ===========================================================================
# OffloadingSpec configuration helper tests
# ===========================================================================


class TestOffloadingSpecConfiguration:
    """The constructor-time extra_config consumption."""

    def test_self_describing_kv_events_default_false(self):
        cfg = _make_offloading_config(enable_kv_cache_events=True)
        spec = _DummySpec(cfg)
        assert spec.kv_events_config.self_describing_kv_events is False

    def test_self_describing_kv_events_enabled_via_extra_config(self):
        cfg = _make_offloading_config(
            enable_kv_cache_events=True,
            extra_config={"self_describing_kv_events": True},
        )
        spec = _DummySpec(cfg)
        assert spec.kv_events_config.self_describing_kv_events is True

    def test_offload_prompt_only_string_coerced(self):
        cfg = _make_offloading_config(
            extra_config={"offload_prompt_only": "yes"},
        )
        spec = _DummySpec(cfg)
        # bool("yes") = True.
        assert spec.offload_prompt_only is True


# ===========================================================================
# Test-only concrete subclasses (must live at module scope for
# OffloadingSpecFactory lazy-loading to find them via getattr).
# ===========================================================================


class _DummySpec(OffloadingSpec):
    """Concrete spec for tests — does NOT implement get_worker/get_manager."""

    def get_manager(self):
        return _DummyManager()

    def get_worker(self, kv_caches):
        return _DummyWorker()


class _DummyManager(OffloadingManager):
    """Concrete manager that uses the ABC defaults + returns MISS."""

    def lookup(self, key, req_context):
        return LookupResult.MISS

    def prepare_load(self, keys, req_context):
        return LoadStoreSpec()

    def prepare_store(self, keys, req_context):
        return None

    def on_new_request(self, req_context):
        return RequestOffloadingContext()


class _DummyWorker(OffloadingWorker):
    """Concrete worker using ABC defaults."""

    def submit_store(self, job_id, src_spec, dst_spec):
        return True

    def submit_load(self, job_id, src_spec, dst_spec):
        return True

    def get_finished(self):
        return []

    def wait(self, job_ids):
        return None