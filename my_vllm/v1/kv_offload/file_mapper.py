"""KV cache offloading — file layout mapping for disk-tier backends.

Mirrors vLLM V1 vllm/v1/kv_offload/file_mapper.py.

C255: `FileMapper` maps `OffloadKey` (block hash + group index) to
a deterministic file path under a configurable root directory.
The layout is engineered so different parallelism layouts can
coexist on disk while still letting identical layouts collapse
into a single shared directory.

Layout:

    <root_dir>/<safe_model_name>_<sha256-prefix>/
        ├── config.json
        └── _r<rank>/
            └── <hhh>/<hh>_g<group_idx>/
                └── <hash>.bin

Where `<hhh>/<hh>` are the first 3 / next 2 hex chars of the
block hash (sub-folder sharding for filesystem fan-out). The
SHA-256 prefix of the canonical config JSON identifies the
(parallelism, dtype, model) tuple; multiple workers sharing the
same canonical config collapse into the same `_r<rank>`
subfolder.

When `parallel_agnostic=True` (and `config.parallel
.is_parallelism_agnostic=True`), all parallelism fields are
forced to 1 and rank to 0 — so e.g. a TP=2 deployment and a TP=4
deployment of the same model land on the same folder, with
different rank-fan-out subfolders each.

my-vllm NOTE: we don't ship a concrete disk-tier spec at C255 —
`FileMapper` is the layout primitive that future `NVMeOffloadingSpec`
(or similar) backends consume.
"""

from __future__ import annotations

import hashlib
import json

from my_vllm.v1.kv_offload.base import (
    OffloadKey,
    OffloadingSpec,
    get_offload_block_hash,
    get_offload_group_idx,
)


_BASE_PATH_HASH_LEN = 12
_CONFIG_FILENAME = "config.json"


class FileMapper:
    """Maps KV blocks (by hash) to file names on disk.

    Mirrors vLLM V1 vllm/v1/kv_offload/file_mapper.py:FileMapper.

    Each worker constructs its own instance but the `config.json`
    is shared across workers (rank lives outside the hash, so
    workers don't write into each other's folder).
    """

    def __init__(
        self,
        root_dir: str,
        model_name: str,
        tokens_per_hash: int,
        blocks_per_file: int,
        tp_size: int,
        pp_size: int,
        pcp_size: int,
        dcp_size: int,
        rank: int,
        dtype: str,
        kv_cache_groups: list[dict] | None = None,
        inference_engine: str = "my-vllm",
        parallel_agnostic: bool = False,
    ):
        if parallel_agnostic:
            tp_size = pp_size = pcp_size = dcp_size = 1
            rank = 0
        self.rank: int = rank
        self.fields: dict = {
            "model_name": model_name,
            "tokens_per_hash": tokens_per_hash,
            "blocks_per_file": blocks_per_file,
            "tp_size": tp_size,
            "pp_size": pp_size,
            "pcp_size": pcp_size,
            "dcp_size": dcp_size,
            "dtype": str(dtype),
            "kv_cache_groups": kv_cache_groups or [],
            "inference_engine": inference_engine,
        }
        self.base_path: str = self._compute_base_path(root_dir, self.fields)

    @classmethod
    def from_offloading_spec(
        cls,
        root_dir: str,
        offloading_spec: OffloadingSpec,
        blocks_per_file: int = 1,
        parallel_agnostic: bool = False,
    ) -> "FileMapper":
        """Build a FileMapper from an OffloadingSpec + root dir.

        Reads the spec's `OffloadingConfig` and translates it into
        the FileMapper constructor args.
        """
        config = offloading_spec.config
        kv_cache_groups = [
            {
                "tokens_per_block": group.tokens_per_block,
                "layer_names": list(group.layer_names),
            }
            for group in config.groups
        ]
        parallel = config.parallel
        return cls(
            root_dir=root_dir,
            model_name=config.model.name,
            tokens_per_hash=config.cache.tokens_per_hash,
            blocks_per_file=blocks_per_file,
            tp_size=parallel.tp_size,
            pp_size=parallel.pp_size,
            pcp_size=parallel.pcp_size,
            dcp_size=parallel.dcp_size,
            rank=parallel.rank,
            dtype=config.model.dtype,
            kv_cache_groups=kv_cache_groups,
            parallel_agnostic=(
                parallel_agnostic and parallel.is_parallelism_agnostic
            ),
        )

    def get_file_name(self, key: OffloadKey) -> str:
        """Map an `OffloadKey` to its file path under this mapper's base."""
        hash_hex = get_offload_block_hash(key).hex()
        group_idx = get_offload_group_idx(key)
        subfolder1, subfolder2 = hash_hex[:3], hash_hex[3:5]
        return (
            f"{self.base_path}_r{self.rank}"
            f"/{subfolder1}/{subfolder2}_g{group_idx}/{hash_hex}.bin"
        )

    def get_run_config(self) -> dict:
        """Return the canonical config fields (for `config.json`)."""
        return dict(self.fields)

    def get_config_file_path(self) -> str:
        """Return the path to the canonical `config.json`."""
        return f"{self.base_path}/{_CONFIG_FILENAME}"

    @staticmethod
    def _compute_base_path(root_dir: str, fields: dict) -> str:
        """Compute the base path from canonical fields.

        Layout: `<root_dir>/<safe_model_name>_<sha256-prefix>/`.
        `safe_model_name` replaces `/` with `_` so HuggingFace IDs
        don't nest.
        """
        canonical = json.dumps(
            fields, sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[
            :_BASE_PATH_HASH_LEN
        ]
        safe_model_name = fields["model_name"].replace("/", "_")
        return f"{root_dir}/{safe_model_name}_{digest}"


__all__ = ["FileMapper"]