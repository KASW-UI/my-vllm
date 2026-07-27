from dataclasses import dataclass


@dataclass
class KVCacheBlock:
    block_id: int
    is_free: bool = True
