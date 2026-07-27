"""Bridge: re-exports Request and RequestStatus from v1 namespace.

C015: Request migrated to my_vllm/v1/request.py.
This module exists for backward compatibility until all
callers are updated to import from v1 directly (C016+).
"""

from my_vllm.v1.request import Request, RequestStatus  # noqa: F401
