# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_DFLASH_SCHEDULER_INIT: ContextVar[bool] = ContextVar(
    "dflash_scheduler_init",
    default=False,
)


@contextmanager
def dflash_scheduler_init_scope() -> Iterator[None]:
    """Mark KV manager construction triggered by a DFlash scheduler."""
    token = _DFLASH_SCHEDULER_INIT.set(True)
    try:
        yield
    finally:
        _DFLASH_SCHEDULER_INIT.reset(token)


def resolve_kv_use_eagle(use_eagle: bool) -> bool:
    """Keep EAGLE scheduling for DFlash without EAGLE KV-tail semantics."""
    return use_eagle and not _DFLASH_SCHEDULER_INIT.get()
