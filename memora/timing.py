"""Optional stderr timing traces for slow integration diagnostics."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

_TRUE_VALUES = {"1", "true", "yes", "on"}


def timing_enabled() -> bool:
    return os.environ.get("MEMORA_TRACE_TIMING", "").strip().lower() in _TRUE_VALUES


@contextmanager
def trace_timing(label: str) -> Iterator[None]:
    if not timing_enabled():
        yield
        return
    print(f"[memora-timing] start {label}", file=sys.stderr, flush=True)
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        print(f"[memora-timing] done {label} elapsed={elapsed:.2f}s", file=sys.stderr, flush=True)
