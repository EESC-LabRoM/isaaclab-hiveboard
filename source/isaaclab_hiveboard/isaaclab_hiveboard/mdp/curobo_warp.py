"""Guard cuRobo's warp usage against simulator-owned CUDA/warp state."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def curobo_compatible_warp() -> Iterator[None]:
    """Ensure ``warp`` is initialized before cuRobo touches it.

    Isaac Newton/MJWarp owns the warp context inside sim processes while
    cuRobo also initializes warp for its collision kernels. A second init is
    safe to skip, so an already-initialized warp is tolerated here.
    """
    import warp

    try:
        warp.init()
    except Exception:  # noqa: BLE001 - warp already owned by the simulator stack
        pass
    yield
