"""Make cuRobo v2 importable after Isaac Sim has replaced ``warp``.

Isaac Sim 5.1 loads ``omni.warp.core`` 1.8.x into ``sys.modules['warp']``.
cuRobo v2 registers collision overloads with ``wp.func(..., module=...)``,
which only exists in warp-lang >= 1.9 (this env has 1.11.1).
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager


def _warp_func_supports_module(wp_mod) -> bool:
    try:
        return "module" in inspect.signature(wp_mod.func).parameters
    except (TypeError, ValueError, AttributeError):
        return False


@contextmanager
def curobo_compatible_warp():
    """Use site-packages warp-lang while importing / constructing cuRobo.

    Isaac Sim's Warp stays in ``sys.modules`` afterwards so Kit extensions
    keep the 1.8 API. Already-imported cuRobo modules retain their 1.11
    ``import warp as wp`` binding.
    """
    import warp

    if _warp_func_supports_module(warp):
        yield
        return

    # Isaac Sim's Warp 1.8 lacks only the newer optional ``module`` keyword
    # that cuRobo uses while registering overloaded helpers.  Importing a
    # second Warp package inside Kit is unsafe (its import hook still routes
    # submodules to the extension), so adapt that keyword in place instead.
    original_func = warp.func

    def func_with_module(func=None, *, module=None, **kwargs):
        del module
        if func is None:
            return lambda decorated: original_func(decorated, **kwargs)
        return original_func(func, **kwargs)

    warp.func = func_with_module
    try:
        yield
    finally:
        warp.func = original_func
