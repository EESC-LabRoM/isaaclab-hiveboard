"""Make cuRobo v2 importable after Isaac Sim has replaced ``warp``.

Isaac Sim 5.1 loads ``omni.warp.core`` 1.8.x into ``sys.modules['warp']``.
cuRobo v2 registers collision overloads with ``wp.func(..., module=...)``,
which only exists in warp-lang >= 1.9 (this env has 1.11.1).
"""

from __future__ import annotations

import inspect
import sys
from contextlib import contextmanager
from pathlib import Path


def _warp_func_supports_module(wp_mod) -> bool:
    try:
        return "module" in inspect.signature(wp_mod.func).parameters
    except (TypeError, ValueError, AttributeError):
        return False


def _site_packages_warp_root() -> Path | None:
    import site

    roots = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        roots.append(user_site)
    for root in roots:
        init = Path(root) / "warp" / "__init__.py"
        if not init.is_file():
            continue
        root_str = str(Path(root))
        if "omni.warp" in root_str or "extscache" in root_str:
            continue
        return Path(root)
    return None


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

    warp_root = _site_packages_warp_root()
    if warp_root is None:
        raise ImportError(
            "cuRobo v2 requires warp-lang with wp.func(module=...), but Isaac "
            "Sim replaced `warp` with an older build and no site-packages "
            "warp-lang was found. Install `warp-lang>=1.9` in the venv."
        )

    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "warp" or name.startswith("warp.")
    }
    for name in saved:
        del sys.modules[name]

    root_str = str(warp_root)
    inserted = False
    if sys.path[:1] != [root_str]:
        sys.path.insert(0, root_str)
        inserted = True

    try:
        import warp as rebound_warp

        if not _warp_func_supports_module(rebound_warp):
            raise ImportError(
                f"Rebound warp from {root_str} still lacks wp.func(module=). "
                f"version={getattr(rebound_warp, '__version__', None)}"
            )
        yield
    finally:
        sys.modules.update(saved)
        if inserted and sys.path[:1] == [root_str]:
            sys.path.pop(0)
