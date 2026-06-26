"""context.py - shared-context utilities.

A single mutable ``Ctx`` is threaded through the ported modules (``userOpts(ctx)``,
``vehModel(ctx)``, ...), replacing MATLAB's shared base workspace and making every
data dependency explicit. Parameter groups (``vp``, ``pt``, ``track``, ``c`` ...)
are nested SimpleNamespaces, preserving the MATLAB dot syntax (``vp.tyre.mu``,
``pt.Pmax``, ``c.ub.T_motor``).
"""

from types import SimpleNamespace


class Ctx(SimpleNamespace):
    """A namespace carrying parameters, symbols, options and results between the
    ported MLTP modules. Use plain attribute access, e.g. ``ctx.vp``, ``ctx.pt``,
    ``ctx.track``, ``ctx.opts``, ``ctx.data``."""
    pass


def ns(**kwargs):
    """Shorthand for a SimpleNamespace."""
    return SimpleNamespace(**kwargs)


def dict2ns(d):
    """Recursively convert nested dicts to SimpleNamespaces."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict2ns(v) for k, v in d.items()})
    return d
