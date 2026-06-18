"""context.py - shared-context utilities.

MATLAB scripts share a single base workspace; each ``run('foo.m')`` injects its
variables into scope. Python has no equivalent, so the ported scripts pass one
mutable ``Ctx`` object between the module functions (``userOpts(ctx)``,
``vehModel(ctx)``, ...). This keeps one file per MATLAB file while making every
data dependency explicit.

Parameter groups (``vp``, ``pt``, ``track``, ``c`` ...) are stored as nested
SimpleNamespaces so they preserve the MATLAB dot syntax (``vp.tyre.mu``,
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
