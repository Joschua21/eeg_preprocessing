"""Deterministic application of the configured figure style.

Styles live in hypnose-behavior-analysis (`hypnose_behavior.io.save`) so every repo that plots
Hypnose data produces figures with one look. This module is the only place that
mutates global matplotlib state, and it does so at figure-creation time rather
than import time — see `config.DEFAULT_FIGURE_STYLE` for why that ordering
matters.

hypnose-behavior-analysis is an optional dependency: without it, `ensure_style` is a
silent no-op and figures render unstyled rather than raising.
"""

from __future__ import annotations

from ..config import DEFAULT_FIGURE_STYLE

# The style currently applied to this process, or None if none has been applied.
_applied: str | None = None


def ensure_style(style: str | None = None, force: bool = False) -> str | None:
    """Apply the configured figure style once per process; return what is active.

    Idempotent: repeated calls with the same style do nothing, so every plotting
    entry point can call it unconditionally.

    Arguments:
    ----------
    style -- style name (e.g. "nature", "poster", "presentation"), or None to use
        `config.DEFAULT_FIGURE_STYLE`. An explicit None in config means "leave
        matplotlib alone".

    force -- re-apply even if the style is already active. Use this to switch
        style mid-session without restarting the kernel.

    Returns:
    --------
    The active style name, or None if no style was applied (either because it is
    disabled in config or because hypnose-behavior-analysis is not installed).
    """
    global _applied

    style = DEFAULT_FIGURE_STYLE if style is None else style
    if style is None:
        return None
    if _applied == style and not force:
        return _applied

    try:
        from hypnose_behavior.io.save import use_style
    except ImportError:
        # hypnose-behavior-analysis not installed — plot unstyled rather than failing.
        return None

    use_style(style)
    _applied = style
    return _applied


def active_style() -> str | None:
    """Return the style applied by `ensure_style`, without applying anything."""
    return _applied
