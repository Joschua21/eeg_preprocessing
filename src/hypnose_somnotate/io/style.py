"""Deterministic application of the configured figure style.

The styles and the once-per-process application logic live in hypnose-helpers
(`hypnose_helpers.viz.styles`) so every repo that plots Hypnose data produces figures
with one look. What stays here is only this package's *default*, which is config.

Previously this reached across into hypnose-behavior-analysis and silently no-opped when
that package was absent, so a missing install showed up as unstyled figures rather than
an error. hypnose-helpers is a hard dependency, so that failure mode is gone.
"""

from __future__ import annotations

from hypnose_helpers.viz.styles import active_style, ensure_style as _ensure_style

from ..config import DEFAULT_FIGURE_STYLE

__all__ = ["ensure_style", "active_style"]


def ensure_style(style: str | None = None, force: bool = False) -> str | None:
    """Apply the configured figure style once per process; return what is active.

    `style=None` means "use `config.DEFAULT_FIGURE_STYLE`"; an explicit None *in config*
    means "leave matplotlib alone".
    """
    return _ensure_style(DEFAULT_FIGURE_STYLE if style is None else style, force=force)
