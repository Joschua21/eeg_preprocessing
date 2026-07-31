"""hypnose-somnotate --- Somnotate-based EEG preprocessing and scoring pipeline.

Subpackages are imported by the caller (e.g.
``from hypnose_somnotate.scoring import score_recordings``) so that
importing the top-level package stays cheap.
"""

# Read from the installed package metadata so pyproject.toml stays the single source
# of truth for the version. The fallback covers running from a checkout that has not
# been installed (e.g. via PYTHONPATH).
try:
    from importlib.metadata import PackageNotFoundError, version as _version

    __version__ = _version("hypnose-somnotate")
except (ImportError, PackageNotFoundError):  # pragma: no cover
    __version__ = "unknown"
