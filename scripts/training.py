#!/usr/bin/env python
"""CLI entry point for training a somnotate state-annotation model.

Placeholder. The CLI is added in a follow-up step; the library API it will wrap
is available now as::

    from hypnose_eeg_preprocessing.training import run_training
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    raise SystemExit("scripts/training.py is not implemented yet.")


if __name__ == "__main__":
    raise SystemExit(main())
