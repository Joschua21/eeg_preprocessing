#!/usr/bin/env python
"""Run the scoring CLI as a plain script.

Equivalent to `hypnose-eeg-preprocess score` (installed entry point) and to
`python -m hypnose_eeg_preprocessing.cli score`. Kept for running straight from a
checkout; the implementation lives in `hypnose_eeg_preprocessing.cli.scoring`.
"""

from hypnose_eeg_preprocessing.cli.scoring import main

if __name__ == "__main__":
    raise SystemExit(main())
