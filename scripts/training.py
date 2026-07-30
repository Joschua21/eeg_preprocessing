#!/usr/bin/env python
"""Run the training CLI as a plain script.

Equivalent to `hypnose-eeg-preprocess train` (installed entry point) and to
`python -m hypnose_eeg_preprocessing.cli train`. Kept for running straight from a
checkout; the implementation lives in `hypnose_eeg_preprocessing.cli.training`.
"""

from hypnose_eeg_preprocessing.cli.training import main

if __name__ == "__main__":
    raise SystemExit(main())
