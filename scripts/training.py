#!/usr/bin/env python
"""Run the training CLI as a plain script.

Equivalent to `hypnose-somnotate train` (installed entry point) and to
`python -m hypnose_somnotate.cli train`. Kept for running straight from a
checkout; the implementation lives in `hypnose_somnotate.cli.training`.
"""

from hypnose_somnotate.cli.training import main

if __name__ == "__main__":
    raise SystemExit(main())
