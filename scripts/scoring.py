#!/usr/bin/env python
"""Run the scoring CLI as a plain script.

Equivalent to `hypnose-somnotate score` (installed entry point) and to
`python -m hypnose_somnotate.cli score`. Kept for running straight from a
checkout; the implementation lives in `hypnose_somnotate.cli.scoring`.
"""

from hypnose_somnotate.cli.scoring import main

if __name__ == "__main__":
    raise SystemExit(main())
