"""Entry point for `python -m hypnose_eeg_preprocessing.cli`.

Guarded so that merely *importing* this module does not run the CLI — otherwise
anything that walks the package (pkgutil.walk_packages, doc builders, test
collectors) would execute a command and exit.
"""

from . import main

if __name__ == "__main__":
    raise SystemExit(main())
