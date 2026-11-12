"""Entry script of the frozen Windows build: ``<name>.exe`` (console) and ``<name>App.exe`` (the window) are both
built from it. Everything it does is in ``streamsafe/app/entry.py``."""

import sys

from streamsafe.app.entry import main

if __name__ == "__main__":
    sys.exit(main())
