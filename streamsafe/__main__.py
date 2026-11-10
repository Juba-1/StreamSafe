import multiprocessing
import sys

from streamsafe.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()  # a frozen Windows build re-runs this module in the analyzer process
    sys.exit(main())
