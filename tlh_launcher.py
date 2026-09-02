"""PyInstaller entry point (keeps `python -m tlh` semantics for the frozen build)."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from tlh.__main__ import main
    sys.exit(main())
