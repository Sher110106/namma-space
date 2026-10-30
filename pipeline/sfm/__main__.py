"""python -m pipeline.sfm --run <run_dir> [options]   (stdlib only; runs inside namma-colmap:0.2)"""
import sys

from .stage import main

if __name__ == "__main__":
    sys.exit(main())
