#!/usr/bin/env python3
"""LensForge Scientific Reconstruction Pipeline: Unveiling the Unlensed Galaxy."""

from pathlib import Path
import sys

# Ensure src/ is on python path
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from lensforge.inversion.pipeline import run_galaxy_reconstruction


if __name__ == "__main__":
    out_dir = project_root / "outputs"
    run_galaxy_reconstruction(out_dir)
