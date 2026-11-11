"""Command-Line Interface (CLI) for LensForge.

Allows running end-to-end galaxy lens reconstructions directly from the command line:
    lensforge-reconstruct --output-dir outputs/
"""

import argparse
from pathlib import Path
import sys

from lensforge.inversion.pipeline import run_galaxy_reconstruction


def main() -> None:
    """CLI entry point for LensForge galaxy reconstruction."""
    parser = argparse.ArgumentParser(
        prog="lensforge-reconstruct",
        description="LensForge: Bayesian Gravitational Lens Source Reconstruction Pipeline",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where reconstructed images and dashboards are saved (default: outputs/)",
    )

    args = parser.parse_args()
    print(f"Executing LensForge Reconstruction Pipeline (Output Directory: {args.output_dir})...\n")
    dashboard_path = run_galaxy_reconstruction(args.output_dir)
    print(f"\n[CLI] Success! Reconstruction dashboard available at: {dashboard_path.resolve()}")


if __name__ == "__main__":
    main()

