#!/usr/bin/env python3
"""LensForge Real Astronomical Data Downloader (NASA MAST Archive).

Downloads calibrated Hubble Space Telescope (HST) / JWST science observations
and extracts ready-to-reconstruct cutouts of known or custom gravitational lenses.

Usage examples:
    # 1. Download iconic SLACS double Einstein ring SDSS J0946+1006
    python scripts/download_real_lens_data.py --target SDSSJ0946+1006

    # 2. Download Cosmic Horseshoe
    python scripts/download_real_lens_data.py --target SDSSJ1148+1930 --filter F814W --cutout-size 10.0

    # 3. Download custom target by RA and Dec
    python scripts/download_real_lens_data.py --ra 146.73617 --dec 10.11461 --cutout-size 8.0
"""

import argparse
from pathlib import Path
import sys

# Ensure src/ is on python path
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from lensforge.data.archive import (
    KNOWN_LENS_CATALOG,
    fetch_and_prepare_lens_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="download_real_lens_data",
        description="Download and extract calibrated HST/JWST cutouts of gravitational lenses from NASA MAST.",
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--target",
        type=str,
        help=f"Celebrated lens preset: {list(KNOWN_LENS_CATALOG.keys())} or SIMBAD resolvable name.",
    )
    target_group.add_argument(
        "--ra",
        type=float,
        help="Right Ascension of lens center in degrees (requires --dec).",
    )

    parser.add_argument(
        "--dec",
        type=float,
        default=None,
        help="Declination of lens center in degrees (required if --ra is used).",
    )
    parser.add_argument(
        "--filter",
        dest="filter_name",
        type=str,
        default="F814W",
        help="HST optical filter name (default: F814W).",
    )
    parser.add_argument(
        "--cutout-size",
        type=float,
        default=8.0,
        help="Cutout side length in arcseconds (default: 8.0).",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=project_root / "real_data",
        help="Directory to save downloaded products (default: real_data/).",
    )

    args = parser.parse_args()

    if args.ra is not None and args.dec is None:
        parser.error("--dec is required when using --ra.")

    target_query = args.target if args.target is not None else args.ra
    try:
        cutout_path = fetch_and_prepare_lens_data(
            target_name_or_ra=target_query,
            dec=args.dec,
            filter_name=args.filter_name,
            cutout_size_arcsec=args.cutout_size,
            download_dir=args.download_dir,
        )
        print(f"\n[SUCCESS] Archival science cutout ready for LensForge at:\n  --> {cutout_path.resolve()}\n")
    except Exception as e:
        print(f"\n[ERROR] Archival retrieval failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
