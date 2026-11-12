"""Archival Astronomical Data Ingestion Engine (MAST / HST / JWST).

Provides robust tools to search, download, and extract science-grade cutouts
of gravitationally lensed galaxies from NASA's Mikulski Archive for Space Telescopes (MAST).
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import requests

try:
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.nddata import Cutout2D
    import astropy.units as u
    from astropy.wcs import WCS
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

try:
    from astroquery.mast import Observations
    ASTROQUERY_AVAILABLE = True
except ImportError:
    ASTROQUERY_AVAILABLE = False


# Catalog of celebrated gravitational lenses with verified celestial coordinates (ICRS)
KNOWN_LENS_CATALOG: Dict[str, Dict[str, Any]] = {
    "SDSSJ0946+1006": {
        "name": "SDSS J0946+1006 (SLACS Double Einstein Ring)",
        "ra": 146.73617,
        "dec": 10.11461,
        "z_lens": 0.222,
        "z_source": 0.609,
        "preferred_filter": "F814W",
    },
    "SDSSJ1148+1930": {
        "name": "SDSS J1148+1930 (The Cosmic Horseshoe)",
        "ra": 177.13808,
        "dec": 19.50089,
        "z_lens": 0.444,
        "z_source": 2.381,
        "preferred_filter": "F814W",
    },
    "RXJ1131-1231": {
        "name": "RX J1131-1231 (Gravitationally Lensed Quasar Quad)",
        "ra": 172.96433,
        "dec": -12.53289,
        "z_lens": 0.295,
        "z_source": 0.658,
        "preferred_filter": "F814W",
    },
    "SDSSJ1430+4105": {
        "name": "SDSS J1430+4105 (SLACS Einstein Ring)",
        "ra": 217.65342,
        "dec": 41.08536,
        "z_lens": 0.285,
        "z_source": 0.575,
        "preferred_filter": "F814W",
    },
}


def resolve_coordinates(
    target_or_ra: Union[str, float],
    dec: Optional[float] = None,
) -> Tuple[float, float, str]:
    """Resolve celestial coordinates (RA, Dec) from target name or explicit coordinates.

    Args:
        target_or_ra: Lens target name (e.g. 'SDSSJ0946+1006') or Right Ascension in degrees.
        dec: Declination in degrees if target_or_ra is numerical RA.

    Returns:
        Tuple of (ra_deg, dec_deg, target_label).
    """
    if isinstance(target_or_ra, (int, float)):
        if dec is None:
            raise ValueError("Declination must be specified when RA is provided as a number.")
        return float(target_or_ra), float(dec), f"Target_RA{target_or_ra:.4f}_Dec{dec:.4f}"

    clean_name = str(target_or_ra).replace(" ", "").replace("-", "-").upper()
    for key, data in KNOWN_LENS_CATALOG.items():
        if key in clean_name or clean_name in key:
            return data["ra"], data["dec"], key

    if ASTROPY_AVAILABLE:
        try:
            coord = SkyCoord.from_name(str(target_or_ra))
            return float(coord.ra.deg), float(coord.dec.deg), str(target_or_ra).replace(" ", "_")
        except Exception as e:
            raise ValueError(
                f"Could not resolve target name '{target_or_ra}' via SIMBAD/NED. "
                f"Please provide explicit (ra, dec) in degrees. Details: {e}"
            )
    else:
        raise ValueError(f"Unknown target '{target_or_ra}' and astropy is not available to resolve it.")


def fetch_mast_observation(
    ra: float,
    dec: float,
    obs_collection: str = "HST",
    instrument_name: str = "ACS/WFC",
    filter_name: str = "F814W",
    search_radius_arcsec: float = 10.0,
    download_dir: Union[str, Path] = "./real_data",
    max_files: int = 1,
) -> List[Path]:
    """Query NASA MAST and download science-calibrated drizzled FITS products.

    Filters specifically for calibrated Level 3 science products (_drz.fits or _drc.fits),
    avoiding raw exposures, previews, and metadata tables.

    Args:
        ra: Right Ascension in degrees (ICRS).
        dec: Declination in degrees (ICRS).
        obs_collection: Observatory ('HST' or 'JWST').
        instrument_name: Instrument (e.g. 'ACS/WFC', 'WFC3/UVIS', 'WFC3/IR').
        filter_name: Optical filter (e.g. 'F814W', 'F606W', 'F475W').
        search_radius_arcsec: Search cone radius in arcseconds.
        download_dir: Directory where downloaded FITS files are stored.
        max_files: Maximum number of observation files to retrieve.

    Returns:
        List of Path objects pointing to downloaded science FITS files.
    """
    if not ASTROQUERY_AVAILABLE or not ASTROPY_AVAILABLE:
        raise ImportError(
            "astroquery and astropy are required for MAST data retrieval. "
            "Install them via 'pip install astroquery astropy'."
        )

    out_dir = Path(download_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    radius_quantity = search_radius_arcsec * u.arcsec

    print(f"[MAST Query] Searching {obs_collection} {instrument_name} ({filter_name}) around "
          f"RA={ra:.5f} deg, Dec={dec:.5f} deg (r={search_radius_arcsec} arcsec)...")

    # 1. Query MAST Criteria
    query_kwargs: Dict[str, Any] = {
        "coordinates": coord,
        "radius": radius_quantity,
        "obs_collection": obs_collection,
    }
    if instrument_name:
        query_kwargs["instrument_name"] = instrument_name
    if filter_name:
        query_kwargs["filters"] = filter_name

    try:
        obs_table = Observations.query_criteria(**query_kwargs)
    except Exception as e:
        raise RuntimeError(f"MAST criteria query failed: {e}")

    if len(obs_table) == 0:
        warnings.warn(
            f"No observations found matching filter {filter_name}. Retrying without filter restriction...",
            RuntimeWarning,
        )
        query_kwargs.pop("filters", None)
        obs_table = Observations.query_criteria(**query_kwargs)

    if len(obs_table) == 0:
        raise FileNotFoundError(
            f"No {obs_collection} observations found in MAST within {search_radius_arcsec}'' "
            f"of RA={ra:.5f}, Dec={dec:.5f}."
        )

    print(f"[MAST Query] Found {len(obs_table)} matching observation record(s). Fetching science product list...")

    # 2. Get product list and filter strictly for high-level calibrated science FITS
    data_products = Observations.get_product_list(obs_table)

    science_filter = (data_products["productType"] == "SCIENCE")
    filtered_products = data_products[science_filter]

    # Prioritize drizzled Level 3 products (DRZ or DRC)
    if "productSubGroupDescription" in filtered_products.colnames:
        drizzled = filtered_products[
            (filtered_products["productSubGroupDescription"] == "DRZ") |
            (filtered_products["productSubGroupDescription"] == "DRC")
        ]
        if len(drizzled) > 0:
            filtered_products = drizzled

    # Prefer direct target visits over massive multi-degree skycell mosaic tiles
    if "productFilename" in filtered_products.colnames:
        non_skycell = filtered_products[["skycell" not in str(fn).lower() for fn in filtered_products["productFilename"]]]
        if len(non_skycell) > 0:
            filtered_products = non_skycell

    if len(filtered_products) == 0:
        raise FileNotFoundError("Found observation records in MAST, but no calibrated FITS science products available.")

    # Sort to prioritize targeted single-visit products with compact file size
    if "size" in filtered_products.colnames:
        filtered_products.sort("size")

    target_products = filtered_products[:max_files]
    filenames = target_products["productFilename"] if "productFilename" in target_products.colnames else []
    print(f"[MAST Download] Selected {len(target_products)} science product(s): {list(filenames)}")

    # 3. Check for existing local files to prevent redundant re-downloads
    downloaded_paths: List[Path] = []
    products_to_download = []

    for row in target_products:
        fname = str(row["productFilename"])
        potential_existing = list(out_dir.rglob(fname))
        valid_cached = False
        if potential_existing:
            cand = potential_existing[0]
            expected_size = row["size"] if "size" in row.colnames else 0
            if cand.stat().st_size > 0 and (expected_size == 0 or cand.stat().st_size >= expected_size * 0.95):
                print(f"[MAST Cache] Found verified cached product: {cand.name}")
                downloaded_paths.append(cand)
                valid_cached = True
            else:
                cand.unlink(missing_ok=True)

        if not valid_cached:
            products_to_download.append(row)

    if products_to_download:
        for row in products_to_download:
            fname = str(row["productFilename"])
            data_uri = str(row["dataURI"])
            dest_file = out_dir / fname
            download_url = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={data_uri}"
            print(f"[MAST Streaming] Downloading {fname}...")
            temp_file = dest_file.with_suffix(".tmp")
            try:
                with requests.get(download_url, stream=True, timeout=60) as resp:
                    resp.raise_for_status()
                    total_bytes = int(resp.headers.get("content-length", 0))
                    downloaded_bytes = 0
                    with open(temp_file, "wb") as f_out:
                        for chunk in resp.iter_content(chunk_size=1024 * 512):
                            if chunk:
                                f_out.write(chunk)
                                downloaded_bytes += len(chunk)
                                if total_bytes > 0:
                                    pct = (downloaded_bytes / total_bytes) * 100.0
                                    mb_done = downloaded_bytes / (1024 * 1024)
                                    mb_tot = total_bytes / (1024 * 1024)
                                    sys.stdout.write(f"\r  --> Progress: {mb_done:.1f}/{mb_tot:.1f} MB ({pct:.1f}%)")
                                    sys.stdout.flush()
                print()
                if temp_file.exists():
                    temp_file.replace(dest_file)
                downloaded_paths.append(dest_file)
            except Exception as dl_err:
                if temp_file.exists():
                    temp_file.unlink(missing_ok=True)
                raise RuntimeError(f"Failed to download {fname} from MAST: {dl_err}")

    return downloaded_paths


def extract_fits_cutout(
    fits_path: Union[str, Path],
    ra: float,
    dec: float,
    cutout_size_arcsec: float = 8.0,
    output_path: Optional[Union[str, Path]] = None,
    overwrite: bool = True,
) -> Path:
    """Extract a calibrated 2D cutout centered on the lens coordinates with updated WCS.

    Full Hubble ACS/WFC frames are 4096 x 4096 pixels (~70-140 MB). This extracts
    the exact sub-region containing the deflector and gravitational arcs, preserving
    WCS metadata for downstream LensForge reconstruction.

    Args:
        fits_path: Path to the input full-frame FITS image.
        ra: Center Right Ascension in degrees (ICRS).
        dec: Center Declination in degrees (ICRS).
        cutout_size_arcsec: Cutout side length in arcseconds (e.g. 8.0'' x 8.0'').
        output_path: Destination path for the extracted cutout FITS file.
        overwrite: If True, overwrite existing cutout file.

    Returns:
        Path to the saved cutout FITS file.
    """
    if not ASTROPY_AVAILABLE:
        raise ImportError("astropy is required for FITS cutout extraction.")

    in_path = Path(fits_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input FITS file not found: {in_path}")

    if output_path is None:
        out_path = in_path.parent / f"{in_path.stem}_cutout_{cutout_size_arcsec:.1f}arcsec.fits"
    else:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        print(f"[Cutout Cache] Using existing cutout: {out_path.name}")
        return out_path

    with fits.open(in_path) as hdul:
        # Locate science extension ('SCI' or Primary HDU)
        sci_ext = 0
        if "SCI" in hdul:
            sci_ext = "SCI"
        elif len(hdul) > 1 and hdul[1].data is not None:
            sci_ext = 1

        sci_data = np.asarray(hdul[sci_ext].data, dtype=np.float64)
        header = hdul[sci_ext].header

        # Parse WCS celestial transformation
        wcs = WCS(header)
        center_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        size_quantity = cutout_size_arcsec * u.arcsec

        # Extract 2D Cutout
        cutout = Cutout2D(
            sci_data,
            position=center_coord,
            size=size_quantity,
            wcs=wcs,
            mode="partial",
            fill_value=0.0,
        )

        # Build updated FITS HDU with cutout WCS
        new_header = header.copy()
        new_header.update(cutout.wcs.to_header())
        new_header["HISTORY"] = f"LensForge 2D cutout centered on RA={ra:.5f}, Dec={dec:.5f}"
        new_header["CUTOUT_S"] = (float(cutout_size_arcsec), "Cutout size in arcseconds")

        # Determine pixel scale
        try:
            from astropy.wcs.utils import proj_plane_pixel_scales
            scales = proj_plane_pixel_scales(cutout.wcs)
            scale_arcsec = float(np.mean(scales) * 3600.0)
            new_header["PIXSCALE"] = (scale_arcsec, "Pixel scale in arcsec/pixel")
        except Exception:
            pass

        primary_hdu = fits.PrimaryHDU(data=np.asarray(cutout.data, dtype=np.float32), header=new_header)
        primary_hdu.writeto(out_path, overwrite=True)

    print(f"[Cutout Generated] Saved {cutout.shape[1]}x{cutout.shape[0]} cutout to: {out_path.resolve()}")
    return out_path


def fetch_and_prepare_lens_data(
    target_name_or_ra: Union[str, float],
    dec: Optional[float] = None,
    filter_name: str = "F814W",
    cutout_size_arcsec: float = 8.0,
    download_dir: Union[str, Path] = "./real_data",
) -> Path:
    """One-line automated pipeline to retrieve, filter, and extract a lens FITS cutout from MAST.

    Args:
        target_name_or_ra: Named lens ('SDSSJ0946+1006') or numerical RA (deg).
        dec: Numerical Dec (deg) if target_name_or_ra is float.
        filter_name: HST filter name (default 'F814W').
        cutout_size_arcsec: Cutout field of view in arcseconds.
        download_dir: Directory for downloaded FITS files.

    Returns:
        Path to the standardized, calibrated cutout FITS ready for LensForge reconstruction.
    """
    ra, dec_val, label = resolve_coordinates(target_name_or_ra, dec)
    print("=" * 78)
    print(f" LENSFORGE ARCHIVAL RETRIEVAL: {label}")
    print(f" Coordinates: RA = {ra:.5f} deg, Dec = {dec_val:.5f} deg")
    print("=" * 78)

    # 1. Download full drizzled science product from MAST
    downloaded_files = fetch_mast_observation(
        ra=ra,
        dec=dec_val,
        obs_collection="HST",
        instrument_name="ACS/WFC",
        filter_name=filter_name,
        download_dir=download_dir,
        max_files=1,
    )

    if not downloaded_files:
        raise RuntimeError("No science files could be downloaded from MAST.")

    # 2. Extract calibrated sub-aperture cutout around the lens
    target_fits = downloaded_files[0]
    out_dir = Path(download_dir)
    cutout_path = out_dir / f"{label}_{filter_name}_cutout_{cutout_size_arcsec:.1f}arcsec.fits"

    prepared_cutout = extract_fits_cutout(
        fits_path=target_fits,
        ra=ra,
        dec=dec_val,
        cutout_size_arcsec=cutout_size_arcsec,
        output_path=cutout_path,
        overwrite=True,
    )

    print("=" * 78)
    print(f"[READY FOR RECONSTRUCTION] Prepared FITS File: {prepared_cutout.resolve()}")
    print("=" * 78)
    return prepared_cutout
