import os
import argparse
from datetime import datetime
from pathlib import Path

import dask
from dask.diagnostics import ProgressBar
from dask import delayed
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.stats import norm
from skimage import filters, morphology
from skimage.measure import regionprops_table

# --- Configuration ---------------------------------------------------------

# Segmentation parameters
INTENSITY_SCALING_PARAM = [1, 7]
BLUR_SIGMA = 1
MIN_AREA = 3500

# Which channel to segment on.
CHANNEL_NAME = "DAPI"

# Default input/output dirs (chain from the metadata script).
DEFAULT_INPUT_DIR = Path("./output/image_metadata")
DEFAULT_OUTPUT_DIR = Path("./output/nuclei_features")

REGIONPROPS = (
    "label", "area", "mean_intensity", "max_intensity", "min_intensity",
    "std_intensity", "centroid", "eccentricity", "solidity", "perimeter",
    "feret_diameter_max", "orientation", "major_axis_length", "minor_axis_length",
)


# --- Segmentation ----------------------------------------------------------

@delayed
def process_nucleus_image(image_path):
    """Segment nuclei in one image and return a per-object feature DataFrame."""
    nuc = tifffile.imread(image_path)

    # Normalize via a fitted-Gaussian contrast stretch
    m, s = norm.fit(nuc.flatten())
    stretch_min = max(m - INTENSITY_SCALING_PARAM[0] * s, nuc.min())
    stretch_max = min(m + INTENSITY_SCALING_PARAM[1] * s, nuc.max())
    nuc_stretch = np.clip(nuc, stretch_min, stretch_max)
    image_norm = (nuc_stretch - stretch_min) / (stretch_max - stretch_min)

    blurred = filters.gaussian(image_norm, sigma=BLUR_SIGMA)

    # Step 1: low-level threshold
    triangle_cutoff = filters.threshold_triangle(blurred)
    global_median_cutoff = np.percentile(blurred, 50)
    th_low_cutoff = (triangle_cutoff + global_median_cutoff) / 2
    img_low_level = blurred > th_low_cutoff
    img_low_level = morphology.remove_small_objects(
        img_low_level.astype(bool), min_size=MIN_AREA
    )
    img_low_level = morphology.dilation(img_low_level, footprint=morphology.disk(2))

    # Step 2: per-object high-level threshold
    otsu_cutoff = 0.333 * filters.threshold_otsu(blurred)
    img_high_level = np.zeros_like(img_low_level)
    lab_low, num_obj = morphology.label(img_low_level, return_num=True)
    for idx in range(num_obj):
        single_obj = lab_low == (idx + 1)
        local_otsu = filters.threshold_otsu(blurred[single_obj])
        if local_otsu > otsu_cutoff:
            img_high_level[np.logical_and(blurred > 0.98 * local_otsu, single_obj)] = 1

    filled = ndi.binary_fill_holes(img_high_level)
    filled = morphology.dilation(filled, footprint=morphology.disk(2))
    # Label the filled mask, then drop objects below MIN_AREA. We filter the
    # boolean mask (unambiguous for remove_small_objects) and relabel, which
    # avoids the "only one label provided" warning that occurs when a label
    # image happens to contain a single object.
    filled_clean = morphology.remove_small_objects(filled.astype(bool), min_size=MIN_AREA)
    nuc_seg = morphology.label(filled_clean)

    # Extract features
    props = regionprops_table(
        label_image=nuc_seg,
        intensity_image=nuc,
        properties=REGIONPROPS,
    )
    df = pd.DataFrame(props)
    df["image_name"] = os.path.basename(image_path)
    df["filepath"] = str(image_path)
    return df


# --- IO helpers ------------------------------------------------------------

def discover_parquets(input_dir, selected=None):
    """Return metadata parquet paths under input_dir.

    If `selected` is given, only parquets whose filename starts with one of
    those experiment names are returned.
    """
    all_parquets = sorted(input_dir.glob("*.parquet"))
    if selected is None:
        return all_parquets

    chosen = []
    for name in selected:
        matches = [p for p in all_parquets if p.name.startswith(name)]
        if matches:
            chosen.extend(matches)
        else:
            print(f"[warning] no parquet found for experiment: {name}")
    return chosen


def experiment_name_from_parquet(parquet_path, df):
    """Best-effort experiment name: prefer the column, fall back to filename."""
    if "Experiment_name" in df.columns and df["Experiment_name"].notna().any():
        return str(df["Experiment_name"].iloc[0])
    # Strip a trailing _metadata_<date> if present, else use the stem.
    stem = parquet_path.stem
    return stem.split("_metadata_")[0] if "_metadata_" in stem else stem


def process_metadata_parquet(parquet_path, output_dir, today, scheduler):
    """Run segmentation for every DAPI image in one metadata parquet."""
    meta = pd.read_parquet(parquet_path)

    if "Channel_name" not in meta.columns:
        print(f"[warning] {parquet_path.name} has no Channel_name column, skipping.")
        return None

    samples = meta[meta["Channel_name"] == CHANNEL_NAME].copy()
    if samples.empty:
        print(f"[warning] no {CHANNEL_NAME} images in {parquet_path.name}, skipping.")
        return None

    filepaths = samples["filepath"].tolist()
    exp_name = experiment_name_from_parquet(parquet_path, meta)
    print(f"\n{exp_name}: segmenting {len(filepaths)} {CHANNEL_NAME} image(s)...")

    tasks = [process_nucleus_image(fp) for fp in filepaths]
    with ProgressBar():
        dfs = dask.compute(*tasks, scheduler=scheduler)

    features_df = pd.concat(dfs, ignore_index=True)
    features_df["Experiment_name"] = exp_name

    out_path = output_dir / f"{exp_name}_nuclei_features_{today}.parquet"
    features_df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}  ({len(features_df)} nuclei)")

    return features_df


# --- Main ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment nuclei from metadata parquet files and write "
                    "per-experiment nuclei-feature parquets."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the metadata parquet files "
             f"(default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "-e", "--experiments",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Specific experiment names to process (matched against parquet "
             "filename prefixes). If omitted, all parquets are processed.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write the nuclei-feature parquets "
             f"(default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "-s", "--scheduler",
        default="processes",
        choices=["processes", "threads", "single-threaded", "synchronous"],
        help="Dask scheduler to use (default: processes).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"[error] input dir is not a directory: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")

    parquets = discover_parquets(args.input_dir, args.experiments)
    print(f"Found {len(parquets)} metadata parquet(s) to process.")

    for parquet_path in parquets:
        process_metadata_parquet(parquet_path, args.output_dir, today, args.scheduler)


if __name__ == "__main__":
    main()