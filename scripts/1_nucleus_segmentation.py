import os
import sys
import argparse
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

# --- run_utils bootstrap ---------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_utils as ru  # noqa: E402

# --- Configuration ---------------------------------------------------------

# Segmentation parameters
INTENSITY_SCALING_PARAM = [1, 7]
BLUR_SIGMA = 1
MIN_AREA = 3500

# Which channel to segment on.
CHANNEL_NAME = "DAPI"

# Full-frame contrast gate: frames whose std is below this have nothing to
# segment and are skipped before the (expensive) segmentation steps.
CONTRAST_CUTOFF = 100

# Stage name + the sub-dirs this stage reads from / writes to inside a run.
STAGE = "nucleus_segmentation"
INPUT_STAGE_DIR = "image_metadata"
OUTPUT_STAGE_DIR = "nuclei_features"

REGIONPROPS = (
    "label", "area", "intensity_mean", "intensity_max", "intensity_min",
    "intensity_std", "centroid", "eccentricity", "solidity", "perimeter",
    "feret_diameter_max", "orientation", "axis_major_length", "axis_minor_length",
)


def collect_params():
    """Tuning constants recorded in run_metadata.json for this stage."""
    return {
        "intensity_scaling_param": INTENSITY_SCALING_PARAM,
        "blur_sigma": BLUR_SIGMA,
        "min_area": MIN_AREA,
        "channel_name": CHANNEL_NAME,
        "contrast_cutoff": CONTRAST_CUTOFF,
        "regionprops": list(REGIONPROPS),
    }


# --- Segmentation ----------------------------------------------------------

@delayed
def process_nucleus_image(image_path):
    """Segment nuclei in one image.

    Returns (features_df, status_dict). features_df is per-nucleus (empty if the
    frame failed the contrast gate). status_dict is one per-image QC record.
    """
    image_name = os.path.basename(image_path)
    nuc = tifffile.imread(image_path)
    frame_std = float(nuc.astype(np.float64).std())

    status = {
        "image_name": image_name,
        "filepath": str(image_path),
        "frame_std": frame_std,
        "contrast_check": "pass",
        "n_nuclei": 0,
        "filter_status": "pass",
        "fail_reason": "",
    }

    # Full-frame contrast gate: skip frames with nothing to segment.
    if frame_std < CONTRAST_CUTOFF:
        status.update(
            contrast_check="fail",
            filter_status="fail",
            fail_reason="low_contrast",
        )
        return _empty_feature_df(image_path), status

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
    filled_clean = morphology.remove_small_objects(filled.astype(bool), min_size=MIN_AREA)
    nuc_seg = morphology.label(filled_clean)

    # Extract features
    props = regionprops_table(
        label_image=nuc_seg,
        intensity_image=nuc,
        properties=REGIONPROPS,
    )
    df = pd.DataFrame(props)
    df["image_name"] = image_name
    df["filepath"] = str(image_path)
    df["frame_std"] = frame_std

    status["n_nuclei"] = len(df)
    # A frame that passed contrast but yielded no nuclei is worth flagging.
    if len(df) == 0:
        status.update(filter_status="fail", fail_reason="no_nuclei_detected")

    return df, status


# --- helpers ------------------------------------------------------------

def experiment_name_from_parquet(parquet_path, df):
    """Best-effort experiment name: prefer the column, fall back to filename."""
    if "Experiment_name" in df.columns and df["Experiment_name"].notna().any():
        return str(df["Experiment_name"].iloc[0])
    stem = parquet_path.stem
    # Filenames are now '<exp>_metadata.parquet' (run dir carries the date).
    return stem.split("_metadata")[0] if "_metadata" in stem else stem

def _empty_feature_df(image_path):
    """Zero-row feature frame with the correct columns, for a frame that
    produced no nuclei (contrast gate or empty segmentation)."""
    cols = []
    for p in REGIONPROPS:
        if p == "centroid":
            cols += ["centroid-0", "centroid-1"]
        else:
            cols.append(p)
    df = pd.DataFrame(columns=cols)
    df["image_name"] = pd.Series(dtype="object")
    df["filepath"] = pd.Series(dtype="object")
    df["frame_std"] = pd.Series(dtype="float64")
    return df

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


def process_metadata_parquet(parquet_path, output_dir, scheduler, rec):
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
    rec.log(f"{exp_name}: segmenting {len(filepaths)} {CHANNEL_NAME} image(s)")

    tasks = [process_nucleus_image(fp) for fp in filepaths]
    with ProgressBar():
        results = dask.compute(*tasks, scheduler=scheduler)

    dfs = [r[0] for r in results]
    statuses = [r[1] for r in results]

    features_df = pd.concat(dfs, ignore_index=True)
    features_df = features_df.merge(samples, on="filepath", how="left")

    qc_df = pd.DataFrame(statuses)
    qc_df = qc_df.merge(samples, on="filepath", how="left", suffixes=("", "_meta"))

    out_path = output_dir / f"{exp_name}_nuclei_features.parquet"
    features_df.to_parquet(out_path, index=False)

    qc_path = output_dir / f"{exp_name}_image_qc.parquet"
    qc_df.to_parquet(qc_path, index=False)

    # --- Per-experiment summary ---
    total_images = len(qc_df)
    passed_contrast = int((qc_df["contrast_check"] == "pass").sum())
    total_nuclei = len(features_df)

    print(f"\n--- {exp_name} summary ---")
    print(f"  Total images:              {total_images}")
    print(f"  Passed contrast check:     {passed_contrast}")
    print(f"  Total nuclei detected:     {total_nuclei}")
    print(f"  Wrote features -> {out_path}")
    print(f"  Wrote QC       -> {qc_path}")
    rec.log(f"{exp_name}: {total_nuclei} nuclei from {passed_contrast}/"
            f"{total_images} frames -> {out_path.name}")

    return {
        "exp_name": exp_name,
        "total_images": total_images,
        "passed_contrast": passed_contrast,
        "total_nuclei": total_nuclei,
    }


# --- Main ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment nuclei from metadata parquet files and write "
                    "per-experiment nuclei-feature parquets."
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
        "-s", "--scheduler",
        default="processes",
        choices=["processes", "threads", "single-threaded", "synchronous"],
        help="Dask scheduler to use (default: processes).",
    )
    # --output-root + --run-id (required here: reuse the run minted by 0_).
    ru.add_run_args(parser, mints_run_id=False)
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve the existing run dir (errors clearly if the run ID is wrong).
    run_dir = ru.resolve_run_dir(args.output_root, args.run_id)
    input_dir = ru.stage_dir(run_dir, INPUT_STAGE_DIR)
    output_dir = ru.stage_dir(run_dir, OUTPUT_STAGE_DIR)

    rec = ru.StageRecorder(
        run_dir, stage=STAGE, run_id=args.run_id,
        params=collect_params(),
        inputs={
            "input_dir": str(input_dir),
            "scheduler": args.scheduler,
            "experiments_requested": args.experiments or "ALL",
        },
    )
    print(f"\n=== RUN ID: {args.run_id} ===")
    print(f"=== run dir: {run_dir} ===\n")

    parquets = discover_parquets(input_dir, args.experiments)
    print(f"Found {len(parquets)} metadata parquet(s) to process.")
    rec.log(f"found {len(parquets)} metadata parquet(s) in {input_dir}")

    all_stats = []
    for parquet_path in parquets:
        stats = process_metadata_parquet(parquet_path, output_dir, args.scheduler, rec)
        if stats is not None:
            all_stats.append(stats)

    # --- Grand total across all experiments ---
    total_images = sum(s["total_images"] for s in all_stats)
    passed_contrast = sum(s["passed_contrast"] for s in all_stats)
    total_nuclei = sum(s["total_nuclei"] for s in all_stats)

    if all_stats:
        print("\n" + "=" * 40)
        print("OVERALL SUMMARY")
        print(f"  Experiments processed:     {len(all_stats)}")
        print(f"  Total images:              {total_images}")
        print(f"  Passed contrast check:     {passed_contrast}")
        print(f"  Total nuclei detected:     {total_nuclei}")
        print("=" * 40)

    rec.finish(
        outputs={"nuclei_features_dir": str(output_dir)},
        summary={
            "experiments_processed": len(all_stats),
            "total_images": total_images,
            "passed_contrast": passed_contrast,
            "total_nuclei": total_nuclei,
            "per_experiment": all_stats,
        },
    )

    print(f"\n=== RUN ID: {args.run_id} (pass to downstream stages) ===")


if __name__ == "__main__":
    main()