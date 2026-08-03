import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

# --- run_utils bootstrap ---------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_utils as ru  # noqa: E402

# --- Configuration ---------------------------------------------------------

# ROI / filtering parameters
ROI_RADIUS = 120          # px; circle centered on each nucleus centroid
AREA_MAX = 22500          # px^2; upper area cutoff (low end handled in segmentation)
OVERLAP_FRACTION = 0.05   # fail if a pair overlaps by more than this fraction of ROI area

# Default frame dimensions (Phenix full-frame). Used only if the shape can't
# be read from an image and --frame-size isn't given.
DEFAULT_FRAME_H = 2160
DEFAULT_FRAME_W = 2160

# Stage name + the sub-dirs this stage reads from / writes to inside a run.
STAGE = "roi_filtering"
INPUT_STAGE_DIR = "nuclei_features"
OUTPUT_STAGE_DIR = "nuclei_filtered"

# The per-ROI check columns this script produces.
CHECK_COLS = ["area_check", "edge_check", "overlap_check"]


def collect_params():
    """Tuning constants recorded in run_metadata.json for this stage."""
    return {
        "roi_radius": ROI_RADIUS,
        "area_max": AREA_MAX,
        "overlap_fraction": OVERLAP_FRACTION,
        "default_frame_h": DEFAULT_FRAME_H,
        "default_frame_w": DEFAULT_FRAME_W,
        "check_cols": list(CHECK_COLS),
    }


# --- Geometry --------------------------------------------------------------

def lens_area(d, r):
    """Intersection area of two equal-radius (r) circles whose centers are d apart."""
    if d >= 2 * r:
        return 0.0
    if d <= 0:
        return np.pi * r ** 2
    return 2 * r ** 2 * np.arccos(d / (2 * r)) - (d / 2) * np.sqrt(4 * r ** 2 - d ** 2)


# --- Frame size resolution -------------------------------------------------

def resolve_frame_size(features_df, override=None):
    """Return (H, W) for the edge check.

    Priority: explicit --frame-size override > read one image's shape from disk
    > module default. Reads at most one image regardless of experiment size.
    """
    if override is not None:
        return override

    if "filepath" in features_df.columns and features_df["filepath"].notna().any():
        sample_fp = features_df["filepath"].dropna().iloc[0]
        try:
            shape = tifffile.imread(sample_fp).shape
            h, w = shape[-2], shape[-1]
            print(f"  Frame size read from image: {h} x {w}")
            return h, w
        except Exception as exc:  # noqa: BLE001 - best-effort, fall back below
            print(f"  [warning] could not read frame shape from {sample_fp}: {exc}")

    print(f"  [warning] falling back to default frame size "
          f"{DEFAULT_FRAME_H} x {DEFAULT_FRAME_W}")
    return DEFAULT_FRAME_H, DEFAULT_FRAME_W


# --- Filters ---------------------------------------------------------------

def add_area_check(df):
    """Flag nuclei whose area exceeds AREA_MAX. Low end is enforced upstream."""
    df["area_check"] = np.where(df["area"] > AREA_MAX, "fail", "pass")
    return df


def add_edge_check(df, H, W, radius=ROI_RADIUS):
    """Flag ROIs whose circle extends past any frame border.

    centroid-0 is row (y), centroid-1 is column (x).
    """
    x = df["centroid-1"]
    y = df["centroid-0"]
    overflow = (
        (x - radius < 0) | (x + radius > W)
        | (y - radius < 0) | (y + radius > H)
    )
    df["edge_check"] = np.where(overflow, "fail", "pass")
    return df


def add_overlap_check(df, radius=ROI_RADIUS, overlap_fraction=OVERLAP_FRACTION):
    """Flag ROIs that overlap another ROI on the same frame by more than
    overlap_fraction of a single ROI's area.

    Overlap is inherently within-frame, so this runs per image_name group.
    """
    circle_area = np.pi * radius ** 2
    area_thresh = overlap_fraction * circle_area

    df["overlap_check"] = "pass"

    for _, grp in df.groupby("image_name"):
        idx = grp.index.to_numpy()
        xs = grp["centroid-1"].to_numpy()
        ys = grp["centroid-0"].to_numpy()

        # pairwise center distances
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist = np.sqrt(dx ** 2 + dy ** 2)

        n = len(idx)
        failed = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if dist[i, j] < 2 * radius:  # circles touch at all
                    if lens_area(dist[i, j], radius) > area_thresh:
                        failed[i] = True
                        failed[j] = True

        df.loc[idx[failed], "overlap_check"] = "fail"

    return df


def apply_filters(features_df, H, W):
    """Add all per-ROI check columns and a combined `selected` boolean."""
    df = features_df.copy()
    df = add_area_check(df)
    df = add_edge_check(df, H, W)
    df = add_overlap_check(df)

    # `contrast_check` originates in segmentation (per-frame). Include it in the
    # final selection if it's present so all gates combine in one place.
    passes = (df[CHECK_COLS] == "pass").all(axis=1)
    if "contrast_check" in df.columns:
        passes &= (df["contrast_check"] == "pass")
    df["selected"] = passes

    return df


# --- Per-experiment driver -------------------------------------------------

def experiment_name_from_parquet(parquet_path):
    """Best-effort experiment name from the features-parquet filename."""
    stem = parquet_path.stem
    return stem.split("_nuclei_features")[0] if "_nuclei_features" in stem else stem


def discover_feature_parquets(input_dir, selected=None):
    """Return nuclei-feature parquet paths under input_dir.

    If `selected` is given, only parquets whose filename starts with one of
    those experiment names are returned.
    """
    all_parquets = sorted(input_dir.glob("*_nuclei_features*.parquet"))
    if selected is None:
        return all_parquets

    chosen = []
    for name in selected:
        matches = [p for p in all_parquets if p.name.startswith(name)]
        if matches:
            chosen.extend(matches)
        else:
            print(f"[warning] no feature parquet found for experiment: {name}")
    return chosen


def process_feature_parquet(parquet_path, output_dir, frame_size_override, rec):
    """Apply ROI filters to one experiment's feature parquet."""
    features_df = pd.read_parquet(parquet_path)
    exp_name = experiment_name_from_parquet(parquet_path)

    if features_df.empty:
        print(f"[warning] {parquet_path.name} has no nuclei, skipping.")
        return None

    required = {"area", "centroid-0", "centroid-1", "image_name"}
    missing = required - set(features_df.columns)
    if missing:
        print(f"[warning] {parquet_path.name} missing columns {missing}, skipping.")
        return None

    print(f"\n{exp_name}: filtering {len(features_df)} nuclei "
          f"across {features_df['image_name'].nunique()} frame(s)...")
    rec.log(f"{exp_name}: filtering {len(features_df)} nuclei")

    H, W = resolve_frame_size(features_df, frame_size_override)
    annotated = apply_filters(features_df, H, W)

    out_path = output_dir / f"{exp_name}_nuclei_filtered.parquet"
    annotated.to_parquet(out_path, index=False)

    # --- Per-experiment summary ---
    total = len(annotated)
    n_area_fail = int((annotated["area_check"] == "fail").sum())
    n_edge_fail = int((annotated["edge_check"] == "fail").sum())
    n_overlap_fail = int((annotated["overlap_check"] == "fail").sum())
    n_selected = int(annotated["selected"].sum())

    print(f"\n--- {exp_name} summary ---")
    print(f"  Total nuclei:              {total}")
    print(f"  Failed area (> {AREA_MAX}):   {n_area_fail}")
    print(f"  Failed edge check:         {n_edge_fail}")
    print(f"  Failed overlap check:      {n_overlap_fail}")
    print(f"  Selected (pass all):       {n_selected} "
          f"({n_selected / total * 100:.1f}%)")
    print(f"  Wrote -> {out_path}")
    rec.log(f"{exp_name}: {n_selected}/{total} selected "
            f"({n_selected / total * 100:.1f}%) -> {out_path.name}")

    return {
        "exp_name": exp_name,
        "total": total,
        "n_area_fail": n_area_fail,
        "n_edge_fail": n_edge_fail,
        "n_overlap_fail": n_overlap_fail,
        "n_selected": n_selected,
    }


# --- Main ------------------------------------------------------------------

def _parse_frame_size(value):
    """Parse a 'HxW' or 'H,W' frame-size string into (H, W)."""
    parts = value.replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("frame size must be 'HxW' or 'H,W'")
    return int(parts[0]), int(parts[1])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply ROI-selection filters (area, edge, overlap) to "
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
        "--frame-size",
        type=_parse_frame_size,
        default=None,
        metavar="HxW",
        help="Frame dimensions as 'HxW' (e.g. 2160x2160). If omitted, the shape "
             "is read from one image per experiment, falling back to "
             f"{DEFAULT_FRAME_H}x{DEFAULT_FRAME_W}.",
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
            "frame_size_override": (
                f"{args.frame_size[0]}x{args.frame_size[1]}"
                if args.frame_size else None
            ),
            "experiments_requested": args.experiments or "ALL",
        },
    )
    print(f"\n=== RUN ID: {args.run_id} ===")
    print(f"=== run dir: {run_dir} ===\n")

    parquets = discover_feature_parquets(input_dir, args.experiments)
    print(f"Found {len(parquets)} feature parquet(s) to process.")
    rec.log(f"found {len(parquets)} feature parquet(s) in {input_dir}")

    all_stats = []
    for parquet_path in parquets:
        stats = process_feature_parquet(
            parquet_path, output_dir, args.frame_size, rec
        )
        if stats is not None:
            all_stats.append(stats)

    # --- Grand total across all experiments ---
    total = sum(s["total"] for s in all_stats)
    n_area_fail = sum(s["n_area_fail"] for s in all_stats)
    n_edge_fail = sum(s["n_edge_fail"] for s in all_stats)
    n_overlap_fail = sum(s["n_overlap_fail"] for s in all_stats)
    n_selected = sum(s["n_selected"] for s in all_stats)

    if all_stats:
        print("\n" + "=" * 40)
        print("OVERALL SUMMARY")
        print(f"  Experiments processed:     {len(all_stats)}")
        print(f"  Total nuclei:              {total}")
        print(f"  Failed area check:         {n_area_fail}")
        print(f"  Failed edge check:         {n_edge_fail}")
        print(f"  Failed overlap check:      {n_overlap_fail}")
        print(f"  Selected (pass all):       {n_selected} "
              f"({n_selected / total * 100:.1f}%)")
        print("=" * 40)

    rec.finish(
        outputs={"nuclei_filtered_dir": str(output_dir)},
        summary={
            "experiments_processed": len(all_stats),
            "total_nuclei": total,
            "n_area_fail": n_area_fail,
            "n_edge_fail": n_edge_fail,
            "n_overlap_fail": n_overlap_fail,
            "n_selected": n_selected,
            "selected_fraction": round(n_selected / total, 4) if total else None,
            "per_experiment": all_stats,
        },
    )

    print(f"\n=== RUN ID: {args.run_id} (pass to downstream stages) ===")


if __name__ == "__main__":
    main()