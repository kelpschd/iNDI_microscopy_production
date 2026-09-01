import sys
import random
import argparse
from pathlib import Path
from importlib import import_module

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, no display
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage.measure import label

# --- run_utils bootstrap ---------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_utils as ru  # noqa: E402

# Reuse the real segmentation + assignment logic so QC can't drift from the
# pipeline. These come straight from script 3.
_seg = import_module("3_organelle_segmentation")
SEGMENTERS_BY_STRUCTURE = _seg.SEGMENTERS_BY_STRUCTURE
assign_objects_to_rois = _seg.assign_objects_to_rois
build_samplesheet = _seg.build_samplesheet
# Same per-stain contrast gate as script 3, so QC skips exactly the channels
# the pipeline would skip.
CONTRAST_CUTOFF_BY_STAIN = _seg.CONTRAST_CUTOFF_BY_STAIN
CONTRAST_CUTOFF_DEFAULT = _seg.CONTRAST_CUTOFF_DEFAULT

# --- Configuration ---------------------------------------------------------

ROI_RADIUS = 120
DEFAULT_N_FRAMES = 5              # random frames to QC per experiment
QC_MAX_TILES_PER_STRUCT = 8       # tiles in the gallery (matches original)
SELECTED_ONLY = True

DEFAULT_SRC_BASE = Path("/data/CARDPB2/iNDI/Production/AbPanel1")
DEFAULT_PANEL = 1

# Stage name + the sub-dirs this stage reads from / writes to inside a run.
STAGE = "segmentation_qc"
INPUT_STAGE_DIR = "nuclei_filtered"
OUTPUT_STAGE_DIR = "qc"


def collect_params():
    """Config recorded in run_metadata.json for this stage."""
    return {
        "roi_radius": ROI_RADIUS,
        "n_frames": DEFAULT_N_FRAMES,
        "max_tiles_per_struct": QC_MAX_TILES_PER_STRUCT,
        "selected_only": SELECTED_ONLY,
        "segmenters": sorted(SEGMENTERS_BY_STRUCTURE.keys()),
        "contrast_cutoff_by_stain": CONTRAST_CUTOFF_BY_STAIN,
        "contrast_cutoff_default": CONTRAST_CUTOFF_DEFAULT,
    }


# --- Rendering (mirrors the original notebook QC) --------------------------

def _draw_circles(ax, xs, ys, r):
    for y, x in zip(ys, xs):
        ax.add_patch(Circle((x, y), r, fill=False, edgecolor="red", linewidth=1))


def qc_show_three_panel(nuc_img, ch_img, global_seg, xs, ys, r,
                        title_prefix, out_path):
    """DAPI | Channel | Channel+segmentation overlay, all with ROI circles.
    Matches the original notebook's three-panel figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(nuc_img, cmap="gray"); axes[0].set_title(f"{title_prefix} - DAPI");    axes[0].axis("off")
    axes[1].imshow(ch_img,  cmap="gray"); axes[1].set_title(f"{title_prefix} - Channel"); axes[1].axis("off")
    axes[2].imshow(ch_img,  cmap="gray"); axes[2].imshow(global_seg, alpha=0.4)
    axes[2].set_title(f"{title_prefix} - Overlay"); axes[2].axis("off")
    for ax in axes:
        _draw_circles(ax, xs, ys, r)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def qc_tile_gallery(tiles, seg_tiles, centers_local, radius, title_prefix,
                    out_path, max_tiles=8):
    """A row of per-ROI tiles with the ASSIGNED segmentation overlaid and the
    ROI circle drawn. seg_tiles contain only objects assigned to that ROI
    (full-containment), matching script 3. centers_local gives the circle
    center within each tile's local coordinates."""
    if max_tiles <= 0 or len(tiles) == 0:
        return
    n = min(max_tiles, len(tiles))
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, tile, seg, (cy, cx) in zip(
        axes, tiles[:n], seg_tiles[:n], centers_local[:n]
    ):
        ax.imshow(tile, cmap="gray")
        ax.imshow(seg, alpha=0.4)
        ax.add_patch(Circle((cx, cy), radius, fill=False,
                            edgecolor="red", linewidth=1))
        ax.axis("off")
    fig.suptitle(f"{title_prefix} - example ROI tiles", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def bbox_from_center(y, x, r, H, W):
    yi, xi = int(round(y)), int(round(x))
    return max(yi - r, 0), min(yi + r + 1, H), max(xi - r, 0), min(xi + r + 1, W)


# --- Per-frame QC ----------------------------------------------------------

def qc_one_frame(frame_name, site_df, nuc_grp, dapi_img, radius, out_dir):
    """Segment each structure on the WHOLE frame, then build the same two QC
    figures as the original notebook (three-panel + tile gallery).

    A per-stain contrast gate (identical to script 3) runs first: channels
    whose raw-frame std falls below their cutoff are skipped entirely, so QC
    never renders a mask for a channel the pipeline would have dropped.

    The mask shown is the whole-frame segmentation (the current pipeline
    behavior); tiles are crops of that mask around each ROI, so the gallery
    reflects what actually gets assigned rather than per-tile re-segmentation.
    """
    site_df = site_df.copy()
    for c in ("Stain", "Structure", "filename"):
        if c in site_df.columns:
            site_df[c] = site_df[c].astype(str).str.strip()

    ys = nuc_grp["centroid-0"].astype(float).to_numpy()
    xs = nuc_grp["centroid-1"].astype(float).to_numpy()
    if len(xs) == 0:
        return 0
    centroids_yx = np.column_stack([ys, xs])

    H, W = dapi_img.shape[:2]
    todo = sorted(set(site_df["Structure"].unique()) & set(SEGMENTERS_BY_STRUCTURE))
    written = 0

    for structure in todo:
        seg_fn = SEGMENTERS_BY_STRUCTURE[structure]
        row = site_df.loc[site_df["Structure"] == structure].iloc[0]
        ch_img = tifffile.imread(row["filepath"])
        if ch_img.ndim == 3 and ch_img.shape[0] == 1:
            ch_img = ch_img[0]

        # Contrast gate (per stain) — identical to script 3. Skip figures for
        # channels the pipeline would drop, so QC reflects real behavior.
        frame_std = float(ch_img.astype(np.float64).std())
        stain = str(row.get("Stain", "")).strip()
        cutoff = CONTRAST_CUTOFF_BY_STAIN.get(stain, CONTRAST_CUTOFF_DEFAULT)
        if frame_std < cutoff:
            print(f"    [contrast-fail] {structure} ({stain}) "
                  f"std={frame_std:.1f} < {cutoff} -> skipped")
            continue

        # whole-frame segmentation (current pipeline behavior)
        global_seg = seg_fn(ch_img)

        # run the REAL assignment: which objects belong to which ROI under
        # full-containment + nearest-centroid (identical to script 3).
        obj_labels = label(global_seg, connectivity=1)
        assignment = assign_objects_to_rois(obj_labels, centroids_yx, radius)

        # group assigned object labels by ROI index
        by_roi = {}
        for obj_lab, roi_idx in assignment.items():
            by_roi.setdefault(roi_idx, []).append(obj_lab)

        # the overlay panel shows only the ASSIGNED mask (what the pipeline
        # actually measures), not every segmented pixel.
        assigned_seg = np.isin(obj_labels, list(assignment.keys())) if assignment \
            else np.zeros_like(global_seg, dtype=bool)

        title = f"{structure} | {frame_name}"
        safe_frame = frame_name.replace("/", "_")

        # --- three-panel overlay (assigned mask) ---
        panel_path = out_dir / f"{safe_frame}__{structure}__panel.png"
        qc_show_three_panel(dapi_img, ch_img, assigned_seg, xs, ys, radius,
                            title_prefix=title, out_path=panel_path)
        written += 1

        # --- tile gallery: each tile shows only that ROI's assigned objects ---
        example_tiles, example_segs, centers_local = [], [], []
        # prefer ROIs that actually have assigned objects, so tiles are useful
        ranked = sorted(by_roi, key=lambda k: len(by_roi[k]), reverse=True)
        if not ranked:
            ranked = list(range(len(centroids_yx)))
        for roi_idx in ranked:
            if len(example_tiles) >= QC_MAX_TILES_PER_STRUCT:
                break
            cy, cx = centroids_yx[roi_idx]
            y0, y1, x0, x1 = bbox_from_center(cy, cx, radius, H, W)
            example_tiles.append(ch_img[y0:y1, x0:x1])
            # only this ROI's assigned objects, cropped to the tile
            roi_labs = by_roi.get(roi_idx, [])
            roi_mask = np.isin(obj_labels, roi_labs) if roi_labs \
                else np.zeros_like(obj_labels, dtype=bool)
            example_segs.append(roi_mask[y0:y1, x0:x1])
            centers_local.append((cy - y0, cx - x0))

        gallery_path = out_dir / f"{safe_frame}__{structure}__tiles.png"
        qc_tile_gallery(example_tiles, example_segs, centers_local, radius,
                        title_prefix=title, out_path=gallery_path,
                        max_tiles=QC_MAX_TILES_PER_STRUCT)
        written += 1

    return written


# --- Per-experiment driver -------------------------------------------------

def experiment_name_from_parquet(parquet_path):
    stem = parquet_path.stem
    # Filenames are now '<exp>_nuclei_filtered.parquet' (run dir carries date).
    return stem.split("_nuclei_filtered")[0] if "_nuclei_filtered" in stem else stem


def discover_filtered_parquets(input_dir, selected=None):
    all_parquets = sorted(input_dir.glob("*_nuclei_filtered*.parquet"))
    if selected is None:
        return all_parquets
    chosen = []
    for name in selected:
        matches = [p for p in all_parquets if p.name.startswith(name)]
        if matches:
            chosen.extend(matches)
        else:
            print(f"[warning] no filtered parquet found for experiment: {name}")
    return chosen


def _dapi_row(site_df):
    """Return the DAPI channel row for this site (Stain == DAPI)."""
    m = site_df["Stain"].astype(str).str.strip().str.upper() == "DAPI"
    sub = site_df[m]
    return None if sub.empty else sub.iloc[0]


def qc_experiment(parquet_path, src_base, output_dir, panel, radius,
                  n_frames, seed, rec):
    nuclei_features = pd.read_parquet(parquet_path)
    exp_name = experiment_name_from_parquet(parquet_path)

    if SELECTED_ONLY and "selected" in nuclei_features.columns:
        nuclei_features = nuclei_features[nuclei_features["selected"]].copy()
    if nuclei_features.empty:
        print(f"[warning] no selected nuclei in {parquet_path.name}, skipping.")
        return 0

    nuclei_features["image_name"] = nuclei_features["image_name"].astype(str).str.strip()

    experiment_path = src_base / exp_name
    if not (experiment_path / "images").is_dir():
        print(f"[warning] no images dir for {exp_name}, skipping.")
        return 0

    samplesheet = build_samplesheet(experiment_path, panel)
    if samplesheet is None:
        print(f"[warning] could not build samplesheet for {exp_name}, skipping.")
        return 0
    samplesheet["filename"] = samplesheet["filename"].astype(str).str.strip()

    frames = sorted(nuclei_features["image_name"].unique())
    rng = random.Random(seed)
    sample = rng.sample(frames, min(n_frames, len(frames)))

    exp_out = output_dir / exp_name
    exp_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{exp_name}: QC on {len(sample)} random frame(s) "
          f"(of {len(frames)} with selected nuclei)")
    rec.log(f"{exp_name}: QC on {len(sample)} of {len(frames)} frame(s)")

    site_keys = ["Row", "Column", "Frame", "Plane", "Time"]
    total_written = 0

    for frame_name in sample:
        nuc_grp = nuclei_features[nuclei_features["image_name"] == frame_name]
        matched = samplesheet.loc[samplesheet["filename"] == frame_name]
        if matched.empty:
            print(f"  [skip] {frame_name} not in samplesheet")
            continue
        row0 = matched.iloc[0]
        mask = np.ones(len(samplesheet), dtype=bool)
        for k in site_keys:
            if k in samplesheet.columns:
                mask &= samplesheet[k].astype(str).str.strip() == str(row0[k]).strip()
        site_df = samplesheet.loc[mask].copy()

        # read the DAPI image for the three-panel figure
        dapi_row = _dapi_row(site_df)
        if dapi_row is None:
            print(f"  [skip] {frame_name} has no DAPI channel in samplesheet")
            continue
        dapi_img = tifffile.imread(dapi_row["filepath"])
        if dapi_img.ndim == 3 and dapi_img.shape[0] == 1:
            dapi_img = dapi_img[0]

        written = qc_one_frame(frame_name, site_df, nuc_grp, dapi_img,
                               radius, exp_out)
        total_written += written
        print(f"  {frame_name} -> {written} figure(s)")

    print(f"  wrote {total_written} PNG(s) to {exp_out}")
    rec.log(f"{exp_name}: wrote {total_written} PNG(s)")
    return total_written


# --- Main ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate organelle QC figures (three-panel overlay + "
                    "per-ROI tile gallery, matching the original notebook QC) "
                    "for random frames from filtered experiments."
    )
    parser.add_argument("-e", "--experiments", nargs="+", default=None, metavar="NAME",
                        help="Specific experiment names (parquet prefixes).")
    parser.add_argument("-b", "--src-base", type=Path, default=DEFAULT_SRC_BASE,
                        help=f"Raw experiment base path (default: {DEFAULT_SRC_BASE}).")
    parser.add_argument("-p", "--panel", type=int, default=DEFAULT_PANEL, choices=[1, 2])
    parser.add_argument("-r", "--radius", type=int, default=ROI_RADIUS)
    parser.add_argument("-n", "--n-frames", type=int, default=DEFAULT_N_FRAMES,
                        help=f"Random frames per experiment (default: {DEFAULT_N_FRAMES}).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for frame sampling (default: 42).")
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
            "src_base": str(args.src_base),
            "panel": args.panel,
            "radius": args.radius,
            "n_frames": args.n_frames,
            "seed": args.seed,
            "experiments_requested": args.experiments or "ALL",
        },
    )
    print(f"\n=== RUN ID: {args.run_id} ===")
    print(f"=== run dir: {run_dir} ===\n")

    parquets = discover_filtered_parquets(input_dir, args.experiments)
    print(f"Found {len(parquets)} filtered parquet(s) to QC.")
    rec.log(f"found {len(parquets)} filtered parquet(s) in {input_dir}")

    per_experiment = []
    grand_total = 0
    for parquet_path in parquets:
        exp_name = experiment_name_from_parquet(parquet_path)
        written = qc_experiment(
            parquet_path, args.src_base, output_dir, args.panel,
            args.radius, args.n_frames, args.seed, rec,
        )
        grand_total += written
        per_experiment.append({"experiment": exp_name, "pngs_written": written})

    print("\n" + "=" * 40)
    print(f"QC COMPLETE - wrote {grand_total} PNG(s) total")
    print("=" * 40)

    rec.finish(
        outputs={"qc_dir": str(output_dir)},
        summary={
            "experiments_qcd": len([e for e in per_experiment if e["pngs_written"]]),
            "total_pngs": grand_total,
            "per_experiment": per_experiment,
        },
    )

    print(f"\n=== RUN ID: {args.run_id} ===")


if __name__ == "__main__":
    main()