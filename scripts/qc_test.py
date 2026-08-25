import sys
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
from scipy import ndimage as ndi
from scipy.stats import norm
from skimage import filters, morphology
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

# Nucleus segmentation constants, MIRRORED from 1_nucleus_segmentation.py so
# the DAPI mask shown here matches what stage 1 produced. Keep in sync if the
# stage-1 segmentation params change.
_nuc = import_module("1_nucleus_segmentation")
NUC_INTENSITY_SCALING_PARAM = _nuc.INTENSITY_SCALING_PARAM
NUC_BLUR_SIGMA = _nuc.BLUR_SIGMA
NUC_MIN_AREA = _nuc.MIN_AREA


def segment_nuclei(nuc):
    """Reproduce 1_nucleus_segmentation.py's DAPI segmentation, returning the
    labeled nucleus mask. Mirrors process_nucleus_image's steps exactly (minus
    the contrast gate / feature extraction, which QC doesn't need)."""
    m, s = norm.fit(nuc.flatten())
    stretch_min = max(m - NUC_INTENSITY_SCALING_PARAM[0] * s, nuc.min())
    stretch_max = min(m + NUC_INTENSITY_SCALING_PARAM[1] * s, nuc.max())
    nuc_stretch = np.clip(nuc, stretch_min, stretch_max)
    image_norm = (nuc_stretch - stretch_min) / (stretch_max - stretch_min)

    blurred = filters.gaussian(image_norm, sigma=NUC_BLUR_SIGMA)

    triangle_cutoff = filters.threshold_triangle(blurred)
    global_median_cutoff = np.percentile(blurred, 50)
    th_low_cutoff = (triangle_cutoff + global_median_cutoff) / 2
    img_low_level = blurred > th_low_cutoff
    img_low_level = morphology.remove_small_objects(
        img_low_level.astype(bool), min_size=NUC_MIN_AREA
    )
    img_low_level = morphology.dilation(img_low_level, footprint=morphology.disk(2))

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
    filled_clean = morphology.remove_small_objects(filled.astype(bool), min_size=NUC_MIN_AREA)
    return morphology.label(filled_clean)

# --- Configuration ---------------------------------------------------------

ROI_RADIUS = 120
SELECTED_ONLY = True

DEFAULT_SRC_BASE = Path("/data/CARDPB2/iNDI/Production/AbPanel1")
DEFAULT_PANEL = 1

# Stage name + the sub-dirs this stage reads from / writes to inside a run.
STAGE = "roi_targeted_qc"
INPUT_STAGE_DIR = "nuclei_filtered"
OUTPUT_STAGE_DIR = "qc_roi_targeted"


def collect_params():
    """Config recorded in run_metadata.json for this stage."""
    return {
        "roi_radius": ROI_RADIUS,
        "selected_only": SELECTED_ONLY,
        "segmenters": sorted(SEGMENTERS_BY_STRUCTURE.keys()),
    }


# --- Target ROI loading ----------------------------------------------------

def load_target_rois(csv_path):
    """Load target ROIs from a CSV.

    Requires columns 'Experiment_name', 'DAPI_filename', and 'Nucleus_ID'.
    Experiment_name is REQUIRED because (DAPI_filename, Nucleus_ID) is only
    unique within one experiment -- positional filenames like
    'r02c03f04p01-ch01t01.tiff' and per-frame Nucleus_ID labels both recur
    across experiments, so keying without it renders ROIs against the wrong
    image.

    An optional 'Structure' column, if present, is treated as the FLAGGED
    structure for that ROI (the one that hit count==1) and is marked in the
    figure; all channels are still drawn. Multiple flagged structures for the
    same nucleus are unioned.

    Returns
    -------
    dict[tuple[str, str], dict[int, set[str]]]
        (experiment_name, frame_name) -> {nucleus_label -> set(flagged structures)}
    """
    tdf = pd.read_csv(csv_path)
    required = {"Experiment_name", "DAPI_filename", "Nucleus_ID"}
    missing = required - set(tdf.columns)
    if missing:
        raise SystemExit(f"[error] target CSV missing columns: {missing}")

    has_structure = "Structure" in tdf.columns
    targets = {}
    for (exp, fname), grp in tdf.groupby(["Experiment_name", "DAPI_filename"]):
        key = (str(exp).strip(), str(fname).strip())
        per_nuc = {}
        for _, r in grp.iterrows():
            lab = int(r["Nucleus_ID"])
            flagged = per_nuc.setdefault(lab, set())
            if has_structure and pd.notna(r["Structure"]):
                flagged.add(str(r["Structure"]).strip())
        targets[key] = per_nuc
    return targets


# --- Rendering -------------------------------------------------------------

def bbox_from_center(y, x, r, H, W):
    yi, xi = int(round(y)), int(round(x))
    return max(yi - r, 0), min(yi + r + 1, H), max(xi - r, 0), min(xi + r + 1, W)


def qc_roi_figure(structures, raw_tiles, mask_tiles, center_local, radius,
                  flagged, nucleus_id, frame_name, out_path):
    """One figure for a single ROI: a 2 x n_channels grid.

    Top row    : raw channel crop (+ ROI circle)
    Bottom row : same crop with the ASSIGNED organelle mask overlaid
    Column titles name the structure; flagged (count==1) structures are marked.
    """
    n = len(structures)
    if n == 0:
        return False
    cy, cx = center_local
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.4), squeeze=False)

    for j, struct in enumerate(structures):
        is_flagged = struct in flagged
        col_title = f"{struct}  [count=1]" if is_flagged else struct
        title_color = "red" if is_flagged else "black"

        # top: raw
        ax_top = axes[0][j]
        ax_top.imshow(raw_tiles[j], cmap="gray")
        ax_top.add_patch(Circle((cx, cy), radius, fill=False,
                                edgecolor="red", linewidth=1))
        ax_top.set_title(col_title, fontsize=10, color=title_color)
        ax_top.axis("off")

        # bottom: raw + assigned mask
        ax_bot = axes[1][j]
        ax_bot.imshow(raw_tiles[j], cmap="gray")
        ax_bot.imshow(mask_tiles[j], alpha=0.4)
        ax_bot.add_patch(Circle((cx, cy), radius, fill=False,
                                edgecolor="red", linewidth=1))
        ax_bot.axis("off")

    axes[0][0].set_ylabel("raw", fontsize=10)
    axes[1][0].set_ylabel("+ mask", fontsize=10)

    flag_str = ", ".join(sorted(flagged)) if flagged else "none"
    fig.suptitle(f"{frame_name}  |  Nucleus_ID={nucleus_id}  |  "
                 f"flagged: {flag_str}", y=1.01, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return True


# --- Per-frame QC ----------------------------------------------------------

def qc_one_frame(frame_name, site_df, nuc_grp, radius, out_dir, per_nuc_flags):
    """For each target nucleus on this frame, emit ONE PNG showing every
    segmentable channel (raw on top, raw+assigned-mask below), with the
    flagged structure(s) marked.

    A mask column can legitimately be empty: that means the pipeline assigned
    no organelle of that structure to the ROI, which is worth eyeballing for a
    supposed count==1 case.
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

    if "label" in nuc_grp.columns:
        labels = nuc_grp["label"].astype(int).to_numpy()
    else:
        labels = np.arange(1, len(xs) + 1)

    # centroid-array positions of the target nuclei present on this frame
    label_to_pos = {int(lab): i for i, lab in enumerate(labels)}
    target_items = [(lab, label_to_pos[lab]) for lab in per_nuc_flags
                    if lab in label_to_pos]
    if not target_items:
        return 0

    structures = sorted(set(site_df["Structure"].unique())
                        & set(SEGMENTERS_BY_STRUCTURE))
    if not structures:
        return 0

    # DAPI channel for this frame (nucleus reference column). Segment it with
    # stage-1's logic to get the actual nucleus mask.
    dapi_img = None
    dapi_labels = None
    dapi_rows = site_df.loc[
        site_df["Stain"].astype(str).str.strip().str.upper() == "DAPI"
    ]
    if not dapi_rows.empty:
        dapi_img = tifffile.imread(dapi_rows.iloc[0]["filepath"])
        if dapi_img.ndim == 3 and dapi_img.shape[0] == 1:
            dapi_img = dapi_img[0]
        dapi_labels = segment_nuclei(dapi_img)

    # Segment each structure ONCE for the whole frame, run the real assignment,
    # and cache (channel image, per-ROI object labels) for cropping per ROI.
    frame_seg = {}  # structure -> (ch_img, obj_labels, by_roi)
    H = W = None
    for structure in structures:
        seg_fn = SEGMENTERS_BY_STRUCTURE[structure]
        row = site_df.loc[site_df["Structure"] == structure].iloc[0]
        ch_img = tifffile.imread(row["filepath"])
        if ch_img.ndim == 3 and ch_img.shape[0] == 1:
            ch_img = ch_img[0]
        H, W = ch_img.shape[:2]

        global_seg = seg_fn(ch_img)
        obj_labels = label(global_seg, connectivity=1)
        assignment = assign_objects_to_rois(obj_labels, centroids_yx, radius)

        by_roi = {}
        for obj_lab, roi_idx in assignment.items():
            by_roi.setdefault(roi_idx, []).append(obj_lab)

        frame_seg[structure] = (ch_img, obj_labels, by_roi)

    # Fall back to a blank DAPI frame if it's missing, so the crop geometry
    # still works (mask column will be empty for that ROI).
    if dapi_img is None and H is not None:
        dapi_img = np.zeros((H, W), dtype=np.uint8)

    safe_frame = frame_name.replace("/", "_")
    written = 0

    for nucleus_id, roi_idx in target_items:
        cy, cx = centroids_yx[roi_idx]
        y0, y1, x0, x1 = bbox_from_center(cy, cx, radius, H, W)
        center_local = (cy - y0, cx - x0)

        # DAPI column first: raw nucleus crop + the stage-1 nucleus mask for
        # THIS nucleus. The re-segmentation has its own labels, so identify the
        # target nucleus by which segmented object contains its centroid rather
        # than assuming label equality with Nucleus_ID.
        col_structures = ["DAPI"] + structures
        dapi_crop = dapi_img[y0:y1, x0:x1]
        if dapi_labels is not None:
            iy, ix = int(round(cy)), int(round(cx))
            iy = min(max(iy, 0), dapi_labels.shape[0] - 1)
            ix = min(max(ix, 0), dapi_labels.shape[1] - 1)
            this_lab = dapi_labels[iy, ix]
            if this_lab != 0:
                nuc_mask_full = dapi_labels == this_lab
            else:
                # centroid landed on background; fall back to ROI disk so the
                # column still shows the nucleus location.
                yy, xx = np.ogrid[:dapi_labels.shape[0], :dapi_labels.shape[1]]
                nuc_mask_full = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
            dapi_mask_crop = nuc_mask_full[y0:y1, x0:x1]
        else:
            dapi_mask_crop = np.zeros_like(dapi_crop, dtype=bool)
        raw_tiles, mask_tiles = [dapi_crop], [dapi_mask_crop]

        for structure in structures:
            ch_img, obj_labels, by_roi = frame_seg[structure]
            raw_tiles.append(ch_img[y0:y1, x0:x1])
            roi_labs = by_roi.get(roi_idx, [])
            roi_mask = np.isin(obj_labels, roi_labs) if roi_labs \
                else np.zeros_like(obj_labels, dtype=bool)
            mask_tiles.append(roi_mask[y0:y1, x0:x1])

        out_path = out_dir / f"{safe_frame}__nuc{nucleus_id}__roi.png"
        ok = qc_roi_figure(
            col_structures, raw_tiles, mask_tiles, center_local, radius,
            flagged=per_nuc_flags[nucleus_id], nucleus_id=nucleus_id,
            frame_name=frame_name, out_path=out_path,
        )
        written += int(ok)

    return written


# --- Per-experiment driver -------------------------------------------------

def experiment_name_from_parquet(parquet_path):
    stem = parquet_path.stem
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


def qc_experiment(parquet_path, src_base, output_dir, panel, radius,
                  target_rois, rec):
    nuclei_features = pd.read_parquet(parquet_path)
    exp_name = experiment_name_from_parquet(parquet_path)

    if SELECTED_ONLY and "selected" in nuclei_features.columns:
        nuclei_features = nuclei_features[nuclei_features["selected"]].copy()
    if nuclei_features.empty:
        print(f"[warning] no selected nuclei in {parquet_path.name}, skipping.")
        return 0

    nuclei_features["image_name"] = nuclei_features["image_name"].astype(str).str.strip()

    # Restrict targets to THIS experiment's folder. The target dict is keyed on
    # (Experiment_name, DAPI_filename); pull only the frames belonging here so a
    # recurring filename from another experiment can never be rendered against
    # this one's images.
    exp_targets = {
        fname: flags for (exp, fname), flags in target_rois.items()
        if exp == exp_name
    }
    if not exp_targets:
        print(f"  {exp_name}: no target ROIs for this experiment, skipping.")
        return 0

    # Only frames that both have selected nuclei AND contain a target ROI.
    frames = sorted(set(nuclei_features["image_name"].unique())
                    & set(exp_targets.keys()))
    if not frames:
        print(f"  {exp_name}: no target frames present, skipping.")
        return 0

    experiment_path = src_base / exp_name
    if not (experiment_path / "images").is_dir():
        print(f"[warning] no images dir for {exp_name}, skipping.")
        return 0

    samplesheet = build_samplesheet(experiment_path, panel)
    if samplesheet is None:
        print(f"[warning] could not build samplesheet for {exp_name}, skipping.")
        return 0
    samplesheet["filename"] = samplesheet["filename"].astype(str).str.strip()

    exp_out = output_dir / exp_name
    exp_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{exp_name}: QC on {len(frames)} frame(s) containing target ROIs")
    rec.log(f"{exp_name}: QC on {len(frames)} target frame(s)")

    site_keys = ["Row", "Column", "Frame", "Plane", "Time"]
    total_written = 0
    matched_nuclei = 0
    unmatched_nuclei = 0

    for frame_name in frames:
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

        per_nuc_flags = exp_targets[frame_name]

        # sanity: how many requested labels actually exist on this frame?
        present = set(nuc_grp["label"].astype(int)) if "label" in nuc_grp.columns \
            else set(range(1, len(nuc_grp) + 1))
        want = set(per_nuc_flags)
        matched_nuclei += len(want & present)
        unmatched_nuclei += len(want - present)

        written = qc_one_frame(
            frame_name, site_df, nuc_grp, radius, exp_out, per_nuc_flags,
        )
        total_written += written
        print(f"  {frame_name} -> {written} ROI figure(s)")

    if unmatched_nuclei:
        print(f"  [warning] {exp_name}: {unmatched_nuclei} target Nucleus_ID(s) "
              f"not found among frame labels (matched {matched_nuclei}). "
              f"Check that Nucleus_ID maps to the 'label' column.")
        rec.log(f"{exp_name}: {unmatched_nuclei} unmatched target Nucleus_ID(s), "
                f"{matched_nuclei} matched")

    print(f"  wrote {total_written} PNG(s) to {exp_out}")
    rec.log(f"{exp_name}: wrote {total_written} PNG(s)")
    return total_written


# --- Main ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Targeted organelle QC: one PNG per flagged ROI showing "
                    "every channel (raw on top, raw+assigned-mask below) with "
                    "the flagged count==1 structure marked. Reuses script 3's "
                    "segmentation/assignment."
    )
    parser.add_argument("-t", "--target-rois", type=Path, required=True,
                        metavar="CSV",
                        help="CSV of target ROIs. Requires columns "
                             "DAPI_filename and Nucleus_ID; optional Structure "
                             "column marks the flagged (count==1) structure.")
    parser.add_argument("-e", "--experiments", nargs="+", default=None, metavar="NAME",
                        help="Specific experiment names (parquet prefixes).")
    parser.add_argument("-b", "--src-base", type=Path, default=DEFAULT_SRC_BASE,
                        help=f"Raw experiment base path (default: {DEFAULT_SRC_BASE}).")
    parser.add_argument("-p", "--panel", type=int, default=DEFAULT_PANEL, choices=[1, 2])
    parser.add_argument("-r", "--radius", type=int, default=ROI_RADIUS)
    # --output-root + --run-id (required here: reuse the run minted by 0_).
    ru.add_run_args(parser, mints_run_id=False)
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve the existing run dir (errors clearly if the run ID is wrong).
    run_dir = ru.resolve_run_dir(args.output_root, args.run_id)
    input_dir = ru.stage_dir(run_dir, INPUT_STAGE_DIR)
    output_dir = ru.stage_dir(run_dir, OUTPUT_STAGE_DIR)

    target_rois = load_target_rois(args.target_rois)
    n_targets = sum(len(nucs) for nucs in target_rois.values())

    rec = ru.StageRecorder(
        run_dir, stage=STAGE, run_id=args.run_id,
        params=collect_params(),
        inputs={
            "input_dir": str(input_dir),
            "src_base": str(args.src_base),
            "panel": args.panel,
            "radius": args.radius,
            "target_rois_csv": str(args.target_rois),
            "n_target_frames": len(target_rois),
            "n_target_rois": n_targets,
            "experiments_requested": args.experiments or "ALL",
        },
    )
    print(f"\n=== RUN ID: {args.run_id} ===")
    print(f"=== run dir: {run_dir} ===")
    print(f"=== targets: {n_targets} ROI(s) across {len(target_rois)} frame(s) ===\n")

    parquets = discover_filtered_parquets(input_dir, args.experiments)
    print(f"Found {len(parquets)} filtered parquet(s) to QC.")
    rec.log(f"found {len(parquets)} filtered parquet(s) in {input_dir}")

    per_experiment = []
    grand_total = 0
    for parquet_path in parquets:
        exp_name = experiment_name_from_parquet(parquet_path)
        written = qc_experiment(
            parquet_path, args.src_base, output_dir, args.panel,
            args.radius, target_rois, rec,
        )
        grand_total += written
        per_experiment.append({"experiment": exp_name, "pngs_written": written})

    print("\n" + "=" * 40)
    print(f"TARGETED QC COMPLETE - wrote {grand_total} PNG(s) total")
    print("=" * 40)

    rec.finish(
        outputs={"qc_roi_targeted_dir": str(output_dir)},
        summary={
            "experiments_qcd": len([e for e in per_experiment if e["pngs_written"]]),
            "total_pngs": grand_total,
            "n_target_rois": n_targets,
            "per_experiment": per_experiment,
        },
    )

    print(f"\n=== RUN ID: {args.run_id} ===")


if __name__ == "__main__":
    main()