import sys
import argparse
from pathlib import Path
from importlib import import_module

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage.measure import label
import re

sys.path.insert(0, str(Path(__file__).resolve().parent))
_seg = import_module("3_organelle_segmentation")
SEGMENTERS_BY_STRUCTURE = _seg.SEGMENTERS_BY_STRUCTURE
assign_objects_to_rois   = _seg.assign_objects_to_rois
build_samplesheet        = _seg.build_samplesheet

ROI_RADIUS = 120
PANEL      = 1


def bbox_from_center(y, x, r, H, W):
    yi, xi = int(round(y)), int(round(x))
    return max(yi-r, 0), min(yi+r+1, H), max(xi-r, 0), min(xi+r+1, W)


def _same_frame(fname, dapi_fname):
    pref = re.match(r"(r\d+c\d+f\d+p\d+)", str(dapi_fname))
    return bool(pref) and str(fname).startswith(pref.group(1))


def qc_frame(dapi_fname, structure, samplesheet, nuc_all, rep_rois,
             radius, out_dir):
    site = samplesheet.copy()
    for c in ("Stain", "Structure", "filename"):
        if c in site.columns:
            site[c] = site[c].astype(str).str.strip()

    row = site.loc[(site["Structure"] == structure) &
                   (site["filename"].apply(lambda f: _same_frame(f, dapi_fname)))]
    if row.empty:
        print(f"  [skip] no {structure} channel for {dapi_fname}")
        return 0
    ch_img = tifffile.imread(row.iloc[0]["filepath"])
    if ch_img.ndim == 3 and ch_img.shape[0] == 1:
        ch_img = ch_img[0]

    drow = site.loc[(site["Stain"].str.upper() == "DAPI") &
                    (site["filename"] == dapi_fname)]
    dapi_img = None
    if not drow.empty:
        dapi_img = tifffile.imread(drow.iloc[0]["filepath"])
        if dapi_img.ndim == 3 and dapi_img.shape[0] == 1:
            dapi_img = dapi_img[0]

    H, W = ch_img.shape[:2]
    ys = nuc_all["centroid-0"].astype(float).to_numpy()
    xs = nuc_all["centroid-1"].astype(float).to_numpy()
    centroids_yx = np.column_stack([ys, xs])

    global_seg = SEGMENTERS_BY_STRUCTURE[structure](ch_img)
    obj_labels = label(global_seg, connectivity=1)
    assignment = assign_objects_to_rois(obj_labels, centroids_yx, radius)
    assigned = np.isin(obj_labels, list(assignment.keys())) if assignment \
        else np.zeros_like(global_seg, dtype=bool)

    tag = rep_rois["Genotype_Name"].iloc[0]
    safe = f"{tag}__{structure}__{dapi_fname}".replace("/", "_").replace(" ", "_")
    sel_yx = rep_rois[["centroid-0", "centroid-1"]].astype(float).to_numpy()

    # three-panel overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if dapi_img is not None:
        axes[0].imshow(dapi_img, cmap="gray")
    axes[0].axis("off")
    axes[1].imshow(ch_img, cmap="gray")
    axes[1].axis("off")
    axes[2].imshow(ch_img, cmap="gray"); axes[2].imshow(assigned, alpha=0.4)
    axes[2].axis("off")
    for ax in axes:
        for y, x in zip(ys, xs):
            ax.add_patch(Circle((x, y), radius, fill=False, edgecolor="red", linewidth=1))
        for y, x in sel_yx:
            ax.add_patch(Circle((x, y), radius, fill=False, edgecolor="yellow", linewidth=2))
    plt.tight_layout()
    fig.savefig(out_dir / f"{safe}__panel.png", bbox_inches="tight", dpi=100)
    plt.close(fig)

    # tile crops of selected ROIs
    n = len(sel_yx)
    fig, axs = plt.subplots(1, n, figsize=(3*n, 3))
    if n == 1:
        axs = [axs]
    for ax, (cy, cx), (_, r) in zip(axs, sel_yx, rep_rois.iterrows()):
        y0, y1, x0, x1 = bbox_from_center(cy, cx, radius, H, W)
        ax.imshow(ch_img[y0:y1, x0:x1], cmap="gray")
        ax.imshow(assigned[y0:y1, x0:x1], alpha=0.4)
        ax.add_patch(Circle((cx-x0, cy-y0), radius, fill=False,
                            edgecolor="yellow", linewidth=1.5))
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_dir / f"{safe}__tiles.png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    return 2


def parse_args():
    p = argparse.ArgumentParser(description="ROI-level QC from a near-mean CSV.")
    p.add_argument("--rep-csv", type=Path, required=True,
                   help="CSV of selected ROIs (from frames_near_mean).")
    p.add_argument("--run-id", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--src-base", type=Path,
                   default=Path("/data/CARDPB2/iNDI/Production/AbPanel1"))
    p.add_argument("--panel", type=int, default=PANEL)
    p.add_argument("--radius", type=int, default=ROI_RADIUS)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = args.output_root / f"run_{args.run_id}"
    if not run_dir.is_dir():
        raise SystemExit(f"[error] run dir not found: {run_dir}")

    nuclei_dir = run_dir / "nuclei_filtered"
    out_dir = run_dir / "qc_roi_near_mean"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== run dir: {run_dir} ===")
    print(f"=== output:  {out_dir} ===")

    rep = pd.read_csv(args.rep_csv)

    ss_cache, nuc_cache = {}, {}
    written = 0
    # Measurement_ID IS the experiment folder name -> group on it directly
    for (exp, dapi, struct), grp in rep.groupby(
            ["Measurement_ID", "DAPI_filename", "Structure"]):
        exp = str(exp).strip()
        dapi = str(dapi).strip()

        # per-experiment nuclei parquet (cached) — centroids + full nucleus set
        if exp not in nuc_cache:
            pq = next(nuclei_dir.glob(f"{exp}*_nuclei_filtered*.parquet"), None)
            if pq is None:
                print(f"[skip] no nuclei parquet for {exp}"); nuc_cache[exp] = None
            else:
                nf = pd.read_parquet(pq)
                nf["image_name"] = nf["image_name"].astype(str).str.strip()
                if "selected" in nf.columns:
                    nf = nf[nf["selected"]]
                nuc_cache[exp] = nf
        nf = nuc_cache[exp]
        if nf is None:
            continue

        if exp not in ss_cache:
            ss = build_samplesheet(args.src_base / exp, args.panel)
            if ss is not None:
                ss["filename"] = ss["filename"].astype(str).str.strip()
            ss_cache[exp] = ss
        ss = ss_cache[exp]
        if ss is None:
            print(f"[skip] no samplesheet for {exp}"); continue

        nuc_all = nf[nf["image_name"] == dapi]
        if nuc_all.empty:
            print(f"[skip] no nuclei for {dapi}"); continue

        # attach centroids to selected ROIs by per-frame nucleus id
        id_col = "label" if "label" in nuc_all.columns else "Nucleus_ID"
        cen = nuc_all.set_index(id_col)[["centroid-0", "centroid-1"]]
        grp = grp.copy().join(cen, on="Nucleus_ID")
        grp = grp.dropna(subset=["centroid-0", "centroid-1"])
        if grp.empty:
            print(f"[skip] no centroid match for {dapi}"); continue

        print(f"{grp['Genotype_Name'].iloc[0]} | {exp} | {dapi} ({len(grp)} ROI)")
        written += qc_frame(dapi, struct, ss, nuc_all, grp, args.radius, out_dir)

    print(f"\nDone. {written} PNG(s) in {out_dir}")


if __name__ == "__main__":
    main()