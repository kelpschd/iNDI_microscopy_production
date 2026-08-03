import os
import re
import sys
import time
import shutil
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tifffile
from scipy.stats import norm
from scipy.ndimage import (
    distance_transform_edt, gaussian_laplace, binary_fill_holes,
)
from skimage import filters, morphology
from skimage.measure import label, regionprops_table
from skimage.filters import threshold_triangle, threshold_otsu
from skimage.morphology import remove_small_objects, erosion, dilation
from skimage.morphology import disk as morph_disk

# --- run_utils bootstrap ---------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_utils as ru  # noqa: E402

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# --- Configuration ---------------------------------------------------------

# ROI geometry (must match the ROI definition used in 2_roi_filtering.py)
ROI_RADIUS = 120

# Only nuclei that passed all upstream filters are used as ROI centers.
SELECTED_ONLY = True

# Which experimental panel to join against (stain -> structure map below).
DEFAULT_PANEL = 1

# Parallelism: honor SLURM allocation, else fall back to cpu_count.
N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

# lscratch staging (Biowulf). If SLURM_JOB_ID is unset we stage nowhere.
SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID")
SCRATCH_BASE = f"/lscratch/{SLURM_JOB_ID}" if SLURM_JOB_ID else None

# Raw imaging tree (experiment folders live directly under here).
DEFAULT_SRC_BASE = Path("/data/CARDPB2/iNDI/Production/AbPanel1")

# Stage name + the sub-dirs this stage reads from / writes to inside a run.
STAGE = "organelle_segmentation"
INPUT_STAGE_DIR = "nuclei_filtered"
OUTPUT_STAGE_DIR = "organelle_features"

# Harmony XML namespace.
NS = {"h": "43B2A954-E3C3-47E1-B392-6635266B0DD3/HarmonyV7"}

# --- Stain / structure design ---------------------------------------------

ANTIGENS = {
    "GM130": "Golgi", "LAMP1": "Lysosome", "G3BP1": "Stress granule",
    "EEA1": "Endosome", "TOMM20": "Mitochondria", "TUBB3": "Microtubules",
    "TGN46": "Golgi", "MAP2": "Microtubules", "TUJ1": "Microtubules",
    "RAB11A": "Endosome", "DAPI": "Nuclei",
}

# Panel -> {Channel_name: Stain}
PANEL_DESIGN = {
    1: {"DAPI": "DAPI", "Alexa 488": "TOMM20", "Alexa 568": "EEA1", "Alexa 647": "LAMP1"},
    2: {"DAPI": "DAPI", "Alexa 488": "RAB11A", "Alexa 568": "GM130", "Alexa 647": "TUJ1"},
}

# --- Per-object feature aggregation ---------------------------------------

# Per-object shape props to measure with regionprops_table.
SHAPE_PROPS = (
    "area", "perimeter", "eccentricity", "solidity",
    "extent", "axis_major_length", "axis_minor_length",
    "equivalent_diameter_area", "orientation",
)

# Per-object intensity props (need intensity_image passed to regionprops).
INTENSITY_PROPS = (
    "intensity_mean", "intensity_max", "intensity_min",
)

# How to summarize each per-object feature across the ROI's objects.
AGG_STATS = {
    "mean":   np.mean,
    "median": np.median,
    "std":    np.std,
    "p10":    lambda a: np.percentile(a, 10),
    "p90":    lambda a: np.percentile(a, 90),
}


def collect_params():
    """Full config snapshot recorded in run_metadata.json for this stage.

    Captures the segmenter set, panel/stain design, and every feature-extraction
    knob so that each (re-)run records exactly which features were computed and
    how. This is what makes feature-iteration self-documenting.
    """
    return {
        "roi_radius": ROI_RADIUS,
        "selected_only": SELECTED_ONLY,
        "segmenters": sorted(SEGMENTERS_BY_STRUCTURE.keys()),
        "panel_design": PANEL_DESIGN,
        "antigens": ANTIGENS,
        "shape_props": list(SHAPE_PROPS),
        "intensity_props": list(INTENSITY_PROPS),
        "agg_stats": list(AGG_STATS.keys()),
    }


# --- Segmenters (unchanged logic; now applied to the whole frame) ---------

def process_golgi_mask(image, intensity_scaling_param=(9, 19), blur_sigma=1,
                       log_sigma=1.6, log_cutoff=0.02, low_thresh_minArea=1200,
                       minArea=10, thin_dist=1):
    m, s = norm.fit(image.ravel())
    stretch_min = max(m - intensity_scaling_param[0] * s, image.min())
    stretch_max = min(m + intensity_scaling_param[1] * s, image.max())
    image_norm = (np.clip(image, stretch_min, stretch_max) - stretch_min) / (stretch_max - stretch_min + 1e-12)
    blurred = filters.gaussian(image_norm, sigma=blur_sigma)
    thresh = threshold_triangle(blurred)
    img_low = remove_small_objects(blurred > thresh, min_size=low_thresh_minArea, connectivity=1)
    img_low = dilation(img_low, footprint=morph_disk(2))
    img_high = np.zeros_like(img_low)
    lab_low, num_obj = label(img_low, return_num=True, connectivity=1)
    for idx in range(num_obj):
        single_obj = lab_low == (idx + 1)
        vals = blurred[single_obj]
        if vals.size == 0 or vals.min() == vals.max():
            continue
        local_otsu = threshold_otsu(vals)
        img_high[np.logical_and(blurred > local_otsu * 0.98, single_obj)] = 1
    skeleton = morphology.medial_axis(img_high > 0)
    dist = distance_transform_edt(skeleton == 0)
    mask = dist > 1 + 1e-5
    thinned = np.logical_xor(img_high > 0, erosion(img_high > 0, morph_disk(thin_dist)))
    skele_mask = np.where(np.logical_and(mask, thinned), 0, img_high)
    log = -1 * (log_sigma ** 2) * gaussian_laplace(blurred, sigma=log_sigma)
    golgi = remove_small_objects(np.logical_or(log > log_cutoff, skele_mask) > 0, min_size=minArea, connectivity=1)
    return binary_fill_holes(golgi)


def process_lysosome_mask(image, intensity_scaling_param=(3, 19), blur_sigma=1,
                          log_params=((5.0, 0.09), (2.5, 0.07), (1.0, 0.01)),
                          vesselness_sigma=(1,), vesselness_cutoff=0.15, min_area=15):
    m, s = norm.fit(image.ravel())
    stretch_min = max(m - intensity_scaling_param[0] * s, image.min())
    stretch_max = min(m + intensity_scaling_param[1] * s, image.max())
    image_norm = (np.clip(image, stretch_min, stretch_max) - stretch_min) / (stretch_max - stretch_min + 1e-12)
    blurred = filters.gaussian(image_norm, sigma=blur_sigma)
    log_mask = np.logical_or.reduce([(-1.0 * sig ** 2 * gaussian_laplace(blurred, sigma=sig)) > cut for sig, cut in log_params])
    vessel_mask = filters.frangi(blurred, sigmas=vesselness_sigma) > vesselness_cutoff
    return remove_small_objects(binary_fill_holes(np.logical_or(log_mask, vessel_mask)), min_size=min_area, connectivity=1)


def process_endosome_mask(image, intensity_scaling_param=(3, 19), blur_sigma=1.0,
                          log_params=((1.0, 0.03),), min_area=3):
    m, s = norm.fit(image.ravel())
    stretch_min = max(m - intensity_scaling_param[0] * s, image.min())
    stretch_max = min(m + intensity_scaling_param[1] * s, image.max())
    image_norm = (np.clip(image, stretch_min, stretch_max) - stretch_min) / (stretch_max - stretch_min + 1e-12)
    blurred = filters.gaussian(image_norm, sigma=blur_sigma)
    log_mask = np.logical_or.reduce([(-1.0 * sig ** 2 * gaussian_laplace(blurred, sigma=sig)) > cut for sig, cut in log_params])
    return remove_small_objects(binary_fill_holes(log_mask), min_size=min_area, connectivity=1)


def process_mitochondria_mask(image, intensity_scaling_param=(3.5, 15), blur_sigma=1.0,
                              log_params=((5.0, 0.09), (2.5, 0.07), (1.0, 0.01)),
                              vesselness_sigmas=(1.5,), vesselness_cutoff=0.16,
                              black_ridges=False, min_area=10, fill_holes=False):
    m, s = norm.fit(image.ravel())
    stretch_min = max(m - intensity_scaling_param[0] * s, image.min())
    stretch_max = min(m + intensity_scaling_param[1] * s, image.max())
    image_norm = (np.clip(image, stretch_min, stretch_max) - stretch_min) / (stretch_max - stretch_min + 1e-12)
    blurred = filters.gaussian(image_norm, sigma=blur_sigma)
    log_mask = np.logical_or.reduce([(-1.0 * sig ** 2 * gaussian_laplace(blurred, sigma=sig)) > cut for sig, cut in log_params])
    vessel_mask = filters.frangi(blurred, sigmas=vesselness_sigmas, black_ridges=black_ridges) > vesselness_cutoff
    combined = np.logical_or(log_mask, vessel_mask)
    if fill_holes:
        combined = binary_fill_holes(combined)
    return remove_small_objects(combined, min_size=min_area, connectivity=1)


SEGMENTERS_BY_STRUCTURE = {
    "Golgi": process_golgi_mask,
    "Lysosome": process_lysosome_mask,
    "Endosome": process_endosome_mask,
    "Mitochondria": process_mitochondria_mask,
}


# --- Samplesheet construction ---------------------------------------------

FILENAME_RE = re.compile(r"r(\d+)c(\d+)f(\d+)p(\d+)-ch(\d+)t(\d+)")


def _parse_filename(name):
    m = FILENAME_RE.match(str(name))
    return [int(g) for g in m.groups()] if m else [None] * 6


def _find_text(element, path, cast=None):
    node = element.find(path, NS)
    if node is None or node.text is None:
        return None
    return cast(node.text) if cast else node.text


def build_samplesheet(experiment_path, panel):
    """Build the channel/stain samplesheet for one experiment folder.

    Mirrors 0_img_metadata.py but also joins the panel stain/structure design.
    """
    import xml.etree.ElementTree as ET

    img_dir = experiment_path / "images"
    exp_xml = next(experiment_path.glob("*.xml"), None)
    index_xml = next((experiment_path / "index").glob("*.xml"), None)
    if exp_xml is None or index_xml is None:
        return None

    exp_root = ET.parse(exp_xml).getroot()
    index_root = ET.parse(index_xml).getroot()
    meas_id = _find_text(exp_root, "h:MeasurementID")
    date = _find_text(exp_root, "h:Date")
    plate_id = _find_text(index_root, ".//h:PlateID")

    # channel metadata
    channels = []
    for map_el in index_root.findall(".//h:Map", NS):
        first = map_el.find("h:Entry", NS)
        if first is not None and first.find("h:ChannelName", NS) is not None:
            for entry in map_el.findall("h:Entry", NS):
                ch_id = entry.attrib.get("ChannelID")
                channels.append({
                    "ChannelID": int(ch_id) if ch_id else None,
                    "Channel_name": _find_text(entry, "h:ChannelName"),
                })
            break
    channel_df = pd.DataFrame(channels).dropna(subset=["ChannelID"])
    channel_df["Measurement_ID"] = meas_id
    channel_df["Measurement_date"] = date
    channel_df["Plate_ID"] = plate_id

    # files
    files = sorted(f for f in img_dir.rglob("*") if f.suffix.lower() == ".tiff")
    if not files:
        return None
    file_df = pd.DataFrame({
        "filepath": [str(f) for f in files],
        "filename": [f.name for f in files],
        "subdirectory": [str(f.parent.relative_to(img_dir)) for f in files],
    })
    file_df[["Row", "Column", "Frame", "Plane", "ChannelID", "Time"]] = (
        file_df["filename"].apply(lambda x: pd.Series(_parse_filename(x)))
    )

    merged = pd.merge(file_df, channel_df, on="ChannelID")

    # panel design: stain + structure
    design = pd.DataFrame(
        [{"Channel_name": ch, "Stain": st} for ch, st in PANEL_DESIGN[panel].items()]
    )
    design["Structure"] = design["Stain"].str.upper().map(
        {k.upper(): v for k, v in ANTIGENS.items()}
    ).fillna("Unknown")
    samplesheet = pd.merge(merged, design, on="Channel_name", how="left")
    samplesheet["Panel"] = panel
    return samplesheet


# --- Whole-frame segmentation + ROI assignment ----------------------------

def assign_objects_to_rois(obj_labels, centroids_yx, radius):
    """Assign each labeled object to an ROI under full-containment rules.

    An object is kept only if ALL of its pixels lie inside a single ROI circle.
    Objects touching or crossing an ROI boundary are dropped. Where an object
    is fully contained in more than one (overlapping) ROI, it goes to the
    nearest nucleus centroid.

    Parameters
    ----------
    obj_labels : 2D int array
        Connected-component labels of the whole-frame organelle mask.
    centroids_yx : (N, 2) float array
        Nucleus centroids (row, col) that define ROI centers.
    radius : float
        ROI radius in pixels.

    Returns
    -------
    dict[int, int]
        Maps object label -> index into centroids_yx (the assigned ROI).
        Objects not assignable are absent from the dict.
    """
    assignment = {}
    if len(centroids_yx) == 0:
        return assignment

    props = regionprops_table(obj_labels, properties=("label", "coords", "centroid"))
    r2 = radius * radius

    for lab, coords, cy, cx in zip(
        props["label"], props["coords"],
        props["centroid-0"], props["centroid-1"],
    ):
        # distance^2 from every ROI center to this object's own centroid,
        # used only to order candidate ROIs (nearest first for tie-break).
        d2_centers = ((centroids_yx[:, 0] - cy) ** 2
                      + (centroids_yx[:, 1] - cx) ** 2)
        order = np.argsort(d2_centers)

        obj_y = coords[:, 0]
        obj_x = coords[:, 1]
        for roi_idx in order:
            ry, rx = centroids_yx[roi_idx]
            # full containment: the object's farthest pixel must be inside.
            max_d2 = np.max((obj_y - ry) ** 2 + (obj_x - rx) ** 2)
            if max_d2 <= r2:
                assignment[int(lab)] = int(roi_idx)
                break  # nearest qualifying ROI wins
    return assignment


# --- Per-ROI feature helpers ----------------------------------------------

def _summarize_props(prop_table, keep_mask, props):
    """Aggregate a per-object regionprops table down to per-ROI stats.

    prop_table : dict from regionprops_table (has a 'label' column)
    keep_mask  : boolean array (aligned to prop_table rows) selecting this
                 ROI's objects
    props      : which property columns to summarize
    Returns a flat {feature_stat: value} dict.
    """
    out = {}
    for p in props:
        vals = np.asarray(prop_table[p], dtype=float)[keep_mask]
        vals = vals[np.isfinite(vals)]
        for stat, fn in AGG_STATS.items():
            out[f"{p}_{stat}"] = float(fn(vals)) if vals.size else np.nan
    return out


def _radial_distances(prop_table, keep_mask, roi_center_yx):
    """Distance from ROI center to each assigned object's centroid."""
    cy = np.asarray(prop_table["centroid-0"], dtype=float)[keep_mask]
    cx = np.asarray(prop_table["centroid-1"], dtype=float)[keep_mask]
    ry, rx = roi_center_yx
    return np.sqrt((cy - ry) ** 2 + (cx - rx) ** 2)


def segment_frame_and_measure(structure, ch_img, seg_fn, centroids_yx,
                              nucleus_ids, radius, site_meta, row):
    """Segment one channel across the whole frame, assign objects to ROIs,
    and return per-nucleus organelle feature rows.

    Per-object shape/intensity and radial distance are summarized across each
    ROI's objects; texture and granularity are computed once per ROI over its
    assigned-object region.
    """
    mask = seg_fn(ch_img)
    obj_labels = label(mask, connectivity=1)
    if obj_labels.max() == 0:
        return []

    assignment = assign_objects_to_rois(obj_labels, centroids_yx, radius)
    if not assignment:
        return []

    # Group object labels by assigned ROI index.
    by_roi = {}
    for obj_lab, roi_idx in assignment.items():
        by_roi.setdefault(roi_idx, []).append(obj_lab)

    # One per-object table for the whole frame; filtered per ROI below.
    # Object identity is preserved exactly as assign_objects_to_rois decided.
    prop_table = regionprops_table(
        obj_labels,
        intensity_image=ch_img,
        properties=("label", "centroid") + SHAPE_PROPS + INTENSITY_PROPS,
    )
    table_labels = np.asarray(prop_table["label"])

    rows = []
    for roi_idx, obj_labs in by_roi.items():
        keep_mask = np.isin(table_labels, obj_labs)

        roi_mask = np.isin(obj_labels, obj_labs)
        vals = ch_img[roi_mask]           # pooled-pixel intensity (kept as-is)
        area_px = int(roi_mask.sum())     # total organelle area in ROI
        count_obj = len(obj_labs)

        # Pooled-pixel intensity summary (unchanged behavior).
        if vals.size:
            int_max = float(vals.max())
            int_sum = float(vals.sum())
            int_mean = float(vals.mean())
            int_median = float(np.median(vals))
            int_std = float(vals.std())
        else:
            int_max = int_sum = 0.0
            int_mean = int_median = int_std = np.nan
        int_cv = (int_std / int_mean) if (not np.isnan(int_mean) and int_mean != 0) else np.nan

        # Per-object shape + intensity, summarized across this ROI's objects.
        shape_summary = _summarize_props(
            prop_table, keep_mask, SHAPE_PROPS + INTENSITY_PROPS
        )

        # Radial distance (per-object -> summarized).
        dists = _radial_distances(prop_table, keep_mask, centroids_yx[roi_idx])
        dists = dists[np.isfinite(dists)]
        dist_summary = {
            f"dist_to_centroid_{s}": (float(fn(dists)) if dists.size else np.nan)
            for s, fn in AGG_STATS.items()
        }
        dist_summary["dist_to_centroid_norm_mean"] = (
            float(np.mean(dists)) / radius if dists.size else np.nan
        )

        rows.append({
            "Measurement_ID": site_meta.get("Measurement_ID"),
            "subdirectory": row.get("subdirectory"),
            "Row": row.get("Row"), "Column": row.get("Column"),
            "Frame": row.get("Frame"), "Plane": row.get("Plane"), "Time": row.get("Time"),
            "DAPI_filename": site_meta.get("dapi_filename"),
            "channel_filename": row["filename"],
            "Structure": structure,
            "Stain": row.get("Stain"),
            "Nucleus_ID": int(nucleus_ids[roi_idx]),
            # ROI-level (count-level) metrics
            "area_px": area_px,
            "organelle_count": count_obj,
            "average_organelle_area": (area_px / count_obj) if count_obj else 0.0,
            "roi_occupancy": area_px / (np.pi * radius ** 2),
            # pooled-pixel intensity (existing columns, kept for continuity)
            "max_f_intensity": int_max,
            "sum_f_intensity": int_sum,
            "mean_f_intensity": int_mean,
            "median_f_intensity": int_median,
            "CoefOfVar_intensity": int_cv,
            # expanded per-object distribution summaries + radial distance
            **shape_summary,
            **dist_summary,
        })
    return rows


# --- Core: one imaging site (one frame, all structures) -------------------

def process_site(site_meta, site_df, nuc_sel, radius=ROI_RADIUS):
    """Segment every organelle channel across one frame and assign to ROIs."""
    site_df = site_df.copy()
    for c in ("Stain", "Structure", "filename"):
        if c in site_df.columns:
            site_df[c] = site_df[c].astype(str).str.strip()

    if nuc_sel.empty:
        return None

    ys = nuc_sel["centroid-0"].astype(float).to_numpy()
    xs = nuc_sel["centroid-1"].astype(float).to_numpy()
    centroids_yx = np.column_stack([ys, xs])
    nucleus_ids = (
        nuc_sel["label"].astype(int).to_numpy()
        if "label" in nuc_sel.columns else np.arange(1, len(xs) + 1)
    )

    todo = sorted(set(site_df["Structure"].unique()) & set(SEGMENTERS_BY_STRUCTURE))
    if not todo:
        return None

    out_rows = []
    for structure in todo:
        seg_fn = SEGMENTERS_BY_STRUCTURE[structure]
        row = site_df.loc[site_df["Structure"] == structure].iloc[0]
        ch_img = tifffile.imread(row["filepath"])
        if ch_img.ndim == 3 and ch_img.shape[0] == 1:
            ch_img = ch_img[0]
        out_rows.extend(
            segment_frame_and_measure(
                structure, ch_img, seg_fn, centroids_yx,
                nucleus_ids, radius, site_meta, row,
            )
        )

    return pd.DataFrame(out_rows) if out_rows else None


# --- Site iteration --------------------------------------------------------

def iter_sites(samplesheet, nuclei_features):
    """Yield (site_meta, site_df, nuc_sel) for each frame that has nuclei.

    samplesheet keys on 'filename'; nuclei_features keys on 'image_name'.
    """
    ss = samplesheet.copy()
    ss["filename"] = ss["filename"].astype(str).str.strip()
    nf = nuclei_features.copy()
    nf["image_name"] = nf["image_name"].astype(str).str.strip()

    site_keys = ["Row", "Column", "Frame", "Plane", "Time"]

    for fname, nuc_grp in nf.groupby("image_name"):
        matched = ss.loc[ss["filename"] == fname]
        if matched.empty:
            continue
        row0 = matched.iloc[0]

        mask = np.ones(len(ss), dtype=bool)
        for k in site_keys:
            if k in ss.columns:
                mask &= ss[k].astype(str).str.strip() == str(row0[k]).strip()
        site_df = ss.loc[mask].copy()
        if site_df.empty:
            continue

        site_meta = {
            "Measurement_ID": site_df["Measurement_ID"].iloc[0] if "Measurement_ID" in site_df.columns else None,
            "dapi_filename": fname,
        }
        yield site_meta, site_df, nuc_grp.copy()


# --- Worker ----------------------------------------------------------------

def _worker(args):
    site_meta, site_df, nuc_sel, radius = args
    try:
        return process_site(site_meta, site_df, nuc_sel, radius=radius)
    except Exception as exc:  # noqa: BLE001
        print(f"[worker] {site_meta.get('dapi_filename')} failed: {exc}")
        return None


# --- Parallel driver for one experiment ------------------------------------

def run_experiment(samplesheet, nuclei_features, radius, n_workers):
    t0 = time.time()
    sites = list(iter_sites(samplesheet, nuclei_features))
    total = len(sites)
    print(f"  built {total} site(s) in {time.time() - t0:.1f}s")

    args_list = [(sm, sd, ns, radius) for sm, sd, ns in sites]
    out, completed = [], 0

    print(f"  starting {n_workers} workers...")
    t_pool = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, a): a[0] for a in args_list}
        first_done = True
        completed_iter = as_completed(futures)
        if _HAS_TQDM:
            completed_iter = tqdm(
                completed_iter, total=total, desc="  segmenting",
                unit="frame", mininterval=5.0,
            )
        for fut in completed_iter:
            sm = futures[fut]
            completed += 1
            if first_done:
                print(f"  first task returned {time.time() - t_pool:.1f}s "
                      f"after pool start (includes worker spawn)")
                first_done = False
            df = fut.result()
            if df is not None and not df.empty:
                out.append(df)
                n_rows = len(df)
            else:
                n_rows = 0
            if not _HAS_TQDM:
                print(f"  [{completed}/{total}] {sm['dapi_filename']} -> {n_rows} rows")

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# --- Scratch staging -------------------------------------------------------

def stage_to_scratch(src_images_dir, mid):
    """rsync one experiment's images to lscratch; return the staged path (or
    None if no scratch is configured). Caller is responsible for cleanup."""
    if SCRATCH_BASE is None:
        return None
    scratch_dir = f"{SCRATCH_BASE}/{mid}"
    subprocess.run(
        ["rsync", "-a", "--info=progress2", f"{src_images_dir}/", f"{scratch_dir}/"],
        check=True,
    )
    return scratch_dir


# --- Per-experiment processing ---------------------------------------------

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


def process_filtered_parquet(parquet_path, src_base, output_dir, version_stamp,
                             panel, radius, n_workers, rec):
    nuclei_features = pd.read_parquet(parquet_path)
    exp_name = experiment_name_from_parquet(parquet_path)

    if SELECTED_ONLY and "selected" in nuclei_features.columns:
        nuclei_features = nuclei_features[nuclei_features["selected"]].copy()

    if nuclei_features.empty:
        print(f"[warning] no selected nuclei in {parquet_path.name}, skipping.")
        return None

    experiment_path = src_base / exp_name
    if not (experiment_path / "images").is_dir():
        print(f"[warning] no images dir for {exp_name} under {src_base}, skipping.")
        return None

    print(f"\n{exp_name}: {len(nuclei_features)} selected nuclei "
          f"across {nuclei_features['image_name'].nunique()} frame(s)")
    rec.log(f"{exp_name}: {len(nuclei_features)} selected nuclei, "
            f"{nuclei_features['image_name'].nunique()} frame(s)")

    t0 = time.time()
    samplesheet = build_samplesheet(experiment_path, panel)
    if samplesheet is None:
        print(f"[warning] could not build samplesheet for {exp_name}, skipping.")
        return None
    print(f"  built samplesheet ({len(samplesheet)} rows) in {time.time() - t0:.1f}s")

    # Stage to scratch and remap filepaths, if scratch is available.
    src_images = experiment_path / "images"
    scratch_dir = None
    try:
        t_stage = time.time()
        scratch_dir = stage_to_scratch(src_images, exp_name)
        if scratch_dir is not None:
            print(f"  staged to scratch in {time.time() - t_stage:.1f}s")
            samplesheet["filepath"] = (
                samplesheet["filepath"].astype(str)
                .str.replace(str(src_images), scratch_dir, regex=False)
            )

        result = run_experiment(samplesheet, nuclei_features, radius, n_workers)
    finally:
        if scratch_dir is not None:
            shutil.rmtree(scratch_dir, ignore_errors=True)
            print(f"  cleared scratch: {scratch_dir}")

    if result.empty:
        print(f"[warning] {exp_name} produced no organelle rows.")
        return None

    # Versioned output: one stamp per 3_ invocation, so re-runs don't overwrite
    # and files from the same run share a stamp.
    out_path = output_dir / f"{exp_name}_organelle_features__{version_stamp}.parquet"
    result.to_parquet(out_path, index=False)

    n_rows = len(result)
    n_nuc = result["Nucleus_ID"].nunique()
    n_feature_cols = result.shape[1]
    print(f"\n--- {exp_name} summary ---")
    print(f"  Organelle feature rows:    {n_rows}")
    print(f"  Nuclei with organelles:    {n_nuc}")
    print(f"  Feature columns:           {n_feature_cols}")
    for structure, grp in result.groupby("Structure"):
        print(f"    {structure:<14} {len(grp)} rows")
    print(f"  Wrote -> {out_path}")
    rec.log(f"{exp_name}: {n_rows} rows, {n_nuc} nuclei, "
            f"{n_feature_cols} cols -> {out_path.name}")

    return {
        "exp_name": exp_name,
        "n_rows": n_rows,
        "n_nuc": n_nuc,
        "n_feature_cols": int(n_feature_cols),
        "output_file": out_path.name,
        "per_structure": {s: int(len(g)) for s, g in result.groupby("Structure")},
    }


# --- Main ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Whole-frame organelle segmentation with ROI assignment. "
                    "Reads filtered nuclei parquets and writes per-experiment "
                    "organelle-feature parquets."
    )
    parser.add_argument(
        "-e", "--experiments", nargs="+", default=None, metavar="NAME",
        help="Specific experiment names (matched against parquet prefixes).",
    )
    parser.add_argument(
        "-b", "--src-base", type=Path, default=DEFAULT_SRC_BASE,
        help=f"Base path of raw experiment folders (default: {DEFAULT_SRC_BASE}).",
    )
    parser.add_argument(
        "-p", "--panel", type=int, default=DEFAULT_PANEL, choices=[1, 2],
        help=f"Panel design to use (default: {DEFAULT_PANEL}).",
    )
    parser.add_argument(
        "-r", "--radius", type=int, default=ROI_RADIUS,
        help=f"ROI radius in px (default: {ROI_RADIUS}).",
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=N_WORKERS,
        help=f"Worker processes (default: {N_WORKERS}, from SLURM allocation).",
    )
    parser.add_argument(
        "--version-stamp", default=None, metavar="YYYYMMDD_HHMMSS",
        help="Shared version stamp for output filenames. If omitted, one is "
             "generated at startup. Pass the SAME stamp to every task of an "
             "array so all its outputs share one version.",
    )
    parser.add_argument(
        "--merge-only", action="store_true",
        help="Do not segment. Fold this run's per-experiment metadata/log "
             "shards into run_metadata.json / run.log, then exit. Run this "
             "once after an array of per-experiment tasks completes.",
    )
    # --output-root + --run-id (required here: reuse the run minted by 0_).
    ru.add_run_args(parser, mints_run_id=False)
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve the existing run dir (errors clearly if the run ID is wrong).
    run_dir = ru.resolve_run_dir(args.output_root, args.run_id)

    # --merge-only: consolidate array shards and exit. No segmentation.
    if args.merge_only:
        print(f"\n=== RUN ID: {args.run_id} | merge-only ===")
        n = ru.merge_shard_records(run_dir)
        print(f"Merged {n} shard record(s) into the run metadata.")
        return

    input_dir = ru.stage_dir(run_dir, INPUT_STAGE_DIR)
    output_dir = ru.stage_dir(run_dir, OUTPUT_STAGE_DIR)

    # One version stamp per invocation (or shared across an array via the flag)
    # -> versioned, non-overwriting outputs.
    version_stamp = args.version_stamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sharded (array-safe) recording when exactly one experiment is requested;
    # that's how the job array invokes this (-e "$EXP"). Zero or multiple
    # experiments -> normal shared writer (a sequential single-process run).
    shard = (
        args.experiments[0]
        if args.experiments and len(args.experiments) == 1
        else None
    )

    rec = ru.StageRecorder(
        run_dir, stage=STAGE, run_id=args.run_id,
        params=collect_params(),
        inputs={
            "input_dir": str(input_dir),
            "src_base": str(args.src_base),
            "panel": args.panel,
            "radius": args.radius,
            "workers": args.workers,
            "version_stamp": version_stamp,
            "experiments_requested": args.experiments or "ALL",
        },
        shard=shard,
    )
    print(f"\n=== RUN ID: {args.run_id} ===")
    print(f"=== run dir: {run_dir} ===")
    print(f"=== version stamp: {version_stamp} ===")
    if shard:
        print(f"=== sharded record: {shard} ===")
    print()
    rec.log(f"version stamp {version_stamp}")

    parquets = discover_filtered_parquets(input_dir, args.experiments)
    print(f"Found {len(parquets)} filtered parquet(s) to process.")
    print(f"Workers: {args.workers} | Panel: {args.panel} | ROI radius: {args.radius}")
    rec.log(f"found {len(parquets)} filtered parquet(s); "
            f"workers={args.workers} panel={args.panel} radius={args.radius}")

    all_stats = []
    for parquet_path in parquets:
        stats = process_filtered_parquet(
            parquet_path, args.src_base, output_dir, version_stamp,
            args.panel, args.radius, args.workers, rec,
        )
        if stats is not None:
            all_stats.append(stats)

    total_rows = sum(s["n_rows"] for s in all_stats)
    total_nuc = sum(s["n_nuc"] for s in all_stats)

    if all_stats:
        print("\n" + "=" * 40)
        print("OVERALL SUMMARY")
        print(f"  Experiments processed:     {len(all_stats)}")
        print(f"  Organelle feature rows:    {total_rows}")
        print(f"  Total nuclei measured:     {total_nuc}")
        print("=" * 40)

    rec.finish(
        outputs={
            "organelle_features_dir": str(output_dir),
            "version_stamp": version_stamp,
            "output_files": [s["output_file"] for s in all_stats],
        },
        summary={
            "experiments_processed": len(all_stats),
            "organelle_feature_rows": total_rows,
            "nuclei_measured": total_nuc,
            "per_experiment": all_stats,
        },
    )

    print(f"\n=== RUN ID: {args.run_id} | version: {version_stamp} ===")


if __name__ == "__main__":
    main()