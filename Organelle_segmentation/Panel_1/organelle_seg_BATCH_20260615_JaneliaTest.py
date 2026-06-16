import os
import re
import uuid
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from PIL import Image

from skimage.measure import label, regionprops_table
from scipy.stats import norm
from scipy.ndimage import distance_transform_edt, gaussian_laplace, binary_fill_holes
from skimage import filters, morphology
from skimage.filters import threshold_triangle, threshold_otsu
from skimage.morphology import remove_small_objects, erosion, dilation
from skimage.morphology import disk as morph_disk
import xml.etree.ElementTree as ET


# =========================
# EXPERIMENT METADATA
# =========================

BASE_PATH = Path("/data/CARDPB2/iNDI/Production/AbPanel1")
NS = {'h': '43B2A954-E3C3-47E1-B392-6635266B0DD3/HarmonyV7'}

pseudocolor_map = {
    "DAPI":        "blue",
    "Brightfield": "gray",
    "Alexa 488":   "green",
    "Alexa 568":   "red",
    "Alexa 647":   "magenta",
}

antigens = {
    "GM130":  "Golgi",
    "LAMP1":  "Lysosome",
    "G3BP1":  "Stress granule",
    "EEA1":   "Endosome",
    "TOMM20": "Mitochondria",
    "TUBB3":  "Microtubules",
    "TGN46":  "Golgi",
    "MAP2":   "Microtubules",
    "TUJ1":   "Microtubules",
    "DAPI":   "Nuclei",
}

experimental_design = pd.DataFrame({
    "Panel":     [1,        2],
    "DAPI":      ["DAPI",   "DAPI"],
    "Alexa 488": ["TOMM20", "RAB11A"],
    "Alexa 568": ["EEA1",   "GM130"],
    "Alexa 647": ["LAMP1",  "TUJ1"],
}).melt(
    id_vars=["Panel"],
    value_vars=["DAPI", "Alexa 488", "Alexa 568", "Alexa 647"],
    var_name="Channel_name",
    value_name="Stain",
)
antigen_keys = list(antigens.keys())
_pattern = r"\b(" + "|".join(re.escape(k) for k in antigen_keys) + r")\b"
_matched = experimental_design["Stain"].str.extract(_pattern, flags=re.IGNORECASE)[0]
experimental_design["Structure"] = (
    _matched.str.upper()
    .map({k.upper(): v for k, v in antigens.items()})
    .fillna("Unknown")
)


# =========================
# SAMPLESHEET BUILDER
# =========================

def _parse_filename(name):
    match = re.match(r"r(\d+)c(\d+)f(\d+)p(\d+)-ch(\d+)t(\d+)", name)
    return [int(g) for g in match.groups()] if match else [None] * 6


def build_samplesheet(measurement_id: str, panel: int = 1) -> pd.DataFrame:
    exp_path  = BASE_PATH / measurement_id
    img_dir   = exp_path / "images"
    exp_xml   = next(exp_path.glob("*.xml"), None)
    index_xml = next((exp_path / "index").glob("*.xml"), None)

    exp_root   = ET.parse(exp_xml).getroot()
    index_root = ET.parse(index_xml).getroot()

    meas_id  = exp_root.find('h:MeasurementID', NS).text
    date     = exp_root.find('h:Date', NS).text
    plate_id = index_root.find('.//h:PlateID', NS).text
    x_res    = float(index_root.find('.//h:ImageResolutionX', NS).text) * 1e6
    y_res    = float(index_root.find('.//h:ImageResolutionY', NS).text) * 1e6

    channels = []
    for map_el in index_root.findall(".//h:Map", NS):
        first = map_el.find("h:Entry", NS)
        if first is not None and first.find("h:ChannelName", NS) is not None:
            for entry in map_el.findall("h:Entry", NS):
                ch_id = entry.attrib.get("ChannelID")
                channels.append({
                    "ChannelID":     int(ch_id) if ch_id else None,
                    "Channel_name":  entry.find("h:ChannelName", NS).text,
                    "Excitation_nm": entry.findtext("h:MainExcitationWavelength", default=None, namespaces=NS),
                    "Emission_nm":   entry.findtext("h:MainEmissionWavelength",   default=None, namespaces=NS),
                })
            break

    channel_df = pd.DataFrame(channels).sort_values("ChannelID").reset_index(drop=True)
    channel_df["Measurement_ID"]   = meas_id
    channel_df["Measurement_date"] = date
    channel_df["Plate_ID"]         = plate_id
    channel_df["res_x"]            = x_res
    channel_df["res_y"]            = y_res

    files   = sorted(f for f in img_dir.rglob("*") if f.suffix.lower() == ".tiff")
    file_df = pd.DataFrame({
        "filepath":     files,
        "filename":     [f.name for f in files],
        "subdirectory": [str(f.parent.relative_to(img_dir)) for f in files],
    })
    file_df[["Row", "Column", "Frame", "Plane", "ChannelID", "Time"]] = (
        file_df["filename"].apply(lambda x: pd.Series(_parse_filename(x)))
    )

    merged     = pd.merge(file_df, channel_df, on="ChannelID")
    merged["Panel"] = panel
    design     = experimental_design[experimental_design["Panel"] == panel]
    samplesheet = pd.merge(merged, design, on=["Channel_name", "Panel"])
    return samplesheet


# =========================
# SEGMENTERS
# =========================

def process_golgi_mask(image, intensity_scaling_param=[9, 19], blur_sigma=1,
                       log_sigma=1.6, log_cutoff=0.02, low_thresh_minArea=1200,
                       minArea=10, thin_dist=1):
    m, s = norm.fit(image)
    stretch_min = max(m - intensity_scaling_param[0] * s, image.min())
    stretch_max = min(m + intensity_scaling_param[1] * s, image.max())
    image_norm = (np.clip(image, stretch_min, stretch_max) - stretch_min) / (stretch_max - stretch_min + 1e-12)
    blurred = filters.gaussian(image_norm, sigma=blur_sigma)
    thresh = threshold_triangle(blurred)
    img_low_thresh = remove_small_objects(blurred > thresh, min_size=low_thresh_minArea, connectivity=1)
    img_low_thresh = dilation(img_low_thresh, footprint=morph_disk(2))
    img_high_thresh = np.zeros_like(img_low_thresh)
    lab_low, num_obj = label(img_low_thresh, return_num=True, connectivity=1)
    for idx in range(num_obj):
        single_obj = lab_low == (idx + 1)
        local_otsu = threshold_otsu(blurred[single_obj > 0])
        img_high_thresh[np.logical_and(blurred > local_otsu * 0.98, single_obj)] = 1
    skeleton = morphology.medial_axis(img_high_thresh > 0)
    dist = distance_transform_edt(skeleton == 0)
    mask = dist > 1 + 1e-5
    thinned = np.logical_xor(img_high_thresh > 0, erosion(img_high_thresh > 0, morph_disk(thin_dist)))
    skele_mask = np.where(np.logical_and(mask, thinned), 0, img_high_thresh)
    log = -1 * (log_sigma**2) * gaussian_laplace(blurred, sigma=log_sigma)
    golgi_mask = remove_small_objects(np.logical_or(log > log_cutoff, skele_mask) > 0, min_size=minArea, connectivity=1)
    return binary_fill_holes(golgi_mask)


def process_lysosome_mask(image, intensity_scaling_param=[3, 19], blur_sigma=1,
                          log_params=((5.0, 0.09), (2.5, 0.07), (1.0, 0.01)),
                          vesselness_sigma=[1], vesselness_cutoff=0.15, min_area=15):
    m, s = norm.fit(image.ravel())
    stretch_min = max(m - intensity_scaling_param[0] * s, image.min())
    stretch_max = min(m + intensity_scaling_param[1] * s, image.max())
    image_norm = (np.clip(image, stretch_min, stretch_max) - stretch_min) / (stretch_max - stretch_min + 1e-12)
    blurred = filters.gaussian(image_norm, sigma=blur_sigma)
    log_mask = np.logical_or.reduce([(-1.0 * sig**2 * gaussian_laplace(blurred, sigma=sig)) > cut for sig, cut in log_params])
    vessel_mask = filters.frangi(blurred, sigmas=vesselness_sigma) > vesselness_cutoff
    return remove_small_objects(binary_fill_holes(np.logical_or(log_mask, vessel_mask)), min_size=min_area, connectivity=1)


def process_endosome_mask(image, intensity_scaling_param=[3, 19], blur_sigma=1.0,
                          log_params=((1.0, 0.03),), min_area=3):
    m, s = norm.fit(image.ravel())
    stretch_min = max(m - intensity_scaling_param[0] * s, image.min())
    stretch_max = min(m + intensity_scaling_param[1] * s, image.max())
    image_norm = (np.clip(image, stretch_min, stretch_max) - stretch_min) / (stretch_max - stretch_min + 1e-12)
    blurred = filters.gaussian(image_norm, sigma=blur_sigma)
    log_mask = np.logical_or.reduce([(-1.0 * sig**2 * gaussian_laplace(blurred, sigma=sig)) > cut for sig, cut in log_params])
    return remove_small_objects(binary_fill_holes(log_mask), min_size=min_area, connectivity=1)


def process_mitochondria_mask(image, intensity_scaling_param=[3.5, 15], blur_sigma=1.0,
                              log_params=((5.0, 0.09), (2.5, 0.07), (1.0, 0.01)),
                              vesselness_sigmas=(1.5,), vesselness_cutoff=0.16,
                              black_ridges=False, min_area=10, fill_holes=False):
    m, s = norm.fit(image.ravel())
    stretch_min = max(m - intensity_scaling_param[0] * s, image.min())
    stretch_max = min(m + intensity_scaling_param[1] * s, image.max())
    image_norm = (np.clip(image, stretch_min, stretch_max) - stretch_min) / (stretch_max - stretch_min + 1e-12)
    blurred = filters.gaussian(image_norm, sigma=blur_sigma)
    log_mask = np.logical_or.reduce([(-1.0 * sig**2 * gaussian_laplace(blurred, sigma=sig)) > cut for sig, cut in log_params])
    vessel_mask = filters.frangi(blurred, sigmas=vesselness_sigmas, black_ridges=black_ridges) > vesselness_cutoff
    combined = np.logical_or(log_mask, vessel_mask)
    if fill_holes:
        combined = binary_fill_holes(combined)
    return remove_small_objects(combined, min_size=min_area, connectivity=1)


# =========================
# CONFIG
# =========================

SEGMENTERS_BY_STRUCTURE = {
    "Golgi":        process_golgi_mask,
    "Lysosome":     process_lysosome_mask,
    "Endosome":     process_endosome_mask,
    "Mitochondria": process_mitochondria_mask,
}

ROI_RADIUS = 120
N_WORKERS  = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

# Timeout (seconds) per site — prevents a single hung worker stalling everything
SITE_TIMEOUT = 600  # 10 min

REGIONPROPS_PROPERTIES = [
    "label", "area", "area_bbox", "area_convex", "area_filled",
    "axis_major_length", "axis_minor_length", "bbox",
    "centroid", "centroid_local", "centroid_weighted", "centroid_weighted_local",
    "eccentricity", "equivalent_diameter_area", "euler_number", "extent",
    "feret_diameter_max", "inertia_tensor", "inertia_tensor_eigvals",
    "intensity_max", "intensity_mean", "intensity_min", "intensity_std",
    "moments", "moments_central", "moments_hu", "moments_normalized",
    "moments_weighted", "moments_weighted_central", "moments_weighted_hu",
    "moments_weighted_normalized", "num_pixels", "orientation",
    "perimeter", "perimeter_crofton", "solidity",
]

SHOW_QC                 = False   # disabled for batch runs; enable for interactive QC
QC_MAX_SITES            = 5
QC_MAX_TILES_PER_STRUCT = 8
QC_OUTPUT_DIR           = "./qc_figures"


# =========================
# HELPERS
# =========================

_site_re = re.compile(
    r"r(?P<Row>\d+)c(?P<Column>\d+)f(?P<Frame>\d+)p(?P<Plane>\d+)"
    r"-ch(?P<ChannelID>\d+)t(?P<Time>\d+)", re.IGNORECASE)

def parse_site_keys_from_filename(fname: str) -> dict:
    m = _site_re.search(str(fname))
    if not m:
        return {}
    d = m.groupdict()
    out = {k: d[k] for k in ("Row", "Column", "Frame", "Plane", "Time") if d.get(k)}
    out["subdirectory"] = f"r{out['Row'].zfill(2)}c{out['Column'].zfill(2)}"
    return out

def imread_plane(path):
    """Always reads eagerly — no dask/lazy overhead inside workers."""
    arr = tifffile.imread(str(path))
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr

def bbox_from_center(y, x, r, H, W):
    yi, xi = int(round(y)), int(round(x))
    return max(yi - r, 0), min(yi + r + 1, H), max(xi - r, 0), min(xi + r + 1, W)

def get_row(site_df: pd.DataFrame, **filters):
    sel = site_df.copy()
    for k, v in filters.items():
        sel = sel[sel[k] == v]
    return None if sel.empty else sel.iloc[0]

def _save_roi_mask(seg_tile, structure, dapi_fname, nucleus_id, out_dir):
    dapi_s = str(dapi_fname).replace("/", "_")
    fname  = f"{structure}__{dapi_s}__NucID={nucleus_id}.png"
    Image.fromarray((seg_tile.astype(np.uint8) * 255)).save(os.path.join(out_dir, fname))

def _write_site_parquet(summary_df, instance_df, base_dir, mid, dapi_name):
    os.makedirs(base_dir, exist_ok=True)
    uid    = uuid.uuid4().hex[:8]
    mid_s  = str(mid).replace("/", "_")
    dapi_s = str(dapi_name).replace("/", "_")
    summary_df.to_parquet(
        os.path.join(base_dir, f"summary__MID={mid_s}__DAPI={dapi_s}__{uid}.parquet"), index=False)
    instance_df.to_parquet(
        os.path.join(base_dir, f"instance__MID={mid_s}__DAPI={dapi_s}__{uid}.parquet"), index=False)

def _regionprops_for_tile(seg_tile, tile, y0, x0):
    obj_labels = label(seg_tile, connectivity=1)
    if obj_labels.max() == 0:
        return pd.DataFrame()
    rpt = regionprops_table(obj_labels, intensity_image=tile, properties=REGIONPROPS_PROPERTIES)
    df  = pd.DataFrame(rpt)
    for col in df.columns:
        if col.startswith("centroid-0") or col.startswith("bbox-0") or col.startswith("bbox-2"):
            df[col] += y0
        elif col.startswith("centroid-1") or col.startswith("bbox-1") or col.startswith("bbox-3"):
            df[col] += x0
    return df


# =========================
# CORE: process_site
# =========================

def process_site(site_meta, site_df, nuclei_features, segmenters_by_structure,
                 r=ROI_RADIUS, mask_output_dir=None):
    site_df = site_df.copy()
    for c in ("Stain", "Structure", "filename"):
        if c in site_df.columns:
            site_df[c] = site_df[c].astype(str).str.strip()
    site_df["Stain"] = site_df["Stain"].str.upper()

    dapi_row = get_row(site_df, Stain="DAPI")
    if dapi_row is None:
        return None, None

    dapi_fname = str(dapi_row["filename"]).strip()
    nuc_img    = imread_plane(dapi_row["filepath"])
    H, W       = nuc_img.shape[:2]

    nf = nuclei_features.copy()
    nf["filename"] = nf["filename"].astype(str).str.strip()

    props_fname = str(site_meta.get("props_filename", site_meta.get("dapi_filename", ""))).strip()
    nf_sel = nf[nf["filename"] == props_fname]
    if nf_sel.empty:
        nf_sel = nf[nf["filename"] == dapi_fname]
    if nf_sel.empty:
        return None, None

    ys     = nf_sel["centroid-0"].astype(float).to_numpy()
    xs     = nf_sel["centroid-1"].astype(float).to_numpy()
    labels = nf_sel["label"].astype(int).to_numpy() if "label" in nf_sel.columns else np.arange(1, len(xs) + 1)

    canon    = {k.lower(): k for k in segmenters_by_structure}
    site_df["Structure"] = site_df["Structure"].str.lower().map(canon).fillna(site_df["Structure"])
    todo     = [s for s in sorted(set(site_df["Structure"].unique()) & set(segmenters_by_structure))
                if callable(segmenters_by_structure.get(s))]
    if not todo:
        return None, None

    summary_rows, instance_frames = [], []

    _site_meta_cols = {
        "Measurement_ID": site_meta.get("Measurement_ID"),
        "DAPI_filename":  dapi_fname,
    }

    for structure in todo:
        seg_fn = segmenters_by_structure[structure]
        row    = site_df.loc[site_df["Structure"] == structure].iloc[0]
        ch_img = imread_plane(row["filepath"])
        if ch_img.shape[:2] != (H, W):
            raise ValueError(f"Shape mismatch @ {structure}: {ch_img.shape} vs {(H, W)}")

        _struct_meta = {
            **_site_meta_cols,
            "subdirectory":     row.get("subdirectory"),
            "Row":              row.get("Row"),
            "Column":           row.get("Column"),
            "Frame":            row.get("Frame"),
            "Plane":            row.get("Plane"),
            "Time":             row.get("Time"),
            "channel_filename": row["filename"],
            "Structure":        structure,
            "Stain":            row.get("Stain"),
        }

        for lab, y, x in zip(labels, ys, xs):
            y0, y1, x0, x1 = bbox_from_center(y, x, r, H, W)
            tile   = ch_img[y0:y1, x0:x1]
            yy, xx = np.ogrid[y0:y1, x0:x1]
            roi      = (yy - y)**2 + (xx - x)**2 <= r**2
            seg_tile = seg_fn(np.where(roi, tile, 0)) & roi

            if mask_output_dir:
                _save_roi_mask(seg_tile, structure, dapi_fname, int(lab), mask_output_dir)

            obj_labels_arr = label(seg_tile, connectivity=1)
            count_obj  = int(obj_labels_arr.max())
            vals       = tile[seg_tile]
            area_px    = int(seg_tile.sum())
            roi_area   = int(roi.sum())
            coverage_frac = (area_px / roi_area) if roi_area else 0.0
            area_px_mean  = (area_px / count_obj) if count_obj else 0.0

            if vals.size:
                int_max    = float(vals.max())
                int_sum    = float(vals.sum())
                int_mean   = float(vals.mean())
                int_median = float(np.median(vals))
                int_std    = float(vals.std())
            else:
                int_max = int_sum = 0.0
                int_mean = int_median = int_std = np.nan

            int_cv = (int_std / int_mean) if (not np.isnan(int_mean) and int_mean != 0) else np.nan

            rad      = np.sqrt((yy - y)**2 + (xx - x)**2)
            rad_norm = np.zeros_like(rad, dtype=float)
            rad_norm[roi] = rad[roi] / float(r)
            rad_obj  = rad_norm[seg_tile]
            rad_n    = rad_obj.size

            if rad_n:
                radial_mean         = float(rad_obj.mean())
                radial_p25          = float(np.percentile(rad_obj, 25))
                radial_p50          = float(np.percentile(rad_obj, 50))
                radial_p75          = float(np.percentile(rad_obj, 75))
                inner_frac          = float((rad_obj < 0.33).sum()) / rad_n
                mid_frac            = float(((rad_obj >= 0.33) & (rad_obj < 0.66)).sum()) / rad_n
                outer_frac          = float((rad_obj >= 0.66).sum()) / rad_n
                boundary_touch_frac = float(((r - rad[seg_tile]) <= 5).sum()) / rad_n
            else:
                radial_mean = radial_p25 = radial_p50 = radial_p75 = np.nan
                inner_frac  = mid_frac = outer_frac = boundary_touch_frac = 0.0

            summary_rows.append({
                **_struct_meta,
                "Nucleus_ID": int(lab),
                "area_px": area_px, "coverage_frac": coverage_frac,
                "organelle_count": count_obj, "average_organelle_area": area_px_mean,
                "max_f_intensity": int_max, "sum_f_intensity": int_sum,
                "mean_f_intensity": int_mean, "median_f_intensity": int_median,
                "CoefOfVar_intensity": int_cv,
                "radial_mean": radial_mean,
                "radial_p25": radial_p25, "radial_p50": radial_p50, "radial_p75": radial_p75,
                "inner_frac": inner_frac, "mid_frac": mid_frac, "outer_frac": outer_frac,
                "boundary_touch_frac": boundary_touch_frac,
            })

            inst_df = _regionprops_for_tile(seg_tile, tile, y0, x0)
            if not inst_df.empty:
                inst_df.insert(0, "Nucleus_ID", int(lab))
                for k, v in reversed(list(_struct_meta.items())):
                    inst_df.insert(0, k, v)
                instance_frames.append(inst_df)

    summary_df  = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    instance_df = pd.concat(instance_frames, ignore_index=True) if instance_frames else pd.DataFrame()
    return summary_df, instance_df


# =========================
# SITE ITERATION
# =========================

def iter_sites_single_panel(samplesheet, nuclei_features):
    ss = samplesheet.copy()
    ss["filename"] = ss["filename"].astype(str).str.strip()
    nf = nuclei_features.copy()
    nf["image_name"] = nf["image_name"].astype(str).str.strip()

    site_keys = ["Row", "Column", "Frame", "Plane", "Time"]

    for fname in nf["image_name"].unique():
        matched = ss.loc[ss["filename"] == fname]
        if matched.empty:
            keys = parse_site_keys_from_filename(fname)
            if not keys:
                continue
            mask = np.ones(len(ss), dtype=bool)
            for k, v in keys.items():
                if k in ss.columns:
                    mask &= ss[k].astype(str).str.strip() == str(v).strip()
            matched = ss.loc[mask]
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
            "dapi_filename":  fname,
            "props_filename": fname,
        }
        yield site_meta, site_df


# =========================
# WORKER
# forkserver start method avoids inheriting parent locks from matplotlib/scipy/dask
# =========================

def _process_site_worker(args):
    site_meta, site_df, nuclei_features, r, mask_output_dir, segmenters = args

    # nuclei_features arrives already subsetted and renamed
    try:
        return process_site(
            site_meta, site_df, nuclei_features, segmenters,
            r=r, mask_output_dir=mask_output_dir,
        )
    except Exception as e:
        print(f"[worker] {site_meta.get('dapi_filename')} failed: {e}", flush=True)
        return None, None


# =========================
# PARALLEL run_all
# =========================

def run_all_parallel(
    samplesheet, nuclei_features, segmenters_by_structure,
    r=ROI_RADIUS, n_workers=N_WORKERS, output_dir=None,
    site_timeout=SITE_TIMEOUT,
):
    samplesheet     = samplesheet.copy()
    nuclei_features = nuclei_features.copy()
    samplesheet["filename"]       = samplesheet["filename"].astype(str).str.strip()
    nuclei_features["image_name"] = nuclei_features["image_name"].astype(str).str.strip()

    all_sites = list(iter_sites_single_panel(samplesheet, nuclei_features))
    total     = len(all_sites)
    print(f"Found {total} sites — dispatching to {n_workers} workers", flush=True)

    args_list = []
    for site_meta, site_df in all_sites:
        fname   = site_meta["props_filename"]
        nf_site = nuclei_features[nuclei_features["image_name"] == fname].copy()
        if nf_site.empty:
            keys = parse_site_keys_from_filename(fname)
            mask = np.ones(len(nuclei_features), dtype=bool)
            for k, v in keys.items():
                if k in nuclei_features.columns:
                    mask &= nuclei_features[k].astype(str).str.strip() == str(v).strip()
            nf_site = nuclei_features[mask].copy()

        # Rename here so workers don't need to
        nf_site = nf_site.rename(columns={"image_name": "filename"})

        args_list.append((
            site_meta, site_df, nf_site, r, output_dir, segmenters_by_structure
        ))

    summary_parts, instance_parts = [], []
    completed = 0

    # forkserver: fresh interpreter per worker — no inherited locks from dask/matplotlib/scipy
    ctx = mp.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {pool.submit(_process_site_worker, args): args[0] for args in args_list}

        for fut in as_completed(futures):
            site_meta  = futures[fut]
            completed += 1
            try:
                summary_df, instance_df = fut.result(timeout=site_timeout)
                has_summary  = summary_df  is not None and not summary_df.empty
                has_instance = instance_df is not None and not instance_df.empty

                if has_summary or has_instance:
                    if output_dir:
                        _write_site_parquet(
                            summary_df  if has_summary  else pd.DataFrame(),
                            instance_df if has_instance else pd.DataFrame(),
                            output_dir,
                            site_meta.get("Measurement_ID", "unknown"),
                            site_meta["dapi_filename"],
                        )
                    else:
                        if has_summary:  summary_parts.append(summary_df)
                        if has_instance: instance_parts.append(instance_df)

                    print(f"[{completed}/{total}] ✓ {site_meta['dapi_filename']} "
                          f"→ {len(summary_df) if has_summary else 0} summary, "
                          f"{len(instance_df) if has_instance else 0} instance rows", flush=True)
                else:
                    print(f"[{completed}/{total}] – {site_meta['dapi_filename']} → no data", flush=True)

            except TimeoutError:
                print(f"[{completed}/{total}] ✗ {site_meta['dapi_filename']} → timed out after {site_timeout}s", flush=True)
            except Exception as e:
                print(f"[{completed}/{total}] ✗ {site_meta['dapi_filename']} → {e}", flush=True)

    print("\nAll sites processed.", flush=True)

    if output_dir:
        s_parts = [pd.read_parquet(os.path.join(output_dir, f))
                   for f in os.listdir(output_dir) if f.startswith("summary__")  and f.endswith(".parquet")]
        i_parts = [pd.read_parquet(os.path.join(output_dir, f))
                   for f in os.listdir(output_dir) if f.startswith("instance__") and f.endswith(".parquet")]
        return (pd.concat(s_parts, ignore_index=True) if s_parts else pd.DataFrame(),
                pd.concat(i_parts, ignore_index=True) if i_parts else pd.DataFrame())

    return (pd.concat(summary_parts,  ignore_index=True) if summary_parts  else pd.DataFrame(),
            pd.concat(instance_parts, ignore_index=True) if instance_parts else pd.DataFrame())


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    MEASUREMENT_IDS = [
        '4405a3b2-6b88-49b1-91f3-992e09ccbd16',
    ]

    NUCLEI_FEATURES_DIR = Path('/data/CARDPB2/iNDI/Production/nucleus_segmentation_result')
    SCRATCH_BASE        = f"/lscratch/{os.environ['SLURM_JOB_ID']}"
    SRC_BASE            = "/data/CARDPB2/iNDI/Production/AbPanel1"
    OUTPUT_DIR          = "/data/CARDPB2/iNDI/JaneliaTest/organelle_features"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build samplesheets
    all_samplesheets = {mid: build_samplesheet(mid) for mid in MEASUREMENT_IDS}

    # Load nuclei features
    all_nuclei_features = {}
    for f in NUCLEI_FEATURES_DIR.iterdir():
        match = next((mid for mid in MEASUREMENT_IDS if mid in f.name), None)
        if match:
            nf = pd.read_csv(f)
            nf["experiment_name"] = match
            all_nuclei_features[match] = nf

    all_summary_dfs, all_instance_dfs = [], []

    for mid in MEASUREMENT_IDS:
        print(f"\n{'='*60}\nProcessing: {mid}\n{'='*60}", flush=True)

        if mid not in all_samplesheets or mid not in all_nuclei_features:
            print(f"  Skipping {mid} — missing samplesheet or nuclei features", flush=True)
            continue

        ss = all_samplesheets[mid].copy()
        nf = all_nuclei_features[mid].copy()

        scratch_dir = f"{SCRATCH_BASE}/{mid}"
        src_dir     = f"{SRC_BASE}/{mid}/images"

        try:
            subprocess.run(
                ["rsync", "-a", "--info=progress2", f"{src_dir}/", f"{scratch_dir}/"],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"  rsync failed for {mid}: {e} — skipping", flush=True)
            continue

        ss["filepath"] = ss["filepath"].astype(str).str.replace(src_dir, scratch_dir, regex=False)

        try:
            summary_df, instance_df = run_all_parallel(
                ss, nf, SEGMENTERS_BY_STRUCTURE,
                r=ROI_RADIUS,
                n_workers=N_WORKERS,
                output_dir=OUTPUT_DIR,
            )
            print(f"  Done: {len(summary_df)} summary rows, {len(instance_df)} instance rows", flush=True)
            if not summary_df.empty:  all_summary_dfs.append(summary_df)
            if not instance_df.empty: all_instance_dfs.append(instance_df)

        except Exception as e:
            print(f"  run_all_parallel failed for {mid}: {e}", flush=True)

        finally:
            if os.path.exists(scratch_dir):
                result = subprocess.run(["rm", "-rf", scratch_dir])
                if result.returncode == 0:
                    print(f"  Cleared scratch: {scratch_dir}", flush=True)
                else:
                    print(f"  Warning: scratch cleanup may have failed for {scratch_dir}", flush=True)

    organelle_features = pd.concat(all_summary_dfs,  ignore_index=True) if all_summary_dfs  else pd.DataFrame()
    instance_features  = pd.concat(all_instance_dfs, ignore_index=True) if all_instance_dfs else pd.DataFrame()

    print(f"\nFinal summary shape:  {organelle_features.shape}", flush=True)
    print(f"Final instance shape: {instance_features.shape}", flush=True)