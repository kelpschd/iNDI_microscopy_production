# %% [markdown]
# ## Samplesheet setup

# %%
import re
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# =========================
# EXPERIMENT METADATA
# =========================

BASE_PATH = Path("/data/CARDPB2/iNDI/Production/AbPanel1")
NS = {'h': '43B2A954-E3C3-47E1-B392-6635266B0DD3/HarmonyV7'}

pseudocolor_map = {
    "DAPI":       "blue",
    "Brightfield": "gray",
    "Alexa 488":  "green",
    "Alexa 568":  "red",
    "Alexa 647":  "magenta",
}

mpl_colormaps = {
    "blue":    LinearSegmentedColormap.from_list("black_blue",    [(0,0,0), (0,0,1)]),
    "green":   LinearSegmentedColormap.from_list("black_green",   [(0,0,0), (0,1,0)]),
    "red":     LinearSegmentedColormap.from_list("black_red",     [(0,0,0), (1,0,0)]),
    "magenta": LinearSegmentedColormap.from_list("black_magenta", [(0,0,0), (1,0,1)]),
    "gray":    LinearSegmentedColormap.from_list("black_gray",    [(0,0,0), (1,1,1)]),
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
    "Panel":      [1,       2],
    "DAPI":       ["DAPI",  "DAPI"],
    "Alexa 488":  ["TOMM20","RAB11A"],
    "Alexa 568":  ["EEA1",  "GM130"],
    "Alexa 647":  ["LAMP1", "TUJ1"],
}).melt(
    id_vars=["Panel"],
    value_vars=["DAPI", "Alexa 488", "Alexa 568", "Alexa 647"],
    var_name="Channel_name",
    value_name="Stain",
)

# Map stain → structure
antigen_keys = list(antigens.keys())
pattern = r"\b(" + "|".join(re.escape(k) for k in antigen_keys) + r")\b"
matched = experimental_design["Stain"].str.extract(pattern, flags=re.IGNORECASE)[0]
experimental_design["Structure"] = (
    matched.str.upper()
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
    """
    Build a samplesheet for one measurement (plate).

    Parameters
    ----------
    measurement_id : str
        The experiment UUID, e.g. '028ebee9-afaf-4ff8-b435-af11714285dc'
    panel : int
        Which panel to join against in experimental_design (default 1).

    Returns
    -------
    pd.DataFrame
        Full samplesheet with filepaths, metadata, stain, and structure columns.
    """
    exp_path = BASE_PATH / measurement_id
    img_dir  = exp_path / "images"

    # --- parse experiment XML ---
    exp_xml   = next(exp_path.glob("*.xml"), None)
    index_xml = next((exp_path / "index").glob("*.xml"), None)

    exp_root   = ET.parse(exp_xml).getroot()
    index_root = ET.parse(index_xml).getroot()

    meas_id   = exp_root.find('h:MeasurementID', NS).text
    date      = exp_root.find('h:Date', NS).text
    plate_id  = index_root.find('.//h:PlateID', NS).text
    x_res     = float(index_root.find('.//h:ImageResolutionX', NS).text) * 1e6
    y_res     = float(index_root.find('.//h:ImageResolutionY', NS).text) * 1e6

    # --- parse channel info ---
    channels = []
    for map_el in index_root.findall(".//h:Map", NS):
        first = map_el.find("h:Entry", NS)
        if first is not None and first.find("h:ChannelName", NS) is not None:
            for entry in map_el.findall("h:Entry", NS):
                ch_id = entry.attrib.get("ChannelID")
                channels.append({
                    "ChannelID":    int(ch_id) if ch_id else None,
                    "Channel_name": entry.find("h:ChannelName", NS).text,
                    "Type":         entry.findtext("h:ChannelType", default=None, namespaces=NS),
                    "Excitation_nm": entry.findtext("h:MainExcitationWavelength", default=None, namespaces=NS),
                    "Emission_nm":   entry.findtext("h:MainEmissionWavelength", default=None, namespaces=NS),
                })
            break

    channel_df = pd.DataFrame(channels).sort_values("ChannelID").reset_index(drop=True)
    channel_df["Pseudocolor"]    = channel_df["Channel_name"].map(pseudocolor_map).fillna("gray")
    channel_df["MPL_colormap"]   = channel_df["Pseudocolor"].str.lower().map(mpl_colormaps)
    channel_df["Measurement_ID"] = meas_id
    channel_df["Measurement_date"] = date
    channel_df["Plate_ID"]       = plate_id
    channel_df["res_x"]          = x_res
    channel_df["res_y"]          = y_res

    # --- collect tiff files ---
    files = sorted(f for f in img_dir.rglob("*") if f.suffix.lower() == ".tiff")
    file_df = pd.DataFrame({
        "filepath":    files,
        "filename":    [f.name for f in files],
        "subdirectory": [str(f.parent.relative_to(img_dir)) for f in files],
    })
    file_df[["Row", "Column", "Frame", "Plane", "ChannelID", "Time"]] = (
        file_df["filename"].apply(lambda x: pd.Series(_parse_filename(x)))
    )

    # --- merge channel metadata ---
    merged = pd.merge(file_df, channel_df, on="ChannelID")
    merged["Panel"] = panel

    # --- merge experimental design (stain + structure) ---
    design = experimental_design[experimental_design["Panel"] == panel]
    samplesheet = pd.merge(merged, design, on=["Channel_name", "Panel"])

    return samplesheet

# %%
# =========================
# BUILD ALL SAMPLESHEETS
# =========================

# MEASUREMENT_IDS = [
#     "028ebee9-afaf-4ff8-b435-af11714285dc",
#     #"07fc5da6-9d7d-4c97-858b-4b76df1859a5",
#     # ... add remaining IDs
# ]

MEASUREMENT_IDS = [p.name for p in BASE_PATH.iterdir() if p.is_dir()]
all_samplesheets = {mid: build_samplesheet(mid) for mid in MEASUREMENT_IDS}

# Access one:
# samplesheet = all_samplesheets["028ebee9-afaf-4ff8-b435-af11714285dc"]

# Or concatenate all into one big samplesheet:
samplesheet_all = pd.concat(all_samplesheets.values(), ignore_index=True)

# %%
print(MEASUREMENT_IDS)

# %%
print(samplesheet_all)

# %% [markdown]
# ## Find nuclei features

# %%
import pandas as pd
from pathlib import Path

NUCLEI_FEATURES_DIR = Path('/data/CARDPB2/iNDI/Production/nucleus_segmentation_result')

# Load all nuclei features files, keyed by measurement ID
all_nuclei_features = {}
for f in NUCLEI_FEATURES_DIR.iterdir():
    match = next((mid for mid in MEASUREMENT_IDS if mid in f.name), None)
    if match:
        nf = pd.read_csv(f)
        nf["experiment_name"] = match
        all_nuclei_features[match] = nf

nuclei_features = pd.concat(all_nuclei_features.values(), ignore_index=True)

# %%
print(nuclei_features.shape)
print(nuclei_features.dtypes)
print(nuclei_features.head())

# %% [markdown]
# ## Pipeline

# %%
import os
import re
import uuid
import numpy as np
import pandas as pd
import tifffile
import dask.array as da
import dask
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage.measure import label
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import norm
from scipy.ndimage import distance_transform_edt, gaussian_laplace, binary_fill_holes
from skimage import filters, morphology
from skimage.filters import threshold_triangle, threshold_otsu
from skimage.morphology import remove_small_objects, erosion, dilation
from skimage.morphology import disk as morph_disk

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
ALLOWED_STRUCTURES = set(SEGMENTERS_BY_STRUCTURE.keys())

ROI_RADIUS  = 120
N_WORKERS   = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
SITE_KEYS   = ["subdirectory", "Row", "Column", "Frame", "Plane", "Time"]

# --- QC controls ---
SHOW_QC                 = True
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

def imread_plane(path, prefer_lazy=True, enforce_2d=True):
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        shape = page.shape if page.ndim <= 2 else tuple(page.shape[1:])
        dtype = page.dtype
    if not prefer_lazy:
        arr = tifffile.imread(path)
        if enforce_2d and arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    @dask.delayed
    def _read(path_):
        arr = tifffile.imread(path_)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    return da.from_delayed(_read(path), shape=shape, dtype=dtype)

def bbox_from_center(y, x, r, H, W):
    yi, xi = int(round(y)), int(round(x))
    return max(yi - r, 0), min(yi + r + 1, H), max(xi - r, 0), min(xi + r + 1, W)

def get_row(site_df: pd.DataFrame, **filters):
    sel = site_df.copy()
    for k, v in filters.items():
        sel = sel[sel[k] == v]
    return None if sel.empty else sel.iloc[0]

def _draw_circles(ax, xs, ys, r):
    for y, x in zip(ys, xs):
        ax.add_patch(Circle((x, y), r, fill=False, edgecolor="red", linewidth=1))

def qc_show_three_panel(nuc_img, ch_img, global_seg, xs, ys, r, title_prefix):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(nuc_img, cmap="gray"); axes[0].set_title(f"{title_prefix} – DAPI");    axes[0].axis("off")
    axes[1].imshow(ch_img,  cmap="gray"); axes[1].set_title(f"{title_prefix} – Channel"); axes[1].axis("off")
    axes[2].imshow(ch_img,  cmap="gray"); axes[2].imshow(global_seg, alpha=0.4)
    axes[2].set_title(f"{title_prefix} – Overlay"); axes[2].axis("off")
    for ax in axes:
        _draw_circles(ax, xs, ys, r)
    plt.tight_layout()
    plt.show()

def qc_tile_gallery(ch_img, tiles, seg_tiles, title_prefix, max_tiles=8):
    if max_tiles <= 0 or len(tiles) == 0:
        return
    n = min(max_tiles, len(tiles))
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, (tile, seg) in zip(axes, list(zip(tiles, seg_tiles))[:n]):
        ax.imshow(tile, cmap="gray"); ax.imshow(seg, alpha=0.4); ax.axis("off")
    fig.suptitle(f"{title_prefix} – example ROI tiles", y=1.02)
    plt.tight_layout()
    plt.show()

def _write_site_parquet(df, base_dir, mid, dapi_name):
    os.makedirs(base_dir, exist_ok=True)
    uid   = uuid.uuid4().hex[:8]
    mid_s  = str(mid).replace("/", "_")
    dapi_s = str(dapi_name).replace("/", "_")
    out_path = os.path.join(base_dir, f"MID={mid_s}__DAPI={dapi_s}__{uid}.parquet")
    df.to_parquet(out_path, index=False)
    return out_path


# =========================
# CORE: process_site
# =========================

def process_site(site_meta, site_df, nuclei_features, segmenters_by_structure,
                 r=ROI_RADIUS, do_qc=False):
    site_df = site_df.copy()
    for c in ("Stain", "Structure", "filename"):
        if c in site_df.columns:
            site_df[c] = site_df[c].astype(str).str.strip()
    site_df["Stain"] = site_df["Stain"].str.upper()

    dapi_row = get_row(site_df, Stain="DAPI")
    if dapi_row is None:
        return None

    dapi_fname = str(dapi_row["filename"]).strip()
    nuc_da     = imread_plane(dapi_row["filepath"])
    nuc_img    = np.asarray(nuc_da.compute())  # always read — used for shape and QC
    H, W       = nuc_img.shape[:2]

    nf = nuclei_features.copy()
    key_col = "filename"   # renamed before entering worker
    nf[key_col] = nf[key_col].astype(str).str.strip()

    props_fname = str(site_meta.get("props_filename", site_meta.get("dapi_filename", ""))).strip()
    nf_sel = nf[nf[key_col] == props_fname]
    if nf_sel.empty:
        nf_sel = nf[nf[key_col] == dapi_fname]
    if nf_sel.empty:
        return None

    ys     = nf_sel["centroid-0"].astype(float).to_numpy()
    xs     = nf_sel["centroid-1"].astype(float).to_numpy()
    labels = nf_sel["label"].astype(int).to_numpy() if "label" in nf_sel.columns else np.arange(1, len(xs) + 1)

    canon  = {k.lower(): k for k in segmenters_by_structure}
    site_df["Structure"] = site_df["Structure"].str.lower().map(canon).fillna(site_df["Structure"])
    todo   = [s for s in sorted(set(site_df["Structure"].unique()) & set(segmenters_by_structure))
              if callable(segmenters_by_structure.get(s))]
    if not todo:
        return None

    out_rows = []
    for structure in todo:
        seg_fn = segmenters_by_structure[structure]
        row    = site_df.loc[site_df["Structure"] == structure].iloc[0]
        ch_da  = imread_plane(row["filepath"])
        if ch_da.shape[:2] != (H, W):
            raise ValueError(f"Shape mismatch @ {structure}: {ch_da.shape} vs {(H, W)}")

        # Read full channel image once — avoids repeated per-nucleus I/O
        ch_img = np.asarray(ch_da.compute())

        global_seg                  = np.zeros((H, W), dtype=bool)
        example_tiles, example_segs = [], []

        for lab, y, x in zip(labels, ys, xs):
            y0, y1, x0, x1 = bbox_from_center(y, x, r, H, W)
            tile = ch_img[y0:y1, x0:x1]
            yy, xx = np.ogrid[y0:y1, x0:x1]
            roi      = (yy - y)**2 + (xx - x)**2 <= r**2
            seg_tile = seg_fn(np.where(roi, tile, 0)) & roi
            global_seg[y0:y1, x0:x1] |= seg_tile

            if do_qc and len(example_tiles) < QC_MAX_TILES_PER_STRUCT:
                example_tiles.append(tile)
                example_segs.append(seg_tile)

            obj_labels = label(seg_tile, connectivity=1)
            count_obj  = int(obj_labels.max())
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

            out_rows.append({
                "Measurement_ID": site_meta.get("Measurement_ID"),
                "subdirectory": row.get("subdirectory"),
                "Row": row.get("Row"), "Column": row.get("Column"),
                "Frame": row.get("Frame"), "Plane": row.get("Plane"), "Time": row.get("Time"),
                "DAPI_filename": dapi_fname,
                "channel_filename": row["filename"],
                "Structure": structure,
                "Stain": row.get("Stain"),
                "Nucleus_ID": int(lab),
                "area_px": area_px,
                "coverage_frac": coverage_frac,
                "organelle_count": count_obj,
                "average_organelle_area": area_px_mean,
                "max_f_intensity": int_max,
                "sum_f_intensity": int_sum,
                "mean_f_intensity": int_mean,
                "median_f_intensity": int_median,
                "CoefOfVar_intensity": int_cv,
                "radial_mean": radial_mean,
                "radial_p25": radial_p25, "radial_p50": radial_p50, "radial_p75": radial_p75,
                "inner_frac": inner_frac, "mid_frac": mid_frac, "outer_frac": outer_frac,
                "boundary_touch_frac": boundary_touch_frac,
            })

        if do_qc:
            title = f"{structure} | {dapi_fname}"
            qc_show_three_panel(nuc_img, ch_img, global_seg, xs, ys, r, title_prefix=title)
            if QC_MAX_TILES_PER_STRUCT > 0:
                qc_tile_gallery(ch_img, example_tiles, example_segs,
                                title_prefix=title, max_tiles=QC_MAX_TILES_PER_STRUCT)

    return pd.DataFrame(out_rows) if out_rows else None


# =========================
# SITE ITERATION
# samplesheet     → "filename"
# nuclei_features → "image_name"
# =========================

def iter_sites_single_panel(samplesheet, nuclei_features):
    ss = samplesheet.copy()
    ss["filename"] = ss["filename"].astype(str).str.strip()
    nf = nuclei_features.copy()
    nf["image_name"] = nf["image_name"].astype(str).str.strip()

    site_keys = ["Row", "Column", "Frame", "Plane", "Time"]

    for fname in nf["image_name"].unique():
        # Find the matching row for this filename
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

        # Expand to ALL rows sharing the same site keys (all channels)
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
            "props_filename": fname,
        }
        yield site_meta, site_df


# =========================
# WORKER
# =========================

def _process_site_worker(args):
    site_meta, site_df, nuclei_features, r, do_qc, qc_output_dir, segmenters = args

    # process_site expects "filename" in nuclei_features
    nuclei_features = nuclei_features.copy().rename(columns={"image_name": "filename"})

    if do_qc:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(qc_output_dir, exist_ok=True)
        _orig_show = plt.show
        _site_id   = str(site_meta.get("dapi_filename", "site")).replace("/", "_")
        _fig_count = [0]

        def _save_instead(*a, **kw):
            fig  = plt.gcf()
            path = os.path.join(qc_output_dir, f"{_site_id}_fig{_fig_count[0]}.png")
            fig.savefig(path, bbox_inches="tight", dpi=100)
            plt.close(fig)
            _fig_count[0] += 1

        plt.show = _save_instead

    try:
        return process_site(site_meta, site_df, nuclei_features, segmenters, r=r, do_qc=do_qc)
    except Exception as e:
        print(f"[worker] {site_meta.get('dapi_filename')} failed: {e}")
        return None
    finally:
        if do_qc:
            plt.show = _orig_show


# =========================
# PARALLEL run_all
# =========================

def run_all_parallel(
    samplesheet, nuclei_features, segmenters_by_structure,
    r=ROI_RADIUS, n_workers=N_WORKERS, output_dir=None,
    show_qc=SHOW_QC, qc_max_sites=QC_MAX_SITES, qc_output_dir=QC_OUTPUT_DIR,
):
    samplesheet     = samplesheet.copy()
    nuclei_features = nuclei_features.copy()
    samplesheet["filename"]       = samplesheet["filename"].astype(str).str.strip()
    nuclei_features["image_name"] = nuclei_features["image_name"].astype(str).str.strip()

    all_sites = list(iter_sites_single_panel(samplesheet, nuclei_features))
    total     = len(all_sites)
    print(f"Found {total} sites — dispatching to {n_workers} workers")

    qc_count, args_list = 0, []
    for site_meta, site_df in all_sites:
        do_qc = show_qc and (qc_count < qc_max_sites)
        if do_qc:
            qc_count += 1
        args_list.append((site_meta, site_df, nuclei_features, r, do_qc, qc_output_dir, segmenters_by_structure))

    out, completed = [], 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_site_worker, args): args[0] for args in args_list}
        for fut in as_completed(futures):
            site_meta  = futures[fut]
            completed += 1
            try:
                df = fut.result()
                if df is not None and not df.empty:
                    if output_dir:
                        _write_site_parquet(df, output_dir,
                                            site_meta.get("Measurement_ID", "unknown"),
                                            site_meta["dapi_filename"])
                    else:
                        out.append(df)
                    print(f"[{completed}/{total}] ✓ {site_meta['dapi_filename']} → {len(df)} rows")
                else:
                    print(f"[{completed}/{total}] – {site_meta['dapi_filename']} → no data")
            except Exception as e:
                print(f"[{completed}/{total}] ✗ {site_meta['dapi_filename']} → {e}")

    print("\nAll sites processed.")

    if output_dir:
        parts = [pd.read_parquet(os.path.join(output_dir, f))
                 for f in os.listdir(output_dir) if f.endswith(".parquet")]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

# %%
# import subprocess, os

# MEASUREMENT = "028ebee9-afaf-4ff8-b435-af11714285dc"
# SCRATCH = f"/lscratch/{os.environ['SLURM_JOB_ID']}/{MEASUREMENT}"
# SRC = f"/data/CARDPB2/iNDI/Production/{MEASUREMENT}/images"

# # Stage just one well to scratch
# WELL = "r02c11"
# os.makedirs(SCRATCH, exist_ok=True)
# subprocess.run(["rsync", "-a", f"{SRC}/{WELL}/", f"{SCRATCH}/{WELL}/"], check=True)

# # Point samplesheet to scratch for just this well
# test_ss = samplesheet[samplesheet["subdirectory"] == WELL].copy()
# test_ss["filepath"] = test_ss["filepath"].astype(str).str.replace(SRC, SCRATCH, regex=False)

# # Subset nuclei features to match
# nuclei_features = all_nuclei_features["028ebee9-afaf-4ff8-b435-af11714285dc"]
# test_nf = nuclei_features[nuclei_features["image_name"].str.contains(WELL, regex=False)]

# # Run on a few frames only
# test_nf = test_nf[test_nf["image_name"].str.contains("f01|f02|f03", regex=True)]

# # Run
# results = run_all_parallel(test_ss, test_nf, SEGMENTERS_BY_STRUCTURE, r=ROI_RADIUS)
# print(results.shape)
# results.head()

# %%
import subprocess, os

SCRATCH_BASE = f"/lscratch/{os.environ['SLURM_JOB_ID']}"
SRC_BASE     = "/data/CARDPB2/iNDI/Production/AbPanel1"
OUTPUT_DIR   = "./organelle_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_results = []

for mid in MEASUREMENT_IDS:
    print(f"\n{'='*60}")
    print(f"Processing: {mid}")
    print(f"{'='*60}")

    # Skip if no samplesheet or nuclei features available
    if mid not in all_samplesheets or mid not in all_nuclei_features:
        print(f"  Skipping {mid} — missing samplesheet or nuclei features")
        continue

    ss = all_samplesheets[mid].copy()
    nf = all_nuclei_features[mid].copy()

    # Stage to scratch
    scratch_dir = f"{SCRATCH_BASE}/{mid}"
    src_dir     = f"{SRC_BASE}/{mid}/images"

    try:
        subprocess.run(
            ["rsync", "-a", "--info=progress2", f"{src_dir}/", f"{scratch_dir}/"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"  rsync failed for {mid}: {e} — skipping")
        continue

    # Remap filepaths to scratch
    ss["filepath"] = ss["filepath"].astype(str).str.replace(src_dir, scratch_dir, regex=False)

    # Run
    try:
        result = run_all_parallel(
            ss, nf, SEGMENTERS_BY_STRUCTURE,
            r=ROI_RADIUS,
            output_dir=OUTPUT_DIR,
        )
        print(f"  Done: {len(result)} rows")
        all_results.append(result)
    except Exception as e:
        print(f"  run_all_parallel failed for {mid}: {e}")
        continue
    finally:
        # Always clean up scratch even if processing failed
        subprocess.run(["rm", "-rf", scratch_dir], check=True)
        print(f"  Cleared scratch: {scratch_dir}")

# Concatenate everything
organelle_features = pd.concat(all_results, ignore_index=True)
print(f"\nFinal shape: {organelle_features.shape}")

# %%
# OUTPUT_DIR = f"/data/CARDPB2/iNDI/Production/Organelle_segmentation/Panel_1/results/{MEASUREMENT}"
# /data/kelpschdj/iNDI/Production/Organelle_segmentation/Panel_1


# results = run_all_parallel(
#     samplesheet,
#     nuclei_features,
#     SEGMENTERS_BY_STRUCTURE,
#     r=ROI_RADIUS,
#     output_dir=OUTPUT_DIR
# )

# %%
# import dask.dataframe as dd

# df = dd.read_parquet("/data/kelpschdj/iNDI/Production/Organelle_segmentation/Panel_1/results/028ebee9-afaf-4ff8-b435-af11714285dc/*.parquet")
# combined = df.compute()

# print(combined.shape)
# print(combined.dtypes)

# combined.to_parquet(
#     "combined_organelle_segmentation_results_028ebee9-afaf-4ff8-b435-af11714285dc.parquet",
#     engine="pyarrow",
#     index=False,
#     compression="zstd"
# )


