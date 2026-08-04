# iNDI Organelle Morphology Pipeline

A staged image-analysis pipeline for high-content microscopy of iPSC-derived neurons. It extracts imaging metadata, segments nuclei, defines and filters per-nucleus ROIs, segments organelles within those ROIs, and generates QC overlays.

The pipeline is built around **immunofluorescence antibody panels** imaged on Revvity Opera Phenix system. Each stage reads the previous stage's output and writes into a shared, per-run output directory, recording its parameters and a summary into the run metadata so that every run is self-documenting and reproducible.

## Pipeline overview

The stages run in numeric order. Each stage is a `<n>_<stage>.py` script with a matching `<n>_<stage>.sh` SLURM submission wrapper.

| Stage | Script | Reads | Writes |
|-------|--------|-------|--------|
| 0 | `0_img_metadata.py` | Raw experiment folders (Harmony XML + `.tiff` images) | `image_metadata/` (per-experiment metadata parquets) |
| 1 | `1_nucleus_segmentation.py` | `image_metadata/` | `nuclei_features/` (per-nucleus features + per-image QC) |
| 2 | `2_roi_filtering.py` | `nuclei_features/` | `nuclei_filtered/` (nuclei annotated with ROI-selection checks) |
| 3 | `3_organelle_segmentation.py` | `nuclei_filtered/` | `organelle_features/` (per-nucleus organelle features) |
| 4 | `4_segmentation_qc.py` | `nuclei_filtered/` | `qc/` (three-panel + tile-gallery PNGs) |

All stages share a run directory `run_<RUN_ID>/`, with one sub-directory per stage.

## Run IDs

Stage 0 **mints** a run ID (printed prominently to stdout as `=== RUN ID: ... ===`). Every downstream stage **requires** that run ID via `--run-id` so all outputs land in the same `run_<RUN_ID>/` directory. Copy the run ID from stage 0's output and pass it to stages 1–4.

## Requirements

- Python 3 with: `numpy`, `pandas`, `pyarrow`, `tifffile`, `scipy`, `scikit-image`, `dask`, `matplotlib`, and optionally `tqdm`.
- `run_utils.py` (referenced as `ru`) must live in the `scripts/` directory alongside the stage scripts. It provides run-ID minting, run/stage directory creation, and the `StageRecorder` used for per-run metadata and logging.
- A conda environment (the SLURM scripts activate one named `indi_project`).
- For stage 3's scratch staging, `rsync` must be available and the job must run under SLURM (uses `/lscratch/$SLURM_JOB_ID`).

## Usage

### Local / interactive

```bash
# Stage 0 — mints the run ID (note it from stdout)
python scripts/0_img_metadata.py /data/CARDPB2/iNDI/Production/AbPanel1 \
    --output-root /path/to/outputs

# Stages 1–4 — pass the run ID from stage 0
python scripts/1_nucleus_segmentation.py --run-id <RUN_ID> --output-root /path/to/outputs
python scripts/2_roi_filtering.py        --run-id <RUN_ID> --output-root /path/to/outputs
python scripts/3_organelle_segmentation.py --run-id <RUN_ID> --output-root /path/to/outputs --panel 1
python scripts/4_segmentation_qc.py      --run-id <RUN_ID> --output-root /path/to/outputs --panel 1
```

Common options across stages:
- `-e / --experiments NAME [NAME ...]` — process only specific experiment folders. If omitted, all experiments in the input are processed.
- `--output-root` — root directory containing the `run_<RUN_ID>/` folders.

### SLURM

Each stage has a `.sh` wrapper. Set the `RUN_ID` in the wrapper (or the command line) before submitting stages 1–4.

```bash
sbatch scripts/0_img_metadata.sh
# copy the RUN_ID from the job output, then set it in the downstream wrappers
sbatch scripts/1_nucleus_segmentation.sh
sbatch scripts/2_roi_filtering.sh
```

Stage 3 runs as a **job array**, one task per experiment, and requires a two-step submit (see below).

## Stage details

### 0 — Image metadata (`0_img_metadata.py`)
Walks each experiment folder, parses the experiment-level and index Harmony XML (measurement ID, date, plate, resolution, channel definitions), enumerates `.tiff` files, and parses `rNcNfNpN-chNtN` filenames into Row/Column/Frame/Plane/ChannelID/Time. Assigns a pseudocolor per channel and writes one metadata parquet per experiment. This is the stage that mints the run ID.

### 1 — Nucleus segmentation (`1_nucleus_segmentation.py`)
Segments nuclei on the **DAPI** channel using a fitted-Gaussian contrast stretch, a triangle/median low-level threshold, a per-object Otsu high-level threshold, hole-filling, and small-object removal (modified from Allen Cell Segmenter). Frames below a contrast cutoff are skipped via a full-frame contrast gate. Extracts per-nucleus region properties and writes a features parquet plus a per-image QC parquet. Uses Dask for parallelism (default `processes` scheduler).

### 2 — ROI filtering (`2_roi_filtering.py`)
Defines a fixed-radius circular ROI centered on each nucleus centroid and applies three checks: **area** (upper cutoff), **edge** (ROI must not extend past the frame border), and **overlap** (fails ROIs overlapping a neighbor by more than a set fraction of ROI area; 5%). Combines these with the upstream `contrast_check` into a single `selected` boolean. Frame dimensions are read from an image, overridable with `--frame-size HxW`.

### 3 — Organelle segmentation (`3_organelle_segmentation.py`)
For each selected nucleus's ROI, segments organelle channels across the whole frame using structure-specific segmenters (Golgi, Lysosome, Endosome, Mitochondria; each modified from Allen Cell Segmenter), then assigns objects to ROIs under a **full-containment, nearest-centroid** rule. Aggregates per-object shape/intensity/radial-distance features into per-nucleus summary statistics. Which stains map to which structures is set by `--panel` (1 or 2).

Outputs are **version-stamped** (`..._organelle_features__<YYYYMMDD_HHMMSS>.parquet`) so re-runs don't overwrite. Pass a shared `--version-stamp` to all array tasks so they share one stamp.

#### Running stage 3 as an array
```bash
# 1) One shared version stamp; submit the array (one task per experiment)
export ORG_VERSION_STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="<RUN_ID>"
FILT="/path/to/outputs/run_${RUN_ID}/nuclei_filtered"
N=$(ls "$FILT"/*_nuclei_filtered*.parquet | wc -l)
sbatch --export=ALL,ORG_VERSION_STAMP --array=0-$((N-1))%4 scripts/3_organelle_segmentation.sh

# 2) After the array finishes, fold per-experiment shards into run metadata (run once)
python scripts/3_organelle_segmentation.py --run-id <RUN_ID> \
    --output-root /path/to/outputs --merge-only
```

### 4 — Segmentation QC (`4_segmentation_qc.py`)
Imports the segmenters and ROI-assignment logic from stage 3 and renders, for a random sample of frames per experiment, a three-panel figure (DAPI | channel | assigned-segmentation overlay) and a per-ROI tile gallery. `-n` controls the number of random frames; `--seed` makes the sampling reproducible.

## Panels

Panel design maps imaging channels to stains/structures:

- **Panel 1:** DAPI → Nuclei, Alexa 488 → TOMM20 (Mitochondria), Alexa 568 → EEA1 (Endosome), Alexa 647 → LAMP1 (Lysosome)
- **Panel 2:** DAPI → Nuclei, Alexa 488 → RAB11A (Endosome), Alexa 568 → GM130 (Golgi), Alexa 647 → TUJ1 (Microtubules)

Pass the matching `--panel` to stages 3 and 4, and point `--src-base` at the corresponding raw imaging tree.

## Output layout

```
outputs/
└── run_<RUN_ID>/
    ├── image_metadata/       # stage 0
    ├── nuclei_features/      # stage 1
    ├── nuclei_filtered/      # stage 2
    ├── organelle_features/   # stage 3 (version-stamped)
    ├── qc/                   # stage 4
    ├── run_metadata.json     # per-stage params + summaries
    └── run.log
```

## Notes

- Keep BLAS single-threaded (`OMP_NUM_THREADS=1`, etc.) when using process-based parallelism so it doesn't contend with the worker pool — the SLURM wrappers set this.
- Default base paths are set at the top of each script and can be overridden on the command line.