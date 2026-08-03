import re
import sys
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# --- run_utils bootstrap ---------------------------------------------------
# Make `import run_utils` work regardless of where the script is launched from
# (repo root, scripts/, or a SLURM job with an arbitrary cwd).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_utils as ru  # noqa: E402

# --- Configuration ---------------------------------------------------------

# Defaults; can be overridden on the command line (see parse_args).
DEFAULT_METADATA_DIR = Path("/data/CARDPB2/iNDI/Production/metadata")
DEFAULT_BASE_PATH = Path("/data/CARDPB2/iNDI/Production/AbPanel2")

# Harmony XML namespace (consistent across experiments)
NS = {"h": "43B2A954-E3C3-47E1-B392-6635266B0DD3/HarmonyV7"}

# Pseudocolor mapping rules (used to generate images on the fly)
PSEUDOCOLOR_MAP = {
    "DAPI": "blue",
    "Brightfield": "gray",
    "Alexa 488": "green",
    "Alexa 568": "red",
    "Alexa 647": "magenta",
}

MPL_COLORMAPS = {
    "blue": LinearSegmentedColormap.from_list("black_blue", [(0, 0, 0), (0, 0, 1)]),
    "green": LinearSegmentedColormap.from_list("black_green", [(0, 0, 0), (0, 1, 0)]),
    "red": LinearSegmentedColormap.from_list("black_red", [(0, 0, 0), (1, 0, 0)]),
    "magenta": LinearSegmentedColormap.from_list("black_magenta", [(0, 0, 0), (1, 0, 1)]),
    "gray": LinearSegmentedColormap.from_list("black_gray", [(0, 0, 0), (1, 1, 1)]),
}

FILENAME_RE = re.compile(r"r(\d+)c(\d+)f(\d+)p(\d+)-ch(\d+)t(\d+)")

# Stage name used in run metadata / logs.
STAGE = "image_metadata"


def collect_params():
    """Tuning constants recorded in run_metadata.json for this stage."""
    return {
        "pseudocolor_map": PSEUDOCOLOR_MAP,
        "harmony_namespace": NS["h"],
        "filename_regex": FILENAME_RE.pattern,
    }


# --- Helpers ---------------------------------------------------------------

def find_text(element, path, cast=None):
    """Return the text of the first matching element, optionally cast."""
    node = element.find(path, NS)
    if node is None or node.text is None:
        return None
    return cast(node.text) if cast else node.text


def parse_filename(name):
    """Extract (Row, Column, Frame, Plane, ChannelID, Time) from a filename."""
    match = FILENAME_RE.match(name)
    if match:
        return [int(g) for g in match.groups()]
    return [None] * 6


def parse_channels(index_root):
    """Extract channel metadata from the index XML."""
    for map_el in index_root.findall(".//h:Map", NS):
        first_entry = map_el.find("h:Entry", NS)
        if first_entry is None or first_entry.find("h:ChannelName", NS) is None:
            continue

        channels = []
        for entry in map_el.findall("h:Entry", NS):
            ch_id = entry.attrib.get("ChannelID")
            channels.append({
                "ChannelID": int(ch_id) if ch_id is not None else None,
                "Channel_name": find_text(entry, "h:ChannelName"),
                "Type": find_text(entry, "h:ChannelType"),
                "Excitation_nm": find_text(entry, "h:MainExcitationWavelength"),
                "Emission_nm": find_text(entry, "h:MainEmissionWavelength"),
            })
        return channels
    return []


def discover_experiments(base_path, selected=None):
    """Return a list of experiment directories under base_path.

    If `selected` is given, only those folder names are returned (preserving
    the given order); a warning is printed for any that are missing.
    """
    if selected is not None:
        experiments = []
        for name in selected:
            path = base_path / name
            if path.is_dir():
                experiments.append(path)
            else:
                print(f"[warning] experiment folder not found, skipping: {path}")
        return experiments

    # Default: every subdirectory that contains an index folder with an XML.
    experiments = []
    for path in sorted(p for p in base_path.iterdir() if p.is_dir()):
        if next((path / "index").glob("*.xml"), None) is not None:
            experiments.append(path)
    return experiments


def process_experiment(experiment_path):
    """Parse one experiment folder and return its merged metadata DataFrame.

    Returns None if required files are missing.
    """
    experiment_img_dir = experiment_path / "images"
    experiment_xml_file = next(experiment_path.glob("*.xml"), None)
    index_xml = next((experiment_path / "index").glob("*.xml"), None)

    if experiment_xml_file is None or index_xml is None:
        print(f"[warning] missing XML files in {experiment_path.name}, skipping.")
        return None

    # Experiment-level XML
    experiment_root = ET.parse(experiment_xml_file).getroot()
    measurement_id = find_text(experiment_root, "h:MeasurementID")
    date = find_text(experiment_root, "h:Date")
    plate = find_text(experiment_root, "h:InitialPlateName")

    # Index XML
    index_root = ET.parse(index_xml).getroot()
    plate_id = find_text(index_root, ".//h:PlateID")
    x_res = find_text(index_root, ".//h:ImageResolutionX", float) * 1e6
    y_res = find_text(index_root, ".//h:ImageResolutionY", float) * 1e6

    # Channel metadata
    channel_df = (
        pd.DataFrame(parse_channels(index_root))
        .sort_values("ChannelID")
        .reset_index(drop=True)
    )
    channel_df["Pseudocolor"] = channel_df["Channel_name"].map(PSEUDOCOLOR_MAP).fillna("gray")
    channel_df["MPL_colormap"] = channel_df["Pseudocolor"].str.lower().map(MPL_COLORMAPS)
    channel_df["Measurement_ID"] = measurement_id
    channel_df["Measurement_date"] = date
    channel_df["Plate_ID"] = plate_id
    channel_df["res_x"] = x_res
    channel_df["res_y"] = y_res

    # Image files
    files = sorted(f for f in experiment_img_dir.rglob("*") if f.suffix.lower() == ".tiff")
    if not files:
        print(f"[warning] no .tiff files in {experiment_path.name}, skipping.")
        return None

    df = pd.DataFrame({
        "filepath": files,
        "filename": [f.name for f in files],
        "subdirectory": [f.parent.relative_to(experiment_img_dir) for f in files],
    })
    df[["Row", "Column", "Frame", "Plane", "ChannelID", "Time"]] = df["filename"].apply(
        lambda x: pd.Series(parse_filename(x))
    )

    merged_df = pd.merge(df, channel_df, on="ChannelID")
    merged_df["Experiment_name"] = experiment_path.name

    # Per-experiment summary
    summary = {
        "wells": merged_df[["Row", "Column"]].drop_duplicates().shape[0],
        "channels": merged_df["ChannelID"].nunique(),
        "z_planes": merged_df["Plane"].nunique(),
        "frames": merged_df["Frame"].nunique(),
        "timepoints": merged_df["Time"].nunique(),
    }

    print(f"""
Experiment ID: {measurement_id}
Experiment folder: {experiment_path.name}
Plate ID: {plate}
Wells imaged: {summary["wells"]}
Frames per well: {summary["frames"]}
Channels per image: {summary["channels"]}
Z-slices per image: {summary["z_planes"]}
Timepoints per image: {summary["timepoints"]}
""")

    return merged_df


# --- Main ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract imaging metadata for one or more experiments."
    )
    parser.add_argument(
        "base_path",
        nargs="?",
        type=Path,
        default=DEFAULT_BASE_PATH,
        help="Directory containing experiment folders "
             f"(default: {DEFAULT_BASE_PATH}).",
    )
    parser.add_argument(
        "-e", "--experiments",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Specific experiment folder names to process. "
             "If omitted, all experiments under base_path are processed.",
    )
    parser.add_argument(
        "-m", "--metadata-dir",
        type=Path,
        default=DEFAULT_METADATA_DIR,
        help=f"Directory containing the plate metadata CSVs "
             f"(default: {DEFAULT_METADATA_DIR}).",
    )
    # --output-root + --run-id (optional here: stage 0_ mints the run ID).
    ru.add_run_args(parser, mints_run_id=True)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.base_path.is_dir():
        raise SystemExit(f"[error] base path is not a directory: {args.base_path}")

    # Mint (or accept) the run ID and create the run directory.
    run_id = args.run_id or ru.new_run_id()
    run_dir = ru.create_run_dir(args.output_root, run_id)
    out_dir = ru.stage_dir(run_dir, STAGE)

    rec = ru.StageRecorder(
        run_dir, stage=STAGE, run_id=run_id,
        params=collect_params(),
        inputs={
            "base_path": str(args.base_path),
            "metadata_dir": str(args.metadata_dir),
            "experiments_requested": args.experiments or "ALL",
        },
    )
    # Make the run ID obvious in stdout — downstream stages need it.
    print(f"\n=== RUN ID: {run_id} ===")
    print(f"=== run dir: {run_dir} ===\n")
    rec.log(f"run id {run_id}")

    # Load shared metadata CSVs once.
    automated_plates = pd.read_csv(args.metadata_dir / "all_automated_plates_combined.csv")
    manual_plates = pd.read_csv(args.metadata_dir / "all_manual_plates_combined.csv")
    plate_id_map = pd.read_csv(args.metadata_dir / "indi_plateID_to_folderID.csv")

    experiments = discover_experiments(args.base_path, args.experiments)
    rec.log(f"found {len(experiments)} experiment(s) to process")
    print(f"Found {len(experiments)} experiment(s) to process.\n")

    all_metadata = {}
    per_experiment = []
    for experiment_path in experiments:
        merged_df = process_experiment(experiment_path)
        if merged_df is None:
            per_experiment.append({
                "experiment": experiment_path.name,
                "status": "skipped",
                "n_images": 0,
            })
            continue
        all_metadata[experiment_path.name] = merged_df

        out_path = out_dir / f"{experiment_path.name}_metadata.parquet"
        # Drop the colormap objects (not serializable) and cast Path
        # columns to str so pyarrow can write them.
        out_df = merged_df.drop(columns=["MPL_colormap"], errors="ignore").copy()
        for col in ("filepath", "subdirectory"):
            if col in out_df.columns:
                out_df[col] = out_df[col].astype(str)
        out_df.to_parquet(out_path, index=False)
        rec.log(f"{experiment_path.name}: wrote {len(out_df)} rows -> {out_path.name}")
        print(f"Wrote {out_path}")

        per_experiment.append({
            "experiment": experiment_path.name,
            "status": "ok",
            "n_images": int(len(out_df)),
            "n_dapi": int((merged_df["Channel_name"] == "DAPI").sum()),
        })

    # Combine everything into one big table (with an Experiment_name column).
    if all_metadata:
        combined_df = pd.concat(all_metadata.values(), ignore_index=True)
    else:
        combined_df = pd.DataFrame()
        print("No experiments were successfully processed.")

    # DAPI samples across all processed experiments.
    n_dapi_total = 0
    if not combined_df.empty:
        dapi_samples = combined_df[combined_df["Channel_name"] == "DAPI"].copy()
        filepaths = dapi_samples["filepath"].tolist()
        n_dapi_total = len(filepaths)

        print(f"\nFound {n_dapi_total} DAPI images across all experiments.")
        print("Sample filepaths:")
        for filepath in filepaths[:5]:
            print(f" - {filepath}")

    # One overall stage record, with per-experiment summaries nested inside.
    n_ok = sum(1 for e in per_experiment if e["status"] == "ok")
    rec.finish(
        outputs={"image_metadata_dir": str(out_dir)},
        summary={
            "experiments_processed": n_ok,
            "experiments_skipped": len(per_experiment) - n_ok,
            "total_dapi_images": int(n_dapi_total),
            "per_experiment": per_experiment,
        },
    )

    print(f"\n=== RUN ID: {run_id} (pass to downstream stages with --run-id) ===")
    return combined_df


if __name__ == "__main__":
    combined_df = main()