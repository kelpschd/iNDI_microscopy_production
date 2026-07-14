import re
import argparse
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

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
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write one dated Parquet per experiment. "
             "Files are named <experiment>_metadata_<YYYYMMDD>.parquet.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.base_path.is_dir():
        raise SystemExit(f"[error] base path is not a directory: {args.base_path}")

    # Load shared metadata CSVs once.
    automated_plates = pd.read_csv(args.metadata_dir / "all_automated_plates_combined.csv")
    manual_plates = pd.read_csv(args.metadata_dir / "all_manual_plates_combined.csv")
    plate_id_map = pd.read_csv(args.metadata_dir / "indi_plateID_to_folderID.csv")

    experiments = discover_experiments(args.base_path, args.experiments)
    print(f"Found {len(experiments)} experiment(s) to process.\n")

    today = datetime.now().strftime("%Y%m%d")
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    all_metadata = {}
    for experiment_path in experiments:
        merged_df = process_experiment(experiment_path)
        if merged_df is None:
            continue
        all_metadata[experiment_path.name] = merged_df

        if args.output_dir is not None:
            out_path = args.output_dir / f"{experiment_path.name}_metadata_{today}.parquet"
            # MPL_colormap holds colormap objects that don't serialize; drop it.
            merged_df.drop(columns=["MPL_colormap"], errors="ignore").to_parquet(
                out_path, index=False
            )
            print(f"Wrote {out_path}")

    # Combine everything into one big table (with an Experiment_name column).
    if all_metadata:
        combined_df = pd.concat(all_metadata.values(), ignore_index=True)
    else:
        combined_df = pd.DataFrame()
        print("No experiments were successfully processed.")

    # DAPI samples across all processed experiments.
    if not combined_df.empty:
        dapi_samples = combined_df[combined_df["Channel_name"] == "DAPI"].copy()
        filepaths = dapi_samples["filepath"].tolist()

        print(f"\nFound {len(filepaths)} DAPI images across all experiments.")
        print("Sample filepaths:")
        for filepath in filepaths[:5]:
            print(f" - {filepath}")

    return combined_df


if __name__ == "__main__":
    combined_df = main()