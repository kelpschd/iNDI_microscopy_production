import re
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# --- Configuration ---------------------------------------------------------

METADATA_DIR = Path("/data/CARDPB2/iNDI/Production/metadata")
BASE_PATH = Path("/data/CARDPB2/iNDI/Production/AbPanel2")
EXPERIMENT_NAME = "e61d5e8c-faef-4c2c-8df6-6cc72032f19e"

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


# --- Load metadata CSVs ----------------------------------------------------

automated_plates = pd.read_csv(METADATA_DIR / "all_automated_plates_combined.csv")
manual_plates = pd.read_csv(METADATA_DIR / "all_manual_plates_combined.csv")
plate_id_map = pd.read_csv(METADATA_DIR / "indi_plateID_to_folderID.csv")


# --- Locate experiment files -----------------------------------------------

experiment_path = BASE_PATH / EXPERIMENT_NAME
experiment_img_dir = experiment_path / "images"
experiment_xml_file = next(experiment_path.glob("*.xml"), None)
index_xml = next((experiment_path / "index").glob("*.xml"), None)


# --- Parse experiment XML --------------------------------------------------

experiment_root = ET.parse(experiment_xml_file).getroot()
measurement_id = find_text(experiment_root, "h:MeasurementID")
date = find_text(experiment_root, "h:Date")
plate = find_text(experiment_root, "h:InitialPlateName")

print("Measurement ID:", measurement_id)
print("Date:", date)
print("Plate:", plate)


# --- Parse index XML -------------------------------------------------------

index_root = ET.parse(index_xml).getroot()
plate_id = find_text(index_root, ".//h:PlateID")
x_res = find_text(index_root, ".//h:ImageResolutionX", float) * 1e6
y_res = find_text(index_root, ".//h:ImageResolutionY", float) * 1e6

print("Plate:", plate_id)
print(f"X resolution: {x_res} µm")
print(f"Y resolution: {y_res} µm")


# --- Build channel DataFrame -----------------------------------------------

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


# --- Build image file DataFrame --------------------------------------------

files = sorted(f for f in experiment_img_dir.rglob("*") if f.suffix.lower() == ".tiff")

df = pd.DataFrame({
    "filepath": files,
    "filename": [f.name for f in files],
    "subdirectory": [f.parent.relative_to(experiment_img_dir) for f in files],
})

df[["Row", "Column", "Frame", "Plane", "ChannelID", "Time"]] = df["filename"].apply(
    lambda x: pd.Series(parse_filename(x))
)

merged_df = pd.merge(df, channel_df, on="ChannelID")


# --- Summary ---------------------------------------------------------------

summary = {
    "wells": merged_df[["Row", "Column"]].drop_duplicates().shape[0],
    "channels": merged_df["ChannelID"].nunique(),
    "z_planes": merged_df["Plane"].nunique(),
    "frames": merged_df["Frame"].nunique(),
    "timepoints": merged_df["Time"].nunique(),
}

print(f"""
Experiment ID: {measurement_id}
Plate ID: {plate}
Wells imaged: {summary["wells"]}
Frames per well: {summary["frames"]}
Channels per image: {summary["channels"]}
Z-slices per image: {summary["z_planes"]}
Timepoints per image: {summary["timepoints"]}
""")


# --- Extract DAPI samples --------------------------------------------------

dapi_samples = merged_df[merged_df["Channel_name"] == "DAPI"].copy()
filepaths = dapi_samples["filepath"].tolist()

print(f"Found {len(filepaths)} DAPI images.")
print("Sample filepaths:")
for filepath in filepaths[0:5]:  # Show first 5 samples
    print(f" - {filepath}")