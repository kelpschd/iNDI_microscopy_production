

## Take input of nucleus segmentation result (check output, i think its just a csv)
# load nucleus parquet file
nuc_feat_fn = "/data/kelpschdj/iNDI/Production/Nucleus_segmentation/Nucleus_feature_data/f8062781-891b-4688-a59e-fccbf48325cf_nuclei_features_test_20260318.csv"
nuc_feat_df = pd.read_csv(nuc_feat_fn)

print(nuc_feat_df.head())
print("************************************************")
print(f"Total number of nuclei in experiment: {len(nuc_feat_df)}")

## Edge and ROI overlap filtering
RADIUS = 120
H, W = 2160, 2160 ## is this in my metadata already?
circle_area = np.pi * RADIUS**2
area_thresh = 0.05 * circle_area

x = nuc_feat_df["centroid-1"]
y = nuc_feat_df["centroid-0"]

overflow = (
    (x - RADIUS < 0) | (x + RADIUS > W) |
    (y - RADIUS < 0) | (y + RADIUS > H)
)

nuc_feat_df["edge_check"] = np.where(overflow, "fail", "pass")

def lens_area(d, RADIUS):
    """Intersection area of two circles, equal radius r, center distance d."""
    if d >= 2*RADIUS:
        return 0.0
    if d <= 0:
        return circle_area
    return 2 * RADIUS**2 * np.arccos(d / (2*RADIUS)) - (d/2) * np.sqrt(4*RADIUS**2 - d**2)

nuc_feat_df["overlap_check"] = "pass"

for name, grp in nuc_feat_df.groupby("image_name"):
    idx = grp.index.to_numpy()
    xs = grp["centroid-1"].to_numpy()
    ys = grp["centroid-0"].to_numpy()

    # pairwise center distances
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx**2 + dy**2)

    n = len(idx)
    failed = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(i+1, n):
            if dist[i, j] < 2*RADIUS:  # circles touch at all
                if lens_area(dist[i, j], RADIUS) > area_thresh:
                    failed[i] = True
                    failed[j] = True

    nuc_feat_df.loc[idx[failed], "overlap_check"] = "fail"

print(f"Total number of nuclei in experiment: {len(nuc_feat_df)}")
print("************************************************")
print(nuc_feat_df["edge_check"].value_counts())
print("************************************************")
print(nuc_feat_df["overlap_check"].value_counts())
print("************************************************")
print(pd.crosstab(nuc_feat_df["edge_check"], nuc_feat_df["overlap_check"]))