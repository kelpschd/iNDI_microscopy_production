#!/usr/bin/env python3
"""
Convert binary mask images (.png / .tiff) to pairwise distance matrices (.npy).

Expected input layout:
    input_dir/
        class1/  image_a.png  image_b.tiff  ...
        class2/  ...
        classN/  ...

Output layout mirrors the input:
    output_dir/
        class1/  image_a.npy  image_b.npy  ...
        class2/  ...
        classN/  ...

Pipeline (matching the prepare_BBBC010 notebook):
  1. Load the mask with PIL and convert to a NumPy array.
  2. Extract contours with skimage; keep only the longest one (single object).
  3. Resample the contour uniformly along a fitted spline.
  4. Build the pairwise distance matrix with scipy.spatial.distance_matrix.
"""

import argparse
import sys
import numpy as np
from PIL import Image
from pathlib import Path
import tqdm
from scipy.spatial import distance_matrix

from helpers import find_longest_contour, contour_spline_resample

MASK_SUFFIXES = {'.png', '.tif', '.tiff'}


def mask_to_dm(fpath: Path, n_samples: int, sparsity: int) -> np.ndarray:
    with Image.open(fpath) as img:
        arr = np.array(img)
    contour = find_longest_contour(arr)
    resampled = contour_spline_resample(contour, n_samples=n_samples, sparsity=sparsity)
    return distance_matrix(resampled, resampled)


def process_dataset(input_dir: Path, output_dir: Path, n_samples: int, sparsity: int) -> int:
    class_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
    if not class_dirs:
        print(f"Error: no class subdirectories found in {input_dir}", file=sys.stderr)
        return 1

    all_masks = [
        f
        for class_dir in class_dirs
        for f in sorted(class_dir.iterdir())
        if f.suffix.lower() in MASK_SUFFIXES
    ]
    if not all_masks:
        print(f"Error: no .png/.tif/.tiff files found under {input_dir}", file=sys.stderr)
        return 1

    n_errors = 0
    for fpath in tqdm.tqdm(all_masks, desc="masks → distance matrices"):
        out_dir = output_dir / fpath.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (fpath.stem + '.npy')
        try:
            dm = mask_to_dm(fpath, n_samples, sparsity)
            np.save(out_path, dm)
        except Exception as exc:
            tqdm.tqdm.write(f"SKIP {fpath.name}: {exc}")
            n_errors += 1

    n_ok = len(all_masks) - n_errors
    print(f"Done: {n_ok}/{len(all_masks)} saved to {output_dir}"
          + (f"  ({n_errors} failed)" if n_errors else ""))
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert binary mask images (.png/.tiff) to pairwise distance matrices (.npy). "
            "Input must be structured as dataset/class1, dataset/class2, ..."
        )
    )
    parser.add_argument(
        "input_dir", type=Path,
        help="Root folder containing one subdirectory per class.",
    )
    parser.add_argument(
        "output_dir", type=Path,
        help="Destination folder; class structure is mirrored with .npy files.",
    )
    # Will likely set this to max of 16 for my sample
    parser.add_argument(
        "--n_samples", type=int, default=64,
        help="Number of points sampled on the fitted spline (default: 64).",
    )
    parser.add_argument(
        "--sparsity", type=int, default=1,
        help=(
            "Step size when subsampling the raw contour before spline fitting "
            "(default: 1 — use every point). Increase to reduce pixel-edge artifacts "
            "on low-resolution images."
        ),
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"Input directory not found: {args.input_dir}")

    sys.exit(process_dataset(args.input_dir, args.output_dir, args.n_samples, args.sparsity))


if __name__ == "__main__":
    main()
