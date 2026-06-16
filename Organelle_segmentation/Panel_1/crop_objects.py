#!/usr/bin/env python3
"""
Crop individual objects from binary mask images (.png / .tiff).

Each connected component (pixel value > 0, background = 0) is tightly
cropped to its bounding box and saved as a separate PNG containing only
that object's pixels.

Usage
-----
Single file:
    python crop_objects.py mask.png output_dir/

Directory (all .png/.tiff files, subdirectory structure is preserved):
    python crop_objects.py masks_dir/ output_dir/

Options:
    --min_pixels N   skip objects smaller than N pixels (default: 1)
"""

import argparse
import sys
import numpy as np
from PIL import Image
from pathlib import Path
import skimage.measure
import tqdm

MASK_SUFFIXES = {'.png', '.tif', '.tiff'}


def crop_objects_from_mask(mask_path: Path, output_dir: Path, min_pixels: int = 1) -> int:
    with Image.open(mask_path) as img:
        arr = np.array(img)

    # Binarize: any non-zero pixel is foreground
    binary = (arr > 0).astype(np.uint8)

    # Label each connected component
    labeled = skimage.measure.label(binary, connectivity=2)
    regions = skimage.measure.regionprops(labeled)

    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for region in regions:
        if region.area < min_pixels:
            continue

        minr, minc, maxr, maxc = region.bbox

        # Isolate this object only — zero out every other label in the crop
        crop = (labeled[minr:maxr, minc:maxc] == region.label).astype(np.uint8)

        saved += 1
        out_path = output_dir / f"{mask_path.stem}_c{saved}.png"
        # Save as a standard 8-bit PNG (0 = background, 255 = object)
        Image.fromarray(crop * 255).save(out_path)

    return saved


def collect_masks(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in MASK_SUFFIXES:
            print(f"Warning: {input_path.name} does not look like a mask image", file=sys.stderr)
        return [input_path]
    return sorted(f for f in input_path.rglob("*") if f.suffix.lower() in MASK_SUFFIXES)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Crop individual objects from binary mask images (.png/.tiff). "
            "Each connected component is saved as a separate tightly-cropped PNG."
        )
    )
    parser.add_argument(
        "input", type=Path,
        help="Binary mask image or directory of mask images.",
    )
    parser.add_argument(
        "output_dir", type=Path,
        help="Root directory where cropped object images are saved.",
    )
    parser.add_argument(
        "--min_pixels", type=int, default=1,
        help="Minimum object size in pixels to be saved (default: 1).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input not found: {args.input}")

    masks = collect_masks(args.input)
    if not masks:
        print(f"No mask images found under {args.input}", file=sys.stderr)
        sys.exit(1)

    total_crops = 0
    for mask_path in tqdm.tqdm(masks, desc="Cropping objects"):
        # Preserve subdirectory structure when input is a directory
        if args.input.is_dir():
            rel = mask_path.relative_to(args.input)
            out_dir = args.output_dir / rel.parent
        else:
            out_dir = args.output_dir

        n = crop_objects_from_mask(mask_path, out_dir, args.min_pixels)
        tqdm.tqdm.write(f"{mask_path.name}: {n} object(s)")
        total_crops += n

    print(f"\nTotal: {total_crops} crop(s) saved to {args.output_dir}")


if __name__ == "__main__":
    main()
