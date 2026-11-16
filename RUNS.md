# iNDI HCS pipeline: run history

All runs live under `/data/kelpschdj/iNDI/Production/outputs/run_<RUN_ID>/`. Each run mints one run ID in stage 0 (`image_metadata`) and reuses it through `nucleus_segmentation`, `roi_filtering`, `organelle_segmentation` (stage 3) and the QC stages. Stage 3 outputs are versioned by a stamp in the filename: `<exp>_organelle_features__<stamp>.parquet`.

## Summary

| Run ID | Panel | Experiments | Created | Stage 3 stamp(s) | Status |
|---|---|---|---|---|---|
| `20260803_124616_6otz` | 1 | 21 | 2026-08-03 | `20260803_171522` | Archived; updated to frame-level organelle segmentation and removal of organelle masks that touch the ROI boundary |
| `20260901_155524_uoe5` | 1 | 21 | 2026-09-01 | `20260902_114905` | Archived; Golgi body often lost due to touching the boundary, updating to mask clipping at the ROI boundary |
| `20260921_131450_n3r6` | 2 | 13 | 2026-09-21 | `20261113_142458` (10 exp), `20261116_110411` (3 exp) | Archived; Golgi body often lost due to touching the boundary, updating to mask clipping at the ROI boundary |
| `20261116_164909_nu3h` | 1 | 21 | UPDATE | UPDATE | Active |
| `20261116_165635_xanf` | 2 | 13 | UPDATE | UPDATE | Active |

Panel 1 stains: TOMM20 (Alexa 488), EEA1 (568), LAMP1 (647), plus DAPI. Panel 2 stains: RAB11A (Alexa 488), GM130 (568), TUJ1 (647), plus DAPI.