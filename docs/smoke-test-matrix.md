# Smoke-Test Matrix

Production-ready baseline for v1.0.0. The matrix tracks node-level workflow
scaffolds, JH real-ComfyUI smoke status, screenshot coverage, and originating
batch. Workflow screenshot PNG files live next to their JSON scaffold under
`workflows/`.

| Node | Public node / coverage | Workflow scaffold | JH smoke status | Screenshot committed | Batch |
|---|---|---|---|---|---|
| N-01 | Frequency Separation | `workflows/S-01-frequency-separation.json` | PASS | `workflows/S-01-frequency-separation-screenshot.png` | Batch 1 |
| N-02 | Sub-Pixel Mask Refiner | `workflows/S-02-subpixel-mask-refiner.json` | PASS | `workflows/S-02-subpixel-mask-refiner-screenshot.png` | Batch 1 |
| N-03 | Edge-Aware Skin Smoother | `workflows/S-03-edge-aware-smoother.json` | PASS | `workflows/S-03-edge-aware-smoother-screenshot.png` | Batch 1 |
| N-04 | High-Frequency Detail Masker | `workflows/S-04-hf-detail-masker.json` | PASS | `workflows/S-04-hf-detail-masker-screenshot.png` | Batch 2 |
| N-05 | Luminosity Masking | `workflows/S-05-luminosity-masking.json` | PASS | `workflows/S-05-luminosity-masking-screenshot.png` | Batch 2 |
| N-06 | Landmark Facial Aligner | `workflows/S-06-facial-aligner.json` | PASS | `workflows/S-06-facial-aligner-screenshot.png` | Batch 2 |
| N-07 | Lens Distortion Corrector | `workflows/S-07-lens-distortion.json` | PASS | `workflows/S-07-lens-distortion-screenshot.png` | Batch 2 |
| N-08 | Color Matcher (LAB) | `workflows/S-08-color-matcher.json` | PASS | `workflows/S-08-color-matcher-screenshot.png` | Batch 3 |
| N-09 | Tone Curve (RGB) | `workflows/S-09-tone-curve.json` | PASS | `workflows/S-09-tone-curve-screenshot.png` | Batch 3 |
| N-10 | Face Detect | `workflows/S-10-face-detect.json` | PASS | `workflows/S-10-face-detect-screenshot.png` | Batch 4 |
| N-11 | Unwrap Face | `workflows/S-11-unwrap-face.json` | PASS | `workflows/S-11-unwrap-face-screenshot.png` | Batch 4 |
| N-12 | HALD Identity | `workflows/S-14-lut-export.json` | PASS | `workflows/S-14-lut-export-screenshot.png` | Batch 5 |
| N-13 | LUT Export (.cube) | `workflows/S-14-lut-export.json` | PASS | `workflows/S-14-lut-export-screenshot.png` | Batch 5 |
| N-14 | LUT Import (.cube) | `workflows/S-15-lut-import.json` | PASS | `workflows/S-15-lut-import-screenshot.png` | Batch 5 |
| N-15 | Hue/Saturation per Range | `workflows/S-16-selective-color.json` | PASS | `workflows/S-16-selective-color-screenshot.png` | Batch 6 |
| N-16 | Saturation Mask Builder | `workflows/S-16-selective-color.json` | PASS | `workflows/S-16-selective-color-screenshot.png` | Batch 6 |
| N-17 | Tone Match LUT | `workflows/S-17-tone-match-lut.json` | PASS | `workflows/S-17-tone-match-lut-screenshot.png` | Batch 6 |
| N-18 | Color Balance selective-color core coverage | `workflows/S-16-selective-color.json` | PASS (core-covered, no standalone public wrapper) | `workflows/S-16-selective-color-screenshot.png` | Batch 6 |
| N-19 | Face Landmarks (MediaPipe 468) | `workflows/S-18-face-pipeline-v2.json` | PASS | `workflows/S-18-face-pipeline-v2-screenshot.png` | Batch 6 |
| N-20 | Face Warp (Delaunay per-triangle) | `workflows/S-18-face-pipeline-v2.json` | PASS | `workflows/S-18-face-pipeline-v2-screenshot.png` | Batch 6 |
| N-21 | Face Beauty Blend | `workflows/S-18-face-pipeline-v2.json` | PASS | `workflows/S-18-face-pipeline-v2-screenshot.png` | Batch 6 |
| N-22 | Look Select Preset | `workflows/S-19-look-select-single.json`, `workflows/S-20-look-select-compare-6up.json` | PASS | `workflows/S-19-look-select-single-screenshot.png`, `workflows/S-20-look-select-compare-6up-screenshot.png` | Batch 8 |
| N-23 | ColorLab (ACR) | `workflows/S-21-colorlab-basic-only.json`, `workflows/S-22-colorlab-hsl-teal-orange.json`, `workflows/S-23-colorlab-color-grading-cinematic.json`, `workflows/S-24-colorlab-gray-mix-bw.json`, `workflows/S-25-colorlab-full-acr-preset.json` | PASS for Basic; preset variants pending visual review | `workflows/S-21-colorlab-basic-only-screenshot.png` | Batch 9 |
| N-24 | Layer Stack Start | `workflows/S-26-layer-compositing-2layer-overlay.json`, `workflows/S-27-layer-compositing-5layer-cinematic.json`, `workflows/S-28-layer-compositing-group-clipping.json` | PASS | `workflows/S-26-layer-compositing-2layer-overlay-screenshot.png`, `workflows/S-27-layer-compositing-5layer-cinematic-screenshot.png`, `workflows/S-28-layer-compositing-group-clipping-screenshot.png` | Batch 9 |
| N-25 | Layer Add | `workflows/S-26-layer-compositing-2layer-overlay.json`, `workflows/S-27-layer-compositing-5layer-cinematic.json`, `workflows/S-28-layer-compositing-group-clipping.json` | PASS | `workflows/S-26-layer-compositing-2layer-overlay-screenshot.png`, `workflows/S-27-layer-compositing-5layer-cinematic-screenshot.png`, `workflows/S-28-layer-compositing-group-clipping-screenshot.png` | Batch 9 |
| N-26 | Layer Group | `workflows/S-28-layer-compositing-group-clipping.json` | PASS | `workflows/S-28-layer-compositing-group-clipping-screenshot.png` | Batch 9 |
| N-27 | Layer Flatten | `workflows/S-26-layer-compositing-2layer-overlay.json`, `workflows/S-27-layer-compositing-5layer-cinematic.json`, `workflows/S-28-layer-compositing-group-clipping.json` | PASS | `workflows/S-26-layer-compositing-2layer-overlay-screenshot.png`, `workflows/S-27-layer-compositing-5layer-cinematic-screenshot.png`, `workflows/S-28-layer-compositing-group-clipping-screenshot.png` | Batch 9 |

Notes:

- Rows marked pending are visual-review gaps only; their workflow JSON scaffolds
  are present and loadable.
- N-18 is tracked as selective-color core coverage because the v1.0 public pack
  exposes 26 nodes while the batch planning index reserves N-18 for the same
  color-balance primitive family.
