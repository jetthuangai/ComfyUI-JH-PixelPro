# Color

The color category covers grading, masking by color statistics, and LUT round-tripping. It is the pack's bridge between practical retouch controls and reusable show-look workflows.

- [**N-05 Luminosity Masking**](../nodes/luminosity-masking.md) — Builds shadow, midtone, and highlight masks with a partition-of-unity style split.
- [**N-08 Color Matcher (LAB)**](../nodes/color-matcher.md) — Transfers color statistics from a reference image with optional masked matching.
- [**N-09 Tone Curve (RGB)**](../nodes/tone-curve.md) — Applies master or per-channel Catmull-Rom tone curves.
- [**N-12 HALD Identity**](../nodes/hald-identity.md) — Generates identity HALD images for LUT authoring pipelines.
- [**N-13 LUT Export (.cube)**](../nodes/lut-export.md) — Writes Adobe Cube 1.0 LUTs from graded HALD images.
- [**N-14 LUT Import (.cube)**](../nodes/lut-import.md) — Applies external or round-tripped LUTs with optional masked blending.
- [**N-15 Hue/Saturation per Range**](../nodes/hue-saturation-range.md) — Isolates a hue band and shifts hue or saturation inside that band only.
- [**N-16 Saturation Mask Builder**](../nodes/saturation-mask.md) — Creates masks from HLS saturation thresholds.
- [**N-17 Tone Match LUT**](../nodes/tone-match-lut.md) — Derives a reusable LUT from a graded reference frame.
- [**N-22 Look Select**](../nodes/look-select.md) — Applies the pack's JSON-backed creative looks from a single dropdown node.
- [**N-23 ColorLab (ACR)**](../nodes/color-lab.md) — Packs an Adobe Camera Raw style grading stack into one node.
- [**N-34 Preset Pack LUT**](../nodes/lut-preset.md) — Applies bundled generic Adobe Cube LUT presets from a dropdown.

Flagship pages: [N-29 Alpha Matte Extractor](../nodes/n29-alpha-matte-extractor.md), [N-33 Mask Edge Smoother](../nodes/n33-mask-edge-smoother.md), and [N-10 Face Detect](../nodes/n10-face-detect.md).
