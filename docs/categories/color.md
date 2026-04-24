# Color

The color category covers grading, masking by color statistics, and LUT round-tripping. It is the pack's bridge between practical retouch controls and reusable show-look workflows.

- **N-05 Luminosity Masking** — Builds shadow, midtone, and highlight masks with a partition-of-unity style split.
- **N-08 Color Matcher (LAB)** — Transfers color statistics from a reference image with optional masked matching.
- **N-09 Tone Curve (RGB)** — Applies master or per-channel Catmull-Rom tone curves.
- **N-12 HALD Identity** — Generates identity HALD images for LUT authoring pipelines.
- **N-13 LUT Export (.cube)** — Writes Adobe Cube 1.0 LUTs from graded HALD images.
- **N-14 LUT Import (.cube)** — Applies external or round-tripped LUTs with optional masked blending.
- **N-15 Hue/Saturation per Range** — Isolates a hue band and shifts hue or saturation inside that band only.
- **N-16 Saturation Mask Builder** — Creates masks from HLS saturation thresholds.
- **N-17 Tone Match LUT** — Derives a reusable LUT from a graded reference frame.
- **N-23 ColorLab (ACR)** — Packs an Adobe Camera Raw style grading stack into one node.

Lean-batch flagship pages: [N-29 Alpha Matte Extractor](../nodes/n29-alpha-matte-extractor.md), [N-33 Mask Edge Smoother](../nodes/n33-mask-edge-smoother.md), and [N-10 Face Detect](../nodes/n10-face-detect.md).
