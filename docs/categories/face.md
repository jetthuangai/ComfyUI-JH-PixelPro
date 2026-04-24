# Face

The face category provides the pack's portrait workflow spine: detect, align, edit, unwrap, and blend. It is designed so a single detection step can feed the rest of the face chain with minimal glue.

- **N-10 Face Detect** — Detects 5-point face landmarks and bounding boxes via MediaPipe.
- **N-11 Unwrap Face** — Warps an edited aligned face back onto the original canvas.
- **N-19 Face Landmarks** — Extracts dense 468-point face landmarks for downstream geometry work.
- **N-20 Face Warp** — Performs Delaunay-based per-triangle face warping.
- **N-21 Face Beauty Blend** — Blends a retouched face plate back onto the original image with mask-aware feathering.

Flagship face page: [N-10 Face Detect](../nodes/n10-face-detect.md).
