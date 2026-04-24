# Face

The face category provides the pack's portrait workflow spine: detect, align, edit, unwrap, and blend. It is designed so a single detection step can feed the rest of the face chain with minimal glue.

- [**N-06 Landmark Facial Aligner**](../nodes/facial-aligner.md) — Aligns a detected face to a canonical square frame.
- [**N-10 Face Detect**](../nodes/n10-face-detect.md) — Detects 5-point face landmarks and bounding boxes via MediaPipe.
- [**N-11 Unwrap Face**](../nodes/unwrap-face.md) — Warps an edited aligned face back onto the original canvas.
- [**N-19 Face Landmarks**](../nodes/face-landmarks.md) — Extracts dense 468-point face landmarks for downstream geometry work.
- [**N-20 Face Warp**](../nodes/face-warp.md) — Performs Delaunay-based per-triangle face warping.
- [**N-21 Face Beauty Blend**](../nodes/face-beauty-blend.md) — Blends a retouched face plate back onto the original image with mask-aware feathering.

Flagship face page: [N-10 Face Detect](../nodes/n10-face-detect.md).
