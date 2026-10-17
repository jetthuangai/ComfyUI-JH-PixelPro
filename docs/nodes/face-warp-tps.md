---
title: N-42 Face Warp (TPS smooth)
description: Warp a face from source to destination landmarks with one smooth thin-plate-spline field.
---

# N-42 Face Warp (TPS smooth)

`JHPixelProFaceWarpTPS` moves image content so that features at `src_landmarks` end up at `dst_landmarks`, like [Face Warp](face-warp.md) (N-20), but with a single smooth thin-plate-spline (TPS) displacement field instead of one affine patch per Delaunay triangle. Use it for subtle reshaping (slimming, eye or jaw adjustments) where triangle seams or kinks would show.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image` | Input | `IMAGE` | Image to warp, `(B,H,W,3)`. |
| `src_landmarks` | Input | `LANDMARKS` | Where the features are now (from Face Landmarks). |
| `dst_landmarks` | Input | `LANDMARKS` | Where the features should go. Batch 1 broadcasts. |
| `max_points` | Widget | `INT`, default `468`, 16 to 468 | Max landmark pairs used for the spline. Pairs that move more than 0.5 px are always kept (even beyond the budget); the rest is filled half with the unmoved pairs nearest to the edit and half with evenly spaced ones. Keep 468 unless speed matters: fewer pins let nearby features drift. |
| `anchor_border` | Widget | `BOOLEAN`, default `True` | Pin fixed points along the image border (corners plus 3 per edge) so the background stays put. |
| `smoothness` | Optional widget | `FLOAT`, default `1.0`, 0 to 20 | Spline regularisation. `0` lands every landmark exactly (hand-edited landmarks). `1` absorbs detection noise when the target landmarks come from another photo, which would otherwise ripple or fold the image; clean edits still land within ~1 px. Raise it if ripples remain. It also relaxes the border anchors slightly (well under 1 px at the default). |
| `warped` | Output | `IMAGE` | Warped image, float32 in `[0,1]`. |

Landmarks follow the same convention as N-20: normalised `[0,1]` or pixel coordinates, `(B,468,2)` or `(B,F,468,2)` (first face is used). Non-finite points are dropped; with fewer than 3 valid pairs, or when one side has no face at all (all NaN), the image is returned unchanged.

## How it works

The spline is fitted on the output-to-input mapping (kernel `r^2 log r` plus an affine part), so every output pixel samples the source at the spot the spline points to and a feature at `src` lands on `dst` (exactly with `smoothness = 0`). The field is evaluated exactly (in memory-bounded blocks) on a grid with a 4 px step, upsampled bicubically to full size, then used to resample the image bilinearly. On a real 1536x2048 portrait with 468 landmarks plus 16 border anchors, with `smoothness = 0` control points land within about 0.6 px of their destination (independent review: max 0.57 px at landmark pixels, p99 0.11 px over the frame; the spline itself is exact to 1e-4 px). Landmarks are subsampled to `max_points` (deterministic) only when you lower it: every moved pair is kept, so a jaw or eye edit is never dropped. With `anchor_border` on, extra pairs with identical source and destination pin the border.

TPS vs Delaunay (N-20):

| | Delaunay (N-20) | TPS (N-42) |
|---|---|---|
| Field | Piecewise affine, one patch per triangle | One smooth global field |
| Seams / kinks | Possible on triangle edges for large moves | None |
| Outside the landmarks | Unchanged | Smoothly decays only if `anchor_border` is on |
| Cost | Grows with triangle count, CPU (OpenCV) | Grows with point count cubed for the solve, runs on GPU |
| Follows every landmark | Exactly | Exactly for the chosen points; a long-range effect between them |

Choose TPS for gentle, natural deformations and when you want GPU speed; choose Delaunay when you need the change confined strictly to the face mesh or want every one of the 468 points honoured locally.

## Tips

- Target landmarks from a second detection (another photo, a zoomed copy) differ from the source by detector noise as well as by the real shape change — on the sample portrait the noise is about 13–17 px (median) for a 3–6% zoomed copy. Exact interpolation of that noise folds the image; keep `smoothness` at 1 or higher for such targets.

- Leave `max_points` at 468 (the solve is only about 484x484). Lower it only for speed; moved landmarks are always kept, but fewer unmoved points means less pinning of the rest of the face.
- Leave `anchor_border` on unless the whole frame should move with the face (pure translation or scale).
- Large moves of a few isolated points create wide, rubbery effects; move groups of neighbouring landmarks together.
- Measured on an RTX 5090 / desktop CPU, 1536x2048 with 468 points plus anchors (whole node incl. resampling): about 19 ms on CUDA (peak about 480 MiB, mostly image buffers), about 0.85 s on CPU.

## Workflow JSON

[workflows/N-42-face-warp-tps.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-42-face-warp-tps.json) — load it in ComfyUI after copying `workflows/sample_portrait.jpg` into `ComfyUI/input/`.

![N-42-face-warp-tps result](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/N-42-face-warp-tps-preview.png)
