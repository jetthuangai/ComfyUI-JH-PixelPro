---
title: N-39 Geometry Auto Align
description: Align an image to a reference frame with feature matching, RANSAC and a perspective warp.
---

# N-39 Geometry: Auto Align

`JHPixelProAutoAlign` finds the same points in two images, fits a transform that maps one onto the other, and warps the image into the reference frame. Use it to line up exposure brackets, before/after retouching pairs, or any shot that must match a reference before [Color Matcher](color-matcher.md) or a layer blend.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image` | Input | `IMAGE` | Image to warp. |
| `reference` | Input | `IMAGE` | Target frame; the output has this size. Batch 1, or the same batch as `image`. |
| `method` | Widget | `COMBO`, default `sift` | `sift` (no download) or `loftr` (learned matcher, ~45 MB download on first use). |
| `model` | Widget | `COMBO`, default `homography` | `homography` (perspective) or `affine` (rotate, scale, shear, shift). |
| `max_side` | Widget | `INT`, default `1024` (widget 256-2048, step 64; the function accepts 64-4096) | Longest side used for matching. The final warp uses full resolution. |
| `ransac_threshold` | Widget | `FLOAT`, default `3.0` (0.5-20) | Largest error, in pixels at matching size, for a match to count as agreeing. |
| `min_inliers` | Widget | `INT`, default `12` (4-500) | Fewer agreeing matches than this triggers the fallback. |
| `aligned` | Output | `IMAGE` | `image` warped to the reference size (bilinear). |
| `valid_mask` | Output | `MASK` | 1 where the warped image has real pixels, 0 where the frame is empty. |
| `matrix_json` | Output | `STRING` | JSON list of 3x3 matrices (one per batch item), image to reference pixel coordinates at full resolution. |
| `inliers` | Output | `INT` | Agreeing-match count of the **first batch item only**. For a batch, check the other items by their matrices in `matrix_json` (a pure scale matrix means that item fell back). |

## How it works

1. Both images become grayscale copies, shrunk so the long side is at most `max_side`.
2. Matching: `sift` extracts SIFT keypoints and keeps mutual nearest-neighbour matches. `loftr` runs the LoFTR network, which also works on blurry or low-texture pairs.
3. RANSAC repeatedly fits the chosen model on random match subsets and keeps the fit that most matches agree with. The inlier matches are then refitted by least squares for sub-pixel accuracy, and the matrix is scaled back to full resolution (pixel-centre convention).
4. The image is warped into the reference frame. `valid_mask` is the same warp applied to an all-ones image.

If a pair has too few matches or fewer than `min_inliers` agree, the node does not stop. It logs a warning, resizes the image to the reference size (matrix is a pure scale), returns an all-ones `valid_mask`, and reports the real inlier count. Check `inliers` to know whether the alignment is trustworthy.

## Tips

- **sift or loftr:** start with `sift`; it is fast and needs no download. Switch to `loftr` when SIFT reports few inliers (big lighting change, soft focus, repeated textures). The first LoFTR run downloads about 45 MB.
- **homography or affine:** use `affine` when the camera only rotated or shifted in a flat scene, or when the homography looks warped. Use `homography` for real perspective change.
- **valid_mask:** after a rotation or shift the edges of `aligned` are black. Use `valid_mask` to crop, to feather, or to limit the area that Color Matcher or an exposure merge reads statistics from.
- **Before Color Matcher:** align first, then match colour, so both images compare the same content.
- Moving subjects (people, leaves) are ignored by RANSAC as long as the background has enough texture.
- Raise `ransac_threshold` for slightly blurred pairs; lower it when you need a tight fit.
