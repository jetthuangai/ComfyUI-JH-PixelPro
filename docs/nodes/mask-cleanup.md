---
title: N-41 Mask Cleanup
description: Remove specks, fill small holes and feather a mask by distance.
---

# N-41 Mask: Cleanup

`JHPixelProMaskCleanup` (display name "Mask: Cleanup") tidies a segmentation mask. It deletes tiny stray blobs, fills small pin-holes inside the subject, and can turn the hard edge into a smooth ramp whose width is measured in real pixels.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `mask` | Input | `MASK` | Mask to clean. |
| `threshold` | Widget | `FLOAT`, default `0.5`, 0-1 | Pixels `> threshold` are foreground. |
| `min_island_area` | Widget | `INT`, default `64`, 0-1000000 | Foreground blobs with fewer pixels are removed. `0` = off. |
| `max_hole_area` | Widget | `INT`, default `64`, 0-1000000 | Enclosed holes with fewer pixels are filled. `0` = off. |
| `feather_px` | Widget | `FLOAT`, default `0.0`, step `0.5`, 0-256 | Soft edge distance in pixels. `0` = hard black/white output. |
| `connectivity` | Widget | `COMBO`, default `8` | Foreground connectivity: `8` (diagonals connect) or `4` (side neighbours only). Holes use the complementary connectivity, so a diagonal outline still encloses its hole. |
| `mask` | Output | `MASK` | Cleaned mask, float32 `(B,H,W)`. |

## How it works

1. The mask is binarised at `> threshold`.
2. Foreground connected components smaller than `min_island_area` are deleted.
3. Background components smaller than `max_hole_area` are filled, but only if they do not touch the image border. Background that reaches the border is the outside of the picture, never a hole.
4. If `feather_px > 0`, the distance to the cleaned edge is computed on both sides and mapped to `alpha = clip(0.5 + d / (2 * feather_px), 0, 1)`, where `d` is positive inside and negative outside. The 0.5 line stays exactly on the binary edge; alpha reaches 1 at `feather_px` pixels inside and 0 at `feather_px` pixels outside, so the whole ramp is `2 * feather_px` pixels wide.

Components are labelled with OpenCV (`connectedComponentsWithStats`): exact and fast even at 2048 x 2048 on CPU. Items in a batch are processed one by one.

## Tips

- Typical use: place it right after SAM, rembg or any segmentation node, before the mask is used for inpaint, compositing or colour work.
- Area guidance: pixels, not percent. Start at 64 for 1024 px images; use about 256 for 2048 px and 16-32 for 512 px. Raise `min_island_area` until the stray dots disappear but real small parts (earrings, fingers) stay; raise `max_hole_area` to close pin-holes in hair or clothing, but keep it below real gaps such as the space between an arm and the body.
- Use `8` connectivity for natural subjects. Use `4` when thin diagonal bridges should not join separate blobs.
- Chaining: Mask: Cleanup first, then N-02 Mask Refiner (grow/shrink the edge) or N-33 Mask Edge Smoother (clean a jagged edge). If you only need a soft edge, set `feather_px` here instead of adding a blur.
- Feather is a distance ramp, not a Gaussian blur, so thin features stay solid and the edge position does not drift.
