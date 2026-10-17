---
title: Color Matcher Region
description: Region-aware LAB color matching between two images of any size.
---

# Color Matcher Region

`JHPixelProColorMatcherRegion` is the region-aware variant of [N-08 Color Matcher](color-matcher.md). It transfers LAB color statistics from one part of a reference image onto one part of a target image, for example matching the skin of a generated face to the skin of the original plate, or the sky of one shot to the sky of another.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image_target` | Input | `IMAGE` | Image to be recolored. |
| `image_reference` | Input | `IMAGE` | Image that provides the color statistics. It may have a different size than the target. |
| `channels` | Widget | `COMBO`, default `ab` | `ab` matches chroma only and keeps the target luminance; `lab` also matches L* for a full tone transfer. |
| `strength` | Widget | `FLOAT`, default `1.0` | Blend amount. `0` bypasses the node, `1` applies the full match. |
| `target_mask` | Optional input | `MASK` | Target region. It defines the target statistics and where the result is applied. Unconnected = the whole target. |
| `reference_mask` | Optional input | `MASK` | Reference sampling region. Unconnected = the whole reference. |
| `image_matched` | Output | `IMAGE` | Color-matched image. |

## How it works

1. Both images are converted to CIE LAB.
2. Mean and standard deviation of the selected channels are measured inside each mask (weighted by the soft mask value).
3. The target channels are shifted and scaled so their statistics match the reference region (Reinhard transfer).
4. The result is converted back to RGB, blended by `strength`, and composited into the target only where `target_mask` is set.

Because only statistics are compared, the two images do not need the same size or framing.

## Tips

- Use `channels = ab` for skin and product color: brightness and texture stay untouched.
- Build the masks with [N-35 Skin Tone Tri-Region](skin-tone-tri-region.md), [Saturation Mask](saturation-mask.md) or the mask category nodes.
- Soft masks give soft transitions; refine hard selections with [N-02 Sub-Pixel Mask Refiner](sub-pixel-mask-refiner.md) first.
- Use [N-08 Color Matcher](color-matcher.md) when you want to match the whole frame.
