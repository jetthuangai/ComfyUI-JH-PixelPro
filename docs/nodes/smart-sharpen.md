---
title: "N-40 Filters: Smart Sharpen"
description: Unsharp-mask sharpening with soft threshold, luminance-only mode and optional mask.
---

# N-40 Filters: Smart Sharpen

`JHPixelProSmartSharpen` sharpens an image with an unsharp mask. It adds back the difference between the image and a blurred copy, so edges gain contrast. A threshold keeps noise and film grain from being boosted, and an optional mask limits the effect to chosen areas.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image` | Input | `IMAGE` | Image to sharpen. |
| `amount` | Widget | `FLOAT`, default `1.0`, 0 to 5 | Strength. `0` returns the image unchanged. |
| `radius` | Widget | `FLOAT`, default `1.0`, 0.3 to 10 | Blur sigma in pixels; sets the size of the detail that gets boosted. |
| `threshold` | Widget | `FLOAT`, default `0.0`, 0 to 0.5 | Detail weaker than this is left alone. |
| `luminance_only` | Widget | `BOOLEAN`, default `true` | Sharpen brightness only; colours are untouched. |
| `mask` | Optional input | `MASK` | Gate for the effect; 1 = full, 0 = none. Batch 1 broadcasts. |
| `image` | Output | `IMAGE` | Sharpened image. |

## How it works

1. A Gaussian blur (sigma = `radius`, reflect borders) gives a soft copy. `detail = image - blur`.
2. The detail is soft-thresholded: values smaller than `threshold` become 0 and larger ones are reduced by `threshold`. This is a smooth ramp, so no speckle appears at the cut-off.
3. Result = `image + amount * detail * mask`, clamped to 0..1.

With `amount = 1`, `threshold = 0` and `luminance_only` off, the result equals Kornia's `unsharp_mask`.

### Why luminance only

Sharpening each RGB channel separately boosts channels by different amounts at coloured edges, which makes bright or dark colour halos (fringes). With `luminance_only` the node converts to YCbCr, sharpens only Y (brightness) and keeps Cb/Cr (colour) as they were, so edges get crisper without changing colour. Turn it off only when you want the per-channel look.

## Tips

- Radius: 0.5 to 1.0 for fine detail (web size, faces); 1.5 to 3 for large prints or soft images; above 5 acts as local contrast, not sharpening.
- Amount: start at 0.5 to 1.0. Above 2 expect halos around strong edges.
- Noisy or grainy source: raise `threshold` to about 0.02 to 0.05 so grain is not amplified while real edges are still sharpened.
- Pair with N-04 High-Frequency Detail Masker: feed its mask into `mask` to sharpen only textured regions (hair, eyes, fabric) and keep skin and sky smooth.
- Radius is limited by image size: the image must be larger than `ceil(3 * radius)` pixels on both sides.

## Workflow JSON

[workflows/N-40-smart-sharpen.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-40-smart-sharpen.json) — load it in ComfyUI after copying `workflows/sample_portrait.jpg` into `ComfyUI/input/`.

![N-40-smart-sharpen result](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/N-40-smart-sharpen-preview.png)
