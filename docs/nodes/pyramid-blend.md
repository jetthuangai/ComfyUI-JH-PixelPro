---
title: N-36 Pyramid Blend
description: Seamless multi-band Laplacian pyramid compositing of two images through a mask.
---

# N-36 Pyramid Blend

`JHPixelProPyramidBlend` ("Compositing: Pyramid Blend") composites `image_b` over `image_a` through a mask using Burt-Adelson multi-band blending. Use it for seamless composites, face or object swaps, and panorama-style stitching where a plain alpha mask leaves a visible seam or exposure step.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image_a` | Input | `IMAGE` | Base image, kept where the mask is 0. |
| `image_b` | Input | `IMAGE` | Overlay image, taken where the mask is 1. |
| `mask` | Input | `MASK` | `1` takes `image_b`, `0` keeps `image_a`. |
| `levels` | Widget | `INT`, default `5`, range 1-10 | Pyramid depth. `1` is a plain alpha composite. |
| `image` | Output | `IMAGE` | Blended image, float32 in `[0,1]`. |

All three inputs must share the same height and width. A batch of 1 on any input broadcasts against the others.

## How it works

1. Build a Laplacian pyramid (band-pass layers plus a low-pass residual) for both images and a Gaussian pyramid for the mask.
2. Blend each band with the mask at the same scale: `band = band_a * (1 - mask) + band_b * mask`.
3. Collapse the blended pyramid back to full resolution.

Fine detail switches over a narrow region while low frequencies (color, exposure) cross-fade over a wide one, so the seam disappears without ghosting. Identical inputs, an all-0 mask, or an all-1 mask reproduce the source exactly.

Any image size works: the images are replicate-padded to a multiple of `2 ** (levels - 1)` and cropped back. The depth is clamped so the coarsest level stays at least 8 px on its short side, so small images silently use fewer levels.

Pair it with masks from the mask category (for example `Mask: Edge-Aware Refiner` or `Mask: Edge Smoother`) for clean subject cut-outs.

## Tips

- `levels` 4-6 suits most 1-4 MP photos; the transition width grows roughly with `2 ** levels` px.
- Use low values (1-3) when you want a tight edge, and high values (7-10) to hide large exposure or color differences between sources.
- A hard binary mask is fine; the pyramid does the feathering. Very soft masks widen the transition further.
- Large `levels` on strongly different sources can leak color across the seam; reduce `levels` if halos appear.

## Workflow JSON

[workflows/N-36-pyramid-blend.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-36-pyramid-blend.json) — load it in ComfyUI after copying `workflows/sample_portrait.jpg` into `ComfyUI/input/`.

![N-36-pyramid-blend result](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/N-36-pyramid-blend-preview.png)
