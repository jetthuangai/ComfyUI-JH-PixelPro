---
title: N-43 Mask Preview Overlay
description: Visualise any MASK on top of an image for debugging, with overlay, heatmap, mask-only and cutout modes.
---

# N-43 Mask: Preview Overlay

`JHPixelProMaskPreview` turns a `MASK` into an `IMAGE` you can look at. Plug it after any mask node in the pack (or any other mask source) to see what the mask really contains before you feed it to matting, inpaint or compositing.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `mask` | Input | `MASK` | Mask to visualise. |
| `image` | Optional input | `IMAGE` | Background. Omit for a neutral grey (overlay, cutout) or the bare colormap (heatmap). |
| `mode` | Widget | `COMBO`, default `overlay` | `overlay`, `heatmap`, `mask_only`, or `cutout`. |
| `color` | Widget | `COMBO`, default `red` | Tint and edge colour: red, green, blue, yellow, cyan, magenta, white, black. |
| `opacity` | Widget | `FLOAT`, default `0.5` | Strength of the tint (overlay) or heatmap blend. |
| `colormap` | Widget | `COMBO`, default `turbo` | Kornia colormap for heatmap mode (list comes from the installed Kornia). |
| `show_edges` | Widget | `BOOLEAN`, default `false` | Draw the 0.5 contour of the mask in `color`, fully opaque. |
| `preview` | Output | `IMAGE` | RGB preview, same size as the mask. |

Image and mask must have the same height and width. A batch of 1 on either side broadcasts to the other.

## How it works

- **overlay**: `lerp(image, color, opacity * mask)`. Soft mask values give a soft tint.
- **heatmap**: the mask value is mapped through the chosen colormap, then blended with the image by `opacity`. Without an image you get the pure colormap.
- **mask_only**: the mask as a greyscale RGB image, exactly as stored.
- **cutout**: `image * mask` over an 8 px grey checkerboard, like a transparency grid.
- **show_edges**: the mask is thresholded at 0.5, and dilation minus erosion (3x3 max-pool) gives a 1-2 px band along the boundary, painted over the result.

## Tips

- Use `overlay` with `show_edges` to check that a selection hugs the subject.
- Use `heatmap` to see soft values: feather width, leftover gray from a threshold, or noise a binary view hides.
- Use `mask_only` for a quick truth view when colours confuse you.
- Use `cutout` to judge the final alpha as if the subject sat on a transparent layer.
- Pick `color` to contrast with the image (for example cyan on warm skin, magenta on foliage).
