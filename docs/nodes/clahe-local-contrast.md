---
title: N-37 Color CLAHE Local Contrast
description: Adaptive local contrast on the LAB lightness channel while keeping colours intact.
---

# N-37 Color: CLAHE Local Contrast

`JHPixelProCLAHE` applies Contrast Limited Adaptive Histogram Equalisation (CLAHE) to the lightness of an image. Dark areas open up and flat areas gain definition, without the colour shifts you get from running CLAHE on RGB channels.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image` | Input | `IMAGE` | Image `(B,H,W,3)` in `[0,1]`. |
| `clip_limit` | Widget | `FLOAT`, default `2.0`, range `0.1`-`40.0` | How much each tile's histogram may be stretched. Higher = stronger contrast and more noise boost. |
| `grid_size` | Widget | `INT`, default `8`, range `2`-`32` | Tiles per axis. Reduced automatically on small images so every tile is at least 80 px; images under 80 px pass through unchanged. |
| `strength` | Widget | `FLOAT`, default `1.0`, range `0`-`1` | Blend between the original (`0`, exact passthrough) and the CLAHE result (`1`). |
| `mask` | Optional input | `MASK` | Limits where the effect applies. Batch 1 broadcasts over the image batch. |
| `image` | Output | `IMAGE` | Enhanced image, float32 in `[0,1]`. |

## How it works

1. The image is converted from RGB to CIE LAB.
2. Only `L*` (divided by 100) goes through CLAHE: the image is cut into `grid_size x grid_size` tiles, each tile gets its own clipped histogram equalisation, and neighbouring tiles are blended bilinearly to avoid seams.
3. The result is rebuilt with the **original** `a*` and `b*`, converted back to RGB and clamped to `[0,1]`.
4. The final image is `lerp(original, enhanced, strength * mask)`.

Because only `L*` changes, hue and chroma stay the same; skin tones and brand colours do not drift. CLAHE runs on 8-bit histograms, so `L*` is quantised to 256 levels.

### Compared with ColorLab clarity

ColorLab clarity is a smooth, radius-based local contrast (a blurred copy is subtracted) that boosts mid-frequency edges and never remaps tones. CLAHE is histogram-based: it redistributes brightness inside each tile, so it can lift deep shadows and flatten large soft gradients that clarity leaves alone. Use clarity for a gentle punch, CLAHE when a region is too flat, too dark, or too hazy.

## Tips

- Start with `clip_limit` 1.5-3.0. Above ~6 noise and grain are amplified quickly.
- `grid_size` 4-6 gives a broad, natural lift; 12-24 gives a crunchy, detail-heavy look. On small images the grid is reduced automatically so each tile stays at least 80 px (small tiles bias the result).
- Lower `strength` (0.3-0.6) is usually enough for portraits.
- Use `mask` to keep faces or skies out of the effect, or to drive it only into shadows.
- Flat colour areas stay flat (within 8-bit quantisation), so solid backgrounds are not tinted.

## Workflow JSON

[workflows/N-37-clahe-local-contrast.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-37-clahe-local-contrast.json) — load it in ComfyUI after copying `workflows/sample_portrait.jpg` into `ComfyUI/input/`.

![N-37-clahe-local-contrast result](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/N-37-clahe-local-contrast-preview.png)
