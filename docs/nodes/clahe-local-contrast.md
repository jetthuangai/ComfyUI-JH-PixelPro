---
title: CLAHE Local Contrast
description: Adaptive local contrast on lightness only, so dark and flat areas gain definition while colours stay put.
---

# CLAHE Local Contrast

Opens up dark areas and adds definition to flat ones by adjusting contrast tile by tile, on lightness only, so hue and colour do not shift.

![CLAHE Local Contrast — before / after](../assets/nodes/clahe-local-contrast.jpg)

| | |
|---|---|
| **In ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → color → Color: CLAHE Local Contrast` |
| **Good for** | Lifting murky shadows, rescuing hazy or flat shots, adding crunch to product and texture detail |

## When to use it

CLAHE (Contrast Limited Adaptive Histogram Equalisation) looks at small tiles of the picture and spreads out the brightness values inside each one, with a limit so noise is not blown up. The result feels like strong local contrast: dark corners gain detail and flat surfaces gain depth.

Here it runs only on the lightness channel of the CIE LAB colour space, so colours are not touched. Running it on RGB channels, as many tools do, shifts colours and tints skin.

Compared with the `basic_clarity` slider in [ColorLab](color-lab.md): clarity is a gentle, radius-based punch for edges. CLAHE redistributes brightness inside each tile, so it can lift deep shadows and flatten large soft gradients that clarity leaves alone. Use clarity for gentle punch, CLAHE when a region is too flat, too dark or too hazy.

## Quick start

1. **Load Image** → CLAHE Local Contrast → **Preview Image**.
2. Leave `clip_limit` at 2.0 and `grid_size` at 8.
3. Lower `strength` to about 0.5 for portraits, then compare with the original.

## Settings

| Setting | Default | What it does | Tip |
|---|---|---|---|
| `image` | required | The picture to enhance. | |
| `clip_limit` | 2.0 (0.1 to 40) | How strongly each tile may be stretched. Higher means more contrast and more noise. | 1.5 to 3 for most work; above about 6 grain jumps out. |
| `grid_size` | 8 (2 to 32) | Number of tiles across and down. Fewer tiles give a broad, natural lift; more tiles give a crunchy, detailed look. | 4 to 6 natural, 12 to 24 gritty. |
| `strength` | 1.0 (0 to 1) | Blend between the original (0) and the CLAHE result (1). | 0.3 to 0.6 for portraits. |
| `mask` (optional) | none | Limits where the effect applies. A single mask is used for the whole batch. | Keep faces or skies out of the effect. |

**Output:** `image`, the enhanced picture.

On small images the grid is reduced automatically so that every tile is at least 80 px wide. Images smaller than 80 px come back unchanged.

## Tips from retouching

- Start gentle. `clip_limit` 2 at `strength` 0.5 is often better than 4 at full strength.
- For portraits, mask out the face and apply it to clothing, hair or background only; skin rarely likes this much local contrast.
- If shadows turn grainy, lower `clip_limit` first, then `strength`.
- Combine with [Smart Sharpen](smart-sharpen.md) afterwards for crisp edges, and apply [Tone Curve](tone-curve.md) last for the final contrast shape.
- Flat colour areas stay flat, so solid backgrounds do not get tinted.

??? note "Under the hood"
    The image is converted to CIE LAB (Kornia). Only `L*` (divided by 100) goes through `kornia.enhance.equalize_clahe` on a `grid_size × grid_size` tile grid; the original `a*` and `b*` are kept, the result is converted back to RGB and clamped to 0..1. The final image is `lerp(original, enhanced, strength × mask)`. Kornia works with 256-bin histograms, so `L*` is quantised to 256 levels. Its clip threshold biases very small tiles (a flat image can drift brighter), which is why tiles are kept at 80 px or more. `strength = 0` returns the original.

## Related nodes

- [ColorLab](color-lab.md): clarity, texture and dehaze in a Camera Raw style panel
- [Smart Sharpen](smart-sharpen.md): edge sharpening after the contrast lift
- [Tone Curve](tone-curve.md): shape global contrast
- [Luminosity Masking](luminosity-masking.md): restrict the effect to shadows or highlights

## Example workflow

[CLAHE Local Contrast example workflow](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-37-clahe-local-contrast.json). Copy `workflows/sample_portrait.jpg` into `ComfyUI/input/` first.
