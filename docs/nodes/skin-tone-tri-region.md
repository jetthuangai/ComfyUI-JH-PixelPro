---
title: N-35 Skin Tone Tri-Region
description: Split skin tones into shadow, midtone, and highlight masks.
---

# N-35 Skin Tone Tri-Region

`JHPixelProSkinToneTriRegion` splits a portrait into skin-tone shadow, midtone, and highlight masks using Rec.601 luminance. An optional upstream skin mask can constrain the split to detected skin only, and the output masks are normalized so their sum preserves the selected skin region.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image` | Input | `IMAGE` | Source image. |
| `skin_mask` | Optional input | `MASK` | Optional skin-region gate. If omitted, the whole frame is split. |
| `shadow_cutoff` | Widget | `FLOAT`, default `0.33` | Luminance values below this cutoff are treated as shadow tones. |
| `highlight_cutoff` | Widget | `FLOAT`, default `0.66` | Luminance values above this cutoff are treated as highlight tones. |
| `soft_sigma` | Widget | `FLOAT`, default `1.0` | Gaussian sigma for smoother tone-region boundaries. |
| `shadow_mask` | Output | `MASK` | Shadow-region mask. |
| `midtone_mask` | Output | `MASK` | Midtone-region mask. |
| `highlight_mask` | Output | `MASK` | Highlight-region mask. |

## Workflow JSON

[workflows/N-35-skin-tone-tri-region.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-35-skin-tone-tri-region.json)
