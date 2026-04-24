---
title: N-05 Luminosity Masking
description: Build shadow, midtone, and highlight masks from image luminance.
---

# N-05 Luminosity Masking

`JHPixelProLuminosityMasking` creates Photoshop-style luminosity masks with a soft partition across shadows, midtones, and highlights. It is a color-grade and dodge/burn helper for tonal selections.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image` | Input | `IMAGE` | Source image. |
| `luminance_source` | Widget | `COMBO`, default `lab_l` | Uses `lab_l`, `ycbcr_y`, or `hsv_v` as luminance basis. |
| `shadow_end` | Widget | `FLOAT`, default `0.33` | Upper edge of the shadow band. |
| `highlight_start` | Widget | `FLOAT`, default `0.67` | Lower edge of the highlight band. |
| `soft_edge` | Widget | `FLOAT`, default `0.1` | Smooth transition width between tonal bands. |
| `mask_shadows` | Output | `MASK` | Shadow selection mask. |
| `mask_midtones` | Output | `MASK` | Midtone selection mask. |
| `mask_highlights` | Output | `MASK` | Highlight selection mask. |

## Workflow JSON

[workflows/S-05-luminosity-masking.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-05-luminosity-masking.json)
