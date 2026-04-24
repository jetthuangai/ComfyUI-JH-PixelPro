---
title: N-15 Hue/Saturation per Range
description: Shift hue and saturation inside a selected hue band.
---

# N-15 Hue/Saturation per Range

`JHPixelProHueSaturationRange` performs selective-color style hue and saturation edits inside a soft hue band. It is the direct color-tuning partner to N-16 Saturation Mask Builder.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image` | Input | `IMAGE` | Source image. |
| `hue_center` | Widget | `FLOAT`, default `0.0` | Hue center in degrees. |
| `band_width` | Widget | `FLOAT`, default `30.0` | Half-width of the hue band. |
| `hue_shift` | Widget | `FLOAT`, default `0.0` | Hue rotation in degrees inside the band. |
| `sat_mult` | Widget | `FLOAT`, default `1.0` | Saturation multiplier. |
| `sat_add` | Widget | `FLOAT`, default `0.0` | Additive saturation offset. |
| `image` | Output | `IMAGE` | Selectively adjusted image. |

## Workflow JSON

[workflows/S-16-selective-color.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-16-selective-color.json)
