---
title: N-16 Saturation Mask Builder
description: Build a soft mask from image saturation thresholds.
---

# N-16 Saturation Mask Builder

`JHPixelProSaturationMask` extracts masks from saturation ranges. Use it to isolate vivid regions, protect neutral skin, or drive downstream selective color adjustments.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image` | Input | `IMAGE` | Source image. |
| `sat_min` | Widget | `FLOAT`, default `0.3` | Minimum saturation included. |
| `sat_max` | Widget | `FLOAT`, default `1.0` | Maximum saturation included. |
| `feather` | Widget | `FLOAT`, default `0.1` | Soft edge width around thresholds. |
| `mask` | Output | `MASK` | Saturation-range mask. |

## Workflow JSON

[workflows/S-16-selective-color.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-16-selective-color.json)
