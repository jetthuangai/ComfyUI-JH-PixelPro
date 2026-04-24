---
title: N-21 Face Beauty Blend
description: Blend a retouched face plate back into a base image.
---

# N-21 Face Beauty Blend

`JHPixelProFaceBeautyBlend` composites a retouched face image over a base image through a mask, strength control, and optional feather. It is the final merge node for face-pipeline edits.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `base` | Input | `IMAGE` | Original or base plate. |
| `retouched` | Input | `IMAGE` | Retouched face plate. |
| `mask` | Input | `MASK` | Blend mask. |
| `strength` | Widget | `FLOAT`, default `1.0` | Overall blend amount. |
| `feather` | Widget | `INT`, default `0` | Gaussian feather radius in pixels. |
| `blended` | Output | `IMAGE` | Final blended image. |

## Workflow JSON

[workflows/S-18-face-pipeline-v2.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-18-face-pipeline-v2.json)
