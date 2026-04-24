---
title: N-07 Lens Distortion Corrector
description: Correct or simulate Brown-Conrady lens distortion.
---

# N-07 Lens Distortion Corrector

`JHPixelProLensDistortion` corrects wide-angle distortion or simulates lens distortion using Brown-Conrady coefficients. It is useful before face detection, alignment, or portrait geometry cleanup.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image` | Input | `IMAGE` | Source image. |
| `preset` | Widget | `COMBO`, default `no_op_identity` | Built-in presets plus `custom`. |
| `direction` | Widget | `COMBO`, default `inverse` | `inverse` rectifies a source; `forward` simulates distortion. |
| `k1`, `k2`, `k3` | Widget | `FLOAT`, default `0.0` | Radial distortion coefficients for custom mode. |
| `p1`, `p2` | Widget | `FLOAT`, default `0.0` | Tangential distortion coefficients for custom mode. |
| `image_rectified` | Output | `IMAGE` | Corrected or distorted output image. |

## Workflow JSON

[workflows/S-07-lens-distortion.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-07-lens-distortion.json)
