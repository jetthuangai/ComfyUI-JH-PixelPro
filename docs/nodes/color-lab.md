---
title: N-23 ColorLab
description: Adobe Camera Raw style color controls in one node.
---

# N-23 ColorLab

`JHPixelProColorLab` packs Basic, HSL, Color Grading, and Gray Mix controls into a single finishing node. It is the broad color module for ACR-style adjustments without wiring many separate nodes.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image` | Input | `IMAGE` | Source image. |
| `basic_*` | Widget group | `FLOAT`, default `0.0` | Basic exposure, contrast, highlights, shadows, whites, blacks, texture, clarity, dehaze, vibrance, and saturation controls. |
| `hsl_<color>_*` | Widget group | `FLOAT`, default `0.0` | Hue, saturation, and luminance controls for red, orange, yellow, green, aqua, blue, purple, and magenta. |
| `grade_<range>_*` | Widget group | `FLOAT`, default `0.0` | Shadow, midtone, and highlight grading controls for hue, saturation, luminance, and balance. |
| `gray_enable` | Widget | `BOOLEAN`, default `false` | Enables Gray Mix conversion. |
| `gray_<color>` | Widget group | `FLOAT`, default `0.0` | Per-color grayscale mix controls. |
| `image` | Output | `IMAGE` | ColorLab adjusted image. |

## Workflow JSON

[workflows/S-21-colorlab-basic-only.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-21-colorlab-basic-only.json)
