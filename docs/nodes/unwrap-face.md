---
title: N-11 Unwrap Face
description: Warp an edited aligned face back onto the original canvas.
---

# N-11 Unwrap Face

`JHPixelProUnwrapFace` consumes the inverse matrix from N-06 Landmark Facial Aligner and composites an edited aligned crop back into the original image. It closes the detect-align-edit-unwrap face chain.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `edited_aligned` | Input | `IMAGE` | Edited aligned face crop. |
| `original_image` | Input | `IMAGE` | Original full-canvas image. |
| `inverse_matrix_json` | Widget/Input | `STRING`, default identity matrix JSON | Inverse transform from N-06. |
| `feather_radius` | Widget | `FLOAT`, default `16.0` | Edge feather radius for the generated mask. |
| `mask_override` | Optional input | `MASK` | Optional custom blend mask. |
| `image_composited` | Output | `IMAGE` | Full-canvas composited image. |
| `mask_used` | Output | `MASK` | Mask used for compositing. |

## Workflow JSON

[workflows/S-11-unwrap-face.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-11-unwrap-face.json)
