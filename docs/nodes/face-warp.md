---
title: N-20 Face Warp
description: Warp faces with Delaunay per-triangle geometry.
---

# N-20 Face Warp

`JHPixelProFaceWarp` warps an image from source landmarks to destination landmarks using triangle-wise affine transforms. It is the deterministic geometry node for controlled face-shape adjustments.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image` | Input | `IMAGE` | Image to warp. |
| `src_landmarks` | Input | `LANDMARKS` | Source dense landmark set. |
| `dst_landmarks` | Input | `LANDMARKS` | Destination dense landmark set. |
| `warped` | Output | `IMAGE` | Warped face image. |

## Workflow JSON

[workflows/S-18-face-pipeline-v2.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-18-face-pipeline-v2.json)
