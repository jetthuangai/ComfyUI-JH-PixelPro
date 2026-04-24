---
title: N-19 Face Landmarks
description: Extract dense MediaPipe face landmarks and an optional overlay.
---

# N-19 Face Landmarks

`JHPixelProFaceLandmarks` extracts dense 468-point face landmarks for geometry-driven face edits. It returns a custom `LANDMARKS` object that feeds N-20 Face Warp.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image` | Input | `IMAGE` | Source face image. |
| `max_num_faces` | Widget | `INT`, default `1` | Maximum faces to detect per image. |
| `min_detection_confidence` | Widget | `FLOAT`, default `0.5` | MediaPipe detection threshold. |
| `refine_landmarks` | Widget | `BOOLEAN`, default `true` | Kept for API compatibility; dense output is always used. |
| `draw_overlay` | Widget | `BOOLEAN`, default `true` | Draw landmarks on an overlay preview. |
| `landmarks` | Output | `LANDMARKS` | Dense landmark tensor for downstream geometry. |
| `overlay` | Output | `IMAGE` | Preview image with landmark dots. |

## Workflow JSON

[workflows/S-18-face-pipeline-v2.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-18-face-pipeline-v2.json)
