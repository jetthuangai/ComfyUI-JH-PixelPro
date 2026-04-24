---
title: N-06 Landmark Facial Aligner
description: Align a face to a canonical square frame from five landmarks.
---

# N-06 Landmark Facial Aligner

`JHPixelProFacialAligner` creates an FFHQ-style aligned face crop from five facial landmarks and returns the inverse matrix needed for unwrapping. It is the geometry bridge between face detection and face-edit passes.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image` | Input | `IMAGE` | Source portrait image. |
| `landmarks` | Widget/Input | `STRING`, default 5-point JSON | Landmarks in left eye, right eye, nose, left mouth, right mouth order. |
| `target_size` | Widget | `INT`, default `1024` | Square aligned output size. |
| `padding` | Widget | `FLOAT`, default `0.2` | Extra canonical-frame room around the face. |
| `image_aligned` | Output | `IMAGE` | Aligned face crop. |
| `inverse_matrix_json` | Output | `STRING` | Serialized inverse transform for N-11 Unwrap Face. |

## Workflow JSON

[workflows/S-06-facial-aligner.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-06-facial-aligner.json)
