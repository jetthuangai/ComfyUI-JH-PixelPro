---
title: N-08 Color Matcher
description: Match LAB color statistics from a reference image.
---

# N-08 Color Matcher

`JHPixelProColorMatcher` transfers reference-image color statistics onto a target image in LAB space. It is a practical retouch node for harmonizing generated inserts, skin plates, and source photography.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image_target` | Input | `IMAGE` | Image to be recolored. |
| `image_reference` | Input | `IMAGE` | Reference image that provides color statistics. |
| `channels` | Widget | `COMBO`, default `ab` | Match chroma only (`ab`) or full LAB (`lab`). |
| `strength` | Widget | `FLOAT`, default `1.0` | Blend amount for the transfer. |
| `mask` | Optional input | `MASK` | Optional spatial gate for matching. |
| `image_matched` | Output | `IMAGE` | Color-matched image. |

## Workflow JSON

[workflows/S-08-color-matcher.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-08-color-matcher.json)
