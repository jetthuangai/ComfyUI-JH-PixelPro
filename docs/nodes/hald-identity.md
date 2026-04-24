---
title: N-12 HALD Identity
description: Generate an identity HALD image for LUT authoring.
---

# N-12 HALD Identity

`JHPixelProHALDIdentity` produces a HALD identity image that can be graded by color nodes and then exported as a `.cube` LUT. It is the source node for portable look-authoring workflows.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `level` | Widget | `COMBO`, default `8` | HALD level; larger values produce finer LUTs and larger images. |
| `image` | Output | `IMAGE` | Identity HALD image. |
| `level` | Output | `INT` | Numeric level pass-through for LUT export. |

## Workflow JSON

[workflows/S-14-lut-export.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-14-lut-export.json)
