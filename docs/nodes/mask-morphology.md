---
title: N-31 Mask Morphology
description: Apply common elliptical-kernel morphology operations to masks.
---

# N-31 Mask Morphology

`JHPixelProMaskMorphology` runs deterministic mask morphology for expansion, cleanup, and edge analysis. It is the utility node for quick mask shape fixes without leaving ComfyUI.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `mask` | Input | `MASK` | Source mask. |
| `operation` | Widget | `COMBO`, default `dilate` | One of `dilate`, `erode`, `open`, `close`, `gradient`, `tophat`, or `blackhat`. |
| `radius` | Widget | `INT`, default `3` | Elliptical kernel radius. |
| `iterations` | Widget | `INT`, default `1` | Number of repeated morphology passes. |
| `mask` | Output | `MASK` | Resulting mask. |

## Workflow JSON

[workflows/N-31-mask-morphology.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-31-mask-morphology.json)
