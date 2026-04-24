---
title: N-03 Edge-Aware Skin Smoother
description: Smooth skin while preserving stronger image edges.
---

# N-03 Edge-Aware Skin Smoother

`JHPixelProEdgeAwareSmoother` applies bilateral edge-aware smoothing with mask support and explicit device controls. It is intended for controlled skin cleanup without flattening important facial or fabric edges.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image` | Input | `IMAGE` | Source image. |
| `strength` | Widget | `FLOAT`, default `0.4` | Blend amount between original and smoothed image. |
| `sigma_color` | Widget | `FLOAT`, default `0.1` | Intensity similarity sigma on the 0 to 1 scale. |
| `sigma_space` | Widget | `FLOAT`, default `6.0` | Spatial smoothing sigma in pixels. |
| `device` | Widget | `COMBO`, default `auto` | Uses `auto`, `cpu`, or `cuda`. |
| `tile_mode` | Widget | `BOOLEAN`, default `false` | Enables tiled processing for large images. |
| `mask` | Optional input | `MASK` | Optional mask to limit smoothing. |
| `image` | Output | `IMAGE` | Smoothed image. |

## Workflow JSON

[workflows/S-03-edge-aware-smoother.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-03-edge-aware-smoother.json)
