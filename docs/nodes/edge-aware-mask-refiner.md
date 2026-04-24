---
title: N-28 Edge-Aware Mask Refiner
description: Refine a mask so its edge follows an image guide.
---

# N-28 Edge-Aware Mask Refiner

`JHPixelProEdgeAwareMaskRefiner` uses a guide image to align mask edges to local image structure. It is the fast cleanup step before heavier alpha-matte extraction or final edge smoothing.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `mask` | Input | `MASK` | Source mask to refine. |
| `guide` | Input | `IMAGE` | RGB guide image whose edges steer refinement. |
| `radius` | Widget | `INT`, default `8` | Guided-filter neighborhood radius. |
| `eps` | Widget | `FLOAT`, default `0.001` | Edge sensitivity regularizer. |
| `feather_sigma` | Widget | `FLOAT`, default `0.0` | Optional post-refine feather. |
| `refined_mask` | Output | `MASK` | Edge-aware refined mask. |

## Workflow JSON

[workflows/N-28-edge-aware-mask-refiner.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-28-edge-aware-mask-refiner.json)
