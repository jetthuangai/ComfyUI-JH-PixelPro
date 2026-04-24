---
title: N-25 Layer Add
description: Add an image layer with blend mode, opacity, fill, mask, and clipping.
---

# N-25 Layer Add

`JHPixelProLayerAdd` appends an image layer to a `LAYER_STACK` with Photoshop-style blend controls. Use it for deterministic finishing composites after generation or retouch.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `stack` | Input | `LAYER_STACK` | Existing layer stack. |
| `layer_image` | Input | `IMAGE` | Image layer to append. |
| `blend_mode` | Widget | `COMBO`, default `normal` | One of 27 Photoshop-style blend modes. |
| `opacity` | Widget | `FLOAT`, default `1.0` | Layer opacity. |
| `fill` | Widget | `FLOAT`, default `1.0` | Layer fill amount. |
| `clip_to_below` | Widget | `BOOLEAN`, default `false` | Clip this layer to the layer below. |
| `layer_mask` | Optional input | `MASK` | Optional layer mask. |
| `LAYER_STACK` | Output | `LAYER_STACK` | Updated layer stack. |

## Workflow JSON

[workflows/S-26-layer-compositing-2layer-overlay.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-26-layer-compositing-2layer-overlay.json)
