---
title: N-24 Layer Stack Start
description: Start a Photoshop-style layer stack from a background image.
---

# N-24 Layer Stack Start

`JHPixelProLayerStackStart` initializes a `LAYER_STACK` from a background image. It is the required first node for the pack's compositing chain.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `background` | Input | `IMAGE` | Base image for the layer stack. |
| `LAYER_STACK` | Output | `LAYER_STACK` | Initialized layer stack. |

## Workflow JSON

[workflows/S-26-layer-compositing-2layer-overlay.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-26-layer-compositing-2layer-overlay.json)
