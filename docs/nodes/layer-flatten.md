---
title: N-27 Layer Flatten
description: Render a layer stack back into a final image.
---

# N-27 Layer Flatten

`JHPixelProLayerFlatten` renders a `LAYER_STACK` back into a normal ComfyUI image. It is the terminal node for the pack's layer compositing workflow.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `stack` | Input | `LAYER_STACK` | Layer stack to render. |
| `IMAGE` | Output | `IMAGE` | Flattened final image. |

## Workflow JSON

[workflows/S-27-layer-compositing-5layer-cinematic.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-27-layer-compositing-5layer-cinematic.json)
