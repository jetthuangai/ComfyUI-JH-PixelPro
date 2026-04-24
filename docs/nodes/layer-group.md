---
title: N-26 Layer Group
description: Nest a sub-stack into a parent stack as a grouped layer.
---

# N-26 Layer Group

`JHPixelProLayerGroup` flattens a child `LAYER_STACK` into a grouped layer inside a parent stack. It gives larger composites a controlled nesting step while preserving blend and opacity controls.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `parent_stack` | Input | `LAYER_STACK` | Destination stack. |
| `sub_stack` | Input | `LAYER_STACK` | Stack to flatten into a group. |
| `group_blend_mode` | Widget | `COMBO`, default `normal` | Blend mode for the grouped layer. |
| `group_opacity` | Widget | `FLOAT`, default `1.0` | Opacity for the grouped layer. |
| `group_mask` | Optional input | `MASK` | Optional group mask. |
| `LAYER_STACK` | Output | `LAYER_STACK` | Parent stack with grouped layer appended. |

## Workflow JSON

[workflows/S-28-layer-compositing-group-clipping.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-28-layer-compositing-group-clipping.json)
