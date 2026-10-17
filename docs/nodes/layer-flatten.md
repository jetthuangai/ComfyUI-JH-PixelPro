---
title: Layer Flatten
description: Merge a layer stack into one normal image.
---

# Layer Flatten

Merges all layers of a stack into a single image, like Layer → Flatten Image in Photoshop. Use it as the last step of a compositing chain.

![Layer Flatten — before / after](../assets/nodes/layer-flatten.jpg)

| | |
|---|---|
| **In ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → compositing → Compositing: Layer Flatten` |
| **Good for** | Getting a normal image out of a layer stack so you can preview, save or keep editing |

## When to use it

A layer stack is not an image yet. Until you flatten it, you cannot preview or save it. **Layer Flatten** renders the stack from the bottom layer up: each layer is mixed in using its blend mode, opacity, fill, mask and clipping.

## Quick start

1. Build a stack with [Layer Stack Start](layer-stack-start.md) and [Layer Add](layer-add.md).
2. Connect the last `LAYER_STACK` to **Layer Flatten** → `Preview Image` or `Save Image`.

## Settings

| Setting | Default | What it does | Tip |
|---|---|---|---|
| `stack` | — | The layer stack to render. | Connect the last node of your chain. |

| Output | Type | Meaning |
|---|---|---|
| `IMAGE` | `IMAGE` | The merged picture, values clamped to 0..1. |

## Tips from retouching

- Flatten once, at the end. A flattened image no longer lets you adjust the individual layers.
- You can flatten the same stack at two points (for example before and after one layer) to make a before/after comparison.
- Run [QA Image Diff](qa-image-diff.md) on the original and the flattened image to see how much the composite changed.

??? note "Under the hood"
    Runs `compose_stack` from `core/blend_modes.py`. An empty stack raises an error. Source: `nodes/layer_compositing.py`.

## Related nodes

- [Layer Stack Start](layer-stack-start.md)
- [Layer Add](layer-add.md)
- [Layer Group](layer-group.md)

## Example workflow

[Layer compositing: 5-layer cinematic grade](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-27-layer-compositing-5layer-cinematic.json). Copy `workflows/sample_portrait.jpg` into `ComfyUI/input/`.
