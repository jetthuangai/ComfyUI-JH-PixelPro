---
title: Skin Tone Retouch
description: Split a portrait into shadow, midtone and highlight masks and retouch each tone range on its own.
---

# Skin Tone Retouch

Skin rarely needs one fix everywhere. This example shows how to build three tone masks from a portrait with [Skin Tone Tri-Region](../nodes/skin-tone-tri-region.md), then use them to treat shine, mid skin tone and shadows separately.

## The chain

```text
Load Image ──────────────────────────────┐
                                         ▼
Person / skin mask (optional) ──► Skin Tone Tri-Region
                                     ├─ shadow_mask
                                     ├─ midtone_mask
                                     └─ highlight_mask
```

The skin mask can come from any segmentation node. The sample workflow uses the person-mask node from the external A Person Mask Generator pack, which you need to install separately. Without a skin mask the whole frame is split by brightness, including hair, clothing and background.

## Try it

1. Download [the sample workflow](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-35-skin-tone-tri-region.json) and drag it onto the ComfyUI canvas.
2. Copy `workflows/sample_portrait.jpg` into `ComfyUI/input/` and pick it in **Load Image**.
3. Run it and look at the three mask previews. White is where each range applies.

## Using the three masks

- **`highlight_mask`:** soften shine, or pull highlight colour back. Route it into a colour adjustment, or into [Face Beauty Blend](../nodes/face-beauty-blend.md) with a smoothed version of the image.
- **`midtone_mask`:** the main skin-tone grade. Match colour to a reference with [Color Matcher Region](../nodes/color-matcher-region.md), or apply a LUT through [Face Beauty Blend](../nodes/face-beauty-blend.md).
- **`shadow_mask`:** lift or warm the shadows, or limit noise reduction to dark areas.

The three masks always add up to your skin mask, so edits made through them never overlap or leave gaps.

## Settings to try first

| Setting | Default | When to change it |
|---|---|---|
| `shadow_cutoff` | `0.33` | Lower it if the shadow mask grabs too much of the face. |
| `highlight_cutoff` | `0.66` | Raise it to target only the brightest shine. |
| `soft_sigma` | `1.0` | Raise it for softer transitions between ranges. |

For every setting in detail, see [Skin Tone Tri-Region](../nodes/skin-tone-tri-region.md). To check a mask before using it, add [Mask Preview](../nodes/mask-preview.md).
