# Skin Tone Retouch Chain

N-35 `JHPixelProSkinToneTriRegion` is a mask-splitting node for portrait retouching. It divides a source image, or an optional upstream skin mask, into shadow, midtone, and highlight masks so downstream color or texture operations can be targeted more precisely.

## Minimal chain

```text
LoadImage
  -> JHPixelProSkinToneTriRegion
      -> shadow_mask
      -> midtone_mask
      -> highlight_mask
```

Use the bundled [N-35 workflow](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-35-skin-tone-tri-region.json) to place the node on the canvas quickly.

## Practical retouch pattern

1. Feed a portrait image into N-35.
2. Optionally connect a skin-only upstream `MASK` into `skin_mask` so the split ignores hair, clothing, and background.
3. Use `shadow_mask` for local lift, noise control, or warm shadow correction.
4. Use `midtone_mask` for the main skin-tone grade.
5. Use `highlight_mask` for shine control, specular softening, or highlight color correction.

## Controls

- `shadow_cutoff` defaults to `0.33`, using Rec.601 luminance.
- `highlight_cutoff` defaults to `0.66`.
- `soft_sigma` defaults to `1.0` and smooths region boundaries while preserving the selected mask sum.

For node details, see [N-35 Skin Tone Tri-Region](../nodes/skin-tone-tri-region.md).
