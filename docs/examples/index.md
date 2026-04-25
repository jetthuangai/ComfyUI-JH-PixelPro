# Examples

These examples show the shortest practical paths into the v1.3.0 pack. They are not exhaustive tutorials; each page gives a concrete entry chain and the node references needed to expand it.

## Example pages

- [LUT Preset Gallery](lut-preset-gallery.md) — visual before/after comparisons for all six bundled N-34 LUT presets.
- [Skin Tone Retouch Chain](skin-tone-retouch.md) — a conceptual N-35 chain for splitting portrait tones into shadow, midtone, and highlight masks.

## Starter workflows

- [N-34 LUT Preset workflow](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-34-lut-preset.json)
- [N-35 Skin Tone Tri-Region workflow](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-35-skin-tone-tri-region.json)
- [N-29 Alpha Matte Extractor workflow](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-29-alpha-matte-extractor.json)
- [N-33 Mask Edge Smoother workflow](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-33-mask-edge-smoother.json)

## Recommended first path

1. Start with [Quickstart](../quickstart.md).
2. Run N-34 once to confirm image input/output wiring.
3. Move to N-35 if your workflow is portrait-retouch focused.
4. Use N-29 and N-33 when you need production mask extraction and edge cleanup.
