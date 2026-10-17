# LUT Preset Gallery

[LUT Preset](../nodes/lut-preset.md) ships six bundled generic Adobe Cube looks. The gallery below was generated from `workflows/sample_portrait.jpg` by running the actual node and saving before and after previews.

![LUT Preset before and after gallery](../assets/examples/lut-preset/lut-preset-gallery-grid.png)

## What each preset is for

| Preset | Use when |
|---|---|
| `neutral-identity` | You want to check your wiring or compare against a no-change baseline. |
| `warm-portrait` | You want a warmer, softer portrait direction without building a full grade. |
| `cool-portrait` | You want a cleaner, cooler cast for editorial or product-style portraits. |
| `cinematic-teal-orange` | You want a stronger split-tone look with cooler shadows and warmer skin-adjacent tones. |
| `soft-pastel` | You want lower-contrast, lighter colour for gentle beauty or social previews. |
| `high-contrast` | You want punchier separation for quick review images or a stronger starting grade. |

## Try it yourself

Open the [LUT Preset example workflow](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-34-lut-preset.json), choose your image in Load Image, and switch the `preset` dropdown. Lower `intensity` below `1.0` when the full LUT is too strong.

Want a look that is not on this list? Build one with [HALD Identity](../nodes/hald-identity.md) and [LUT Export](../nodes/lut-export.md), then apply it with [LUT Import](../nodes/lut-import.md).
