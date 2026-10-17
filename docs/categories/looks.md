# Looks

Looks are finished creative grades you can audition in one step. Instead of building a long colour chain, you pick a style from a dropdown, set how strong it is, and compare. This is the fastest way to find a direction before you refine it with the colour tools.

<div class="grid cards" markdown>

-   ![](../assets/nodes/look-select.jpg)

    **[Look Select](../nodes/look-select.md)**

    Six ready-made looks (teal and orange, golden hour, faded film and more) with an intensity slider and optional skin protection.

</div>

For `.cube` LUT based looks see [LUT Preset](../nodes/lut-preset.md), [LUT Import](../nodes/lut-import.md) and [Tone Match LUT](../nodes/tone-match-lut.md) in the colour group.

## Typical chains

- Quick audition: Load Image → [Look Select](../nodes/look-select.md) → Preview Image
- Look with fine control: [Look Select](../nodes/look-select.md) → [Tone Curve](../nodes/tone-curve.md) → Preview Image
- Copy a reference's feel: [Tone Match LUT](../nodes/tone-match-lut.md) → [LUT Import](../nodes/lut-import.md) on your photo
- Build and save your own look: [HALD Identity](../nodes/hald-identity.md) → colour nodes → [LUT Export](../nodes/lut-export.md)
