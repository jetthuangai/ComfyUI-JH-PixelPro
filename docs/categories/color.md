# Color & Tone

These nodes handle everything about colour and brightness: matching one image to another, shaping contrast with curves, grading with a Camera Raw style panel, selecting by tone or colour, and saving or applying LUTs. Use them to finish a retouch, to bring AI output back to a reference photo, or to build a repeatable look.

<div class="grid cards" markdown>

-   ![](../assets/nodes/color-matcher.jpg)

    **[Color Matcher](../nodes/color-matcher.md)**

    Pull the colours of one image toward a reference in LAB space.

-   ![](../assets/nodes/color-matcher-region.jpg)

    **[Color Matcher Region](../nodes/color-matcher-region.md)**

    Match one masked region to another, even across different image sizes.

-   ![](../assets/nodes/color-lab.jpg)

    **[ColorLab](../nodes/color-lab.md)**

    Camera Raw style Basic, HSL, Color Grading and Gray Mix in one node.

-   ![](../assets/nodes/clahe-local-contrast.jpg)

    **[CLAHE Local Contrast](../nodes/clahe-local-contrast.md)**

    Adaptive local contrast on lightness only; colours stay intact.

-   ![](../assets/nodes/tone-curve.jpg)

    **[Tone Curve](../nodes/tone-curve.md)**

    Eight-point curves for contrast and colour balance, with presets.

-   ![](../assets/nodes/hue-saturation-range.jpg)

    **[Hue/Saturation per Range](../nodes/hue-saturation-range.md)**

    Shift hue or saturation inside one chosen colour band.

-   ![](../assets/nodes/luminosity-masking.jpg)

    **[Luminosity Masking](../nodes/luminosity-masking.md)**

    Soft shadow, midtone and highlight masks from image brightness.

-   ![](../assets/nodes/saturation-mask.jpg)

    **[Saturation Mask](../nodes/saturation-mask.md)**

    Mask the pixels whose colour strength falls inside a range.

-   ![](../assets/nodes/lut-preset.jpg)

    **[LUT Preset](../nodes/lut-preset.md)**

    Apply a bundled colour look from a dropdown with an intensity slider.

-   ![](../assets/nodes/lut-import.jpg)

    **[LUT Import](../nodes/lut-import.md)**

    Load any `.cube` LUT and apply it, with strength and optional mask.

-   ![](../assets/nodes/lut-export.jpg)

    **[LUT Export](../nodes/lut-export.md)**

    Save a graded chart as a portable `.cube` LUT.

-   ![](../assets/nodes/hald-identity.jpg)

    **[HALD Identity](../nodes/hald-identity.md)**

    Generate the neutral colour chart you grade and bake into a LUT.

-   ![](../assets/nodes/tone-match-lut.jpg)

    **[Tone Match LUT](../nodes/tone-match-lut.md)**

    Build a LUT automatically from the colour and tone of a reference image.

</div>

## Typical chains

- **Match and finish:** Load Image → [Color Matcher](../nodes/color-matcher.md) → [Tone Curve](../nodes/tone-curve.md) → [ColorLab](../nodes/color-lab.md) → Preview Image
- **Grade only the shadows:** Load Image → [Luminosity Masking](../nodes/luminosity-masking.md) → use `mask_shadows` as the mask of [Tone Curve](../nodes/tone-curve.md) or [Hue/Saturation per Range](../nodes/hue-saturation-range.md)
- **Bake a look into a LUT:** [HALD Identity](../nodes/hald-identity.md) → [Tone Curve](../nodes/tone-curve.md) / [ColorLab](../nodes/color-lab.md) → [LUT Export](../nodes/lut-export.md), then reuse it with [LUT Import](../nodes/lut-import.md)
