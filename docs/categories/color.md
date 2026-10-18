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

-   ![](../assets/nodes/shadows-highlights-pro.jpg)

    **[Shadows/Highlights Pro](../nodes/shadows-highlights-pro.md)**

    Lift shadows and recover highlights without halos, Camera Raw or Photoshop style.

-   ![](../assets/nodes/defringe.jpg)

    **[Defringe](../nodes/defringe.md)**

    Removes purple and green fringes along high-contrast edges, and lateral CA.

-   ![](../assets/nodes/hdr-merge.jpg)

    **[HDR Merge (bracket)](../nodes/hdr-merge.md)**

    Fuses bracketed exposures into one image with detail in shadows and sky.

-   ![](../assets/nodes/auto-tone.jpg)

    **[Auto: Tone & White Balance](../nodes/auto-tone.md)**

    One click: exposure, Whites/Blacks, Highlights/Shadows and colour cast, with the settings as a ColorLab preset.

-   ![](../assets/nodes/apply-colorlab-preset.jpg)

    **[Apply ColorLab Preset](../nodes/apply-colorlab-preset.md)**

    Put a ColorLab look on any photo or batch, with strength and mask.

-   ![](../assets/nodes/lightroom-preset.jpg)

    **[Lightroom Preset (XMP)](../nodes/lightroom-preset.md)**

    Your Lightroom / Camera Raw presets with a thumbnail browser and a report.

-   ![](../assets/nodes/lightroom-preset-contact-sheet.jpg)

    **[Lightroom Preset Contact Sheet](../nodes/lightroom-preset-contact-sheet.md)**

    Compare a whole preset collection on one photo.

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

-   ![](../assets/nodes/palette-extract.jpg)

    **[Palette Extract](../nodes/palette-extract.md)**

    Pull the 2–16 dominant colours as swatches and HEX codes, plus a flat-colour (posterised) version.

</div>

## Typical chains

- **Match and finish:** Load Image → [Color Matcher](../nodes/color-matcher.md) → [Tone Curve](../nodes/tone-curve.md) → [ColorLab](../nodes/color-lab.md) → Preview Image
- **Grade only the shadows:** Load Image → [Luminosity Masking](../nodes/luminosity-masking.md) → use `mask_shadows` as the mask of [Tone Curve](../nodes/tone-curve.md) or [Hue/Saturation per Range](../nodes/hue-saturation-range.md)
- **Bake a look into a LUT:** [HALD Identity](../nodes/hald-identity.md) → [Tone Curve](../nodes/tone-curve.md) / [ColorLab](../nodes/color-lab.md) → [LUT Export](../nodes/lut-export.md), then reuse it with [LUT Import](../nodes/lut-import.md)
