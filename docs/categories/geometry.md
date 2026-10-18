# Geometry

Geometry nodes fix the shape and position of the picture before you start retouching: straighten lens distortion, line up two shots, or bring a face into a standard upright frame. Doing this first keeps later edits predictable.

<div class="grid cards" markdown>

-   ![](../assets/nodes/facial-aligner.jpg)

    **[Facial Aligner](../nodes/facial-aligner.md)**

    Rotates, scales and crops a face to a standard square from five landmarks.

-   ![](../assets/nodes/lens-distortion.jpg)

    **[Lens Distortion](../nodes/lens-distortion.md)**

    Corrects (or adds) barrel and pincushion distortion with lens presets.

-   ![](../assets/nodes/auto-align.jpg)

    **[Auto Align](../nodes/auto-align.md)**

    Matches features between two photos and warps one onto the other.

-   ![](../assets/nodes/auto-upright.jpg)

    **[Auto: Upright](../nodes/auto-upright.md)**

    Levels the horizon and makes converging building lines parallel.

</div>

## Typical chains

- **One-click base edit:** Load Image → [Auto: Upright](../nodes/auto-upright.md) → [Auto: Tone & White Balance](../nodes/auto-tone.md) → `params_json` → [ColorLab](../nodes/color-lab.md) to fine-tune
- **Wide-angle portrait to face edit:** Load Image → [Lens Distortion](../nodes/lens-distortion.md) → [Face Detect Lite](../nodes/face-detect-lite.md) → [Facial Aligner](../nodes/facial-aligner.md) → edit → [Unwrap Face](../nodes/unwrap-face.md)
- **Match two shots:** [Auto Align](../nodes/auto-align.md) → [Color Matcher](../nodes/color-matcher.md) → [Layer Add](../nodes/layer-add.md)
