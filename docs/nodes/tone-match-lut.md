---
title: Tone Match LUT
description: Build a reusable .cube LUT automatically from the colour and tone of a reference image.
---

# Tone Match LUT

Give it a photo whose colour you love and it writes a LUT that pushes other images toward that look. No manual grading needed.

![Tone Match LUT — before / after](../assets/nodes/tone-match-lut.jpg)

| | |
|---|---|
| **In ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → color → Tone Match LUT (auto-gen .cube)` |
| **Good for** | Copying the feel of a reference shot, making a house look, matching a set of photos to a hero frame |

## When to use it

[Color Matcher](color-matcher.md) matches one image to another, but only for that pair. Tone Match LUT does the same kind of colour transfer once and saves it as a `.cube` file. You can then apply the LUT to any number of photos with [LUT Import](lut-import.md), or open it in other apps that read Cube 1.0.

It captures overall colour and tone: the average colour, how widely the colours spread, and how they relate to each other. It does not copy local detail or content.

## Quick start

1. Connect **Load Image** (your reference) to `reference`.
2. Leave `level` at `8`. Set `filename` (for example `hero_look.cube`).
3. Queue the prompt. The `lut_path` output shows the new file in `ComfyUI/output/`.
4. Apply it with [LUT Import](lut-import.md). Copy the file to `ComfyUI/input/` or use the full path.

## Settings

| Setting | Default | What it does | Tip |
|---|---|---|---|
| `reference` | — | The image whose look you want to capture. | Choose a well-exposed frame with the colours you want. |
| `level` | 8 | Detail of the LUT. Choices: 4, 6, 8, 10, 12. | 8 gives a 64-point cube, the standard for creative looks. |
| `filename` | `tone_match.cube` | Output name. Relative names go to ComfyUI's `output/` folder; `.cube` is added if missing. | An existing file is overwritten. |
| `title` | `Tone Match LUT` | Name stored inside the file. | Name it after the reference. |

**Output:** `lut_path` — the full path of the written file.

## Tips from retouching

- Pick a reference that is representative. A reference dominated by one colour (a sunset, a green field) gives a strong, one-sided LUT.
- A grey or nearly colourless reference produces a LUT that changes almost nothing.
- Always test with [LUT Import](lut-import.md) at `strength` 0.6 to 0.8 first. Automatic matches can be stronger than they look in the reference.
- If the match is close but not perfect, do small fixes with [Tone Curve](tone-curve.md) after the LUT.

??? note "Under the hood"
    An identity HALD at the chosen level is converted to LAB and passed through an MKL (Monge-Kantorovich linear) transfer: a 3 × 3 matrix derived from the covariance of the HALD's LAB samples and the reference's LAB samples, plus the reference mean. If either covariance is ill-conditioned, it falls back to a mean-only shift. Results are clamped to the LAB range and converted back to RGB. A reference that is almost uniform grey returns the identity. The graded HALD is written with the same exporter as [LUT Export](lut-export.md). Earlier versions used histogram or mean/std matching; the current method preserves cross-channel colour relationships.

## Related nodes

- [Color Matcher](color-matcher.md) — match two images directly, no file.
- [LUT Import](lut-import.md) — apply the generated `.cube`.
- [LUT Export](lut-export.md) — export a hand-built grade instead.
- [HALD Identity](hald-identity.md) — the chart used inside this node.

## Example workflow

[Tone Match LUT example workflow](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-17-tone-match-lut.json). A golden-hour copy of the portrait is the reference. Tone Match LUT writes a `.cube` file to `ComfyUI/output/` and its path goes straight into LUT Import, which applies it to the original. Copy `workflows/sample_portrait.jpg` into `ComfyUI/input/`. It uses only core ComfyUI nodes and this pack.

![Result: Tone Match LUT example workflow](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/S-17-tone-match-lut-preview.png)
