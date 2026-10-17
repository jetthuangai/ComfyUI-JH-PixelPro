---
title: ColorLab
description: A Camera Raw style editing panel in one node, with Basic, HSL, Color Grading and Gray Mix controls.
---

# ColorLab

A full Camera Raw style finishing stage in a single node: exposure and tone, per-colour HSL, three-way colour grading and a black-and-white mixer. Every control starts at neutral.

![ColorLab — before / after](../assets/nodes/color-lab.jpg)

| | |
|---|---|
| **In ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → color → Color: ColorLab (ACR)` |
| **Good for** | A quick global grade, teal-and-orange looks, split toning, black-and-white conversion, one-node finishing after retouching |

## When to use it

If you know Lightroom or Camera Raw, you already know this node. Instead of chaining many small nodes, you get the same four panels in the same order, applied as one pipeline: **Basic → HSL → Color Grading → Gray Mix**. It is a good final step after the heavy retouching is done, or a stand-in for a Camera Raw pass on a generated image.

It is an approximation written for this pack, not Adobe's code, so numbers will not match Camera Raw one-to-one. Judge by eye. With every slider at 0 and Gray Mix off, the image passes through untouched.

The node has 55 controls on one body. Their names tell you the panel: `basic_*`, `hsl_*`, `grade_*`, `gray_*`.

## Quick start

1. **Load Image** → ColorLab → **Preview Image**.
2. Start in Basic: nudge `basic_exposure` and `basic_contrast`, then `basic_highlights` and `basic_shadows`.
3. For colour, move on to HSL or Color Grading. Change one panel at a time and compare against the original.
4. For black and white, turn on `gray_enable` and shape the tones with the `gray_*` sliders.

## Settings

All sliders default to 0, except where noted. "±100" means the slider runs from -100 to 100.

### Basic

Global tone and presence.

| Setting | Range | What it does |
|---|---|---|
| `basic_exposure` | -5 to 5 (stops) | Overall brightness. Each +1 doubles the light. |
| `exposure_mode` (optional) | `display` | `display` multiplies the picture values by 2^EV (the classic ColorLab behaviour). `linear` converts to linear light first, multiplies, then converts back, like a camera or Lightroom. |
| `basic_contrast` | ±100 | Spreads tones away from mid-grey. |
| `basic_highlights` | ±100 | Brightens or recovers the bright half of the picture. |
| `basic_shadows` | ±100 | Opens up or deepens the dark half. |
| `basic_whites` | ±100 | Moves the very brightest tones. |
| `basic_blacks` | ±100 | Moves the very darkest tones. |
| `basic_texture` | ±100 | Fine-detail punch (small radius). |
| `basic_clarity` | ±100 | Mid-tone local contrast (larger radius), strongest in the mid-tones. |
| `basic_dehaze` | ±100 | Adds contrast and slightly deepens tones to cut haze; negative adds a soft haze. |
| `basic_vibrance` | ±100 | Boosts weak colours more than strong ones and protects skin tones. |
| `basic_saturation` | ±100 | Scales all colours equally. |

### HSL

Eight colour bands: `red`, `orange`, `yellow`, `green`, `aqua`, `blue`, `purple`, `magenta`. Each has three sliders named `hsl_<colour>_hue`, `hsl_<colour>_sat` and `hsl_<colour>_lum`, all ±100 (24 in total, e.g. `hsl_orange_sat`).

| Slider | What it does |
|---|---|
| `hsl_<colour>_hue` | Rotates that colour toward its neighbours (up to about 30 degrees at the extremes). |
| `hsl_<colour>_sat` | Makes that colour stronger or weaker. |
| `hsl_<colour>_lum` | Makes that colour lighter or darker. |

Bands overlap softly, so neighbouring colours move a little with the one you edit. This is the panel for teal-and-orange, greener foliage or calmer skin reds.

### Color Grading

Three tonal ranges: `shadow`, `mid`, `highlight`. Each has four sliders, named `grade_<range>_hue`, `grade_<range>_sat`, `grade_<range>_lum` and `grade_<range>_bal` (12 in total, e.g. `grade_shadow_hue`).

| Slider | Range | What it does |
|---|---|---|
| `grade_<range>_hue` | 0 to 360 | The tint colour, as a position on the colour wheel. |
| `grade_<range>_sat` | 0 to 100 | How strong the tint is. The tint only appears when this is above 0. |
| `grade_<range>_lum` | ±100 | Lightens or darkens that tonal range. |
| `grade_<range>_bal` | ±100 | Treats pixels as brighter (positive) or darker (negative) when deciding which range they belong to. Positive shrinks the shadow range and widens the highlight range. |

Set `hue` first, then raise `sat` slowly. A touch of blue in the shadows and warm yellow in the highlights is the classic split-tone.

### Gray Mix

Black-and-white conversion with per-colour control.

| Setting | Default | Range | What it does |
|---|---|---|---|
| `gray_enable` | off | on / off | Converts the image to grey. Nothing in this panel applies while it is off. |
| `gray_red`, `gray_orange`, `gray_yellow`, `gray_green`, `gray_aqua`, `gray_blue`, `gray_purple` | 0 | -200 to 300 | How light each colour appears in the grey result. Positive lightens that colour, negative darkens it. |

There is no `gray_magenta` slider: magenta follows its neighbours.

**Output:** `image`, the adjusted picture.

## Tips from retouching

- For natural exposure moves, set `exposure_mode` to `linear`. Real light adds up in linear space, so +1 stop behaves like opening the camera aperture one stop: mid-grey 0.5 goes to about 0.69, highlights roll off gradually and colours keep their balance. `display` multiplies the stored values instead, so the same +1 sends mid-grey to 1.0 and clips bright areas quickly. Old workflows stay on `display`.
- Work top to bottom: tone in Basic first, colour in HSL or Grading afterwards, so the colour edits sit on a stable base.
- Use small steps. Many sliders act strongly near their ends; 10 to 30 is often plenty.
- Skin mostly sits in the orange band. Move `hsl_orange_*` gently and shift blues and greens harder when you want a look without sickly skin.
- For black and white, darken `gray_blue` to deepen skies, lighten `gray_orange` and `gray_red` to smooth skin.
- ColorLab affects the whole frame. To grade only part of it, blend the result back through a mask (see [Luminosity Masking](luminosity-masking.md)).
- Want only local contrast? [CLAHE Local Contrast](clahe-local-contrast.md) lifts flat or dark regions in a different way than `basic_clarity`.

??? note "Under the hood"
    The pipeline lives in `core/color_lab.py` and uses plain PyTorch plus a Kornia box blur. Input is clamped to 0..1.

    - **Basic:** exposure multiplies by `2^exposure` (`display`, on the sRGB values) or converts to linear RGB with Kornia, multiplies by `2^exposure`, clamps and converts back (`linear`; mid-grey 0.5 at +1 EV becomes about 0.686); contrast pivots around 0.5; highlights, shadows, whites and blacks add smoothstep-weighted offsets driven by luma (Rec. 601 weights) and are range-protected so they do not clip; texture and clarity are unsharp-style boosts using a 5 px and a 9 px box blur with replicated borders (clarity is weighted toward the mid-tones); dehaze is a contrast and offset change; vibrance and saturation work in HSV, with vibrance down-weighted near the skin hue.
    - **HSL:** each band is a Gaussian on the hue circle (sigma 20 degrees) around fixed anchors: red 0, orange 30, yellow 60, green 120, aqua 180, blue 240, purple 280, magenta 310.
    - **Color Grading:** each range uses a smooth luma mask (shifted by `bal`), a tint built from `hue` and `sat`, and `lum` as a lift.
    - **Gray Mix:** luma is modulated by the per-band weights; magenta is not a band.

    The result is passed through `nan_to_num` and clamped to 0..1. If all sliders are 0 and `gray_enable` is off, the input is returned as is.

## Related nodes

- [Tone Curve](tone-curve.md): precise contrast shaping with curves
- [Hue/Saturation per Range](hue-saturation-range.md): a single, adjustable hue band
- [CLAHE Local Contrast](clahe-local-contrast.md): adaptive local contrast on lightness only
- [Luminosity Masking](luminosity-masking.md): grade only shadows, midtones or highlights
- [LUT Preset](lut-preset.md): apply a ready-made look after grading

## Example workflow

Sample workflows (copy `workflows/sample_portrait.jpg` into `ComfyUI/input/` first):

- [ColorLab Basic example workflow](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-21-colorlab-basic-only.json)
- [ColorLab HSL teal-orange example workflow](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-22-colorlab-hsl-teal-orange.json)
- [ColorLab Color Grading example workflow](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-23-colorlab-color-grading-cinematic.json)
- [ColorLab Gray Mix example workflow](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-24-colorlab-gray-mix-bw.json)
- [ColorLab full preset example workflow](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-25-colorlab-full-acr-preset.json)
