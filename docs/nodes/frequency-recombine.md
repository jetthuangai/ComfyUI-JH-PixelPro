---
title: Frequency Recombine
description: Put the low and high layers from Frequency Separation back together, with a slider to boost or soften texture.
---

# Frequency Recombine

Rebuilds a picture from the two layers made by [Frequency Separation](frequency-separation.md). Edit colour on `low`, texture on `high`, then put them back. One slider lets you push pores and fine detail up or down at the same time.

![Frequency Recombine — before / after](../assets/nodes/frequency-recombine.jpg)

| | |
|---|---|
| **In ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → filters → Frequency Recombine (low + high)` |
| **Good for** | Finishing a frequency-separation retouch, boosting or softening skin texture without touching colour |

## When to use it

[Frequency Separation](frequency-separation.md) splits a photo into a smooth `low` layer (colour and light) and a signed `high` layer (texture). The `high` layer holds negative values, so ordinary blend or add nodes cannot join the two again: they clamp the negative half away. **Frequency Recombine** adds them correctly and only clamps the final picture to the 0 to 1 range.

The `detail` setting scales the texture layer before it is added back:

- `1.0` returns the original exactly if you did not edit either layer.
- Above `1.0` boosts texture: pores, hair and fabric get stronger.
- Below `1.0` softens texture. `0.0` gives you the `low` layer alone.

## Quick start

1. `Load Image` → [Frequency Separation](frequency-separation.md).
2. Connect `low` and `high` to the same-named inputs of **Frequency Recombine**.
3. Leave `detail = 1.0` and preview: you should see your original photo.
4. Edit `low` (for example with [Skin Smoother](edge-aware-smoother.md)), then try `detail = 1.2` to `1.6` for more texture.

## Settings

| Setting | Default | What it does | Tip |
|---|---|---|---|
| `low` | — | The colour-and-tone layer, edited or not. | From Frequency Separation. |
| `high` | — | The texture layer. Signed values. | Connect it directly. Do not pass it through a node that clamps. |
| `detail` | `1.0` (0 to 3, step 0.05) | How much texture to add back. | 0.8 to 1.0 for softer skin, 1.2 to 1.6 for stronger texture. |

| Output | Type | Meaning |
|---|---|---|
| `image` | `IMAGE` | `low + detail × high`, clamped to 0..1. |

## Tips from retouching

- Check the round trip first: with no edits and `detail = 1.0` the result should match the original.
- Smooth `low` and keep `detail` at or slightly above 1 to hide blotches but keep pores.
- Keep `radius` in Frequency Separation the same for the whole job. The two layers only match if they come from the same split.
- A large `detail` on a noisy photo also boosts the noise. Try smaller steps.
- Both layers must have the same width and height.

??? note "Under the hood"
    The result is `low + detail × high` in float32, clamped once at the end. Both inputs must be RGB with the same height and width. Batch sizes must match, or one side must be 1 (it broadcasts). `detail` outside 0..3 raises an error. Source: `frequency_recombine` in `core/frequency.py` and the wrapper in `nodes/frequency_node.py`.

## Related nodes

- [Frequency Separation](frequency-separation.md)
- [Skin Smoother](edge-aware-smoother.md)
- [Smart Sharpen](smart-sharpen.md)
- [Detail Mask](high-freq-detail-masker.md)

## Example workflow

[Frequency Separation example workflow](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-01-frequency-separation.json) splits the portrait and rebuilds it with Frequency Recombine at `detail` 1.6. It uses only core ComfyUI nodes and this pack. Copy `workflows/sample_portrait.jpg` into `ComfyUI/input/`.
