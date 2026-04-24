---
title: N-02 Sub-Pixel Mask Refiner
description: Feather binary masks into protected-core soft alpha masks.
---

# N-02 Sub-Pixel Mask Refiner

`JHPixelProSubPixelMaskRefiner` converts hard masks into soft masks while preserving an inside core and an outside core. It is useful before inpaint, composite, or detail-preservation steps that need controllable edge falloff.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `mask` | Input | `MASK` | Binary or near-binary source mask. |
| `erosion_radius` | Widget | `INT`, default `2` | Protected inside-core erosion radius. |
| `dilation_radius` | Widget | `INT`, default `4` | Outside-core dilation radius. |
| `feather_sigma` | Widget | `FLOAT`, default `2.0` | Gaussian feather sigma for the uncertain edge band. |
| `threshold` | Widget | `FLOAT`, default `0.5` | Binarization threshold before morphology. |
| `refined_mask` | Output | `MASK` | Feathered sub-pixel mask. |

## Workflow JSON

[workflows/S-02-subpixel-mask-refiner.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-02-subpixel-mask-refiner.json)
