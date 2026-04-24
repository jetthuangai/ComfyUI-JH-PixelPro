---
title: N-34 Preset Pack LUT
description: Apply a bundled generic Adobe Cube LUT preset from a dropdown.
---

# N-34 Preset Pack LUT

`JHPixelProLUTPreset` applies one of the pack-bundled Adobe Cube `.cube` presets with a single intensity slider. It reuses the same trilinear LUT engine as N-14, but removes file-path handling for faster look auditioning.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image` | Input | `IMAGE` | Source image. |
| `preset` | Widget | `COMBO`, default `neutral-identity` | Bundled preset from the pack `presets/` directory. |
| `intensity` | Widget | `FLOAT`, default `1.0` | Blend factor: `0.0` returns the original image, `1.0` applies the full LUT. |
| `image` | Output | `IMAGE` | LUT-graded image. |

## Workflow JSON

[workflows/N-34-lut-preset.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-34-lut-preset.json)
