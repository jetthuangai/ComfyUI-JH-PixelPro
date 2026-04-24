---
title: N-17 Tone Match LUT
description: Generate a reusable LUT from a graded reference frame.
---

# N-17 Tone Match LUT

`JHPixelProToneMatchLUT` creates a `.cube` LUT by tone-matching a HALD to a reference image. It turns a finished reference grade into a reusable look artifact.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `reference` | Input | `IMAGE` | Graded reference image. |
| `level` | Widget | `COMBO`, default `8` | HALD LUT level to generate. |
| `filename` | Widget | `STRING`, default `tone_match.cube` | Output cube filename. |
| `title` | Widget | `STRING`, default `Tone Match LUT` | Cube file metadata title. |
| `lut_path` | Output | `STRING` | Resolved generated LUT path. |

## Workflow JSON

[workflows/S-17-tone-match-lut.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-17-tone-match-lut.json)
