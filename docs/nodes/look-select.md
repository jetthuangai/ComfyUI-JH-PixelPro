---
title: N-22 Look Select
description: Apply one of the pack's preset creative looks from a dropdown.
---

# N-22 Look Select

`JHPixelProLookSelect` wraps the pack's curated look presets behind one dropdown node. It is the fastest way to audition finished grades without manually building a longer color chain.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image` | Input | `IMAGE` | Source image. |
| `preset` | Widget | `COMBO`, default `cinematic-teal-orange` | Selects one of the JSON-backed creative looks. |
| `intensity` | Widget | `FLOAT`, default `0.7` | Preset blend amount. |
| `protect_skin` | Widget | `BOOLEAN`, default `false` | Attempts to preserve skin tones during look application. |
| `IMAGE` | Output | `IMAGE` | Look-graded image. |

## Workflow JSON

[workflows/S-19-look-select-single.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-19-look-select-single.json)
