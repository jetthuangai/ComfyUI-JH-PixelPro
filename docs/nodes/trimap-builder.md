---
title: N-30 Trimap Builder
description: Convert masks into strict foreground, unknown, and background trimaps.
---

# N-30 Trimap Builder

`JHPixelProTrimapBuilder` turns a binary or soft mask into the pack's strict three-value trimap convention. Pair it upstream of N-29 Alpha Matte Extractor for production matte solves.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `mask` | Input | `MASK` | Source mask used to derive trimap regions. |
| `fg_radius` | Widget | `INT`, default `4` | Radius used to shrink the definite foreground core. |
| `bg_radius` | Widget | `INT`, default `8` | Radius used to expand the definite background core. |
| `smoothing` | Widget | `FLOAT`, default `0.0` | Optional smoothing before quantizing to trimap values. |
| `trimap` | Output | `MASK` | Three-value mask: background `0.0`, unknown `0.5`, foreground `1.0`. |

## Workflow JSON

[workflows/N-30-trimap-builder.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-30-trimap-builder.json)
