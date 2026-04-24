---
title: N-09 Tone Curve
description: Apply preset or custom RGB tone curves.
---

# N-09 Tone Curve

`JHPixelProToneCurve` applies master or per-channel tone curves through a compact LUT path. It covers common contrast shaping and channel-balance work without requiring an external editor.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image` | Input | `IMAGE` | Source image. |
| `preset` | Widget | `COMBO`, default `linear` | Built-in curve preset or `custom`. |
| `channel` | Widget | `COMBO`, default `rgb_master` | Apply to all channels or a single `r`, `g`, or `b` channel. |
| `points_json` | Widget | `STRING`, default 8-point curve JSON | Custom curve control points used when `preset = custom`. |
| `strength` | Widget | `FLOAT`, default `1.0` | Blend amount for the curve result. |
| `image_toned` | Output | `IMAGE` | Tone-adjusted image. |

## Workflow JSON

[workflows/S-09-tone-curve.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-09-tone-curve.json)
