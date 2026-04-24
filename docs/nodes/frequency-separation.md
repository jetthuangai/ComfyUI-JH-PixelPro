---
title: N-01 GPU Frequency Separation
description: Split an image into low-frequency and high-frequency layers for retouch workflows.
---

# N-01 GPU Frequency Separation

`JHPixelProFrequencySeparation` separates form/color from detail by producing a blurred low layer and a signed high layer. It is the pack's base texture-retouch node for rebuildable frequency-separation chains.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image` | Input | `IMAGE` | Source image in ComfyUI BHWC format. |
| `radius` | Widget | `INT`, default `8` | Gaussian blur radius in pixels. |
| `sigma` | Widget | `FLOAT`, default `0.0` | Sigma override; `0.0` auto-derives sigma from radius. |
| `precision` | Widget | `COMBO`, default `float32` | Uses `float32` for exact reconstruction or `float16` for GPU speed. |
| `low` | Output | `IMAGE` | Low-frequency color and form layer. |
| `high` | Output | `IMAGE` | High-frequency detail layer. |

## Workflow JSON

[workflows/S-01-frequency-separation.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-01-frequency-separation.json)
