---
title: N-14 LUT Import
description: Apply an Adobe Cube LUT with optional mask and strength.
---

# N-14 LUT Import

`JHPixelProLUTImport` applies external or round-tripped `.cube` LUTs to an image. It closes the LUT authoring loop by pairing with N-13 LUT Export.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image` | Input | `IMAGE` | Source image. |
| `filename` | Widget | `STRING`, default `pack_lut.cube` | Cube filename or absolute path to load. |
| `strength` | Widget | `FLOAT`, default `1.0` | Blend amount between original and LUT-applied image. |
| `mask` | Optional input | `MASK` | Optional spatial gate. |
| `image` | Output | `IMAGE` | LUT-applied image. |

## Workflow JSON

[workflows/S-15-lut-import.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-15-lut-import.json)
