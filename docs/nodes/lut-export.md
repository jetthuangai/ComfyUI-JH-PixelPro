---
title: N-13 LUT Export
description: Export a graded HALD image as an Adobe Cube LUT.
---

# N-13 LUT Export

`JHPixelProLUTExport` writes a graded HALD image to a portable Adobe Cube 1.0 `.cube` file. Use it after N-12 HALD Identity and color-grade nodes to move looks into Resolve, Premiere, OBS, or other LUT-aware tools.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image` | Input | `IMAGE` | Graded HALD image. |
| `level` | Widget/Input | `INT`, default `8` | HALD level matching the upstream identity image. |
| `filename` | Widget | `STRING`, default `pack_lut.cube` | Output filename or path. |
| `title` | Widget | `STRING`, default `JHPixelPro LUT` | Cube file metadata title. |
| `path` | Output | `STRING` | Resolved output file path. |

## Workflow JSON

[workflows/S-14-lut-export.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-14-lut-export.json)
