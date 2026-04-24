---
title: N-04 High-Frequency Detail Masker
description: Build masks from high-frequency image detail.
---

# N-04 High-Frequency Detail Masker

`JHPixelProHighFreqDetailMasker` detects texture and edge energy to build a detail-protection mask. Use it to protect pores, hair, fabric, or fine edges during retouch and denoise passes.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image` | Input | `IMAGE` | Source image. |
| `kernel_type` | Widget | `COMBO`, default `laplacian` | Uses `laplacian`, `sobel`, or `fs_gaussian` high-pass logic. |
| `sensitivity` | Widget | `FLOAT`, default `0.5` | Fraction of high-detail pixels retained. |
| `threshold_mode` | Widget | `COMBO`, default `relative_percentile` | Adaptive percentile or absolute thresholding. |
| `mask` | Optional input | `MASK` | Optional pre-mask gate. |
| `mask_detail` | Output | `MASK` | High-frequency detail mask. |

## Workflow JSON

[workflows/S-04-hf-detail-masker.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-04-hf-detail-masker.json)
