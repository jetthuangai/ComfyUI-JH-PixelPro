---
title: N-32 Mask Combine
description: Combine two masks with boolean-style and blend-style operations.
---

# N-32 Mask Combine

`JHPixelProMaskCombine` merges two masks using common compositing operations. Use it to build compound selections before matting, inpaint, or final layer stacks.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `mask_a` | Input | `MASK` | First mask input. |
| `mask_b` | Input | `MASK` | Second mask input. |
| `operation` | Widget | `COMBO`, default `union` | `add`, `subtract`, `intersect`, `union`, `difference`, `xor`, or `multiply`. |
| `blend_mode` | Widget | `COMBO`, default `hard` | Hard combine or `soft_feather` combine. |
| `opacity` | Widget | `FLOAT`, default `1.0` | Blend amount for the second mask contribution. |
| `feather_sigma` | Widget | `FLOAT`, default `0.0` | Optional feather after the combine operation. |
| `mask` | Output | `MASK` | Combined mask. |

## Workflow JSON

[workflows/N-32-mask-combine.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-32-mask-combine.json)
