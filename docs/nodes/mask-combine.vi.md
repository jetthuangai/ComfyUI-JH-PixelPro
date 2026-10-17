---
title: Mask Combine
description: Cộng, trừ, giao hoặc nhân hai mask, giống cách kết hợp vùng chọn trong Photoshop.
---

# Mask Combine

Gộp hai mask thành một: cộng lại, cắt cái này khỏi cái kia, chỉ giữ phần chồng nhau, và nhiều cách khác. Có thể feather cả hai trước khi gộp.

![Mask Combine — trước / sau](../assets/nodes/mask-combine.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → mask → Mask: Combine` |
| **Phù hợp để** | Thêm vùng tô tay vào mask tự động, cắt mắt ra khỏi mask da, giữ phần giao của hai vùng chọn |

## Khi nào dùng

Đây là phiên bản mask của các chế độ vùng chọn trong Photoshop: add to selection, subtract from selection, intersect with selection. `mask_a` là mask nền, `mask_b` là mask bạn áp lên nó.

| Phép | Kết quả | Hiểu đơn giản |
|---|---|---|
| `union` | Lấy giá trị lớn hơn ở mỗi pixel | Add to selection (giữ viền mềm) |
| `add` | `a + b`, cắt về 0..1 | Cộng, chỗ chồng nhau sáng lên |
| `subtract` | `a - b`, cắt về 0..1 | Trừ B khỏi A |
| `difference` | `a * (1 - b)` | Cắt B khỏi A, tôn trọng viền mềm |
| `intersect` | Lấy giá trị nhỏ hơn | Chỉ phần cả hai cùng có |
| `multiply` | `a * b` | Phần chồng nhau, các giá trị mềm hòa vào nhau |
| `xor` | Chỗ chỉ một trong hai được chọn | Mọi thứ trừ phần chồng nhau |

## Bắt đầu nhanh

1. Nối mask chính vào `mask_a` và mask thứ hai vào `mask_b`.
2. Chọn `operation`. Muốn "cắt mắt ra khỏi mask da", dùng `difference` với mask mắt làm `mask_b`.
3. Để `blend_mode` ở `hard`, trừ khi viền cần mềm hơn.
4. Kiểm tra bằng [Mask Preview](mask-preview.md).

## Thông số

| Thông số | Mặc định | Tác dụng | Gợi ý |
|---|---|---|---|
| `operation` | `union` | Cách gộp hai mask. | Xem bảng trên. |
| `blend_mode` | `hard` | `hard` dùng nguyên mask. `soft_feather` blur cả hai mask trước. | Dùng `soft_feather` để chuyển tiếp mượt giữa các vùng. |
| `opacity` | 1.0 | Cường độ của phép. 0 trả lại `mask_a` nguyên vẹn; 1 cho kết quả đầy đủ. | 0 đến 1. Giảm xuống để chỉnh nhẹ tay. |
| `feather_sigma` | 0.0 | Độ blur (pixel) mà `soft_feather` dùng. | 0 đến 64. Không có tác dụng ở chế độ `hard`. |

**Đầu ra:** `mask` (MASK).

## Mẹo khi chỉnh ảnh

- Thứ tự quan trọng với `subtract` và `difference`: `mask_b` bị trừ khỏi `mask_a`.
- Hai mask phải cùng chiều rộng và chiều cao. Một trong hai có thể chỉ có một khung để dùng cho cả batch.
- Muốn chỉ làm mềm kết quả, gộp bằng `hard` rồi feather bằng [Mask Edge Smoother](mask-edge-smoother.md).
- Kết hợp mask từ [Luminosity Masking](luminosity-masking.md) (luminosity mask) với mask chủ thể bằng `multiply` hoặc `intersect` để chỉ chỉnh vùng highlight trên chủ thể.

??? note "Bên trong node hoạt động thế nào"
    Đầu vào được cắt về 0..1. Ở chế độ `soft_feather`, cả hai mask được blur Gaussian (`kernel = 2 * round(3 * sigma) + 1`, replicate padding). Phép tính chạy trên các mask (có thể đã blur) rồi cắt về 0..1; kết quả cuối là `a * (1 - opacity) + combined * opacity`. Batch 1 được nhân ra cho batch lớn hơn. Mã nguồn: `core/mask_combine.py`.

## Node liên quan

- [Mask Morphology](mask-morphology.md) để nở hoặc co một đầu vào
- [Mask Edge Smoother](mask-edge-smoother.md) để làm mềm kết quả
- [Luminosity Masking](luminosity-masking.md) cho mask theo tông để gộp
- [Mask Preview](mask-preview.md) để kiểm tra kết quả

## Workflow mẫu

[Workflow mẫu Mask Combine](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-32-mask-combine.json). Chép `workflows/sample_portrait.jpg` và `workflows/sample_binary_mask.png` vào `ComfyUI/input/`. Workflow lấy phần giao giữa mask chủ thể và vùng highlight của Luminosity Masking, có `soft_feather` ở mép, nên chỉ còn lại phần sáng của chủ thể. Workflow chỉ dùng node lõi của ComfyUI và pack này.

![Kết quả: Workflow mẫu Mask Combine](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/N-32-mask-combine-preview.png)
