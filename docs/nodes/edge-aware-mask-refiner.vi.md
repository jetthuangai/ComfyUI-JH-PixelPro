---
title: Edge-Aware Mask Refiner
description: Kéo viền của mask thô bám sát vào cạnh thật trong ảnh bằng guided filter.
---

# Edge-Aware Mask Refiner

Mask của bạn đại khái đúng, nhưng viền cắt xuyên vào chủ thể hoặc để lại quầng sáng (halo). Node này nhìn vào ảnh gốc và kéo viền mask về đúng ranh giới thật.

![Edge-Aware Mask Refiner — trước / sau](../assets/nodes/edge-aware-mask-refiner.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → mask → Mask: Edge-Aware Refiner` |
| **Phù hợp để** | Sửa mask phân vùng AI còn lỏng, siết vùng chọn thô quanh người hoặc sản phẩm, giảm halo |

## Khi nào dùng

Các công cụ phân vùng như SAM thường cho mask lệch vài pixel. Blur không sửa được, vì blur không biết cạnh thật nằm ở đâu. Node này dùng ảnh làm **guide** (ảnh dẫn hướng): chỗ nào ảnh có cạnh rõ, viền mask bám vào đó; chỗ nào ảnh mượt, mask giữ mượt. Kỹ thuật này gọi là *guided filter*.

Hiệu quả nhất với cạnh có độ tương phản rõ giữa chủ thể và nền. Với tóc mỏng, nơi cạnh bán trong suốt, hãy dùng [Alpha Matte](alpha-matte.md) vì nó khôi phục được độ trong suốt thật.

## Bắt đầu nhanh

1. `Load Image` → node phân vùng của bạn → mask.
2. Nối mask vào `mask` và **cùng bức ảnh đó** vào `guide`. Hai bên phải cùng chiều rộng và chiều cao.
3. Thử `radius` 8 và `eps` 0.001. Tăng `radius` nếu mask lệch nhiều.
4. Xem lại bằng [Mask Preview](mask-preview.md) ở chế độ overlay và bật `show_edges`.

## Thông số

| Thông số | Mặc định | Tác dụng | Gợi ý |
|---|---|---|---|
| `radius` | 8 | Bộ lọc nhìn xung quanh mỗi pixel bao xa (pixel). Bán kính lớn sửa được mask lỏng hơn. | 1 đến 32. Chọn cỡ bằng độ lệch bạn cần sửa. |
| `eps` | 0.001 | Độ nhạy với cạnh. Nhỏ thì bám cạnh ảnh sát hơn; lớn thì kết quả mượt và mềm hơn. | 0.000001 đến 1. Cạnh sắc thử 0.0001, cạnh mềm thử 0.01. |
| `feather_sigma` | 0.0 | Blur Gaussian thêm sau khi lọc, tính bằng pixel. 0 = tắt. | 0 đến 64. Thử 1 đến 3 để hoàn thiện nhẹ. |

**Đầu ra:** `refined_mask` (MASK).

## Mẹo khi chỉnh ảnh

- Dọn mask trước bằng [Mask Cleanup](mask-cleanup.md), rồi mới refine viền.
- Nếu kết quả quá mềm, giảm `eps`. Nếu viền bám theo nhiễu hoặc texture, tăng `eps`.
- Đừng dùng `radius` rất lớn cho mask đã gần đúng, vì nó có thể kéo thêm chi tiết nền không mong muốn.
- Chạy trước [Mask Combine](mask-combine.md) hoặc bước ghép ảnh để mọi viền đã gọn sẵn.

??? note "Bên trong node hoạt động thế nào"
    Guided filter thang xám (He và cộng sự) tính trên độ sáng (luma) của ảnh guide (`0.299 R + 0.587 G + 0.114 B`). Thống kê cửa sổ lấy từ box filter gồm hai lần average pool 1 chiều với replicate padding, nên pixel sát mép ảnh không bị kéo về 0. Hệ số là `a = cov(I, p) / (var(I) + eps)` và `b = mean(p) - a * mean(I)`; kết quả là `a` đã box-filter nhân với guide cộng `b` đã box-filter, cắt về 0..1. Sau đó có thể blur Gaussian tùy chọn. Guide phải là RGB dạng float cùng chiều cao và chiều rộng với mask; guide batch 1 sẽ được nhân ra. Mã nguồn: `core/mask_edge_refine.py`.

## Node liên quan

- [Sub-Pixel Mask Refiner](sub-pixel-mask-refiner.md) để feather mà không cần ảnh guide
- [Alpha Matte](alpha-matte.md) cho tóc và cạnh bán trong suốt
- [Trimap Builder](trimap-builder.md) để chuẩn bị đầu vào cho matting
- [Mask Preview](mask-preview.md) để kiểm tra viền

## Workflow mẫu

[Workflow mẫu Edge-Aware Mask Refiner](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-28-edge-aware-mask-refiner.json). Mở workflow rồi trỏ node `Load Image` vào ảnh của bạn.
