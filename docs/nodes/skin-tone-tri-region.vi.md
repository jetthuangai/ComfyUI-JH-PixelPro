---
title: Skin Tone Tri-Region
description: Tách ảnh chân dung thành mask shadow, midtone và highlight để retouch từng dải tông riêng.
---

# Skin Tone Tri-Region

Tách ảnh chân dung thành ba mask mềm: vùng da tối, vùng da trung bình và vùng da sáng nhất. Bạn có thể retouch shadow, midtone và highlight của da một cách độc lập.

![Skin Tone Tri-Region — trước / sau](../assets/nodes/skin-tone-tri-region.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → face → Face: Skin Tone Tri-Region` |
| **Dùng để** | Giảm bóng dầu ở highlight; nâng hoặc làm ấm shadow; chỉnh tông da chính; dodge & burn có chọn lọc |

## Khi nào dùng

Da hiếm khi cần cùng một cách sửa ở mọi chỗ. Highlight cần kiểm soát bóng dầu, shadow có thể cần nâng sáng hoặc làm ấm, còn midtone mang màu da chính. Mask theo độ sáng (luminosity mask, tức mask tạo từ độ sáng của từng pixel) cho phép bạn nhắm vào đúng một dải. Node này tạo ba mask như vậy chỉ trong một bước và đảm bảo chúng cộng lại khớp nhau.

Hãy nối thêm một mask da (tùy chọn) để tóc, quần áo và hậu cảnh không tham gia vào phép tách. Nếu không có, cả khung hình sẽ được tách theo độ sáng.

## Bắt đầu nhanh

1. Nối **Load Image** vào `image`.
2. Tùy chọn nhưng nên làm: nối một mask chỉ có da (từ node tách người hoặc tách da) vào `skin_mask`.
3. Đưa `highlight_mask`, `midtone_mask` hoặc `shadow_mask` vào ô mask của bước chỉnh bạn muốn giới hạn, ví dụ [Face Beauty Blend](face-beauty-blend.md), một node chỉnh màu hoặc LUT.
4. Bắt đầu với ngưỡng mặc định rồi tinh chỉnh theo ảnh của bạn.

## Thông số

| Thông số | Mặc định | Tác dụng | Gợi ý |
|---|---|---|---|
| `image` | — | Ảnh chân dung cần phân tích. | |
| `skin_mask` | tùy chọn | Giới hạn phép tách trong vùng da. Không có thì tách cả khung hình. | Mask tốt cho kết quả sạch hơn nhiều so với cả khung. |
| `shadow_cutoff` | `0.33` (0–0.5) | Độ sáng dưới mức này được tính là shadow. | Hạ xuống để thu nhỏ mask shadow. |
| `highlight_cutoff` | `0.66` (0.5–1) | Độ sáng trên mức này được tính là highlight. | Nâng lên để chỉ giữ lại phần bóng sáng nhất. |
| `soft_sigma` | `1.0` (0–8) | Làm mềm ranh giới giữa ba dải. | Tăng để chuyển tiếp giữa các dải êm hơn. `0` cho viền cứng. |

| Đầu ra | Kiểu | Ý nghĩa |
|---|---|---|
| `shadow_mask` | `MASK` | Vùng da tối. |
| `midtone_mask` | `MASK` | Vùng da trung bình. |
| `highlight_mask` | `MASK` | Vùng da sáng nhất. |

## Mẹo khi chỉnh ảnh

- `shadow_cutoff` phải nhỏ hơn `highlight_cutoff`; node từ chối các cặp giá trị khác.
- Với ảnh chân dung rất tối hoặc rất sáng, hãy dịch cả hai ngưỡng cùng chiều để mỗi mask vẫn chiếm một phần hữu ích của khuôn mặt.
- Ba mask luôn cộng lại bằng mask da của bạn (hoặc cả khung trắng nếu không nối mask), nên không có phần nào bị mất hay bị tính hai lần.
- Dùng `highlight_mask` cho bóng dầu và màu vùng sáng; `midtone_mask` cho chỉnh tông da chính; `shadow_mask` cho nâng sáng và giảm noise.
- Hãy xem trước mask bằng [Mask Preview](mask-preview.md) trước khi dùng.

??? note "Bên trong node hoạt động thế nào"
    Độ sáng là luma Rec. 601, `0.299 R + 0.587 G + 0.114 B`, tính từ ảnh đã kẹp vào `[0, 1]`. Các mask cứng được tạo từ hai ngưỡng, nhân với mask da, làm mờ bằng Gaussian tách trục (đệm replicate), rồi chuẩn hóa lại để ba mask cộng đúng bằng mask da. Ngưỡng phải thỏa `0 ≤ shadow < highlight ≤ 1`, với `shadow ≤ 0.5` và `highlight ≥ 0.5`. `skin_mask` phải khớp chiều cao và chiều rộng của ảnh. Mã nguồn nằm ở `core/skin_tone_region.py`.

## Node liên quan

- [Face Beauty Blend](face-beauty-blend.md) — trộn phần retouch qua một trong các mask này.
- [Luminosity Masking](luminosity-masking.md) — mask độ sáng linh hoạt hơn cho mọi ảnh.
- [Mask Combine](mask-combine.md) — kết hợp các mask này với mask khác.
- [Tổng quan nhóm Khuôn mặt](../categories/face.md)

## Workflow mẫu

[Workflow mẫu Skin Tone Tri-Region](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-35-skin-tone-tri-region.json). Workflow dùng một node tách người từ pack custom node khác để tạo mask da. Xem [ví dụ Retouch tông da](../examples/skin-tone-retouch.md) để có hướng dẫn đầy đủ.
