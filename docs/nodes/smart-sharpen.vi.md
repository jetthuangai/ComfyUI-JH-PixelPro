---
title: Smart Sharpen
description: Làm nét chi tiết bằng unsharp mask, bỏ qua nhiễu, giữ nguyên màu và có thể giới hạn bằng mask.
---

# Smart Sharpen

Làm cạnh và texture sắc nét hơn bằng unsharp mask. Ngưỡng (threshold) giúp không khuếch đại nhiễu và hạt phim, mặc định chỉ làm nét độ sáng nên màu không bị viền, và mask tùy chọn cho phép bạn chỉ làm nét vùng mình chọn.

![Smart Sharpen — trước / sau](../assets/nodes/smart-sharpen.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → filters → Filters: Smart Sharpen` |
| **Dùng cho** | Làm nét bản xuất cuối, lấy lại chi tiết mắt và tóc, làm nét vùng nhiều texture mà để da và bầu trời mịn |

## Khi nào dùng

Unsharp mask cùng ý tưởng với bộ lọc Unsharp Mask của Photoshop: làm mờ một bản sao, trừ khỏi ảnh gốc để tìm chi tiết, rồi cộng chi tiết đó lại với cường độ cao hơn.

Dùng node này ở gần cuối chuỗi, sau khi làm mịn, chỉnh màu và đổi kích thước.

## Bắt đầu nhanh

1. `Load Image` → **Filters: Smart Sharpen** → `Preview Image`.
2. Bắt đầu với `amount = 1.0`, `radius = 1.0`, `threshold = 0`.
3. Với ảnh nhiều hạt, tăng `threshold` lên khoảng 0.02 đến 0.05.
4. Muốn chỉ làm nét vùng có chi tiết, nối mask từ [Detail Mask](high-freq-detail-masker.md) vào `mask`.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `amount` | `1.0` (0 đến 5) | Cường độ. 0 trả ảnh nguyên vẹn. | 0.5 đến 1 thường đủ. Trên 2 dễ xuất hiện quầng sáng (halo). |
| `radius` | `1.0` (0.3 đến 10) | Kích thước chi tiết được tăng cường, tính bằng pixel (sigma của blur). | 0.5 đến 1 cho chi tiết mịn và ảnh web, 1.5 đến 3 cho ảnh in lớn. Trên 5 thì giống tăng local contrast hơn là làm nét. |
| `threshold` | `0.0` (0 đến 0.5) | Chi tiết yếu hơn mức này sẽ không bị làm nét. | Tăng lên để tránh khuếch đại nhiễu hoặc hạt phim. |
| `luminance_only` | bật | Chỉ làm nét độ sáng, giữ nguyên màu. | Để bật để tránh viền màu ở cạnh. |
| `mask` (tùy chọn) | không | Nơi làm nét: trắng = đủ, đen = không. | Batch bằng 1 được dùng cho mọi ảnh. |

Đầu ra: `image`, ảnh đã làm nét (`IMAGE`).

## Mẹo khi chỉnh ảnh

- Làm nét sau [Skin Smoother](edge-aware-smoother.md), đừng làm trước, kẻo bạn làm nét luôn các khuyết điểm sắp bị làm mịn.
- Mask gồm mắt, lông mi, tóc và vải cho chi tiết sắc mà da không bị thô.
- Ảnh phải lớn hơn `ceil(3 × radius)` pixel ở cả hai cạnh, nếu không node sẽ báo lỗi.
- Dùng [QA Image Diff](qa-image-diff.md) để kiểm tra việc làm nét chỉ thay đổi đúng chỗ bạn muốn.

??? note "Bên trong node hoạt động thế nào"
    `detail = x − gaussian_blur(x, sigma = radius)` với biên reflect và kernel `2 × ceil(3 × radius) + 1`. Chi tiết được soft-threshold, `sign(d) × max(|d| − threshold, 0)`, là đường dốc liên tục nên không có đốm ở ngưỡng cắt. Sau đó `out = x + amount × detail × mask`, giới hạn trong 0..1. Ở chế độ luminance, chi tiết lấy từ luma BT.601 và được cộng đều vào R, G, B, nên chỉ đổi độ sáng. Nguồn: `core/sharpen.py` và `nodes/sharpen_node.py`.

## Node liên quan

- [Detail Mask](high-freq-detail-masker.md)
- [Skin Smoother](edge-aware-smoother.md)
- [Frequency Separation](frequency-separation.md)
- [QA Image Diff](qa-image-diff.md)

## Workflow mẫu

[Workflow mẫu Smart Sharpen](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-40-smart-sharpen.json). Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/`.

![Kết quả Smart Sharpen](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/N-40-smart-sharpen-preview.png)
