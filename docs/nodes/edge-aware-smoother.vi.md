---
title: Skin Smoother
description: Làm mịn da, giảm loang màu nhưng giữ sắc nét các cạnh như mắt, môi, tóc.
---

# Skin Smoother

Làm mềm tông da và các khuyết điểm nhỏ bằng bộ lọc blur giữ cạnh (bilateral). Vùng mịn sẽ mịn hơn, còn các cạnh mạnh như mí mắt, môi, sợi tóc vẫn rõ, nên khuôn mặt không bị "nhựa".

![Skin Smoother — trước / sau](../assets/nodes/edge-aware-smoother.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → filters → Edge-Aware Skin Smoother` |
| **Dùng cho** | Làm sạch da chân dung, đều tông da, làm mềm texture bên trong mask da |

## Khi nào dùng

Blur thông thường làm mềm tất cả, kể cả những cạnh bạn muốn giữ. Bộ lọc bilateral chỉ lấy trung bình pixel với các pixel lân cận có độ sáng và màu tương tự. Da (giá trị gần nhau) được làm mịn. Cạnh mắt (giá trị chênh lệch lớn) thì không.

Dùng `strength` để trộn kết quả đã làm mịn với ảnh gốc, giống như hạ opacity của một layer làm mịn. Liều thường dùng khi retouch là 0.3 đến 0.5.

## Bắt đầu nhanh

1. `Load Image` → **Edge-Aware Skin Smoother** → `Preview Image`.
2. Bắt đầu với `strength = 0.4`, `sigma_color = 0.1`, `sigma_space = 6`.
3. **Bật `tile_mode` với ảnh cỡ bình thường** (xem phần bộ nhớ bên dưới).
4. Nối `MASK` vùng da vào `mask` để chỉ làm mịn trên da.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `strength` | `0.4` (0 đến 1) | Trộn giữa ảnh gốc (0) và ảnh làm mịn hoàn toàn (1). | 0.3 đến 0.5 cho da tự nhiên. 0 trả ảnh nguyên vẹn. |
| `sigma_color` | `0.1` (0.01 đến 0.5) | Hai màu được phép chênh nhau bao nhiêu mà vẫn được lấy trung bình. Tính trên thang 0 đến 1, không phải giá trị 8-bit. | Nhỏ thì giữ cạnh nhiều hơn, lớn thì làm mịn qua cả cạnh yếu. |
| `sigma_space` | `6.0` (1 đến 8) | Tầm ảnh hưởng của làm mịn, tính bằng pixel. | Giảm với ảnh nhỏ. Muốn rộng hơn thì thu nhỏ ảnh trước bằng node resize. |
| `device` | `auto` | `auto` dùng CUDA nếu có, không thì CPU. `cuda` báo lỗi nếu không có CUDA. `cpu` ép dùng CPU. | |
| `tile_mode` | tắt | Xử lý ảnh theo từng ô (tile) để hạn chế bộ nhớ. | Xem bên dưới. |
| `mask` (tùy chọn) | không | Nơi áp dụng làm mịn: trắng = làm mịn, đen = giữ nguyên. | Mask da đã feather giúp tránh lộ viền. |

Đầu ra: `image`, ảnh đã làm mịn (`IMAGE`).

### Bộ nhớ và `tile_mode`

Bộ lọc bilateral tốn nhiều bộ nhớ: xấp xỉ số pixel nhân với bình phương kích thước kernel.

- **Tắt tile mode:** node ước tính bộ nhớ cần dùng. Nếu vượt khoảng 2 GB thì dừng và báo lỗi, gợi ý bật `tile_mode`, giảm `sigma_space` hoặc thu nhỏ ảnh. Với `sigma_space = 6` mặc định, giới hạn chỉ khoảng 0.1 megapixel, nên thực tế bạn sẽ cần bật `tile_mode` cho ảnh chụp thật.
- **Bật tile mode:** ảnh được cắt thành các ô có kích thước chọn sao cho mỗi ô nằm trong ngân sách bộ nhớ khoảng 2 GB. Ô rộng tối đa 512 px và nhỏ hơn khi `sigma_space` lớn (ở giá trị mặc định ô chỉ khoảng 120 px). Mỗi ô có thêm một lề pixel lân cận nên bạn không thấy đường nối.

Gợi ý hiển thị trên widget `tile_mode` vẫn nói ô cố định 512 px; thực tế kích thước ô được tính từ ngân sách bộ nhớ như mô tả ở đây.

## Mẹo khi chỉnh ảnh

- Làm nhẹ hai lần tốt hơn làm mạnh một lần. Hai lượt 0.3 trông tự nhiên hơn một lượt 0.8.
- Kết hợp với mask da và loại mắt, môi, lông mày, tóc ra khỏi mask. [Skin Tone Tri-Region](skin-tone-tri-region.md) hoặc các node Mask có thể giúp.
- Thêm lại texture bằng [Smart Sharpen](smart-sharpen.md) (chỉ luminance), hoặc dùng [Frequency Separation](frequency-separation.md) trước để chỉ làm mịn lớp màu.
- Kiểm tra bằng [QA Image Diff](qa-image-diff.md) để xem pixel nào đã thay đổi.

??? note "Bên trong node hoạt động thế nào"
    Dùng `bilateral_blur` của Kornia với khoảng cách màu L1 và biên replicate. Kích thước kernel là `2 × ceil(3 × sigma_space) + 1`. Kết quả là `strength × smoothed + (1 − strength) × original`, rồi trộn với ảnh gốc qua mask. Nếu `strength` bằng 0, hoặc mask toàn 0, ảnh vào được trả lại nguyên vẹn. Khi không chia ô, bộ nhớ ước tính là `B × C × H × W × kernel² × 4` byte và vượt ngân sách 2.000.000.000 byte sẽ báo lỗi. Khi chia ô, cạnh ô là giá trị lớn nhất để ô (kể cả lề) nằm trong ngân sách đó (có hệ số an toàn cho các bộ đệm tạm của Kornia), tối đa 512 và không nhỏ hơn 16. Nguồn: `core/smoother.py` và `nodes/smoother_node.py`.

## Node liên quan

- [Frequency Separation](frequency-separation.md)
- [Smart Sharpen](smart-sharpen.md)
- [Skin Tone Tri-Region](skin-tone-tri-region.md)
- [QA Image Diff](qa-image-diff.md)

## Workflow mẫu

[Workflow mẫu Skin Smoother](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-03-edge-aware-smoother.json). Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/`. Workflow hiển thị ảnh gốc và ảnh đã làm mịn cạnh nhau.
