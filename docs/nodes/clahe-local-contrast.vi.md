---
title: CLAHE Local Contrast
description: Local contrast thích ứng chỉ trên độ sáng, giúp vùng tối và vùng phẳng có chi tiết mà màu không đổi.
---

# CLAHE Local Contrast

Mở vùng tối và thêm chiều sâu cho vùng phẳng bằng cách chỉnh tương phản theo từng ô nhỏ, chỉ trên độ sáng, nên hue và màu sắc không bị lệch.

![CLAHE Local Contrast — trước / sau](../assets/nodes/clahe-local-contrast.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → color → Color: CLAHE Local Contrast` |
| **Dùng để** | Kéo chi tiết vùng bóng tối đục, cứu ảnh mờ sương hoặc phẳng, tăng độ "gắt" cho sản phẩm và kết cấu |

## Khi nào dùng

CLAHE (Contrast Limited Adaptive Histogram Equalisation, cân bằng histogram thích ứng có giới hạn) nhìn từng ô nhỏ của ảnh và trải đều các giá trị sáng trong ô đó, có giới hạn để nhiễu không bị thổi lên. Kết quả giống local contrast rất mạnh: góc tối hiện chi tiết, bề mặt phẳng có chiều sâu.

Node chỉ chạy trên kênh độ sáng của không gian màu CIE LAB, nên màu sắc không bị đụng tới. Nhiều công cụ chạy CLAHE trên các kênh RGB, làm lệch màu và ám da.

So với thanh `basic_clarity` trong [ColorLab](color-lab.md): clarity là cú "punch" nhẹ theo bán kính, tác động vào cạnh. CLAHE phân phối lại độ sáng trong từng ô, nên nó nâng được vùng bóng rất sâu và làm phẳng các dải chuyển mềm lớn mà clarity bỏ qua. Dùng clarity cho độ nổi nhẹ, dùng CLAHE khi vùng đó quá phẳng, quá tối hoặc quá mờ sương.

## Bắt đầu nhanh

1. **Load Image** → CLAHE Local Contrast → **Preview Image**.
2. Giữ `clip_limit` ở 2.0 và `grid_size` ở 8.
3. Hạ `strength` xuống khoảng 0.5 với chân dung, rồi so với ảnh gốc.

## Thông số

| Thông số | Mặc định | Tác dụng | Gợi ý |
|---|---|---|---|
| `image` | bắt buộc | Ảnh cần tăng cường. | |
| `clip_limit` | 2.0 (0.1 đến 40) | Mỗi ô được phép kéo giãn mạnh đến đâu. Cao hơn là tương phản mạnh hơn và nhiễu nhiều hơn. | 1.5 đến 3 cho đa số trường hợp; trên khoảng 6 hạt nổi rõ. |
| `grid_size` | 8 (2 đến 32) | Số ô theo chiều ngang và dọc. Ít ô cho độ nâng rộng, tự nhiên; nhiều ô cho vẻ gắt, nhiều chi tiết. | 4 đến 6 tự nhiên, 12 đến 24 gắt. |
| `strength` | 1.0 (0 đến 1) | Pha trộn giữa ảnh gốc (0) và kết quả CLAHE (1). | 0.3 đến 0.6 cho chân dung. |
| `mask` (tùy chọn) | không | Giới hạn vùng áp dụng. Một mask duy nhất được dùng cho cả batch. | Loại khuôn mặt hoặc bầu trời ra khỏi hiệu ứng. |

**Đầu ra:** `image`, ảnh đã tăng cường.

Với ảnh nhỏ, lưới ô tự giảm để mỗi ô rộng ít nhất 80 px. Ảnh nhỏ hơn 80 px được trả về nguyên vẹn.

## Mẹo khi chỉnh ảnh

- Bắt đầu nhẹ. `clip_limit` 2 với `strength` 0.5 thường đẹp hơn 4 ở full.
- Với chân dung, mask bỏ vùng mặt và chỉ áp lên quần áo, tóc hoặc nền; da hiếm khi hợp local contrast mạnh như vậy.
- Nếu vùng bóng bị hạt, hạ `clip_limit` trước, rồi mới hạ `strength`.
- Sau đó kết hợp [Smart Sharpen](smart-sharpen.md) cho cạnh sắc nét, và đặt [Tone Curve](tone-curve.md) ở cuối để tạo dáng tương phản cuối cùng.
- Vùng màu phẳng vẫn giữ phẳng, nên nền đơn sắc không bị ám màu.

??? note "Bên trong node hoạt động thế nào"
    Ảnh được chuyển sang CIE LAB (Kornia). Chỉ kênh `L*` (chia 100) đi qua `kornia.enhance.equalize_clahe` trên lưới `grid_size × grid_size`; `a*` và `b*` gốc được giữ, kết quả đổi về RGB và kẹp 0..1. Ảnh cuối là `lerp(original, enhanced, strength × mask)`. Kornia dùng histogram 256 bin nên `L*` bị lượng tử hóa còn 256 mức. Ngưỡng cắt của nó làm lệch các ô quá nhỏ (ảnh phẳng có thể sáng dần lên), vì vậy ô luôn được giữ từ 80 px trở lên. `strength = 0` trả lại ảnh gốc.

## Node liên quan

- [ColorLab](color-lab.md): clarity, texture và dehaze trong bảng kiểu Camera Raw
- [Smart Sharpen](smart-sharpen.md): làm nét cạnh sau khi nâng tương phản
- [Tone Curve](tone-curve.md): tạo dáng tương phản toàn ảnh
- [Luminosity Masking](luminosity-masking.md): giới hạn hiệu ứng ở vùng bóng hoặc vùng sáng

## Workflow mẫu

[Workflow mẫu CLAHE Local Contrast](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-37-clahe-local-contrast.json). Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/` trước.
