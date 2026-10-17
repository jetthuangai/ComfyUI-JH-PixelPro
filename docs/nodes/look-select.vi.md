---
title: Look Select
description: Áp một look màu sáng tạo hoàn chỉnh từ danh sách chọn, có thể bảo vệ màu da.
---

# Look Select

Chọn một look hoàn chỉnh như teal and orange, phim phai màu hay golden hour, rồi pha vào ảnh bằng một thanh trượt. Có thể giữ màu da ra khỏi hiệu ứng.

![Look Select — trước / sau](../assets/nodes/look-select.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → looks → Look: Select Preset` |
| **Phù hợp để** | Thử nhanh nhiều look, giữ phong cách đồng nhất cho cả loạt ảnh, thêm không khí mà không phải dựng chuỗi node |

## Khi nào dùng

Mỗi look là một công thức ngắn gồm các bước màu: dịch màu ở vùng tối hoặc vùng sáng (trong không gian màu LAB), một đường cong tương phản nhẹ, và chỉnh saturation ở những dải màu chọn lọc. Look Select chạy công thức đó giúp bạn rồi pha kết quả với ảnh gốc.

Khác với [LUT Preset](lut-preset.md): preset LUT là bảng màu cố định. Look ở đây được dựng từ chính các công cụ màu của pack, nên bảo vệ được da và xử lý màu có chọn lọc. Cả hai đều cho kết quả nhanh; với file `.cube` riêng, dùng [LUT Import](lut-import.md).

## Sáu look

| Preset | Đặc điểm |
|---|---|
| `cinematic-teal-orange` | Look điện ảnh kinh điển: bóng teal, vùng sáng cam, S-curve nhẹ. |
| `warm-skin-tone` | Ấm mà an toàn cho chân dung: nâng dải cam, dịu màu xanh lá, bảo vệ vùng sáng. |
| `moody-green` | Xanh lá đậm kiểu editorial, bóng bị nén, vùng sáng dịu. |
| `faded-film` | Vùng đen được nâng lên, trung tính ấm và giảm saturation nhẹ, như bản in phim mềm. |
| `golden-hour` | Sắc ấm cuối ngày, cam và vàng mạnh hơn, giữ nguyên xanh dương. |
| `desaturated-pop` | Màu nền dịu đi trong khi da cam-đỏ vẫn tươi. |

## Bắt đầu nhanh

1. Nối **Load Image → Look: Select Preset → Preview Image**.
2. Chọn `preset`. Giữ `intensity` ở 0.7.
3. Bật `protect_skin` nếu khuôn mặt bị ngả cam, ngả xanh hoặc nhợt.
4. Muốn so sánh cả sáu look cùng lúc, mở workflow so sánh sáu look bên dưới.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `preset` | `cinematic-teal-orange` | Look cần áp. | Xem bảng ở trên. |
| `intensity` | 0.70 | Pha giữa ảnh gốc (0) và look đầy đủ (1). Ở 0, ảnh được trả về nguyên vẹn. | 0.4 đến 0.7 cho kết quả tự nhiên; 1.0 cho look đậm chất. |
| `protect_skin` | tắt | Giảm tác động của từng bước look lên các pixel có màu da. | Thử với `cinematic-teal-orange`, `golden-hour` và `moody-green`. |

**Ngõ ra:** `image` — ảnh đã chỉnh màu.

## Mẹo khi chỉnh ảnh

- `warm-skin-tone` và `desaturated-pop` được thiết kế xoay quanh màu da. Trong workflow so sánh chúng chạy với `protect_skin` bật, nhưng công tắc không tự bật, nên bạn tự đặt.
- Look đi cùng tốt với một bước chỉnh tương phản cuối bằng [Tone Curve](tone-curve.md).
- Cần kiểm soát chính xác một màu, chuyển sang [Hue/Saturation per Range](hue-saturation-range.md) và [Saturation Mask](saturation-mask.md).
- Muốn giữ cùng một look cho cả loạt ảnh, cố định `preset` và `intensity` rồi chạy cả batch qua cùng node.
- Bảo vệ da dựa trên màu sắc nên chỉ là ước đoán. Gỗ ấm hay cát cũng có thể được bảo vệ. Hãy kiểm tra kết quả.

??? note "Bên trong node hoạt động thế nào"
    Preset là các file JSON trong thư mục `presets/` của pack với danh sách thao tác; `nodes/look_base.py` chạy chúng và pha bằng `lerp(ảnh gốc, ảnh xử lý, intensity)`. Các thao tác gồm: dịch màu LAB (trọng số toàn cục, bóng, vùng sáng, midtone), tone curve RGB, đổi hue/saturation trong một dải hue, và đổi saturation trong một khoảng saturation. Khi bật `protect_skin`, mask da (hue quanh 24° với dải 28°, nhân với khoảng saturation 0.18 đến 0.95) làm yếu thay đổi của từng bước (thay đổi được nhân với một trừ mask). Kết quả được cắt về `[0, 1]`. Danh sách chọn là danh sách cố định trong `nodes/look_select.py`, nên chỉ thêm file JSON thì chưa thêm được look. Trường `protect_skin_default` trong các file JSON không được node đọc.

## Node liên quan

- [LUT Preset](lut-preset.md) — look `.cube` đi kèm.
- [LUT Import](lut-import.md) — file LUT của riêng bạn.
- [Hue/Saturation per Range](hue-saturation-range.md) — công cụ màu dùng bên trong nhiều look.
- [Tone Curve](tone-curve.md) — điều khiển tương phản dùng bên trong nhiều look.
- [Skin Tone Tri-Region](skin-tone-tri-region.md) — chỉnh tông da chuyên dụng.

## Workflow mẫu

- [Look Select single example](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-19-look-select-single.json) — một look trên một ảnh.
- [Look Select six-look compare example](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-20-look-select-compare-6up.json) — cả sáu look cạnh nhau.

Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/` rồi chọn trong Load Image.
