---
title: Apply ColorLab Preset
description: Áp bộ thông số ColorLab đã lưu lên ảnh bất kỳ, với một thanh cường độ và mask tùy chọn.
---

# Apply ColorLab Preset

Lấy bộ thông số từ node ColorLab hoặc Lightroom Preset rồi áp cùng look đó lên ảnh khác. Một thanh cường độ làm look nhẹ đi hay đậm lên, giống thanh Amount của preset trong Lightroom.

![Apply ColorLab Preset — ảnh gốc, preset ở 0.5 và preset ở 1.0](../assets/nodes/apply-colorlab-preset.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → color → Color: Apply ColorLab Preset` |
| **Dùng để** | Áp look đã dựng trên một tấm cho cả loạt ảnh, giảm hoặc tăng độ đậm của look, dùng preset Lightroom ở mức nhẹ hơn, chỉ grade bên trong mask |

## Khi nào dùng

Bạn grade một tấm chủ đạo trong [ColorLab](color-lab.md) cho tới khi ưng. Đầu ra `params_json` của ColorLab giữ bộ thông số đó dưới dạng preset. Nối nó vào node này, mọi ảnh bạn đưa vào đều được grade y như vậy, không cần chép tay từng thanh trượt.

Thanh cường độ co giãn lượng chỉnh của preset, giống Amount của preset trong Lightroom. Exposure, contrast, HSL, độ đậm màu nhuộm grading… cùng giảm hoặc tăng theo. Những gì mô tả *màu nào* hay *hình dạng ra sao* thì giữ nguyên: hue của color grading (vùng tối nhuộm teal vẫn là teal ở mọi cường độ) và các thông số hình dạng của vignette và grain. Đường cong điểm (point curve) duỗi dần về đường thẳng khi giảm cường độ.

Node cũng nhận `params_json` từ [Lightroom Preset](lightroom-preset.md), nên bạn nạp preset một lần rồi dùng lại ở nhiều mức cường độ khác nhau.

## Bắt đầu nhanh

1. **Load Image** → [ColorLab](color-lab.md). Grade ảnh.
2. Thêm **Color: Apply ColorLab Preset**. Nối đầu ra `params_json` của ColorLab vào đầu vào `params_json`.
3. Nối ảnh (hoặc batch ảnh) cần grade vào `image`, rồi đầu ra `image` → **Preview Image**.
4. Bắt đầu với `strength` 1.0, sau đó thử 0.5 đến 0.8 để có bản nhẹ hơn.

## Thông số

| Thông số | Mặc định | Tác dụng | Gợi ý |
|---|---|---|---|
| `image` | | Ảnh hoặc batch ảnh cần grade. | |
| `params_json` | | Preset, dưới dạng thông số ColorLab. Chỉ nhận dây nối: từ ColorLab, Lightroom Preset hoặc một node văn bản chứa preset đã lưu. | JSON sai sẽ dừng lượt chạy với thông báo lỗi; tên thông số lạ bị bỏ qua kèm cảnh báo. |
| `strength` | 1.0 | 0 đến 2. Co giãn mọi lượng chỉnh trong preset. 0 trả lại ảnh gốc không đổi; trên 1 đẩy look mạnh hơn. | 0.5 đến 0.8 cho look tinh tế, 1.2 đến 1.5 để làm nổi một preset quá nhẹ. |
| `mask` (tùy chọn) | không | Chỉ áp preset ở vùng mask trắng, chuyển mềm ở vùng xám. | Mask phải cùng kích thước với ảnh. Tạo bằng [Luminosity Masking](luminosity-masking.md) hoặc mask chủ thể. |

Những gì cường độ **không** co giãn:

- Hue của grading (`grade_*_hue`): màu nhuộm giữ nguyên, chỉ độ đậm thay đổi.
- Midpoint, feather, roundness của vignette, và size, độ thô của grain: giữ hình dạng, chỉ co giãn lượng.
- Công tắc bật/tắt và thông số dạng chữ: `gray_enable`, `exposure_mode`, `tone_engine`. Riêng preset đen trắng thì được trộn dần: dưới cường độ 1, bản đen trắng được phủ lên bản màu, nên 0.5 còn khoảng một nửa màu (giống Amount của preset trong Lightroom).

**Đầu ra**

| Đầu ra | Là gì |
|---|---|
| `image` | Ảnh đã áp preset. |
| `params_json` | Preset ở cường độ đã chọn. Lưu lại, hoặc nối vào [ColorLab](color-lab.md) để tinh chỉnh bằng thanh trượt. |

## Mẹo khi chỉnh ảnh

- Với một bộ ảnh (tiệc cưới, chụp sản phẩm), grade tấm tiêu biểu nhất rồi cho cả loạt qua node này. Kiểm tra tấm tối nhất và sáng nhất trước khi tin look hợp với tất cả.
- Preset làm trên ảnh sáng có thể quá nặng với ảnh tối. Hạ `strength` thay vì dựng lại từ đầu.
- Khi một tấm cần sửa nhỏ, nối `params_json` vào đầu vào `params_json` của một ColorLab: ColorLab cộng thanh trượt của nó lên trên preset.
- Chỉ đưa ảnh gốc vào đây, không đưa ảnh ColorLab đã grade, nếu không look bị áp hai lần.
- Look của bạn đang là file `.cube` chứ không phải thông số ColorLab? Dùng [LUT Import](lut-import.md), node này cũng có thanh cường độ và đầu vào mask.

??? note "Bên trong node hoạt động thế nào"
    Preset được đọc theo cùng quy tắc với đầu vào `params_json` của ColorLab, bắt đầu từ trạng thái trung tính. Mọi giá trị số được nhân với `strength`, trừ hue của grading và các thông số hình dạng của vignette và grain. Đường cong điểm được trộn về đường đồng nhất: mỗi điểm `[x, y]` thành `[x, x + (y - x) · strength]`. Bộ thông số đã co giãn chạy qua toàn bộ chuỗi ColorLab (`core/color_lab.py`, `scale_params_json` và `apply_colorlab_pipeline`). Với `strength` 0, ảnh vào được trả về nguyên vẹn. Khi có mask, kết quả là `gốc + (đã grade - gốc) · mask`; mask phải có dạng `(B, H, W)` cùng chiều cao và chiều rộng với ảnh, nếu không node báo lỗi. Đầu ra `params_json` chỉ liệt kê các giá trị khác trung tính.

## Node liên quan

- [ColorLab](color-lab.md): dựng look và xuất ra `params_json`
- [Lightroom Preset](lightroom-preset.md): đổi preset Lightroom hoặc Camera Raw thành thông số ColorLab
- [LUT Import](lut-import.md): áp look lưu dạng file `.cube`, có cường độ và mask
- [Luminosity Masking](luminosity-masking.md): giới hạn preset trong vùng tối, midtone hoặc vùng sáng

## Workflow mẫu

[Workflow mẫu Apply ColorLab Preset](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-51-apply-colorlab-preset.json): ColorLab grade ảnh chân dung, rồi `params_json` của nó điều khiển Apply ColorLab Preset ở cường độ 0.5. Workflow chỉ dùng node gốc của ComfyUI và pack này. Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/`.

![Kết quả: workflow mẫu Apply ColorLab Preset](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/N-51-apply-colorlab-preset-preview.png)
