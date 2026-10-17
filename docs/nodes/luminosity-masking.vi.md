---
title: Luminosity Masking
description: Tạo ba mask mềm cho vùng bóng, midtone và vùng sáng dựa trên độ sáng của ảnh.
---

# Luminosity Masking

Chia ảnh thành ba mask mềm theo độ sáng (shadow, midtone, highlight), cùng ý tưởng với luminosity mask trong Photoshop, để bạn chỉ grade hoặc retouch một vùng tông.

![Luminosity Masking — trước / sau](../assets/nodes/luminosity-masking.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → color → Luminosity Masking` |
| **Dùng để** | Dodge & burn, grade vùng bóng và vùng sáng riêng, bảo vệ highlight, làm nét hoặc khử nhiễu có chọn lọc |

## Khi nào dùng

Luminosity mask chọn pixel theo độ sáng chứ không theo vị trí. Nhờ vậy ta có vùng chọn sạch, không có viền cắt: chỉ nâng vùng tối, chỉ làm lạnh vùng sáng, hoặc chỉ làm nét midtone mà không phải vẽ mask.

Ở đây ba mask đều mềm và cộng lại bằng 1 tại mọi pixel, nên shadow, midtone và highlight hòa vào nhau mà không hở hay chồng lấp. Bạn điều khiển chỗ các dải chuyển giao (`shadow_end`, `highlight_start`) và độ êm của chuyển tiếp (`soft_edge`).

## Bắt đầu nhanh

1. **Load Image** → Luminosity Masking.
2. Nối đầu ra `mask_shadows` vào **Mask to Image** → **Preview Image** để xem vùng chọn.
3. Dùng mask làm đầu vào mask của node khác, ví dụ [Tone Curve](tone-curve.md) hoặc [Color Matcher](color-matcher.md), để giới hạn hiệu ứng ở vùng bóng.

## Thông số

| Thông số | Mặc định | Tác dụng | Gợi ý |
|---|---|---|---|
| `image` | bắt buộc | Ảnh cần phân tích. | |
| `luminance_source` | `lab_l` | Cách đo độ sáng. `lab_l` theo cảm nhận thị giác (gần với những gì bạn thấy). `ycbcr_y` nhanh, hợp để xem trước. `hsv_v` là giá trị lớn nhất của R, G, B, kém sát cảm nhận hơn. | Bản cuối: `lab_l`. |
| `shadow_end` | 0.33 (0 đến 0.5) | Mức sáng nơi dải shadow kết thúc. | Hạ thấp để chỉ lấy vùng tối nhất. |
| `highlight_start` | 0.67 (0.5 đến 1.0) | Mức sáng nơi dải highlight bắt đầu. Phải lớn hơn `shadow_end`. | Nâng lên để chỉ lấy điểm sáng chói. |
| `soft_edge` | 0.1 (0.01 đến 0.3) | Độ rộng chuyển tiếp mượt ở hai rìa. Nhỏ cho dải sắc hơn, lớn cho pha trộn êm hơn. | 0.1 đến 0.2 để tránh banding trên da. |

**Đầu ra:** `mask_shadows`, `mask_midtones` và `mask_highlights`, mỗi cái là mask mềm từ 0 đến 1.

## Mẹo khi chỉnh ảnh

- Dodge & burn: dùng `mask_shadows` để nâng chi tiết và `mask_highlights` để kéo vùng cháy xuống.
- Giữ `soft_edge` rộng (từ 0.15) khi mask sẽ giới hạn những chỉnh sửa mạnh, để không lộ ranh giới tông.
- Thu hẹp dải midtone (đẩy `shadow_end` lên, hạ `highlight_start` xuống) để có mask "chỉ midtone" cho clarity hoặc làm mịn da.
- Kết hợp hai mask bằng [Mask Combine](mask-combine.md) (union, subtract, intersect...) để tạo dải riêng như "tối cộng midtone".
- Với ảnh lớn, `ycbcr_y` nhanh hơn `lab_l` nhiều, nên dùng khi xem trước rồi chuyển sang `lab_l` cho lần chạy cuối.

??? note "Bên trong node hoạt động thế nào"
    Độ sáng được tính trong không gian đã chọn: `lab_l` chuyển sang RGB tuyến tính rồi dùng CIE L* (0 đến 1), `ycbcr_y` dùng kênh Y của YCbCr (Kornia), `hsv_v` dùng kênh lớn nhất. Shadow là `1 − smoothstep(shadow_end ± soft_edge/2)`, highlight là `smoothstep(highlight_start ± soft_edge/2)`, midtone là phần còn lại (không nhỏ hơn 0). Ba mask sau đó được chia cho tổng, nên cộng lại bằng 1 tại mọi pixel. Chú thích trong code ghi khoảng 120 ms cho `lab_l` ở 2K trên CPU và khoảng 7 ms cho `ycbcr_y` ở 1024 trên CPU; hãy xem đó là số tham khảo. `shadow_end` phải nhỏ hơn `highlight_start`, và đầu vào ngoài 0..1 bị kẹp lại kèm cảnh báo trong log.

## Node liên quan

- [Mask Combine](mask-combine.md): cộng, trừ hoặc giao các mask
- [Tone Curve](tone-curve.md): chỉ áp đường cong lên một vùng tông
- [ColorLab](color-lab.md): tông và màu toàn ảnh; dùng mask để giới hạn
- [Frequency Separation](frequency-separation.md): kết hợp với mask midtone khi retouch da

## Workflow mẫu

[Workflow mẫu Luminosity Masking](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-05-luminosity-masking.json). Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/` trước.
