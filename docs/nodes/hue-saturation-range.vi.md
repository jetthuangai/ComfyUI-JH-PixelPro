---
title: Hue/Saturation per Range
description: Đổi hue hoặc độ bão hòa của một dải màu, giống chỉnh có chọn lọc trong bảng Hue/Saturation.
---

# Hue/Saturation per Range

Chọn một màu trên vòng hue và chỉ đổi màu đó: cho lá xanh ngả vàng hơn một chút, dịu một màu đỏ quá chói, hoặc làm xanh da trời sâu hơn.

![Hue/Saturation per Range — trước / sau](../assets/nodes/hue-saturation-range.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → color → Hue/Saturation per Range` |
| **Dùng để** | Dịu một màu quá chói, sửa màu lá hoặc bầu trời, đổi màu sản phẩm, giảm bão hòa có chọn lọc |

## Khi nào dùng

Đây là thao tác selective color trong bảng Hue/Saturation của phần mềm chỉnh ảnh. Bạn chọn một màu tâm bằng độ trên vòng hue và độ rộng dải. Pixel gần hue đó chịu tác động mạnh nhất; hiệu ứng nhạt dần về 0 ở rìa dải, và pixel xám không bị đụng tới. Màu nằm ngoài dải hoàn toàn không dịch chuyển, và độ sáng không bao giờ thay đổi.

Vị trí trên vòng hue: 0 đỏ, 60 vàng, 120 xanh lá, 180 cyan, 240 xanh lam, 300 magenta.

## Bắt đầu nhanh

1. **Load Image** → Hue/Saturation per Range → **Preview Image**.
2. Đặt `hue_center` ở màu cần đổi (ví dụ 120 cho xanh lá) và `band_width` khoảng 30.
3. Kéo `hue_shift` hoặc `sat_mult` và xem chỉ màu đó phản ứng.

## Thông số

| Thông số | Mặc định | Tác dụng | Gợi ý |
|---|---|---|---|
| `image` | bắt buộc | Ảnh cần chỉnh. | |
| `hue_center` | 0 (0 đến 360) | Màu nhắm tới, tính bằng độ trên vòng hue. | 0 đỏ, 60 vàng, 120 xanh lá, 180 cyan, 240 xanh lam, 300 magenta. |
| `band_width` | 30 (0 đến 180) | Nửa độ rộng của dải, tính bằng độ. Đủ mạnh ở tâm, nhạt về 0 ở khoảng cách này. | 30 hẹp, 60 vừa, 180 cả vòng. |
| `hue_shift` | 0 (-180 đến 180) | Xoay hue trong dải, tính bằng độ. | Giá trị nhỏ (5 đến 20) trông tự nhiên. |
| `sat_mult` | 1.0 (0 đến 2) | Nhân độ bão hòa trong dải. 1 là giữ nguyên, 0 bỏ hết màu, 2 gấp đôi. | 0.6 để dịu màu chói. |
| `sat_add` | 0 (-1 đến 1) | Cộng hoặc trừ độ bão hòa sau phép nhân. | Hữu ích để thêm màu cho vùng nhạt. |

**Đầu ra:** `image`, ảnh đã chỉnh.

## Mẹo khi chỉnh ảnh

- Nới `band_width` lên 45 đến 60 để kết quả tự nhiên; dải quá hẹp có thể lộ cạnh trên các dải chuyển mượt.
- Để dịu các mảng đỏ trên da, nhắm 0 đến 10 với `sat_mult` nhẹ khoảng 0.8.
- Muốn bầu trời sâu hơn mà không động tới da, nhắm 220 đến 240 và tăng `sat_mult`.
- Với nhiều màu, nối nhiều node Hue/Saturation per Range. Muốn cả bảng tám dải, xem [ColorLab](color-lab.md).
- Dùng [Saturation Mask](saturation-mask.md) để bảo vệ các màu vốn đã đậm khỏi bị tăng thêm.

??? note "Bên trong node hoạt động thế nào"
    Ảnh được chuyển sang HSV (Kornia). Mask dải được dựng từ khoảng cách hue vòng tròn tới `hue_center`: bằng 1 ở tâm và giảm tuyến tính về 0 tại `band_width`; pixel không có bão hòa được gán 0. Hue được xoay `hue_shift × mask`, bão hòa thành `s × (1 + (sat_mult − 1) × mask) + sat_add × mask` rồi kẹp 0..1, value giữ nguyên, và kết quả được trộn với ảnh vào bằng chính mask đó. Với `band_width = 0`, chỉ những pixel trùng đúng hue tâm mới được chọn.

## Node liên quan

- [ColorLab](color-lab.md): tám dải màu cùng tông và grading
- [Saturation Mask](saturation-mask.md): tạo mask theo độ đậm màu
- [Color Matcher](color-matcher.md): khớp màu tổng thể theo ảnh tham chiếu
- [Luminosity Masking](luminosity-masking.md): giới hạn thay đổi ở vùng bóng, midtone hoặc sáng

## Workflow mẫu

[Workflow mẫu Hue/Saturation per Range](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-16-selective-color.json). Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/` trước.
