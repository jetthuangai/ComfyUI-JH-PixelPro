---
title: Saturation Mask
description: Tạo mask mềm cho những vùng có độ bão hòa màu nằm trong khoảng bạn chọn.
---

# Saturation Mask

Chọn vùng theo độ rực của màu, không phải theo vị trí. Dùng để lấy riêng các màu gắt, riêng các màu nhạt, hoặc khoảng ở giữa.

![Saturation Mask — trước / sau](../assets/nodes/saturation-mask.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → color → Saturation Mask Builder` |
| **Phù hợp để** | Hạ màu quá gắt của áo hoặc nền, tăng riêng vùng nhợt nhạt, bảo vệ màu đã rực khỏi bước chỉnh màu |

## Khi nào dùng

Độ bão hòa (saturation) là độ đậm của màu: biển đèn neon rất bão hòa, bức tường xám gần như không có. Saturation mask hoạt động giống luminosity mask (vùng chọn theo độ sáng), nhưng dựa trên độ đậm của màu.

Một số cách dùng:

- Chỉ hạ phần đỏ quá gắt của chiếc váy mà không đụng đến da.
- Chỉ nâng những vùng màu bị phẳng, nhợt.
- Nối mask vào bất kỳ node nào có ngõ `MASK`, ví dụ [LUT Import](lut-import.md), để chỉnh màu chỉ rơi vào đúng chỗ bạn muốn.

Giá trị được đo bằng saturation của hệ HLS (hue, lightness, saturation), cùng ý với thanh saturation trong hầu hết phần mềm chỉnh ảnh.

## Bắt đầu nhanh

1. Nối **Load Image → Saturation Mask Builder**.
2. Giữ `sat_min` = `0.30` và `sat_max` = `1.00` để chọn phần có màu rực.
3. Nối ngõ `mask` vào [Mask Preview](mask-preview.md) để xem vùng chọn, hoặc vào ngõ `mask` của node khác.
4. Tăng `sat_min` để thu hẹp vùng chọn, chỉ còn các màu đậm nhất.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `sat_min` | 0.30 | Độ bão hòa thấp nhất được chọn (0 đến 1). | Tăng lên để bỏ qua vùng màu nhạt. |
| `sat_max` | 1.00 | Độ bão hòa cao nhất được chọn (0 đến 1). Phải lớn hơn `sat_min`. | Giảm xuống để loại các màu rực nhất. |
| `feather` | 0.10 | Độ mềm của mép quanh hai giới hạn (0 đến 1). | `0` cho mask cứng bật/tắt; 0.05 đến 0.15 để tránh bậc thang. |

**Ngõ ra:** `mask` — một `MASK` có giá trị từ 0 (không chọn) đến 1 (chọn hoàn toàn), mỗi ảnh trong batch một mask.

## Mẹo khi chỉnh ảnh

- Luôn để một chút `feather`. Ngưỡng saturation cứng sẽ tạo vùng chọn lốm đốm, nhiễu trên da và vải.
- Muốn bảo vệ màu đã rực, chọn dải thấp (ví dụ `sat_min` 0, `sat_max` 0.4) rồi áp chỉnh màu qua mask đó.
- Da nằm ở dải saturation trung bình. Nếu bước chỉnh màu ảnh hưởng đến da, kết hợp mask này với vùng chọn theo hue bằng [Hue/Saturation per Range](hue-saturation-range.md).
- Làm sạch các đốm nhỏ bằng [Mask Cleanup](mask-cleanup.md) trước khi dùng mask cho bản xuất cuối.

??? note "Bên trong node hoạt động thế nào"
    Ảnh được chuyển sang HLS và kênh saturation được so với khoảng đã chọn. Khi `feather` bằng 0 kết quả là mask nhị phân. Ngược lại mỗi giới hạn có một đoạn chuyển smoothstep rộng `feather` (đoạn dưới kết thúc tại `sat_min`, đoạn trên bắt đầu tại `sat_max`), rồi hai phần nhân với nhau. `sat_max` phải lớn hơn hẳn `sat_min`, nếu không node báo lỗi. Cùng hàm này cũng được dùng ở bước chỉnh saturation trong một số look của [Look Select](look-select.md).

## Node liên quan

- [Hue/Saturation per Range](hue-saturation-range.md) — chỉnh màu trong một dải hue.
- [Luminosity Masking](luminosity-masking.md) — bản tương đương theo độ sáng.
- [Mask Combine](mask-combine.md) — nhân hoặc cộng mask này với các vùng chọn khác.
- [Look Select](look-select.md) — dùng saturation mask trong một số look.

## Workflow mẫu

[Selective color example workflow](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-16-selective-color.json). Workflow tạo Saturation Mask từ ảnh chân dung, hiển thị dạng heatmap, và dùng Hue/Saturation per Range trên một dải màu. Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/`. Workflow chỉ dùng node lõi của ComfyUI và pack này.

![Kết quả: Selective color example workflow](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/S-16-selective-color-preview.png)
