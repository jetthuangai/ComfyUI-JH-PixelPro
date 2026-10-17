---
title: Tone Curve
description: Tạo tương phản và cân bằng màu bằng đường cong tám điểm, dùng preset hoặc điểm tự đặt.
---

# Tone Curve

Công cụ curves quen thuộc: uốn đáp ứng tông của cả ảnh hoặc của một kênh màu, bằng preset dựng sẵn hoặc tám điểm do bạn tự đặt.

![Tone Curve — trước / sau](../assets/nodes/tone-curve.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → color → Tone Curve (RGB)` |
| **Dùng để** | Tạo tương phản đường cong chữ S, nâng hoặc nén vùng bóng, sửa ám màu theo từng kênh |

## Khi nào dùng

Đường cong tông (tone curve) ánh xạ mỗi mức sáng đầu vào sang một mức sáng đầu ra mới. Dáng chữ S nhẹ làm vùng tối tối hơn, vùng sáng sáng hơn, tức là thêm tương phản. Nâng phần dưới của đường cong sẽ nâng vùng bóng, tạo vẻ phai (faded). Áp đường cong chỉ lên kênh red, green hoặc blue sẽ đổi cân bằng màu, đó là cách xử lý ám màu hoặc tạo split-tone.

Node dùng đường cong đi qua tám điểm điều khiển, làm mượt bằng spline Catmull-Rom, nên mượt và dễ đoán.

## Bắt đầu nhanh

1. **Load Image** → Tone Curve → **Preview Image**.
2. Chọn `preset` = `s_curve_mild` và `channel` = `rgb_master`.
3. Hạ `strength` xuống 0.6 nếu hiệu ứng quá nặng.
4. Muốn đường cong riêng, đặt `preset` là `custom` và sửa `points_json`.

## Thông số

| Thông số | Mặc định | Tác dụng | Gợi ý |
|---|---|---|---|
| `preset` | `linear` | `linear` (không đổi), `s_curve_mild`, `s_curve_strong`, `lift_shadows`, `crush_blacks` hoặc `custom`. | Chữ S cho tương phản, `lift_shadows` cho vẻ matte. |
| `channel` | `rgb_master` | `rgb_master` áp cùng một đường cong lên red, green, blue (tương phản). `r`, `g` hoặc `b` chỉ áp lên kênh đó (cân bằng màu). | Dùng từng kênh để sửa ám màu. |
| `points_json` | đường chữ S nhẹ | Tám điểm của bạn dạng `[[x1,y1],…,[x8,y8]]`. Chỉ dùng khi `preset` là `custom`. | Xem quy tắc bên dưới. |
| `strength` | 1.0 (0 đến 1) | Pha trộn giữa ảnh gốc (0) và đường cong đầy đủ (1). | 0.5 đến 0.8 cho grade tinh tế. |

**Đầu ra:** `image_toned`, ảnh đã chỉnh.

**Quy tắc cho `points_json`:** đúng 8 điểm; mọi giá trị nằm trong 0 đến 1; điểm đầu là `[0,0]`, điểm cuối là `[1,1]`; giá trị x phải tăng dần nghiêm ngặt. Sai quy tắc sẽ báo lỗi kèm thông điệp rõ ràng. Ví dụ (đường chữ S nhẹ):

```json
[[0.0, 0.0], [0.15, 0.10], [0.30, 0.25], [0.45, 0.43],
 [0.55, 0.57], [0.70, 0.75], [0.85, 0.90], [1.0, 1.0]]
```

Muốn nâng vùng bóng, tăng y của điểm thứ hai và thứ ba. Muốn hạ vùng sáng, giảm y của điểm thứ sáu và thứ bảy.

## Mẹo khi chỉnh ảnh

- Tăng chữ S từng bước nhỏ. `s_curve_mild` ở `strength` 0.7 thường là đủ cho chân dung.
- Muốn bóng lạnh, sáng ấm: dùng `channel` = `b` với đường cong nâng vùng bóng, rồi `r` với đường cong nâng vùng sáng. Nối hai node Tone Curve liên tiếp là được.
- `crush_blacks` kết hợp [Luminosity Masking](luminosity-masking.md) giúp làm sâu vùng bóng mà không đụng tới da.
- Đường cong không thêm được chi tiết cục bộ. Dùng [CLAHE Local Contrast](clahe-local-contrast.md) cho việc đó.
- Muốn lưu đường cong thành look dùng lại, bake thành LUT bằng [HALD Identity](hald-identity.md) và [LUT Export](lut-export.md).

??? note "Bên trong node hoạt động thế nào"
    Tám điểm được biến thành bảng tra 1024 mục bằng spline Catmull-Rom (cubic Hermite); hai đầu được ghim ở 0 và 1. Giá trị pixel được tra bằng nội suy tuyến tính. Với `rgb_master`, bảng áp lên cả ba kênh; với `r`, `g`, `b` chỉ áp lên kênh đó. Kết quả được trộn với ảnh vào theo `strength` và kẹp 0..1. Đường cong áp thẳng lên giá trị RGB node nhận được, không qua linear light. Giá trị ngoài 0..1 bị kẹp lại kèm cảnh báo trong log, và `strength = 0` trả lại ảnh vào.

## Node liên quan

- [ColorLab](color-lab.md): tông và màu kiểu Camera Raw trong một node
- [Luminosity Masking](luminosity-masking.md): chỉ áp đường cong lên vùng bóng, midtone hoặc sáng
- [HALD Identity](hald-identity.md) và [LUT Export](lut-export.md): lưu đường cong thành LUT
- [Color Matcher](color-matcher.md): khớp theo ảnh tham chiếu trước, rồi tinh chỉnh bằng đường cong

## Workflow mẫu

[Workflow mẫu Tone Curve](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-09-tone-curve.json). Workflow áp preset `s_curve_mild` lên ảnh chân dung. Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/`. Workflow chỉ dùng node lõi của ComfyUI và pack này.

![Kết quả: Workflow mẫu Tone Curve](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/S-09-tone-curve-preview.png)
