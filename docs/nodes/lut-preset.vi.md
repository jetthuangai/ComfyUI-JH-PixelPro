---
title: LUT Preset
description: Áp một trong sáu look màu làm sẵn (LUT đi kèm pack) chỉ với một thanh intensity.
---

# LUT Preset

Chọn một look màu có sẵn từ danh sách rồi chỉnh độ mạnh. Các look là file `.cube` đi kèm pack nên bạn không phải tìm hay nạp file.

![LUT Preset — trước / sau](../assets/nodes/lut-preset.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → color → Color: LUT Preset` |
| **Phù hợp để** | Thử nhanh một look, làm điểm khởi đầu cho bước chỉnh màu, áp tông màu đồng nhất cho cả loạt ảnh |

## Khi nào dùng

LUT (look-up table, bảng tra màu) là một bước chỉnh màu được lưu lại: với mỗi màu đầu vào, LUT ghi sẵn màu đầu ra. Colorist dùng LUT để mang cùng một look qua nhiều cảnh và nhiều phần mềm.

Node này là đường tắt. Nó có sáu look phổ thông và dùng cùng cơ chế với [LUT Import](lut-import.md). Khi bạn có file `.cube` riêng, hãy dùng LUT Import.

Xem [thư viện LUT Preset](../examples/lut-preset-gallery.md) để thấy ảnh trước/sau của từng preset.

## Bắt đầu nhanh

1. Nối **Load Image → Color: LUT Preset → Preview Image**.
2. Chọn `preset`. Thử `warm-portrait` hoặc `cinematic-teal-orange` trước.
3. Giảm `intensity` xuống 0.6 đến 0.85 nếu look hơi gắt.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `preset` | `neutral-identity` | Look đi kèm được áp (danh sách bên dưới). | `neutral-identity` không đổi gì; dùng để kiểm tra đấu nối. |
| `intensity` | 1.00 | Pha giữa ảnh gốc (0) và look đầy đủ (1). | 0.6 đến 0.85 là liều tự nhiên cho ảnh chân dung. |

| Preset | Đặc điểm |
|---|---|
| `neutral-identity` | Không thay đổi; dùng làm mốc so sánh. |
| `warm-portrait` | Chân dung ấm, mềm hơn. |
| `cool-portrait` | Sắc lạnh, sạch hơn. |
| `cinematic-teal-orange` | Bóng lạnh, vùng sáng và da ấm. |
| `soft-pastel` | Màu nhạt, tương phản thấp. |
| `high-contrast` | Tách tông mạnh, "gắt" hơn. |

**Ngõ ra:** `image` — ảnh đã chỉnh màu.

## Mẹo khi chỉnh ảnh

- Chỉnh màu sau khi retouch. Xử lý da và tách tần số (frequency separation) cho kết quả tốt nhất trên file trung tính.
- Dùng `intensity` làm thanh điều chỉnh độ mạnh. Look ở 0.7 thường tự nhiên hơn ở 1.0.
- Nếu look đẩy màu da đi quá xa, thử [Look Select](look-select.md), có công tắc `protect_skin`.
- Tạo look riêng bằng [quy trình LUT](lut-export.md) rồi nạp lại bằng [LUT Import](lut-import.md).

??? note "Bên trong node hoạt động thế nào"
    Danh sách chọn lấy từ mọi file `.cube` trong thư mục `presets/` của pack, nên danh sách đi theo thư mục này. Mỗi file được đọc theo chuẩn Adobe Cube 1.0 (mọi file đi kèm là cube 17 điểm) và áp bằng nội suy trilinear qua `grid_sample`, tôn trọng header `DOMAIN_MIN` / `DOMAIN_MAX` nếu có. `intensity` là phép pha tuyến tính `ảnh gốc * (1 - s) + lut * s`. Màu nằm ngoài miền của LUT bị cắt về biên, không ngoại suy.

## Node liên quan

- [LUT Import](lut-import.md) — áp file `.cube` của riêng bạn.
- [LUT Export](lut-export.md) — biến bước chỉnh màu trong ComfyUI thành file `.cube`.
- [Look Select](look-select.md) — các look dựng từ JSON, có bảo vệ da.
- [Tone Curve](tone-curve.md) — tinh chỉnh tương phản trước hoặc sau look.

## Workflow mẫu

[LUT Preset example workflow](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-34-lut-preset.json). Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/` rồi chọn trong Load Image.
