---
title: HALD Identity
description: Tạo một ảnh bảng màu trung tính để bạn chỉnh màu rồi lưu thành LUT.
---

# HALD Identity

Tạo một ảnh đặc biệt chứa toàn bộ màu theo dạng lưới và tự nó không thay đổi gì. Bất kỳ bước chỉnh màu nào bạn áp lên ảnh này đều có thể lưu thành LUT `.cube`.

![HALD Identity — trước / sau](../assets/nodes/hald-identity.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → color → HALD Identity` |
| **Phù hợp để** | Bắt đầu xuất LUT, biến một chuỗi node chỉ chỉnh màu thành look dùng lại được |
| **Tốc độ** | Tức thì. Level cao hơn tạo ảnh lớn hơn (level 12 là 1728 × 1728 px) |

## Khi nào dùng

Hãy xem nó như một bảng màu trắng. Nhìn bề ngoài là một hoa văn nhiều màu, mượt. Nếu cho nó đi qua các node chỉnh màu (tone curve, cân bằng màu, …), hoa văn sẽ đổi đúng như cách các node đó đổi mọi màu trong ảnh chụp. Sau đó [LUT Export](lut-export.md) đọc bảng màu đã chỉnh và ghi thành file `.cube`.

Đây là bước đầu của quy trình LUT:

```text
HALD Identity → chỉ các node chỉnh màu → LUT Export → file .cube
```

!!! warning "Giữa HALD và Export chỉ dùng node chỉnh màu"
    Bảng màu chỉ giữ nguyên qua các thao tác màu xử lý từng pixel độc lập, như [Tone Curve](tone-curve.md) hay [Hue/Saturation per Range](hue-saturation-range.md). Đừng blur, sharpen, resize, crop, xoay hay thêm grain: chúng dịch chuyển hoặc trộn pixel, và LUT xuất ra sẽ sai.

## Bắt đầu nhanh

1. Thêm **HALD Identity**, giữ `level` = `8`.
2. Nối ngõ `image` vào bước chỉnh màu của bạn (ví dụ [Tone Curve](tone-curve.md)).
3. Nối ảnh đã chỉnh vào [LUT Export](lut-export.md) và nối ngõ `level` của HALD Identity vào ngõ `level` của LUT Export.

Hướng dẫn đầy đủ nằm ở trang [LUT Export](lut-export.md).

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `level` | 8 | Độ mịn của LUT. Các lựa chọn: 4, 6, 8, 10, 12. | 8 cho cube 64 điểm, chuẩn thường dùng cho look sáng tạo. |

| Level | Lưới LUT | Kích thước ảnh |
|---|---|---|
| 4 | 16 điểm mỗi cạnh | 64 × 64 px |
| 8 | 64 điểm mỗi cạnh | 512 × 512 px |
| 12 | 144 điểm mỗi cạnh | 1728 × 1728 px |

**Ngõ ra:**

- `image` — ảnh bảng màu trung tính, một ảnh duy nhất.
- `level` — cùng con số đó dạng số nguyên, để nối thẳng vào LUT Export.

## Mẹo khi chỉnh ảnh

- Cứ để 8 nếu không có lý do khác. Level cao cho LUT mượt hơn với các bước chỉnh rất mạnh, nhưng file lớn hơn và tốn bộ nhớ hơn.
- Luôn nối `level` sang LUT Export. Lệch level sẽ báo lỗi.
- Xem trước bảng màu trước và sau bước chỉnh để chắc rằng nó vẫn là hoa văn màu mượt.

??? note "Bên trong node hoạt động thế nào"
    Với level `L`, LUT có `N = L²` điểm mỗi trục và ảnh có kích thước `L³ × L³` pixel, theo quy ước HALD của ImageMagick. Đỏ thay đổi nhanh nhất theo trục x, xanh lá theo trục y, còn xanh dương đi qua các ô `L × L`. Tensor là float32 trong `[0, 1]`, hình dạng `(1, L³, L³, 3)`. Node cho chọn level 4, 6, 8, 10, 12; hàm bên dưới nhận 2 đến 16.

## Node liên quan

- [LUT Export](lut-export.md) — ghi bảng màu đã chỉnh thành `.cube`.
- [LUT Import](lut-import.md) — áp `.cube` lên ảnh chụp.
- [Tone Curve](tone-curve.md) — bước chỉnh màu hay đặt giữa hai node.
- [Tone Match LUT](tone-match-lut.md) — tạo LUT tự động từ ảnh tham chiếu.

## Workflow mẫu

[LUT Export example workflow](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-14-lut-export.json) — HALD Identity, Tone Curve và LUT Export nối liền. Không cần ảnh đầu vào.
