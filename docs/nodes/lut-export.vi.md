---
title: LUT Export
description: Lưu bảng màu HALD đã chỉnh thành LUT .cube để dùng trong phần mềm khác.
---

# LUT Export

Biến bước chỉnh màu dựng trong ComfyUI thành file `.cube` mang đi được. Đem look của bạn sang phần mềm khác, hoặc nạp lại bằng [LUT Import](lut-import.md).

![LUT Export — trước / sau](../assets/nodes/lut-export.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → color → LUT Export (.cube)` |
| **Phù hợp để** | Dùng lại một look cho nhiều ảnh, chia sẻ look, đồng bộ màu giữa video và ảnh tĩnh |

## Khi nào dùng

Bạn đã có look ưng ý, dựng từ nhiều node chỉnh màu trong ComfyUI. LUT đóng băng look đó thành một file nhỏ. Sau đó bạn áp nhanh, nhiều lần, hoặc mở trong phần mềm khác.

### Quy trình LUT

1. [HALD Identity](hald-identity.md) tạo bảng màu trung tính.
2. Các node chỉnh màu chỉnh bảng màu này, ví dụ [Tone Curve](tone-curve.md). Chỉ dùng node chỉnh màu: không blur, sharpen, resize hay crop.
3. **LUT Export** đọc bảng màu đã chỉnh và ghi file `.cube` vào thư mục `output/` của ComfyUI.
4. Dùng file `.cube`:
    - Trong ComfyUI bằng [LUT Import](lut-import.md), trên ảnh bất kỳ, có thể qua mask.
    - Trong các phần mềm đọc file Adobe Cube 1.0. Ghi chú phát hành của dự án nêu DaVinci Resolve, Premiere Pro, OBS Studio và OCIO. Photoshop cũng nạp được `.cube` qua adjustment layer Color Lookup.

Hãy thử file trong phần mềm đích trước khi dựa vào nó. Pack này ghi và đọc đúng định dạng chuẩn, nhưng không có tích hợp riêng với OCIO hay Photoshop.

## Bắt đầu nhanh

1. Thêm [HALD Identity](hald-identity.md) (`level` 8).
2. Nối `image` của nó vào node chỉnh màu như [Tone Curve](tone-curve.md) và đặt look.
3. Thêm **LUT Export**. Nối ảnh đã chỉnh vào `image`, và ngõ `level` của HALD Identity vào `level`.
4. Đặt `filename` (ví dụ `my_look.cube`) và `title`, rồi chạy prompt.
5. Tìm file trong `ComfyUI/output/`. Ngõ `path` của node cho biết file được ghi ở đâu.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `image` | — | Bảng màu HALD đã chỉnh. | Phải là cả bảng màu, đúng kích thước như khi ra khỏi HALD Identity. |
| `level` | 8 (khoảng 2 đến 16) | Level HALD của bảng màu. | Phải trùng với HALD Identity; hãy nối hai ngõ với nhau. |
| `filename` | `pack_lut.cube` | Tên file ra. Tên tương đối nằm trong thư mục `output/` của ComfyUI; đường dẫn tuyệt đối được dùng nguyên. Tự thêm `.cube` nếu thiếu. | File trùng tên sẽ bị ghi đè. Mỗi look nên một tên riêng. |
| `title` | `JHPixelPro LUT` | Tên lưu trong file, hiện trong trình duyệt LUT. | Đặt tên dễ hiểu như `Warm Portrait v2`. |

**Ngõ ra:** `path` — đường dẫn đầy đủ của file `.cube` đã ghi.

## Mẹo khi chỉnh ảnh

- Giữ chuỗi node chỉ chỉnh màu. Nếu LUT trông sai trong phần mềm khác, nguyên nhân thường là một node không gian (blur, sharpen, resize) nằm giữa HALD và Export.
- LUT chỉ lưu được những thay đổi màu phụ thuộc vào chính màu của từng pixel. Chỉnh cục bộ như mask, dodge & burn hay làm mịn da không thể "nướng" vào LUT.
- Level 8 là đủ cho look sáng tạo. Chỉ tăng khi đường cong rất dốc.
- Thư mục xuất phải tồn tại. Đường dẫn tuyệt đối có thư mục cha chưa có sẽ báo lỗi.
- Thử file ngay: dùng [LUT Import](lut-import.md) trên một ảnh với `strength` 1.0.

??? note "Bên trong node hoạt động thế nào"
    Bảng màu phải đúng `L³ × L³` pixel theo `level` và chỉ chứa giá trị hữu hạn, nếu không node báo lỗi. Giá trị được cắt về `[0, 1]`. File có các header `TITLE`, `LUT_3D_SIZE` (`L²`), `DOMAIN_MIN 0 0 0` và `DOMAIN_MAX 1 1 1`, theo sau là `N³` dòng (`N = L²`) dạng `r g b` với sáu chữ số thập phân, đỏ đổi nhanh nhất, rồi đến xanh lá, rồi xanh dương. Xuống dòng kiểu LF. Chỉ ảnh đầu tiên trong batch được xuất. Việc ghi chỉ là thao tác file thông thường nên GPU không giúp nhanh hơn.

## Node liên quan

- [HALD Identity](hald-identity.md) — bảng màu trung tính để bắt đầu.
- [LUT Import](lut-import.md) — áp `.cube` trở lại trong ComfyUI.
- [Tone Match LUT](tone-match-lut.md) — tạo LUT từ ảnh tham chiếu chỉ trong một bước.
- [LUT Preset](lut-preset.md) — look đi kèm, không cần xuất.

## Workflow mẫu

[LUT Export example workflow](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-14-lut-export.json) — HALD Identity, Tone Curve và LUT Export. Không cần ảnh đầu vào. Sau khi chạy, mở [LUT Import](lut-import.md) và nạp file vừa ghi.
