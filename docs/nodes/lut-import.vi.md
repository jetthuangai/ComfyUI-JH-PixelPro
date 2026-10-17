---
title: LUT Import
description: Nạp file LUT .cube và áp lên ảnh, có thể chỉnh độ mạnh và dùng mask.
---

# LUT Import

Áp bất kỳ LUT Adobe Cube 1.0 `.cube` nào lên ảnh của bạn. Dùng look của colorist khác, LUT bạn đã xuất từ ComfyUI, hoặc LUT làm từ phần mềm khác.

![LUT Import — trước / sau](../assets/nodes/lut-import.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → color → LUT Import (.cube)` |
| **Phù hợp để** | Áp look yêu thích, dùng lại LUT từ [LUT Export](lut-export.md), chỉnh màu chỉ một phần ảnh qua mask |
| **Tốc độ** | Khoảng 0,5 giây cho ảnh 2048 × 2048 với LUT 65 điểm trên máy chỉ có CPU (theo ghi chú phát hành của dự án). Có GPU sẽ nhanh hơn. |

## Khi nào dùng

LUT (look-up table, bảng tra màu) là một bước chỉnh màu được lưu thành file nhỏ. LUT Import đọc file đó rồi áp lên từng pixel. Vì chỉ xử lý màu nên rất nhanh và lần nào cũng cho kết quả giống nhau.

Node này khép kín [quy trình LUT](lut-export.md): HALD Identity, chỉnh màu, LUT Export, rồi LUT Import trên ảnh bất kỳ. Nó cũng nạp được file `.cube` không đến từ pack này.

Với sáu look đi kèm pack, [LUT Preset](lut-preset.md) nhanh hơn vì không cần đường dẫn file.

## Bắt đầu nhanh

1. Chép file `.cube` vào `ComfyUI/input/`.
2. Nối **Load Image → LUT Import (.cube) → Preview Image**.
3. Gõ tên file vào `filename`, kèm `.cube` (ví dụ `my_look.cube`).
4. Đặt `strength` 0.7 đến 0.8 cho liều nhẹ, hoặc 1.0 để lấy toàn bộ look.

Muốn chỉnh màu chỉ một phần ảnh, nối một `MASK` (ví dụ từ [Saturation Mask](saturation-mask.md)) vào ngõ `mask` tùy chọn.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `filename` | `pack_lut.cube` | Đường dẫn tới file `.cube`. Tên tương đối được tìm trong thư mục `input/` của ComfyUI; đường dẫn tuyệt đối dùng nguyên. | Phần mở rộng `.cube` không tự thêm. Hãy gõ đủ tên file. |
| `strength` | 1.00 | Pha giữa ảnh gốc (0) và LUT đầy đủ (1). | 0.5 đến 0.8 cho look sáng tạo nhẹ nhàng. |
| `mask` (tùy chọn) | không | Giới hạn LUT vào vùng trắng của mask; vùng xám chỉ chịu một phần hiệu ứng. | Feather mask để chuyển vùng mượt. |

**Ngõ ra:** `image` — ảnh đã chỉnh màu.

!!! note "File được đọc từ đâu"
    LUT Export lưu vào `ComfyUI/output/`, nhưng LUT Import đọc tên tương đối từ `ComfyUI/input/`. Hãy chép file sang, hoặc nhập đường dẫn đầy đủ.

## Mẹo khi chỉnh ảnh

- Nhiều LUT sáng tạo được làm cho một không gian màu riêng (ví dụ log của video). Trên ảnh sRGB thường, chúng có thể trông phẳng hoặc quá gắt. Giảm `strength` hoặc thử LUT khác.
- Retouch da trước, áp look sau. LUT mạnh có thể làm da loang lổ lộ rõ hơn.
- Dùng `mask` để tránh LUT tác động lên da hoặc logo. Tạo vùng chọn bằng [Saturation Mask](saturation-mask.md) hoặc [Luminosity Masking](luminosity-masking.md).
- Chỉ hỗ trợ LUT 3D (file có `LUT_3D_SIZE`). LUT 1D sẽ bị từ chối.

??? note "Bên trong node hoạt động thế nào"
    File được đọc các mục `TITLE`, `LUT_3D_SIZE` (2 đến 256), `DOMAIN_MIN` và `DOMAIN_MAX`; phần thân phải có đúng size³ dòng, mỗi dòng ba số, nếu không node báo lỗi. Ảnh được cắt theo miền, chuẩn hóa rồi lấy mẫu bằng nội suy trilinear qua `grid_sample` (padding biên, không ngoại suy). Kết quả được pha theo `in * (1 - s) + lut(in) * s`, với `s` là `strength` nhân với mask. Mask có thể có dạng `(B, H, W)` hoặc `(B, 1, H, W)`. Chưa có nội suy tetrahedral. Ảnh phải là RGB ba kênh; không hỗ trợ alpha.

## Node liên quan

- [LUT Export](lut-export.md) — tạo file `.cube` mà node này đọc.
- [HALD Identity](hald-identity.md) — bảng màu mở đầu việc xuất LUT.
- [LUT Preset](lut-preset.md) — sáu look đi kèm, không cần file.
- [Tone Match LUT](tone-match-lut.md) — tạo LUT từ ảnh tham chiếu.

## Workflow mẫu

[LUT Import example workflow](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-15-lut-import.json). Workflow nạp file tên `S-14 Demo.cube`, vì vậy hãy chạy trước [LUT Export example](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-14-lut-export.json), chép file đã xuất vào `ComfyUI/input/` rồi đổi tên hoặc sửa `filename`. Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/` để lấy ảnh.
