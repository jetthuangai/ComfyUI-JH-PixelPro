---
title: Tone Match LUT
description: Tự động tạo LUT .cube dùng lại được từ màu sắc và tông của một ảnh tham chiếu.
---

# Tone Match LUT

Đưa vào một bức ảnh có màu bạn thích, node sẽ ghi ra một LUT kéo các ảnh khác về gần look đó. Không cần chỉnh màu thủ công.

![Tone Match LUT — trước / sau](../assets/nodes/tone-match-lut.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → color → Tone Match LUT (auto-gen .cube)` |
| **Phù hợp để** | Bắt chước cảm giác của ảnh tham chiếu, tạo look riêng của studio, đưa cả loạt ảnh về gần ảnh chủ đạo |

## Khi nào dùng

[Color Matcher](color-matcher.md) khớp màu một ảnh với ảnh khác, nhưng chỉ cho đúng cặp đó. Tone Match LUT thực hiện phép chuyển màu tương tự một lần rồi lưu thành file `.cube`. Sau đó bạn áp LUT cho bao nhiêu ảnh cũng được bằng [LUT Import](lut-import.md), hoặc mở trong phần mềm khác đọc được Cube 1.0.

Nó nắm bắt màu và tông tổng thể: màu trung bình, độ trải rộng của màu và mối liên hệ giữa các màu. Nó không sao chép chi tiết cục bộ hay nội dung ảnh.

## Bắt đầu nhanh

1. Nối **Load Image** (ảnh tham chiếu) vào `reference`.
2. Giữ `level` = `8`. Đặt `filename` (ví dụ `hero_look.cube`).
3. Chạy prompt. Ngõ `lut_path` cho thấy file mới trong `ComfyUI/output/`.
4. Áp bằng [LUT Import](lut-import.md). Chép file vào `ComfyUI/input/` hoặc dùng đường dẫn đầy đủ.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `reference` | — | Ảnh có look bạn muốn lấy. | Chọn khung hình đúng sáng, có đúng màu bạn cần. |
| `level` | 8 | Độ mịn của LUT. Các lựa chọn: 4, 6, 8, 10, 12. | 8 cho cube 64 điểm, chuẩn cho look sáng tạo. |
| `filename` | `tone_match.cube` | Tên file ra. Tên tương đối nằm trong thư mục `output/` của ComfyUI; tự thêm `.cube` nếu thiếu. | File trùng tên sẽ bị ghi đè. |
| `title` | `Tone Match LUT` | Tên lưu trong file. | Đặt theo tên ảnh tham chiếu. |

**Ngõ ra:** `lut_path` — đường dẫn đầy đủ của file đã ghi.

## Mẹo khi chỉnh ảnh

- Chọn ảnh tham chiếu mang tính đại diện. Ảnh bị một màu chi phối (hoàng hôn, cánh đồng xanh) sẽ cho LUT mạnh và lệch một phía.
- Ảnh tham chiếu xám hoặc gần như không có màu sẽ cho LUT hầu như không đổi gì.
- Luôn thử bằng [LUT Import](lut-import.md) với `strength` 0.6 đến 0.8 trước. Kết quả tự động có thể mạnh hơn bạn tưởng khi nhìn ảnh tham chiếu.
- Nếu khớp gần được nhưng chưa hoàn hảo, tinh chỉnh nhẹ bằng [Tone Curve](tone-curve.md) sau LUT.

??? note "Bên trong node hoạt động thế nào"
    Một HALD identity ở level đã chọn được chuyển sang LAB rồi đi qua phép chuyển MKL (Monge-Kantorovich linear): ma trận 3 × 3 tính từ hiệp phương sai của mẫu LAB của HALD và của ảnh tham chiếu, cộng với giá trị trung bình của ảnh tham chiếu. Nếu một trong hai ma trận hiệp phương sai không ổn định thì chuyển về dịch trung bình đơn giản. Kết quả được cắt theo miền LAB rồi đổi lại RGB. Ảnh tham chiếu gần như xám đồng nhất trả về identity. HALD đã chỉnh được ghi bằng cùng bộ xuất như [LUT Export](lut-export.md). Các phiên bản cũ dùng khớp histogram hoặc mean/std; cách hiện tại giữ được quan hệ giữa các kênh màu.

## Node liên quan

- [Color Matcher](color-matcher.md) — khớp trực tiếp hai ảnh, không cần file.
- [LUT Import](lut-import.md) — áp file `.cube` đã tạo.
- [LUT Export](lut-export.md) — xuất một look tự chỉnh bằng tay.
- [HALD Identity](hald-identity.md) — bảng màu dùng bên trong node này.

## Workflow mẫu

[Tone Match LUT example workflow](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-17-tone-match-lut.json). Bản golden-hour của ảnh chân dung là ảnh tham chiếu. Tone Match LUT ghi một file `.cube` vào `ComfyUI/output/` và đường dẫn được nối thẳng vào LUT Import để áp lên ảnh gốc. Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/`. Workflow chỉ dùng node lõi của ComfyUI và pack này.

![Kết quả: Tone Match LUT example workflow](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/S-17-tone-match-lut-preview.png)
