---
title: Unwrap Face
description: Dán khuôn mặt đã chỉnh (đã được xoay thẳng) trở lại ảnh gốc với viền mềm.
---

# Unwrap Face

Đặt khuôn mặt đã chỉnh sửa về đúng chỗ ban đầu. Node đảo ngược phép xoay thẳng của [Facial Aligner](facial-aligner.md) rồi hòa kết quả vào ảnh gốc của bạn.

![Unwrap Face — trước / sau](../assets/nodes/unwrap-face.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → face → Unwrap Face (paste back)` |
| **Dùng để** | Hoàn tất chuỗi căn chỉnh → chỉnh sửa → dán lại; giữ phần còn lại của ảnh ở chất lượng gốc |

## Khi nào dùng

AI retouch, upscale và phục hồi khuôn mặt cho kết quả tốt nhất trên một vùng cắt nhỏ, thẳng và nằm giữa khung. [Facial Aligner](facial-aligner.md) tạo ra vùng cắt đó, đồng thời đưa cho bạn phép biến đổi ngược chính xác. Sau khi chỉnh xong, Unwrap Face dùng phép biến đổi này để xoay, co giãn và đặt khuôn mặt về lại ảnh kích thước đầy đủ, nên chỉ vùng mặt thay đổi, phần còn lại giữ nguyên.

## Bắt đầu nhanh

1. Chạy [Face Detect](face-detect.md) (hoặc [Face Detect Lite](face-detect-lite.md)) và đưa `landmarks_json` vào [Facial Aligner](facial-aligner.md).
2. Chỉnh sửa vùng mặt đã căn thẳng (`image_aligned`) bằng các node AI hoặc retouch của bạn.
3. Nối vùng mặt đã chỉnh vào `edited_aligned`, ảnh gốc vào `original_image`, và `inverse_matrix_json` của Facial Aligner vào `inverse_matrix_json`.
4. Xem `image_composited` trong **Preview Image**. Chỉnh `feather_radius` nếu còn thấy viền.

## Thông số

| Thông số | Mặc định | Tác dụng | Gợi ý |
|---|---|---|---|
| `edited_aligned` | — | Vùng mặt đã căn thẳng và đã chỉnh sửa. | Giữ cùng kích thước với đầu ra của aligner. |
| `original_image` | — | Ảnh kích thước đầy đủ để dán vào. | Chính là ảnh bạn đã dùng để nhận diện. |
| `inverse_matrix_json` | ma trận đơn vị | Phép biến đổi ngược từ [Facial Aligner](facial-aligner.md). | Luôn nối vào. Giá trị mặc định không có tác dụng thực tế. |
| `feather_radius` | `16.0` (0–128) | Độ mềm của viền, tính bằng pixel. `0` là viền cứng. | Giá trị lớn che được đường nối nhưng làm mềm mép mặt. Bị bỏ qua khi có nối `mask_override`. |
| `mask_override` | tùy chọn | Mask trộn do bạn tự cung cấp, kích thước bằng ảnh gốc. | Dùng mask mặt hoặc mask da để chỉ dán phần da, không dán cả vùng cắt. |

| Đầu ra | Kiểu | Ý nghĩa |
|---|---|---|
| `image_composited` | `IMAGE` | Ảnh gốc với khuôn mặt đã chỉnh được dán vào. |
| `mask_used` | `MASK` | Mask trộn đã được áp dụng. |

## Mẹo khi chỉnh ảnh

- Mặc định, vùng được dán là toàn bộ ô vuông của vùng căn chỉnh, được làm mềm bằng `feather_radius`. Vùng này gồm cả tóc và hậu cảnh trong khung cắt. Hãy nối mask mặt hoặc mask da vào `mask_override` để chỉ dán phần bạn muốn.
- Nếu bước AI làm đổi độ phân giải của vùng cắt, hãy đưa nó về lại kích thước đầu ra của aligner trước, để phép biến đổi khớp.
- Thấy lệch màu ở mép? Hãy khớp màu trước khi dán bằng [Color Matcher](color-matcher.md), hoặc dùng [Face Beauty Blend](face-beauty-blend.md) cho bước trộn cuối có điều chỉnh cường độ.
- Với batch, `edited_aligned`, `original_image` và danh sách ma trận mỗi thứ phải có một phần tử hoặc bằng kích thước batch.

??? note "Bên trong node hoạt động thế nào"
    Vùng mặt được warp lên canvas gốc bằng `kornia.geometry.transform.warp_affine`, dùng ma trận ngược (3×3 hoặc 2×3, mỗi ảnh một ma trận). Nếu không có `mask_override`, một ô vuông trắng cùng kích thước vùng cắt được warp tương tự, co vào trong 2 × `feather_radius` rồi làm mờ Gaussian (sigma = `feather_radius`), nhờ vậy viền mềm luôn nằm trong vùng cắt và không lẫn phần nền đen bên ngoài. Kết quả là `mask × warped + (1 − mask) × original`, giới hạn trong `[0, 1]`. `mask_override` phải khớp chính xác chiều cao và chiều rộng của ảnh gốc. Giá trị ngoài `[0, 1]` bị kẹp lại kèm cảnh báo. Phép warp chạy trên toàn canvas, nên ảnh lớn trên CPU chậm hơn GPU. Mã nguồn nằm ở `core/unwrap_face.py`.

## Node liên quan

- [Facial Aligner](facial-aligner.md) — tạo vùng cắt và ma trận ngược.
- [Face Beauty Blend](face-beauty-blend.md) — bước trộn cuối bằng mask, có điều chỉnh cường độ.
- [Face Detect](face-detect.md) và [Face Detect Lite](face-detect-lite.md) — tìm landmark.
- [Tổng quan nhóm Khuôn mặt](../categories/face.md)

## Workflow mẫu

[Workflow mẫu Unwrap Face](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-11-unwrap-face.json). Hãy chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/` trước.
