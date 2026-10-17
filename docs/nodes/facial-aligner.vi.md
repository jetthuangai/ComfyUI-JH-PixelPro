---
title: Facial Aligner
description: Xoay, co giãn và cắt khuôn mặt về khung vuông chuẩn từ năm điểm landmark.
---

# Facial Aligner

Làm thẳng và cắt khuôn mặt vào khung vuông chuẩn (cùng bố cục với các bộ dữ liệu mặt như FFHQ), dựa trên năm điểm landmark. Node cũng trả về ma trận để dán khuôn mặt đã chỉnh trở lại ảnh gốc.

![Facial Aligner — trước / sau](../assets/nodes/facial-aligner.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → geometry → Landmark Facial Aligner` |
| **Dùng cho** | Chuẩn bị khuôn mặt cho AI chỉnh sửa mặt, upscale hoặc inpaint; cắt mặt đồng nhất cho cả loạt ảnh |

## Khi nào dùng

Các model về khuôn mặt hoạt động tốt nhất khi mắt, mũi, miệng luôn nằm ở cùng vị trí. Node này xoay ảnh để hai mắt nằm ngang, co giãn để khuôn mặt có kích thước cố định, rồi cắt khung vuông. `padding` điều khiển khoảng trống quanh tóc và cằm.

Đây là nửa đầu của một vòng khép kín: căn mặt, chỉnh sửa ảnh đã cắt, rồi đưa về ảnh gốc bằng [Unwrap Face](unwrap-face.md) với `inverse_matrix_json`.

## Bắt đầu nhanh

1. `Load Image` → [Face Detect Lite](face-detect-lite.md) (hoặc [Face Detect](face-detect.md)).
2. Nối đầu ra `landmarks_json` vào đầu vào `landmarks` của **Landmark Facial Aligner**.
3. Chọn `target_size` (1024 là điểm bắt đầu tốt) và `padding = 0.2`.
4. Nối `image_aligned` vào `Preview Image`.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `image` | — | Ảnh chân dung cần căn. | |
| `landmarks` | chuỗi mẫu cho ảnh chân dung mẫu | Năm điểm dạng JSON: mắt trái, mắt phải, mũi, khóe miệng trái, khóe miệng phải. Dùng được cả giá trị pixel lẫn giá trị 0 đến 1. | Nối từ node phát hiện khuôn mặt thay vì gõ tay. |
| `target_size` | `1024` | Cạnh của ảnh vuông đầu ra: 512, 768 hoặc 1024. | Nhỏ hơn thì nhanh hơn; 1024 hợp với SDXL. |
| `padding` | `0.2` (0 đến 0.5) | Khoảng trống giữ quanh khuôn mặt. 0 là cắt sát, lớn hơn thì thấy nhiều tóc và cằm hơn. | 0.2 đến 0.3 cho hầu hết chân dung. |

| Đầu ra | Kiểu | Ý nghĩa |
|---|---|---|
| `image_aligned` | `IMAGE` | Ảnh khuôn mặt vuông đã căn. |
| `inverse_matrix_json` | `STRING` | Ma trận đưa ảnh đã cắt trở lại ảnh gốc. Nối vào [Unwrap Face](unwrap-face.md). |

## Mẹo khi chỉnh ảnh

- Giá trị từ 1.5 trở xuống được hiểu là tọa độ 0 đến 1; giá trị lớn hơn được hiểu là pixel.
- Hãy chỉnh méo ống kính trước ([Lens Distortion](lens-distortion.md)) nếu mặt nằm gần mép của ảnh góc rộng; mặt bị kéo giãn cho landmark kém chính xác hơn.
- Với cả loạt ảnh, đưa một bộ landmark cho mỗi ảnh (danh sách các bộ năm điểm). Chỉ một bộ thì được dùng lại cho mọi ảnh.

??? note "Bên trong node hoạt động thế nào"
    Năm điểm được khớp với bố cục chuẩn bằng phép biến đổi đồng dạng (xoay, co giãn đều, dịch chuyển; phương pháp Umeyama), rồi áp dụng bằng `warp_affine` của Kornia ra ảnh `target_size × target_size`. Ma trận nghịch đảo được trả về dạng 3×3. Landmark phải không âm; node báo lỗi nếu sau khi chuẩn hóa chúng nằm ngoài ảnh. `target_size` chỉ nhận 512, 768 và 1024. Nguồn: `core/facial_aligner.py` và `nodes/facial_aligner_node.py`.

## Node liên quan

- [Face Detect Lite](face-detect-lite.md)
- [Face Detect](face-detect.md)
- [Unwrap Face](unwrap-face.md)
- [Lens Distortion](lens-distortion.md)

## Workflow mẫu

[Workflow mẫu Facial Aligner](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-06-facial-aligner.json). Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/`. Chuỗi landmark mặc định được đặt cho ảnh chân dung mẫu đó.
