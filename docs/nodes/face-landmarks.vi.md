---
title: Face Landmarks
description: Lấy bản đồ 468 điểm dày đặc của khuôn mặt, kèm ảnh xem trước có vẽ các điểm.
---

# Face Landmarks

Vẽ bản đồ khuôn mặt bằng 468 điểm (mắt, lông mày, mũi, môi, đường hàm, má). Các node warp dùng những điểm này để biến dạng khuôn mặt.

![Face Landmarks — trước / sau](../assets/nodes/face-landmarks.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → face → Face Landmarks (MediaPipe 468)` |
| **Dùng để** | Đưa dữ liệu cho [Face Warp](face-warp.md) hoặc [Face Warp TPS](face-warp-tps.md); kiểm tra lưới điểm khớp với mặt đến đâu |

## Khi nào dùng

Năm điểm là đủ để xoay thẳng khuôn mặt nhưng không đủ để chỉnh dáng. Muốn thon hàm, nâng mắt hoặc lấy biểu cảm từ ảnh khác, bạn cần một bản đồ dày. Mỗi landmark là một điểm đánh số trên khuôn mặt, và cùng một số luôn nằm đúng một vị trí (ví dụ đầu mũi), nên bạn có thể so sánh khuôn mặt ở hai ảnh từng điểm một.

Chạy node một lần trên ảnh gốc để lấy điểm "nguồn" (source), và một lần nữa trên ảnh tham chiếu hoặc bản đã chỉnh để lấy điểm "đích" (destination). Nối cả hai vào node warp.

## Bắt đầu nhanh

1. Nối **Load Image** vào **Face Landmarks (MediaPipe 468)**.
2. Để `draw_overlay` bật và xem `overlay` trong **Preview Image** để chắc chắn các chấm nằm đúng trên mặt.
3. Nối `landmarks` vào `src_landmarks` của [Face Warp TPS](face-warp-tps.md) (hoặc [Face Warp](face-warp.md)).
4. Lấy `dst_landmarks` từ một node Face Landmarks thứ hai chạy trên ảnh đích.

## Thông số

| Thông số | Mặc định | Tác dụng | Gợi ý |
|---|---|---|---|
| `max_num_faces` | `1` (1–10) | Số mặt tối đa tìm trong mỗi ảnh. Mặt thiếu được điền NaN (không có dữ liệu). | Chân dung: giữ 1. Các node warp dùng khuôn mặt đầu tiên. |
| `min_detection_confidence` | `0.5` (0.0–1.0) | Mức chắc chắn tối thiểu của MediaPipe để tính là một khuôn mặt. | Hạ xuống nếu một khuôn mặt thật bị bỏ sót. |
| `refine_landmarks` | bật | Giữ lại để tương thích. Đầu ra luôn là 468 điểm dày. | Để nguyên. |
| `draw_overlay` | bật | Vẽ các điểm thành chấm xanh lá trên ảnh `overlay`. | Tắt nếu không cần xem trước. |

Đầu ra:

| Đầu ra | Kiểu | Ý nghĩa |
|---|---|---|
| `landmarks` | `LANDMARKS` | 468 điểm cho mỗi mặt, theo toạ độ chuẩn hóa `[0, 1]` của ảnh. Chỉ nối vào node nhận `LANDMARKS`. |
| `overlay` | `IMAGE` | Ảnh của bạn với các điểm được vẽ lên. |

## Mẹo khi chỉnh ảnh

- `LANDMARKS` là một kiểu dữ liệu riêng. Nó không nối được vào `landmarks_json` của [Facial Aligner](facial-aligner.md); với aligner, hãy dùng node nhận diện khuôn mặt.
- Chạy node trên ảnh cùng độ phân giải (hoặc ít nhất cùng khung hình) với ảnh bạn sẽ warp, để các điểm khớp nhau.
- Nếu không tìm thấy mặt, vị trí đó chứa NaN. Khi đó node warp trả về ảnh nguyên vẹn.
- Nhiễu khi nhận diện là có thật: hai lần chạy trên hai vùng cắt hơi khác nhau sẽ lệch vài pixel. Vì vậy [Face Warp TPS](face-warp-tps.md) có thông số `smoothness`.

??? note "Bên trong node hoạt động thế nào"
    Node dùng MediaPipe FaceLandmarker (tasks API) và dùng chung file model với [Face Detect](face-detect.md). Kết quả là tensor float `(batch, số mặt, 468, 2)`, được điền NaN khi tìm thấy ít mặt hơn. Ảnh được chuyển sang RGB 8-bit cho MediaPipe. Mã nguồn nằm ở `core/face_pipeline.py`.

## Node liên quan

- [Face Warp](face-warp.md) — warp Delaunay theo từng tam giác.
- [Face Warp TPS](face-warp-tps.md) — warp thin-plate-spline mượt.
- [Face Detect](face-detect.md) — phiên bản 5 điểm để căn chỉnh.
- [Tổng quan nhóm Khuôn mặt](../categories/face.md)

## Workflow mẫu

[Workflow mẫu face pipeline](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-18-face-pipeline-v2.json) (Face Landmarks → Face Warp → Face Beauty Blend). Hãy chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/` trước.
