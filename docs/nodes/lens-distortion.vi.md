---
title: Lens Distortion
description: Nắn thẳng méo thùng hoặc méo gối do ống kính, hoặc thêm méo có chủ đích, với các preset ống kính dựng sẵn.
---

# Lens Distortion

Sửa hiện tượng đường thẳng bị cong do ống kính (méo thùng ở góc rộng, méo gối nhẹ ở tele ngắn), hoặc làm ngược lại để tạo cảm giác ống kính cho ảnh sạch.

![Lens Distortion — trước / sau](../assets/nodes/lens-distortion.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → geometry → Lens Distortion Corrector` |
| **Dùng cho** | Chân dung góc rộng bị kéo giãn mặt hoặc cong mép, cảnh quay từ action camera, hiệu ứng ống kính sáng tạo |

## Khi nào dùng

Ống kính góc rộng làm cong đường thẳng gần mép khung và kéo giãn khuôn mặt. Hãy sửa trước khi phát hiện khuôn mặt, căn mặt hoặc các chỉnh sửa nhạy với hình học, để các bước sau làm việc trên hình dạng tự nhiên hơn.

Chọn preset gần với ống kính của bạn, hoặc chọn `custom` và tự nhập hệ số. Các preset chỉ là giá trị xấp xỉ, không phải profile của một máy cụ thể, nên hãy kiểm tra kết quả bằng mắt.

## Bắt đầu nhanh

1. `Load Image` → **Lens Distortion Corrector** → `Preview Image`.
2. Chọn một `preset`, ví dụ `canon_24mm_wide`.
3. Giữ `direction = inverse` để sửa ảnh.
4. So với ảnh gốc, nếu đường thẳng vẫn cong thì đổi preset hoặc dùng `custom`.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `preset` | `no_op_identity` | `canon_24mm_wide`: sửa méo thùng cho ống kính góc rộng. `sony_85mm_tele`: sửa méo gối nhẹ. `gopro_fisheye`: gỡ méo thùng mạnh. `no_op_identity`: giữ nguyên ảnh. `custom`: dùng năm giá trị bên dưới. | Khi chọn preset, năm giá trị bên dưới bị bỏ qua. |
| `direction` | `inverse` | `inverse` sửa ảnh bị méo. `forward` thêm méo vào ảnh sạch. | Dùng `forward` cho hiệu ứng sáng tạo. |
| `k1`, `k2`, `k3` | `0.0` (−1 đến 1) | Méo xuyên tâm (radial): đường thẳng cong nhiều thế nào về phía mép. | Chỉ dùng với `custom`. k1 âm sửa méo thùng. |
| `p1`, `p2` | `0.0` (−0.1 đến 0.1) | Méo tiếp tuyến (tangential), cho ống kính lệch tâm nhẹ. | Chỉ dùng với `custom`. Hiếm khi cần. |

Đầu ra: `image_rectified`, ảnh đã sửa (hoặc đã làm méo) (`IMAGE`).

## Mẹo khi chỉnh ảnh

- Sửa méo ống kính trước, sau đó mới crop, căn chỉnh và retouch.
- Nhìn vào một cạnh thẳng gần góc khung để đánh giá kết quả.
- Đầu ra giữ nguyên kích thước khung của ảnh gốc.
- Chạy node này trước [Face Detect Lite](face-detect-lite.md) và [Facial Aligner](facial-aligner.md) với chân dung góc rộng.

??? note "Bên trong node hoạt động thế nào"
    Cài đặt mô hình Brown-Conrady với `k1, k2, k3` (xuyên tâm) và `p1, p2` (tiếp tuyến). Camera được giả định có tiêu cự bằng cạnh dài của ảnh và tâm ở giữa ảnh, nên giá trị chỉ xấp xỉ chứ không phải đã hiệu chuẩn. Giá trị preset: `canon_24mm_wide` k1 −0.18, k2 0.08, k3 −0.02; `sony_85mm_tele` k1 0.03, k2 −0.01; `gopro_fisheye` k1 −0.35, k2 0.12, k3 −0.04; các hệ số tiếp tuyến đều bằng 0. Chế độ inverse dùng `undistort_image` của Kornia trên GPU và `remap` của OpenCV trên CPU (và khi dự phòng); chế độ forward lấy mẫu bằng `grid_sample`. Nguồn: `core/lens_distortion.py` và `nodes/lens_distortion_node.py`.

## Node liên quan

- [Facial Aligner](facial-aligner.md)
- [Auto Align](auto-align.md)
- [Face Detect Lite](face-detect-lite.md)

## Workflow mẫu

[Workflow mẫu Lens Distortion](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-07-lens-distortion.json). Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/`.
