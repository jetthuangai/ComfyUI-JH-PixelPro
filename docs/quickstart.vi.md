---
title: Hướng dẫn nhanh
description: Cài PixelPro, tìm node và chạy workflow đầu tiên trong năm phút.
---

# Hướng dẫn nhanh

Năm phút từ lúc cài đặt đến kết quả đầu tiên.

## 1. Cài đặt

=== "ComfyUI Manager (khuyên dùng)"

    1. Mở **Manager → Custom Nodes Manager**.
    2. Tìm `ComfyUI-JH-PixelPro` rồi bấm **Install**.
    3. Khởi động lại ComfyUI.

=== "Cài thủ công (git)"

    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/jetthuangai/ComfyUI-JH-PixelPro.git
    cd ComfyUI-JH-PixelPro
    pip install -r requirements.txt
    ```

    Dùng đúng Python mà ComfyUI đang chạy (bản portable hoặc venv có `python` riêng). Sau đó khởi động lại ComfyUI.

!!! info "Yêu cầu"
    Python 3.10 trở lên, PyTorch đi kèm ComfyUI, Kornia ≥ 0.7, MediaPipe ≥ 0.10, OpenCV ≥ 4.8 và SciPy ≥ 1.10. Tùy chọn: `onnxruntime-gpu` (hoặc `onnxruntime`) cho Depth Anything và Depth Lens Blur.
    Nên có GPU NVIDIA từ 8 GB VRAM. Máy chỉ có CPU vẫn chạy được, chỉ chậm hơn.

## 2. Tìm node

Bấm chuột phải lên canvas (hoặc nhấp đúp để tìm kiếm), chọn **Add Node → ComfyUI-JH-PixelPro**. Các node được chia nhóm giống trang tài liệu này:

`color` · `mask` · `face` · `compositing` · `filters` · `geometry` · `looks` · `qa`

!!! tip "Mẹo"
    Gõ một phần tên node vào ô tìm kiếm, ví dụ `Frequency`, `Matte` hay `CLAHE`.

## 3. Chạy workflow đầu tiên

1. Chép `workflows/sample_portrait.jpg` trong thư mục pack vào `ComfyUI/input/`.
2. Kéo một workflow mẫu thả vào canvas. Nên thử trước:
    - [Ví dụ CLAHE Local Contrast](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-37-clahe-local-contrast.json): một node, có kết quả ngay.
    - [Ví dụ Face Detect Lite](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-38-face-detect-lite.json): khung mặt, landmark và căn mặt.
    - [Ví dụ Mask Cleanup](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-41-mask-cleanup.json): chép thêm `workflows/sample_binary_mask.png` vào `ComfyUI/input/`.
3. Bấm **Queue Prompt**. Mỗi lần chỉ đổi một thông số rồi so sánh.

Danh sách đầy đủ các workflow mẫu nằm ở trang [Ví dụ](examples/index.md).

## 4. Tự dựng chuỗi node

Cách học nhanh nhất là thử một bước retouch kinh điển ở mỗi nhóm:

| Mục tiêu | Dùng thử |
|---|---|
| Làm đều da nhưng giữ lỗ chân lông | [Frequency Separation](nodes/frequency-separation.md) hoặc [Skin Smoother](nodes/edge-aware-smoother.md) |
| Chỉ làm nét mắt và tóc | [Detail Mask](nodes/high-freq-detail-masker.md) → [Smart Sharpen](nodes/smart-sharpen.md) |
| Chỉnh màu kiểu Camera Raw | [ColorLab](nodes/color-lab.md), [Tone Curve](nodes/tone-curve.md) |
| Khớp màu ảnh này theo ảnh khác | [Color Matcher](nodes/color-matcher.md) hoặc [Color Matcher Region](nodes/color-matcher-region.md) |
| Tách tóc thật gọn | [Trimap Builder](nodes/trimap-builder.md) → [Alpha Matte](nodes/alpha-matte.md) |
| Chạy AI trên khuôn mặt rồi dán lại | [Face Detect Lite](nodes/face-detect-lite.md) → [Facial Aligner](nodes/facial-aligner.md) → … → [Unwrap Face](nodes/unwrap-face.md) |
| Kiểm tra ảnh đã thay đổi gì | [QA Image Diff](nodes/qa-image-diff.md) |

## Xử lý sự cố

??? question "Không thấy node trong menu"
    Xem cửa sổ console của ComfyUI có lỗi import từ `ComfyUI-JH-PixelPro` không. Thường là thiếu thư viện: chạy `pip install -r requirements.txt` bằng Python của ComfyUI rồi khởi động lại.

??? question "Face Detect Lite hoặc Auto Align (LoFTR) tự tải file"
    Đúng vậy, chỉ ở lần dùng đầu tiên. Face Detect Lite tải trọng số YuNet (khoảng 0,4 MB), Auto Align với `method = loftr` tải trọng số LoFTR (khoảng 45 MB) vào thư mục cache của torch. Face Detect (MediaPipe) tải model (khoảng 5 MB) vào `ComfyUI/models/mediapipe`.

??? question "Máy chậm hẳn khi xử lý ảnh lớn"
    Bật `tile_mode` trong [Skin Smoother](nodes/edge-aware-smoother.md) với mọi ảnh lớn hơn ảnh xem thử. Từ v1.5.2, kích thước ô được tính để bộ nhớ quanh mức 2 GB. Nếu VRAM đã đầy, hãy đóng bớt workflow nặng khác.

??? question "Đã cập nhật pack nhưng ComfyUI không đổi gì"
    ComfyUI chỉ nạp code của custom node một lần lúc khởi động. Hãy khởi động lại ComfyUI sau khi cập nhật pack.
