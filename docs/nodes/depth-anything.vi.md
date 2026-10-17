---
title: Depth Anything V2
description: Ước lượng bản đồ độ sâu từ một tấm ảnh, để các node sau biết chỗ nào gần, chỗ nào xa.
---

# Depth Anything V2

Biến ảnh phẳng thành bản đồ độ sâu (depth map): trắng là gần máy ảnh, đen là xa. Dùng để làm xoá phông, sương mù, relight hoặc mask theo độ sâu.

![Depth Anything V2 — trước / sau](../assets/nodes/depth-anything.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → depth → Depth: Depth Anything V2` |
| **Dùng cho** | Giả lập xoá phông, tách tiền cảnh/hậu cảnh, mask độ sâu khi chỉnh màu |

## Khi nào dùng

Depth map là ảnh xám mà độ sáng thể hiện khoảng cách. Depth Anything V2 đoán nó từ một tấm ảnh, không cần cặp ảnh stereo hay cảm biến. Kết quả mang tính *tương đối*: cho biết cái nào gần hơn cái nào, không cho biết bao nhiêu mét. Giá trị được chuẩn hoá riêng cho từng ảnh: 1 là vật gần nhất trong khung, 0 là xa nhất.

Nối output `depth` vào [Depth Lens Blur](depth-lens-blur.md) hoặc bất kỳ node nào nhận mask. `depth_preview` là cùng bản đồ đó dạng ảnh xám để xem bằng Preview Image.

## Bắt đầu nhanh

1. Load Image → **Depth: Depth Anything V2** → Preview Image (dùng `depth_preview`).
2. Để `model` là `small` và `resolution` là 518.
3. Nối `depth` vào [Depth Lens Blur](depth-lens-blur.md).

## Cài đặt

- Cài onnxruntime một lần: `pip install onnxruntime-gpu` (hoặc `pip install onnxruntime` nếu chỉ chạy CPU). Thiếu nó node sẽ báo lỗi rõ ràng.
- File model được tải ở lần chạy đầu vào `ComfyUI/models/depth_anything/` (hoặc thư mục `depth_anything` trong `extra_model_paths.yaml` nếu bạn đặt) với tên `depth-anything-v2-<size>.onnx` (small khoảng 99 MB). Thư mục chỉ được tạo khi bắt đầu tải. Việc tải dùng một phiên bản Hugging Face cố định (không phải `main`), giới hạn chờ mạng 60 giây; model small còn được kiểm tra SHA256, còn base và large chỉ ghim theo phiên bản. Nếu không tải được, hãy tự lưu file vào đó từ `https://huggingface.co/onnx-community/depth-anything-v2-<size>/resolve/<revision>/onnx/model.onnx` (các revision nằm trong `core/depth_anything.py`).

## Thông số

| Thông số | Mặc định | Tác dụng | Gợi ý |
|---|---|---|---|
| `model` | `small` | Cỡ mạng nơ-ron. | `small` nhanh và dùng thương mại được. `base`, `large` chi tiết hơn. |
| `resolution` | 518 | Cạnh dài (pixel) mà mạng xử lý; hai cạnh được làm tròn về bội số của 14. | 266 đến 1036. Tăng để viền mịn hơn, giảm để nhanh hơn. |
| `keep_model_loaded` (tùy chọn) | bật | Giữ model Depth Anything trong VRAM giữa các lần chạy. | Tắt để giải phóng VRAM của nó ngay sau khi chạy; nút *Free memory* của ComfyUI không giải phóng được model này. |

**Output:** `depth` (MASK, 1 = gần nhất) và `depth_preview` (IMAGE, độ sâu dạng ảnh xám).

!!! warning "Giấy phép"
    `small` theo Apache-2.0, dùng cho việc thương mại được. `base` và `large` theo **CC-BY-NC-4.0: chỉ dùng phi thương mại**. Đừng dùng cho job có thu tiền.

## Mẹo khi chỉnh ảnh

- Độ sâu chỉ là ước đoán nên vật mảnh (tóc, dây điện) và kính có thể sai. Hãy xem `depth_preview` trước khi làm mờ.
- Bản đồ là tương đối theo từng ảnh. Với batch, mỗi ảnh được chuẩn hoá riêng nên cùng một khoảng cách có thể ra giá trị khác nhau giữa hai khung hình. Với video, depth có thể nhấp nháy từ khung này sang khung khác.
- Ảnh cận mặt không có hậu cảnh thật sẽ cho bản đồ theo hình khối khuôn mặt, không phải "hậu cảnh".
- Có thể làm mềm viền depth bằng [Mask Edge Smoother](mask-edge-smoother.md) trước khi dùng.

??? note "Bên trong node hoạt động thế nào"
    Ảnh được resize (bicubic) sao cho hai cạnh là bội số của 14 và cạnh dài gần `resolution`, chuẩn hoá bằng mean/std của ImageNet rồi chạy model ONNX trực tiếp bằng onnxruntime (dùng CUDA nếu có, không thì CPU; chỉ giữ một session trong cache; với CUDA dùng chiến lược bộ nhớ `kSameAsRequested`, nên bộ nhớ GPU phát sinh chỉ khoảng 0,4 đến 0,5 GB thay vì khoảng 1,3 GB). Output thô là nghịch đảo độ sâu tương đối; được resize về cỡ ảnh gốc (bilinear) và co giãn về 0..1 cho từng ảnh. Ảnh một màu cho ra toàn 0. Batch được xử lý từng ảnh một. Mã nguồn: `core/depth_anything.py`.

## Node liên quan

- [Depth Lens Blur](depth-lens-blur.md) dùng bản đồ này để làm mờ kiểu ống kính
- [Face Detect Lite](face-detect-lite.md) cho mask khuôn mặt để lấy nét
- [Mask Edge Smoother](mask-edge-smoother.md) để làm mềm bản đồ

## Workflow mẫu

[Workflow mẫu Depth Lens Blur](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-45-depth-lens-blur.json) chạy Depth Anything bên trong Depth Lens Blur và hiển thị bản đồ độ mờ. Workflow chỉ dùng node lõi của ComfyUI và pack này. Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/`.

![Kết quả: Workflow mẫu Depth Lens Blur](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/N-45-depth-lens-blur-preview.png)
