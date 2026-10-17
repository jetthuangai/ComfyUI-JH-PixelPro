---
title: Depth Lens Blur
description: Thêm hiệu ứng xoá phông (bokeh) chân thực theo độ sâu của cảnh, với điểm lấy nét do bạn chọn.
---

# Depth Lens Blur

Giả lập ống kính khẩu độ lớn sau khi chụp. Chủ thể giữ nét, vật gần hơn hoặc xa hơn mờ dần thành bokeh mịn theo bản đồ độ sâu.

![Depth Lens Blur — trước / sau](../assets/nodes/depth-lens-blur.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → depth → Depth: Lens Blur` |
| **Dùng cho** | Xoá phông chân dung, ảnh sản phẩm nền kem mịn, cứu hậu cảnh rối |
| **Tốc độ** | Ảnh 2048 × 2048 đến từ ComfyUI (tensor trên CPU), chưa tính model depth: khoảng 0,03 giây với GPU CUDA (RTX 5090, đã gồm chép dữ liệu); chỉ CPU thì khoảng 2 giây ở `max_blur` 24, tối đa khoảng 10 giây ở `max_blur` 96 với 16 lớp |

## Khi nào dùng

Trên ống kính thật chỉ một khoảng cách là nét. Càng xa khoảng cách đó, đĩa mờ (circle of confusion) càng lớn. Node làm điều tương tự cho từng pixel, dựa trên depth map trong đó 1 là gần nhất.

- **Lấy nét** (`focus_depth`) là độ sâu được giữ nét.
- **Độ sâu trường ảnh** (`depth_of_field`) là bề rộng vùng nét. Giá trị nhỏ giống f/1.4, lớn giống f/16.
- **Max blur** (`max_blur`) là độ mờ (pixel) tại chỗ xa vùng nét nhất.

Chưa nối depth map? Node tự ước lượng bằng [Depth Anything V2](depth-anything.md) (model small, 518 px). Cần onnxruntime (`pip install onnxruntime-gpu`).

## Bắt đầu nhanh

1. Load Image → **Depth: Lens Blur** → Preview Image.
2. Để trống `depth` để tự tính, hoặc nối bản đồ của bạn (ví dụ từ [Depth Anything V2](depth-anything.md)).
3. Thử `depth_of_field` 0.1 và `max_blur` 24, rồi tăng `max_blur` nếu muốn mờ mạnh hơn.
4. Muốn tự động giữ nét người, nối mask khuôn mặt vào `focus_mask` (xem bên dưới).

## Thông số

| Thông số | Mặc định | Tác dụng | Gợi ý |
|---|---|---|---|
| `focus_depth` | 0.5 | Độ sâu giữ nét (1 = gần nhất). | Bị bỏ qua khi đã nối `focus_mask`. |
| `depth_of_field` | 0.1 | Bề rộng vùng nét. | 0 đến 1. Càng nhỏ càng nông. |
| `max_blur` | 24 | Bán kính mờ (pixel) tại chỗ xa vùng nét nhất. | 1 đến 96. Ảnh lớn thì tăng lên. |
| `bokeh` | `disc` | `disc` cho bokeh phẳng như ống kính thật; `gaussian` mờ mềm hơn. | `disc` cho cảm giác nhiếp ảnh. |
| `layers` | 8 | Số lớp mờ được trộn với nhau. | 2 đến 16. Nhiều hơn = chuyển mượt hơn. |
| `blur_foreground` | bật | Tắt: chỉ vật phía sau mặt phẳng nét mới bị mờ. | Tắt khi muốn giữ nét tiền cảnh. |
| `depth` (tuỳ chọn) | không | Depth map, 1 = gần nhất. | Không nối thì tự ước lượng. Bản đồ khác cỡ sẽ được resize theo ảnh. |
| `focus_mask` (tuỳ chọn) | không | Mask chủ thể. Độ sâu trung vị trong mask trở thành điểm nét. | Dùng được mask khuôn mặt hoặc SAM. |
| `keep_model_loaded` (tuỳ chọn) | bật | Giữ model Depth Anything tích hợp trong VRAM giữa các lần chạy (chỉ dùng khi không nối `depth`). | Tắt để giải phóng VRAM của nó ngay sau khi chạy; nút *Free memory* của ComfyUI không giải phóng được model này. |

**Output:** `image` (IMAGE) và `coc` (MASK, lượng mờ từ 0 đến 1; nhân với `max_blur` ra pixel), tiện để kiểm tra vùng nào bị mờ.

## Chuỗi node: tự lấy nét vào khuôn mặt

`face_mask` của [Face Detect Lite](face-detect-lite.md) → `focus_mask` của **Depth Lens Blur**. Độ sâu khuôn mặt được đo tự động và mặt luôn nét dù người đứng xa hay gần. Mask khác cũng dùng cách tương tự, ví dụ từ [SAM Mask](sam-mask.md).

## Mẹo khi chỉnh ảnh

- Với video, depth tự động được chuẩn hoá trên từng khung hình nên điểm nét có thể nhấp nháy giữa các khung. Hãy dùng depth ổn định hoặc `focus_depth` cố định cho clip.
- Kiểm tra depth map trước. Độ mờ bám theo viền depth, nên viền depth sai sẽ thành viền mờ sai.
- Chọn `max_blur` theo cỡ ảnh: 24 vừa phải với ảnh 2K, từ 60 trở lên hợp với ảnh 6K.
- Đặt `depth_of_field` nhỏ (0.05 đến 0.15) để giữ nét cả khuôn mặt chứ không chỉ đôi mắt.
- Thêm chút noise/grain sau khi làm mờ để vùng mờ đồng nhất với vùng nét.

??? note "Bên trong node hoạt động thế nào"
    Circle of confusion mỗi pixel: `clamp((|depth − focus| − dof/2) / (1 − dof/2), 0, 1) × max_blur`. Node dựng `layers` ảnh mờ với bán kính từ 0 đến `max_blur`, rồi trộn mỗi pixel giữa hai bán kính gần nhất. Mỗi lớp là một phép tích chập chuẩn hoá (`blur(image·w) / blur(w)`), trong đó trọng số `w` giảm với pixel nét hơn lớp đó, nhờ vậy tiền cảnh nét không lem sang hậu cảnh mờ. Bán kính trên 8 px được làm mờ trên bản thu nhỏ rồi phóng lại, nên bộ nhớ và thời gian có giới hạn (không dùng unfold). Node chạy trên GPU nếu có (ComfyUI đưa ảnh vào ở CPU; kết quả được trả về đúng thiết bị cũ). Đỉnh bộ nhớ GPU khoảng 0,45 GB ở 2048 × 2048. Bản đồ độ mờ được làm sạch bằng một phép closing nhỏ (cửa sổ khoảng `max_blur` / 8 px) để không còn vệt nét mỏng ở chỗ depth cắt qua mặt phẳng nét giữa hai vùng mờ; vùng nét rộng giữ nguyên. Giá trị NaN trong depth hoặc focus mask được đổi thành số hữu hạn, nên một pixel lỗi không làm hỏng cả ảnh. Hạn chế: tiền cảnh bị mờ không lan đè lên hậu cảnh đang nét, và lỗi depth sẽ thành lỗi mờ. Mã nguồn: `core/depth_lens_blur.py`.

## Node liên quan

- [Depth Anything V2](depth-anything.md) để tạo depth map
- [Face Detect Lite](face-detect-lite.md) cho mask khuôn mặt để lấy nét
- [Mask Edge Smoother](mask-edge-smoother.md) để làm mềm viền depth trước

## Workflow mẫu

[Workflow mẫu Depth Lens Blur](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-45-depth-lens-blur.json) tự ước lượng độ sâu bằng Depth Anything tích hợp, lấy nét vào khuôn mặt từ Face Detect Lite và làm mờ cánh tay phía trước; có ảnh cắt cận cánh tay trước và sau. Workflow chỉ dùng node lõi của ComfyUI và pack này. Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/`.

![Kết quả: Workflow mẫu Depth Lens Blur](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/N-45-depth-lens-blur-preview.png)
