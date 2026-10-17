# Độ sâu (Depth)

Các node Depth ước lượng khoảng cách từ máy ảnh tới từng điểm ảnh, rồi dùng bản đồ đó như cách người chụp dùng ống kính: làm mờ hậu cảnh (hoặc tiền cảnh) sau khi đã chụp.

<div class="grid cards" markdown>

-   ![](../assets/nodes/depth-anything.jpg)

    **[Depth Anything V2](../nodes/depth-anything.md)**

    Ước lượng bản đồ độ sâu từ một ảnh (gần = trắng). Bản Small giấy phép Apache-2.0; Base/Large chỉ dùng phi thương mại.

-   ![](../assets/nodes/depth-lens-blur.jpg)

    **[Depth Lens Blur](../nodes/depth-lens-blur.md)**

    Bokeh theo độ sâu: chọn mặt phẳng lấy nét (hoặc để mask mặt/chủ thể tự chọn) và độ sâu trường ảnh.

</div>

## Chuỗi node thường dùng

- **Xóa phông chân dung:** Load Image → [Face Detect Lite](../nodes/face-detect-lite.md) (face_mask) → [Depth Lens Blur](../nodes/depth-lens-blur.md) với face mask làm `focus_mask`. Độ sâu được ước lượng ngay trong node.
- **Ảnh sản phẩm:** [SAM One-Click Mask](../nodes/sam-mask.md) chọn sản phẩm → làm `focus_mask` cho [Depth Lens Blur](../nodes/depth-lens-blur.md).
- **Dùng lại bản đồ độ sâu:** chạy [Depth Anything V2](../nodes/depth-anything.md) một lần → đưa `depth` vào nhiều biến thể Lens Blur, hoặc vào [Mask Preview](../nodes/mask-preview.md) để xem.
