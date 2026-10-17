# Looks / Look màu

Look là các bước chỉnh màu sáng tạo hoàn chỉnh mà bạn thử được chỉ trong một bước. Thay vì dựng một chuỗi node màu dài, bạn chọn phong cách từ danh sách, đặt độ mạnh rồi so sánh. Đây là cách nhanh nhất để tìm hướng đi trước khi tinh chỉnh bằng các công cụ màu.

<div class="grid cards" markdown>

-   ![](../assets/nodes/look-select.jpg)

    **[Look Select](../nodes/look-select.md)**

    Sáu look làm sẵn (teal and orange, golden hour, phim phai màu và nữa) với thanh intensity và tùy chọn bảo vệ da.

</div>

Với look dựa trên LUT `.cube`, xem [LUT Preset](../nodes/lut-preset.md), [LUT Import](../nodes/lut-import.md) và [Tone Match LUT](../nodes/tone-match-lut.md) trong nhóm màu.

## Chuỗi node thường dùng

- Thử nhanh: Load Image → [Look Select](../nodes/look-select.md) → Preview Image
- Look có kiểm soát chi tiết: [Look Select](../nodes/look-select.md) → [Tone Curve](../nodes/tone-curve.md) → Preview Image
- Bắt chước cảm giác của ảnh tham chiếu: [Tone Match LUT](../nodes/tone-match-lut.md) → [LUT Import](../nodes/lut-import.md) trên ảnh của bạn
- Dựng và lưu look riêng: [HALD Identity](../nodes/hald-identity.md) → các node chỉnh màu → [LUT Export](../nodes/lut-export.md)
