# Bộ lọc (Filters)

Nhóm bộ lọc làm việc trực tiếp trên bức ảnh: texture, cạnh và độ mịn. Tách chi tiết khỏi tông màu bằng frequency separation, làm mịn da mà không mất cạnh, và làm nét ở cuối. Ba node này gộp lại là phần lõi của một lượt retouch chân dung.

<div class="grid cards" markdown>

-   ![](../assets/nodes/frequency-separation.jpg)

    **[Frequency Separation](../nodes/frequency-separation.md)**

    Tách ảnh thành lớp màu mượt và lớp chi tiết mịn.

-   ![](../assets/nodes/edge-aware-smoother.jpg)

    **[Skin Smoother](../nodes/edge-aware-smoother.md)**

    Làm mịn da bằng blur giữ cạnh, có mask tùy chọn và xử lý chia ô.

-   ![](../assets/nodes/smart-sharpen.jpg)

    **[Smart Sharpen](../nodes/smart-sharpen.md)**

    Unsharp mask với ngưỡng nhiễu, chế độ chỉ luminance và mask.

</div>

## Chuỗi node thường dùng

- **Da tự nhiên:** Load Image → [Skin Smoother](../nodes/edge-aware-smoother.md) (strength 0.3 đến 0.5, kèm mask da) → [Smart Sharpen](../nodes/smart-sharpen.md) → Preview Image
- **Sửa màu, giữ lỗ chân lông:** Load Image → [Frequency Separation](../nodes/frequency-separation.md) → chỉ chỉnh `low` → cộng `low` và `high` lại với nhau
- **Chỉ làm nét phần chi tiết:** [Detail Mask](../nodes/high-freq-detail-masker.md) → `mask` của [Smart Sharpen](../nodes/smart-sharpen.md)
