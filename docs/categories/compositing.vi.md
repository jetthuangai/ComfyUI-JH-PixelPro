# Ghép ảnh (Compositing)

Nhóm node này biến ComfyUI thành chồng layer kiểu Photoshop: một ảnh nền, các layer xếp lên trên với blend mode, opacity, fill, mask và clipping, group, rồi flatten ở cuối. Chúng dành cho các bước hoàn thiện có tính lặp lại sau khi tạo ảnh hoặc retouch, khi bạn muốn chỉnh một layer mà không làm lại các layer còn lại. Pyramid Blend hơi khác: nó ghép hai ảnh qua mask sao cho không lộ đường nối.

<div class="grid cards" markdown>

-   ![](../assets/nodes/layer-stack-start.jpg)

    **[Layer Stack Start](../nodes/layer-stack-start.md)**

    Tạo layer nền của chồng layer từ một ảnh.

-   ![](../assets/nodes/layer-add.jpg)

    **[Layer Add](../nodes/layer-add.md)**

    Thêm layer với blend mode, opacity, fill, mask và clipping.

-   ![](../assets/nodes/layer-group.jpg)

    **[Layer Group](../nodes/layer-group.md)**

    Gộp một chồng layer con thành một layer, như group trong Photoshop.

-   ![](../assets/nodes/layer-flatten.jpg)

    **[Layer Flatten](../nodes/layer-flatten.md)**

    Render chồng layer về lại ảnh thường.

-   ![](../assets/nodes/pyramid-blend.jpg)

    **[Pyramid Blend](../nodes/pyramid-blend.md)**

    Ghép liền mạch hai ảnh qua mask bằng blend nhiều dải tần.

</div>

## Chuỗi node thường dùng

- **Chỉnh màu bằng layer:** Load Image → [Layer Stack Start](../nodes/layer-stack-start.md) → [Layer Add](../nodes/layer-add.md) (`soft_light`) → [Layer Add](../nodes/layer-add.md) (`screen`) → [Layer Flatten](../nodes/layer-flatten.md) → Preview Image
- **Một look dạng group:** hai chuỗi [Layer Stack Start](../nodes/layer-stack-start.md) → [Layer Group](../nodes/layer-group.md) (`overlay`, opacity 0.8) → [Layer Flatten](../nodes/layer-flatten.md)
- **Dán lại khuôn mặt đã retouch:** [LUT Preset](../nodes/lut-preset.md) trên khuôn mặt → [Pyramid Blend](../nodes/pyramid-blend.md) qua mask khuôn mặt → Preview Image
