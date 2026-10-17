---
title: Layer Add
description: Đặt một ảnh lên chồng layer với blend mode, opacity, fill, mask và clipping.
---

# Layer Add

Thêm một layer vào chồng layer, giống như bạn thả một layer mới vào Photoshop: chọn blend mode, chỉnh opacity và fill, thêm mask nếu cần, và có thể clip vào layer bên dưới.

![Layer Add — trước / sau](../assets/nodes/layer-add.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → compositing → Compositing: Layer Add` |
| **Dùng cho** | Chỉnh màu (color grade), light leak, layer dodge & burn, texture overlay, phủ màu cục bộ qua mask |

## Khi nào dùng

Dùng cho các bước ghép hoàn thiện cần lặp lại được và chỉnh được. Mỗi node **Layer Add** là một layer. Nối nhiều node để dựng chồng layer từ dưới lên: layer đầu tiên bạn thêm nằm ngay trên nền, layer tiếp theo nằm trên nữa, cứ thế.

Cách chồng layer hoạt động:

1. [Layer Stack Start](layer-stack-start.md) tạo layer nền.
2. **Layer Add** thêm layer lên trên, mỗi layer có blend mode, opacity, fill, mask và clipping riêng.
3. [Layer Group](layer-group.md) lồng cả một chồng layer con thành một layer.
4. [Layer Flatten](layer-flatten.md) gộp tất cả thành một ảnh.

## Bắt đầu nhanh

1. `Load Image` → [Layer Stack Start](layer-stack-start.md).
2. Đầu ra stack → `stack` của **Layer Add**. Nối ảnh muốn đặt lên trên vào `layer_image`.
3. Thử `blend_mode = soft_light` và `opacity = 0.5`.
4. Thêm các node **Layer Add** khác nếu cần, rồi kết thúc bằng [Layer Flatten](layer-flatten.md) → `Preview Image`.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `stack` | — | Chồng layer cần thêm vào. | Lấy từ Layer Stack Start hoặc một Layer Add khác. |
| `layer_image` | — | Ảnh đặt lên trên. | Tự co giãn theo kích thước nền nếu khác nhau. |
| `blend_mode` | `normal` | Cách layer này hòa với mọi thứ bên dưới. | Xem danh sách bên dưới. |
| `opacity` | `1.0` (0 đến 1) | Độ mạnh chung của layer. | Layer chỉnh màu thường 0.3 đến 0.6. |
| `fill` | `1.0` (0 đến 1) | Thanh độ mạnh thứ hai. | Nhân với `opacity`. |
| `clip_to_below` | tắt | Giới hạn layer theo mask của layer ngay bên dưới. | Chỉ có tác dụng khi layer dưới có `layer_mask`. |
| `layer_mask` (tùy chọn) | không | Nơi layer hiện ra: trắng = hiện hết, đen = ẩn. | Tự co giãn theo canvas nếu cần. |

Đầu ra: `LAYER_STACK`, chồng layer có thêm layer mới ở trên cùng.

### Các blend mode

Đủ 27 mode trong `core/blend_modes.py`, chia nhóm giống menu của Photoshop:

| Nhóm | Mode |
|---|---|
| Normal | `normal`, `dissolve` |
| Darken (làm tối) | `darken`, `multiply`, `color_burn`, `linear_burn`, `darker_color` |
| Lighten (làm sáng) | `lighten`, `screen`, `color_dodge`, `linear_dodge`, `lighter_color` |
| Contrast (tăng tương phản) | `overlay`, `soft_light`, `hard_light`, `vivid_light`, `linear_light`, `pin_light`, `hard_mix` |
| Comparative (so sánh) | `difference`, `exclusion`, `subtract`, `divide` |
| Màu | `hue`, `saturation`, `color`, `luminosity` |

`dissolve` hoạt động giống Photoshop: mỗi pixel hoặc hoàn toàn là layer, hoặc hoàn toàn là nền, với xác suất bằng `opacity × fill × mask`. Mode này không trộn màu. Họa tiết dùng seed cố định, nên cùng thiết lập luôn cho cùng một kết quả.

## Mẹo khi chỉnh ảnh

- **Soft Light hoặc Overlay** ở opacity thấp là cách nhanh nhất để thêm tương phản hoặc phủ màu mà không bị cháy.
- **Multiply** làm tối, **Screen** làm sáng: dùng với mask đã làm mờ cho kiểu dodge & burn.
- **Color** chỉ lấy hue và saturation của layer, rất tiện để phủ màu da mà không đổi độ sáng. **Luminosity** thì ngược lại.
- Nối mask từ các node Mask (ví dụ [Luminosity Masking](luminosity-masking.md)) vào `layer_mask` để giới hạn chỉnh màu ở vùng highlight hoặc shadow.
- Dùng `clip_to_below` để giữ layer nằm trong vùng mask của layer bên dưới.

??? note "Bên trong node hoạt động thế nào"
    Các layer được render lần lượt. Kết quả blend được tính từ ảnh hiện tại và ảnh của layer (cả hai đều giới hạn trong 0..1), rồi trộn lại bằng `lerp` với `alpha = mask × opacity × fill`, nên opacity và fill cho cùng một kết quả ở đây. Riêng `dissolve` là ngoại lệ: `alpha` được dùng làm xác suất để một pixel lấy hoàn toàn từ layer, với giá trị ngẫu nhiên seed cố định cho từng pixel. Không có mask thì coi như mask trắng hoàn toàn. Với `clip_to_below`, mask của layer được nhân với mask của layer bên dưới; nếu layer dưới không có mask thì coi như trắng, nên clipping không đổi gì. Ảnh và mask của layer được co giãn bilinear theo kích thước nền. Các mode `hue`, `saturation`, `color`, `luminosity` được tính trong không gian HSV nên có thể lệch nhẹ so với Photoshop. Nguồn: `nodes/layer_compositing.py` và `core/blend_modes.py`.

## Node liên quan

- [Layer Stack Start](layer-stack-start.md)
- [Layer Group](layer-group.md)
- [Layer Flatten](layer-flatten.md)
- [Pyramid Blend](pyramid-blend.md)

## Workflow mẫu

- [Layer compositing: 2-layer overlay](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-26-layer-compositing-2layer-overlay.json)
- [Layer compositing: chỉnh màu 5 layer kiểu cinematic](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-27-layer-compositing-5layer-cinematic.json)
- [Layer compositing: group và clipping](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-28-layer-compositing-group-clipping.json)

Hãy chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/` trước.
