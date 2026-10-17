---
title: Color Matcher Region
description: Khớp màu của một vùng trên ảnh tham chiếu vào một vùng trên ảnh đích, kể cả khi hai ảnh khác kích thước.
---

# Color Matcher Region

Khớp màu giữa hai vùng bạn chọn, ví dụ da của khuôn mặt AI về da của ảnh gốc, hoặc bầu trời của tấm này về tấm kia.

![Color Matcher Region — trước / sau](../assets/nodes/color-matcher-region.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → color → Color Matcher Region (LAB)` |
| **Dùng để** | Khớp da với da, khớp trời hoặc phông nền, khớp vùng đã retouch với một ảnh khác |

## Khi nào dùng

[Color Matcher](color-matcher.md) so sánh toàn bộ hai ảnh cùng kích thước. Nhiều lúc bạn không muốn vậy: ảnh ghép mặt có da nằm cạnh bức tường xanh, và ảnh tham chiếu có khi là một khung cắt hoàn toàn khác. Color Matcher Region cho bạn vẽ mask ở mỗi bên. "Tính cách màu" (trung bình và độ trải) chỉ được đo bên trong mask tham chiếu, rồi dùng để đổi màu đúng vùng nằm trong mask đích. Vì chỉ so sánh thống kê nên hai ảnh không cần cùng kích thước hay bố cục.

## Bắt đầu nhanh

1. Nối ảnh đang chỉnh vào `image_target`, ảnh có màu bạn muốn vào `image_reference`.
2. Tạo mask phủ vùng cần sửa (ví dụ bằng [Skin Tone Tri-Region](skin-tone-tri-region.md)) và nối vào `target_mask`.
3. Tạo mask phủ vùng tương ứng trên ảnh tham chiếu và nối vào `reference_mask`.
4. Giữ `channels` là `ab`, đặt `strength` quanh 0.8, rồi xem `image_matched`.

## Thông số

| Thông số | Mặc định | Tác dụng | Gợi ý |
|---|---|---|---|
| `image_target` | bắt buộc | Ảnh cần đổi màu. | |
| `image_reference` | bắt buộc | Ảnh cung cấp màu. Có thể khác kích thước. | |
| `channels` | `ab` | `ab` chỉ khớp màu, giữ độ sáng của ảnh đích. `lab` khớp cả độ sáng. | Da, sản phẩm: dùng `ab`. |
| `strength` | 1.0 (0 đến 1) | 0 giữ nguyên ảnh, 1 áp dụng khớp hoàn toàn. | 0.6 đến 0.9 cho kết quả tự nhiên. |
| `target_mask` (tùy chọn) | toàn ảnh | Vùng của ảnh đích được đo và đổi màu. Viền mềm cho chuyển tiếp mềm. Phải cùng kích thước ảnh đích. | Nên feather mask cứng trước. |
| `reference_mask` (tùy chọn) | toàn ảnh | Vùng của ảnh tham chiếu được đo. Phải cùng kích thước ảnh tham chiếu. | Chọn mảng sạch, sáng đều. |
| `method` (tùy chọn) | `stats` | `stats` khớp giá trị trung bình và độ trải của từng kênh (mượt, an toàn). `histogram` khớp toàn bộ phân bố của từng kênh, nên vùng tối, trung tính và sáng đều rơi đúng chỗ như ảnh tham chiếu. | Thử `histogram` khi `stats` gần đúng nhưng chưa tới. |

**Đầu ra:** `image_matched`, ảnh đã đổi màu. Ngoài `target_mask`, ảnh giữ nguyên.

## Stats hay histogram?

- **`stats`** (mặc định) dịch và co giãn từng kênh để trung bình và độ trải bằng ảnh tham chiếu. Nhẹ nhàng, dễ đoán, hợp với hầu hết việc sửa màu da và cân bằng trắng.
- **`histogram`** uốn từng kênh để toàn bộ phân bố (histogram) theo ảnh tham chiếu. Dùng khi hai ảnh cùng mức trung bình nhưng cảm giác khác hẳn, ví dụ bản render phẳng so với ảnh chụp tương phản. Mạnh hơn, nên hạ `strength` và để ý nhiễu hoặc dải bậc (banding) ở vùng mịn.

Workflow cũ vẫn chạy: khi không có `method`, node dùng `stats`.

## Mẹo khi chỉnh ảnh

- Với khuôn mặt, dùng `ab`: ánh sáng và kết cấu da vẫn còn, chỉ màu thay đổi.
- Chọn vùng tham chiếu không có bóng đổ, kính, son hay tóc; một bên má sạch tốt hơn cả khuôn mặt.
- Làm mềm mask cứng trước khi dùng. [Mask Edge Smoother](mask-edge-smoother.md) hoặc [Sub-Pixel Mask Refiner](sub-pixel-mask-refiner.md) giúp viền không lộ đường nối.
- Nếu chỉ cần khớp cả khung giữa hai ảnh cùng kích thước, [Color Matcher](color-matcher.md) đơn giản hơn là đủ.

??? note "Bên trong node hoạt động thế nào"
    Cả hai ảnh được chuyển sang CIE LAB (Kornia). Trung bình và độ lệch chuẩn của các kênh được chọn được tính có trọng số mask trong từng mask, sau đó ảnh đích được dịch và co giãn theo thống kê của ảnh tham chiếu (Reinhard). Kết quả đổi về RGB, trộn theo `strength`, cuối cùng ghép bằng `lerp(target, matched, target_mask)`. Mỗi mask phải cùng chiều cao, chiều rộng với ảnh của nó và có ít nhất một pixel dương; một mask hoặc một ảnh tham chiếu duy nhất sẽ được dùng cho cả batch. Ảnh và mask ngoài 0..1 bị kẹp lại kèm cảnh báo trong log.

    Với `method = histogram`, mỗi kênh được chọn của ảnh đích và ảnh tham chiếu được chia vào histogram 4096 bin (có trọng số theo mask nếu có). Mỗi pixel đích được ánh xạ sang phân vị của nó trong CDF của ảnh đích, rồi sang giá trị ở cùng phân vị trong CDF của ảnh tham chiếu (nội suy tuyến tính trong bin). Phần trộn theo `strength` và đường ra giống `stats`. Chạy trên GPU vài mili giây ở 2K, dùng dưới 1 GB.

## Node liên quan

- [Color Matcher](color-matcher.md): bản khớp toàn khung
- [Skin Tone Tri-Region](skin-tone-tri-region.md): các vùng da dựng sẵn để làm mask
- [Saturation Mask](saturation-mask.md): chọn vùng theo độ đậm màu
- [Mask Edge Smoother](mask-edge-smoother.md): làm mềm viền mask

## Workflow mẫu

[Workflow mẫu Color Matcher stats và histogram](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-49-color-matcher-histogram.json) khớp ảnh tông lạnh theo ảnh tham chiếu tông ấm bằng cả hai cách, đặt cạnh nhau (dùng Color Matcher; node Region có cùng nút `method`). Workflow chỉ dùng node lõi của ComfyUI và pack này. Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/`.

![Kết quả: Workflow mẫu Color Matcher stats và histogram](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/N-49-color-matcher-histogram-preview.png)
