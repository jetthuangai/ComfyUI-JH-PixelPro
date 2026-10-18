---
title: Defringe
description: Xóa viền tím và viền xanh lá dọc các cạnh tương phản cao, đồng thời căn lại sắc sai ngang (lateral chromatic aberration).
---

# Defringe

Xóa các viền tím hoặc xanh lá xuất hiện dọc cành cây, cạnh ngược sáng và bề mặt chrome, cùng viền đỏ và cyan về phía góc ảnh. Chỉ những pixel thuộc viền bị thay đổi. Phần còn lại của ảnh giữ nguyên.

![Defringe — Original (gốc), Purple fringe (mô phỏng), Defringe](../assets/nodes/defringe.jpg)

!!! note
    Viền tím trong ảnh minh họa này là **mô phỏng**, được thêm vào một ảnh mẫu để bạn thấy rõ hiệu quả. Viền thật trông tương tự nhưng thay đổi tùy ống kính.

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → color → Color: Defringe` |
| **Dùng cho** | Cành lá ngược sáng trên nền trời, cửa sổ sáng, chrome và điểm sáng trên kim loại, ảnh mở khẩu lớn |
| **Tốc độ** | 4K dùng dưới 1 GB VRAM |

## Khi nào dùng

Ống kính hội tụ các màu hơi khác nhau. Điều đó để lại hai loại viền:

- **Viền tím (hoặc xanh lá).** Quầng màu dọc cạnh sáng-tối mạnh. Thường gặp khi mở khẩu lớn và ngược sáng.
- **Sắc sai ngang (lateral chromatic aberration).** Kênh đỏ và xanh dương bị phóng đại hơi khác kênh xanh lá, nên cạnh có viền đỏ và cyan tăng dần về phía góc.

Cả hai được xử lý trong một bước. Sắc sai ngang được căn lại trước, sau đó các pixel màu còn sót dọc cạnh được kéo về màu của vùng bên cạnh.

## Bắt đầu nhanh

1. Load Image → **Color: Defringe** → Preview Image.
2. Giữ mặc định: `purple` 1, `green` 0, `lateral_ca` bật.
3. Zoom 100 % vào một cạnh tương phản cao và so với ảnh gốc.
4. Nếu viền xanh lá còn lại trên kim loại hoặc kính, tăng `green` lên 0.5.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `image` | | Ảnh có viền màu ở cạnh. | |
| `purple` | 1.0 | 0 đến 1. Mức xóa viền tím và magenta cạnh các đường biên mạnh. | Giảm nếu vật thể tím thật bị mất màu ở rìa. |
| `green` | 0.0 | 0 đến 1. Mức xóa viền xanh lá. Mặc định tắt. | Lá cây nhìn trên nền trời sáng có thể bị xám, nên chỉ tăng khi viền xanh lá thật sự là vấn đề. |
| `lateral_ca` | bật | Căn kênh đỏ và xanh dương theo kênh xanh lá. Độ lệch được đo tự động trên bản sao 1024 px và chỉ áp dụng khi rõ ràng làm cạnh khớp hơn. | Tắt nếu ống kính đã được hiệu chỉnh sẵn. |
| `radius` (tùy chọn) | 0 | 0 đến 32 px. Bề rộng viền. 0 là tự động: 0,15 % cạnh dài. | Tăng cho viền rộng trên file rất lớn. |

**Đầu ra**

| Đầu ra | Là gì |
|---|---|
| `image` | Ảnh đã xóa viền, cùng kích thước ảnh vào. |

## Mẹo khi chỉnh ảnh

- Defringe trước khi sharpen, để [Smart Sharpen](smart-sharpen.md) không làm viền đậm thêm.
- Nếu phần mềm xử lý raw đã áp dụng lens correction, hãy tắt `lateral_ca` và chỉ giữ phần xóa viền.
- Cành cây trên nền trời thường chỉ cần `purple`. Thêm `green` khi có cả cặp tím-xanh, và nhìn kỹ lá cây trước và sau.
- Cứ để `radius` ở 0 trước. Chỉ đổi khi sau lượt tự động bạn vẫn còn thấy viền.
- Méo hình học của ống kính là vấn đề khác. Hãy dùng [Lens Distortion](lens-distortion.md) cho việc đó.

??? note "Bên trong node hoạt động thế nào"
    1. **Sắc sai ngang.** Kênh đỏ và xanh dương được co giãn quanh tâm ảnh theo hệ số làm cạnh của chúng khớp nhất với kênh xanh lá. Hệ số này tìm bằng tương quan của các kênh high-pass trên bản sao 1024 px, trong khoảng cộng trừ 0,4 %. Nó chỉ được áp dụng ở kích thước đầy đủ khi tương quan cạnh cải thiện rõ rệt.
    2. **Xóa viền.** Pixel viền có hue tím (270 đến 335 độ) hoặc xanh lá (75 đến 165 độ), màu rõ ràng, và nằm trong phạm vi `radius` của một cạnh độ chói mạnh. Màu Lab a* và b* của nó được kéo về màu của các pixel không phải viền ở gần. Độ sáng được giữ nguyên nên cạnh vẫn sắc. Chỉ pixel viền bị thay đổi.

    Ảnh được xử lý theo từng dải, nên 4K dùng dưới 1 GB VRAM. Mã nguồn: `core/defringe.py`, wrapper `nodes/defringe_node.py`.

## Node liên quan

- [Lens Distortion](lens-distortion.md): sửa méo thùng và méo gối
- [Smart Sharpen](smart-sharpen.md): sharpen sau khi đã hết viền
- [ColorLab](color-lab.md): chỉnh màu tiếp sau khi defringe
- [Decontaminate Colors](decontaminate-colors.md): xóa màu lem ở rìa ảnh tách nền

## Workflow mẫu

[Workflow mẫu Defringe](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-60-defringe.json) cho một ảnh kiến trúc mẫu qua Defringe. Ảnh mẫu có sắc sai ngang màu đỏ nhẹ. Chép `workflows/sample_architecture.jpg` vào `ComfyUI/input/` (Pexels License).
