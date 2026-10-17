---
title: Pyramid Blend
description: Ghép hai ảnh qua mask sao cho không còn lộ đường nối, bằng phương pháp blend nhiều dải tần (Laplacian pyramid).
---

# Pyramid Blend

Ghép một ảnh lên ảnh khác qua mask và làm biến mất đường nối. Chi tiết nhỏ chuyển nhanh ở mép, còn màu sắc và độ sáng chuyển mượt trên vùng rộng, nên bạn không thấy viền cứng hay bậc sáng tối.

![Pyramid Blend — trước / sau](../assets/nodes/pyramid-blend.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → compositing → Compositing: Pyramid Blend` |
| **Dùng cho** | Dán lại khuôn mặt đã chỉnh màu vào ảnh, hoán đổi vật thể hoặc khuôn mặt, ghép hai ảnh khác độ sáng |

## Khi nào dùng

Ghép bằng mask thông thường sẽ lộ mép mỗi khi hai ảnh lệch nhau về độ sáng hay màu. Node này tách cả hai ảnh thành các dải chi tiết (Laplacian pyramid, phương pháp Burt-Adelson kinh điển) rồi blend từng dải riêng: dải chi tiết mịn chuyển trong vùng hẹp, dải thô chuyển trong vùng rộng. Đường nối hòa vào ảnh mà không bị bóng ma (ghosting).

Hãy chọn node này thay vì [Layer Add](layer-add.md) kèm mask khi hai ảnh không khớp hoàn toàn ở mép ghép.

## Bắt đầu nhanh

1. Đặt ảnh gốc vào `image_a` và ảnh đã chỉnh (hoặc ảnh khác) vào `image_b`. Hai ảnh phải cùng kích thước.
2. Nối một `MASK` vào `mask`: trắng lấy `image_b`, đen giữ `image_a`.
3. Để `levels` ở `5`, nếu vẫn thấy đường nối thì tăng lên.
4. `Preview Image`.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `image_a` | — | Ảnh nền, giữ lại nơi mask bằng 0. | |
| `image_b` | — | Ảnh phủ lên, lấy nơi mask bằng 1. | |
| `mask` | — | Nơi `image_b` thay thế `image_a`. | Mask cứng (nhị phân) vẫn ổn; pyramid tự làm mềm mép (feather). |
| `levels` | `5` (1 đến 10) | Độ sâu của pyramid. Nhiều level thì vùng chuyển rộng và mượt hơn. | `1` là ghép alpha thông thường. Thử 4 đến 6 trước. |

Đầu ra: `image`, ảnh đã blend (`IMAGE`).

Ba đầu vào phải cùng chiều cao và chiều rộng. Đầu vào có batch bằng 1 sẽ được dùng lại cho các đầu vào còn lại.

## Mẹo khi chỉnh ảnh

- Dùng ít level (1 đến 3) khi cần mép gọn, nhiều level (7 đến 10) để che chênh lệch lớn về độ sáng hay màu.
- Nếu thấy quầng màu quanh đường nối, hãy giảm `levels`.
- Làm mask trước bằng [Mask Cleanup](mask-cleanup.md) hoặc [Edge-Aware Mask Refiner](edge-aware-mask-refiner.md) để có hình cắt sạch.
- Chỉnh màu khuôn mặt bằng LUT trước ([LUT Preset](lut-preset.md)), rồi blend lại qua mask khuôn mặt.

??? note "Bên trong node hoạt động thế nào"
    Cả hai ảnh được dựng Laplacian pyramid (các lớp band-pass cộng một lớp low-pass cuối), mask được dựng Gaussian pyramid, dùng `pyrdown` / `pyrup` của Kornia. Mỗi dải được blend theo `band_a × (1 − mask) + band_b × mask`, sau đó pyramid được ghép ngược lại. Ảnh được đệm replicate đến bội số của `2 ** (levels - 1)` rồi cắt lại, nên kích thước nào cũng dùng được. Số level tự giảm để lớp nhỏ nhất còn ít nhất 8 px ở cạnh ngắn. Hai đầu vào giống hệt nhau, mask toàn 0 hoặc toàn 1 đều trả đúng ảnh nguồn. Nguồn: `core/pyramid_blend.py`.

## Node liên quan

- [Layer Add](layer-add.md)
- [Mask Cleanup](mask-cleanup.md)
- [Edge-Aware Mask Refiner](edge-aware-mask-refiner.md)
- [LUT Preset](lut-preset.md)

## Workflow mẫu

[Workflow mẫu Pyramid Blend](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-36-pyramid-blend.json). Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/`.

![Kết quả Pyramid Blend](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/N-36-pyramid-blend-preview.png)
