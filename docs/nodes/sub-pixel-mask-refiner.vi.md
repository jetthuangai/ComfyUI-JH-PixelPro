---
title: Sub-Pixel Mask Refiner
description: Biến mask đen trắng cứng thành mask có viền mềm (feather), phần lõi bên trong vẫn đặc.
---

# Sub-Pixel Mask Refiner

Lấy một mask cắt tách cứng, răng cưa và cho nó một viền mềm mượt, trong khi phần giữa chủ thể vẫn đặc hoàn toàn và nền vẫn trong suốt hoàn toàn.

![Sub-Pixel Mask Refiner — trước / sau](../assets/nodes/sub-pixel-mask-refiner.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → mask → Sub-Pixel Mask Refiner` |
| **Phù hợp để** | Làm mềm mask từ SAM hoặc rembg, bỏ cảm giác "dán sticker", chuẩn bị mask để ghép ảnh |

## Khi nào dùng

**Mask nhị phân (binary mask)** chỉ có hai giá trị: 0 (không chọn) và 1 (chọn). Mỗi pixel ở viền hoặc nằm trong hoặc nằm ngoài, nên viền bị bậc thang và trông như dán lên. **Mask mềm (soft mask, còn gọi là alpha)** có các giá trị trung gian dọc viền, giúp chủ thể hòa vào nền mới.

Chỉ blur thì viền mềm đi, nhưng cũng ăn vào chủ thể và để nền lem vào. Node này khôn hơn: nó đánh dấu một vùng an toàn bên trong chủ thể và một vùng an toàn bên ngoài, rồi chỉ feather dải mỏng ở giữa. Dùng khi mask của bạn đã nằm gần đúng chỗ và chỉ cần viền đẹp hơn. Nếu viền phải bám chi tiết nhỏ như tóc, hãy dùng [Edge-Aware Mask Refiner](edge-aware-mask-refiner.md) hoặc [Alpha Matte](alpha-matte.md).

## Bắt đầu nhanh

1. `Load Image` → `Convert Image to Mask` (hoặc node nào xuất ra mask).
2. Nối mask vào **Sub-Pixel Mask Refiner**.
3. Giữ giá trị mặc định (erosion 2, dilation 4, feather 2.0). Tăng `feather_sigma` nếu muốn viền mềm hơn.
4. Kiểm tra bằng [Mask Preview](mask-preview.md).

## Thông số

| Thông số | Mặc định | Tác dụng | Gợi ý |
|---|---|---|---|
| `erosion_radius` | 2 | Co vào từ viền bao nhiêu pixel để tìm lõi "chắc chắn bên trong". Lõi này giữ trắng hoàn toàn. | 0 đến 64. Càng lớn, dải feather bên trong càng rộng. |
| `dilation_radius` | 4 | Đẩy ra từ viền bao nhiêu pixel để tìm vùng "chắc chắn bên ngoài". Vùng này giữ đen hoàn toàn. | Nên bằng hoặc lớn hơn `erosion_radius`. |
| `feather_sigma` | 2.0 | Độ mềm của viền, tính bằng pixel. | 0.1 đến 32. Chân dung thử 1 đến 4. |
| `threshold` | 0.5 | Giá trị lớn hơn mức này được coi là đã chọn, trước mọi bước khác. | Chỉ quan trọng khi mask đầu vào chưa phải đen trắng thuần. |

**Đầu ra:** `refined_mask` (MASK), cùng kích thước với đầu vào.

## Mẹo khi chỉnh ảnh

- Đưa vào mask nhị phân sạch. Nếu mask có đốm nhỏ hoặc lỗ, chạy [Mask Cleanup](mask-cleanup.md) trước.
- Đặt `dilation_radius` lớn hơn `erosion_radius` một chút để dải mềm nằm cân đối quanh viền gốc.
- Đặt cả hai bán kính bằng 0 thì node chỉ còn là blur Gaussian đơn thuần, thường không phải điều bạn muốn.
- Với tóc và lông, feather thôi là chưa đủ. Hãy tạo [Trimap](trimap-builder.md) rồi dùng [Alpha Matte](alpha-matte.md).

??? note "Bên trong node hoạt động thế nào"
    Mask được nhị phân hóa bằng `mask > threshold`. Blur Gaussian của mask nhị phân tạo ra phần feather. Bản mask đã co (erode) là lõi bên trong, còn `1 - bản đã nở (dilate)` là lõi bên ngoài. Kết quả là `inside + (1 - inside - outside) * feather`, cắt về 0..1, nên feather chỉ hiện ở dải giữa hai lõi.

    Erode và dilate dùng cửa sổ vuông (max trượt, làm bằng `max_pool2d` tách theo hàng và cột), không dùng cửa sổ tròn, nên bộ nhớ chỉ tỷ lệ với kích thước ảnh kể cả khi bán kính lớn. Kernel Gaussian được chọn đủ rộng khoảng ba sigma. Node ghi cảnh báo khi hơn 5% pixel đầu vào là xám giữa, vì nó mong đợi mask gần như nhị phân. Mã nguồn: `core/mask_refiner.py`.

## Node liên quan

- [Mask Cleanup](mask-cleanup.md) để loại đốm và lấp lỗ trước khi refine
- [Edge-Aware Mask Refiner](edge-aware-mask-refiner.md) cho viền bám theo ảnh
- [Mask Edge Smoother](mask-edge-smoother.md) để làm mượt viền răng cưa mà vẫn giữ góc
- [Mask Preview](mask-preview.md) để kiểm tra kết quả

## Workflow mẫu

[Workflow mẫu Sub-Pixel Mask Refiner](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-02-subpixel-mask-refiner.json). Hãy chép `workflows/sample_binary_mask.png` vào `ComfyUI/input/` trước.
