---
title: Mask Cleanup
description: Xóa đốm lẻ, lấp lỗ nhỏ và tùy chọn feather viền cho mask phân vùng.
---

# Mask Cleanup

Điểm dừng đầu tiên cho mask từ SAM, rembg hay bất kỳ công cụ cắt tách nào. Nó xóa đốm lẻ, lấp lỗ nhỏ trong chủ thể, và có thể kết thúc bằng một lớp feather đều.

![Mask Cleanup — trước / sau](../assets/nodes/mask-cleanup.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → mask → Mask: Cleanup` |
| **Phù hợp để** | Xóa đốm trôi nổi, lấp lỗ kim trong chủ thể, tạo mask sạch có feather đều |

## Khi nào dùng

Phân vùng bằng AI thường để lại hai loại rác: các **đảo** vùng chọn nhỏ trôi nổi ngoài nền, và các **lỗ** nhỏ bên trong chủ thể. Cả hai đều xuất phát từ việc mask là tập hợp các nhóm pixel liền nhau. Node này tìm từng nhóm và quyết định giữ hay bỏ dựa trên **kích thước tính bằng pixel**, chứ không dựa vào bán kính cọ. Nhờ vậy nó dễ đoán hơn cách erode rồi dilate khi cần dọn dẹp.

Đầu tiên mask được chuyển thành đen trắng thuần tại `threshold`. Sau đó các đảo nhỏ hơn `min_island_area` bị xóa, và các lỗ nhỏ hơn `max_hole_area` được lấp. Lỗ nào chạm mép ảnh sẽ không bao giờ bị lấp, vì thực chất đó là nền.

## Bắt đầu nhanh

1. Nối mask từ node phân vùng của bạn.
2. Giữ mặc định (đảo 64, lỗ 64, không feather). Với ảnh lớn thì tăng các giá trị diện tích.
3. Thêm `feather_px` 3 đến 6 nếu muốn viền mềm.
4. So sánh trước và sau bằng [Mask Preview](mask-preview.md).

## Thông số

| Thông số | Mặc định | Tác dụng | Gợi ý |
|---|---|---|---|
| `threshold` | 0.5 | Pixel lớn hơn giá trị này được tính là chủ thể. | Giảm xuống để lấy cả vùng mờ nhạt. |
| `min_island_area` | 64 | Xóa các mảng chủ thể nhỏ hơn số pixel này. 0 = tắt. | Tăng theo kích thước ảnh. Ảnh cỡ 2K thử 200 đến 2000. |
| `max_hole_area` | 64 | Lấp các lỗ kín nhỏ hơn số pixel này. 0 = tắt. | Không bao giờ lấp vùng nền chạm mép ảnh. |
| `feather_px` | 0.0 | Khoảng cách viền mềm, tính từ viền đã dọn. 0 = đen trắng cứng. | 0 đến 256. Toàn bộ dải chuyển mềm rộng gấp đôi giá trị này. |
| `connectivity` | 8 | `8` coi pixel chéo là liền nhau; `4` chỉ tính các pixel kề cạnh. | Hầu hết mask để 8. |

**Đầu ra:** `mask` (MASK).

## Mẹo khi chỉnh ảnh

- Dùng nó đầu tiên trong chuỗi mask, rồi đến [Edge-Aware Mask Refiner](edge-aware-mask-refiner.md) hoặc [Sub-Pixel Mask Refiner](sub-pixel-mask-refiner.md).
- Feather của nó đều một khoảng cách từ viền, nên đường nửa-trong-suốt vẫn nằm đúng trên viền gốc. Blur thì sẽ làm mỏng các phần nhỏ của chủ thể.
- Nếu một phần thật của chủ thể biến mất (khuyên tai nhỏ, cúc áo), `min_island_area` đang quá cao.
- Nếu một lỗ thật biến mất (khe giữa cánh tay và thân), hãy giảm `max_hole_area`.

??? note "Bên trong node hoạt động thế nào"
    Mask được nhị phân hóa bằng `mask > threshold`. Các thành phần liền nhau được gán nhãn bằng `cv2.connectedComponentsWithStats`. Lỗ được tìm trên mask đảo với kiểu liên thông bù (8 cho chủ thể đi cùng 4 cho lỗ), đây là cặp chuẩn giúp lỗ không bị "rò" theo đường chéo. Feather là `alpha = clip(0.5 + sd / (2 * feather_px), 0, 1)`, trong đó `sd` là khoảng cách có dấu tới biên từ `cv2.distanceTransform` (dương ở bên trong). Chạy trên CPU, từng ảnh một trong batch. Mã nguồn: `core/mask_cleanup.py`.

## Node liên quan

- [Mask Morphology](mask-morphology.md) để đổi kích thước mask theo bán kính
- [Sub-Pixel Mask Refiner](sub-pixel-mask-refiner.md) cho một cách feather khác
- [Edge-Aware Mask Refiner](edge-aware-mask-refiner.md) để kéo viền về sát ảnh
- [Mask Preview](mask-preview.md) để so sánh trước và sau

## Workflow mẫu

[Workflow mẫu Mask Cleanup](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-41-mask-cleanup.json). Hãy chép `workflows/sample_portrait.jpg` và `workflows/sample_binary_mask.png` vào `ComfyUI/input/` trước.
