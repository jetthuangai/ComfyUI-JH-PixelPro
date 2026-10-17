---
title: Alpha Matte
description: Tạo mask trong suốt mềm cỡ tóc từ trimap và ảnh gốc bằng closed-form matting.
---

# Alpha Matte

Node cắt tách chất lượng cao. Đưa cho nó ảnh gốc và một bản đồ ba vùng (chắc chắn là chủ thể, chắc chắn là nền, chưa chắc ở viền), nó sẽ tính từng pixel chưa chắc trong suốt bao nhiêu, đúng thứ mà tóc, lông và viền kính cần.

![Alpha Matte — trước / sau](../assets/nodes/alpha-matte.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → mask → Mask: Alpha Matte` |
| **Phù hợp để** | Cắt tách tóc và lông, cạnh bán trong suốt, tách chủ thể sạch để ghép ảnh |

## Khi nào dùng

Có ba khái niệm cần nắm.

- **Mask nhị phân (binary mask):** mỗi pixel hoặc nằm hẳn trong, hoặc nằm hẳn ngoài. Tóc thành một hình khối cứng bị cắt cụt.
- **Alpha mềm (soft alpha, matte):** mỗi pixel có giá trị từ 0 đến 1 cho biết nó chứa bao nhiêu phần chủ thể. Một sợi tóc phủ nửa pixel được khoảng 0.5, nên hòa tự nhiên vào nền mới.
- **Trimap:** bản đồ gợi ý có ba giá trị: `0.0` = chắc chắn là nền, `0.5` = chưa biết, `1.0` = chắc chắn là chủ thể. Thuật toán chỉ phải giải dải chưa biết, nhờ vậy kết quả chính xác. Tạo trimap bằng [Trimap Builder](trimap-builder.md).

Node dùng **closed-form matting** (Levin, Lischinski và Weiss, 2008). "Closed-form" nghĩa là nó giải toàn bộ alpha cùng lúc bằng toán, thay vì đoán từng pixel. Giả định là trong một cửa sổ nhỏ, màu ảnh nằm xấp xỉ trên một đường thẳng giữa màu chủ thể và màu nền. Nhờ đó nó phân biệt được một sợi tóc mảnh với nền lộn xộn, và dùng chính ảnh làm hướng dẫn.

## Bắt đầu nhanh

1. Tạo mask thô của chủ thể (SAM, rembg hoặc tô tay).
2. Nối vào [Trimap Builder](trimap-builder.md). Nới rộng dải chưa biết cho phủ hết phần tóc.
3. Nối trimap vào `trimap` và ảnh gốc vào `guide`.
4. Giữ nguyên các thông số còn lại. Xem `alpha` bằng [Mask Preview](mask-preview.md) ở chế độ `cutout`.

## Thông số

| Thông số | Mặc định | Tác dụng | Gợi ý |
|---|---|---|---|
| `epsilon` | 1e-7 | Hệ số an toàn số học cho mô hình màu cục bộ. | Khoảng 1e-8 đến 0.01. Giữ mặc định, trừ khi kết quả bị nhiễu. |
| `window_radius` | 1 | Kích thước vùng lân cận dùng để mô hình hóa màu. | 1 đến 3. Tăng với ảnh mờ hoặc tương phản thấp. |
| `lambda_constraint` | 100.0 | Mức bắt kết quả phải tôn trọng vùng chắc chắn là chủ thể và chắc chắn là nền của trimap. | 1 đến 10000. Càng cao, vùng đã biết càng bị khóa chặt. |
| `compute_device` | auto | `auto` dùng GPU khi ảnh guide đã nằm trên GPU. `cuda` ép dùng GPU và báo lỗi nếu không có. `cpu` ép dùng CPU. | Để `auto`. |

**Đầu ra:** `alpha` (MASK), giá trị mềm từ 0 đến 1.

!!! warning "Định dạng trimap"
    Trimap chỉ được dùng các giá trị 0.0, 0.5 và 1.0 (cho phép sai số 0.05). Giá trị khác sẽ báo lỗi. Mask tô tay hoặc đã blur sẽ không qua được, hãy tạo bằng [Trimap Builder](trimap-builder.md).

## Mẹo khi chỉnh ảnh

- Dải chưa biết phải phủ hết chi tiết nhỏ. Sợi tóc mà rơi vào vùng "chắc chắn là chủ thể" thì sẽ đặc cứng. Nền mà rơi vào vùng đó thì sẽ bị giữ lại.
- Giữ dải chưa biết hẹp nhất mà chi tiết cho phép. Dải càng hẹp, kết quả càng sạch.
- Matting cần sự khác biệt màu giữa tóc và nền. Tóc đen trên nền tối là trường hợp khó với mọi phương pháp matting.
- Khi ghép, dùng khử nhiễm màu (colour decontamination) hoặc chỉnh màu viền để loại màu nền lem vào viền tóc.
- Với ảnh rất lớn, nên làm trên vùng đã crop. Node này giải một hệ phương trình lớn và nặng hơn các node mask khác.

??? note "Bên trong node hoạt động thế nào"
    Dựng ma trận Laplacian matting từ các cửa sổ hiệp phương sai RGB cục bộ (bán kính `window_radius`, hệ số chính quy `epsilon`) và giải `(L + lambda D) alpha = lambda b`, trong đó `D` đánh dấu pixel đã biết của trimap và `b` là alpha đã biết. Đường CPU dùng bộ giải thưa của SciPy, có dự phòng conjugate gradient. Đường GPU dựng ma trận thưa CUDA và chạy conjugate gradient có tiền điều kiện Jacobi trong PyTorch; nếu lỗi (ví dụ hết VRAM) thì cảnh báo và tự chuyển sang CPU. Mỗi ảnh trong batch được giải riêng. Guide batch 1 được nhân ra. Mã nguồn: `core/mask_alpha_matte.py`, kiểm tra trimap trong `core/mask_trimap.py`.

## Node liên quan

- [Trimap Builder](trimap-builder.md) để tạo trimap bắt buộc
- [Edge-Aware Mask Refiner](edge-aware-mask-refiner.md) cho cách sửa viền nhẹ hơn
- [Mask Preview](mask-preview.md) để xem matte dạng cutout
- [Mask Cleanup](mask-cleanup.md) để dọn mask thô trước khi tạo trimap

## Workflow mẫu

[Workflow mẫu Alpha Matte](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-29-alpha-matte-extractor.json). Mở workflow rồi trỏ node `Load Image` vào ảnh của bạn.
