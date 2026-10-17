---
title: Layer Stack Start
description: Bắt đầu một chồng layer kiểu Photoshop từ ảnh nền.
---

# Layer Stack Start

Biến một ảnh bình thường thành layer dưới cùng của chồng layer (layer stack), giống layer Background trong Photoshop. Mọi node compositing khác đều xếp lên trên layer này.

![Layer Stack Start — trước / sau](../assets/nodes/layer-stack-start.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → compositing → Compositing: Layer Stack Start` |
| **Dùng cho** | Mở đầu mọi bước ghép hoàn thiện: layer chỉnh màu, hiệu ứng ánh sáng, texture overlay, group lồng nhau |

## Khi nào dùng

ComfyUI thường chuyền ảnh đã xử lý xong từ node này sang node khác. **Layer stack** thì khác: nó giữ từng layer riêng biệt (ảnh, blend mode, opacity, mask) cho đến khi bạn quyết định gộp lại. Nhờ vậy bạn có thể dựng lại cùng một chồng layer bao nhiêu lần tùy ý, và sửa một layer mà không phải làm lại các layer còn lại.

Chồng layer đi qua một kết nối riêng tên `LAYER_STACK`. **Layer Stack Start** tạo ra nó từ ảnh nền của bạn. Sau đó bạn thêm layer bằng [Layer Add](layer-add.md), lồng chồng layer bằng [Layer Group](layer-group.md) nếu cần, rồi đổi kết quả về ảnh thường bằng [Layer Flatten](layer-flatten.md).

## Bắt đầu nhanh

1. `Load Image` → **Layer Stack Start** (`background`).
2. Nối đầu ra `LAYER_STACK` vào [Layer Add](layer-add.md).
3. Kết thúc chuỗi bằng [Layer Flatten](layer-flatten.md) → `Preview Image`.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `background` | — | Ảnh trở thành layer dưới cùng. | Kích thước của nó là kích thước canvas cho cả chồng layer. |

| Đầu ra | Kiểu | Ý nghĩa |
|---|---|---|
| `LAYER_STACK` | `LAYER_STACK` | Chồng layer chỉ có layer nền. |

## Mẹo khi chỉnh ảnh

- Dùng ảnh đã retouch xong ở kích thước đầy đủ làm nền, mọi layer bạn thêm sẽ được co giãn cho khớp.
- Dùng thêm một **Layer Stack Start** thứ hai khi muốn dựng một group riêng (ví dụ một bộ layer chỉnh màu), rồi lồng vào bằng [Layer Group](layer-group.md).
- `LAYER_STACK` không nối thẳng vào `Preview Image` hay `Save Image` được. Luôn kết thúc bằng [Layer Flatten](layer-flatten.md).

??? note "Bên trong node hoạt động thế nào"
    Chồng layer là một danh sách các layer. Layer nền là phần tử đầu tiên với blend mode `normal`, opacity 1, fill 1, không mask và không clipping. Xem `nodes/layer_compositing.py` (phần wrapper) và `compose_stack` trong `core/blend_modes.py` (cách chồng layer được render).

## Node liên quan

- [Layer Add](layer-add.md)
- [Layer Group](layer-group.md)
- [Layer Flatten](layer-flatten.md)

## Workflow mẫu

[Layer compositing: 2-layer overlay](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-26-layer-compositing-2layer-overlay.json). Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/`. Workflow này còn nạp một ảnh thứ hai làm overlay; hãy chọn ảnh của bạn trong node Load Image đó.
