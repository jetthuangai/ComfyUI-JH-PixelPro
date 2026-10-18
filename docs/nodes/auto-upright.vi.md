---
title: Auto Upright
description: Nắn thẳng đường chân trời bị nghiêng và các đường thẳng đứng bị chụm, tự tìm từ các đường thẳng trong ảnh.
---

# Auto Upright

Nắn thẳng ảnh giống Upright của Lightroom: node tìm các đường thẳng trong ảnh, rồi cân bằng đường chân trời hoặc làm các đường đứng của tòa nhà song song. Nếu không chắc chắn, node để nguyên ảnh.

![Auto Upright — trước, crop none và crop valid](../assets/nodes/auto-upright.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → geometry → Auto: Upright` |
| **Dùng cho** | Đường chân trời nghiêng, tòa nhà chụp ngước lên, nội thất, ảnh kiến trúc và bất động sản |
| **Tốc độ** | Khoảng 0,1 đến 0,3 giây cho một ảnh, VRAM dưới 0,3 GB |

## Khi nào dùng

Ngước máy lên chụp tòa nhà thì các đường thẳng đứng nghiêng chụm vào nhau. Cầm máy hơi lệch thì đường chân trời bị xiên. Node đo cả hai từ các đường thẳng trong ảnh và nắn lại mà không cần bạn đoán thanh trượt.

Node chỉ chỉnh khi bằng chứng rõ ràng. Chân dung, cây cối, mây hoặc cảnh nhiễu không có đường đáng tin, nên ảnh được trả về nguyên vẹn và `confidence` bằng 0. Đoán sai còn tệ hơn là không chỉnh.

## Bắt đầu nhanh

1. Load Image → **Auto: Upright** → Preview Image.
2. Giữ `mode` là `auto`. Đặt `crop` là `valid` để có ảnh cuối gọn.
3. Nếu tòa nhà cao trông quá cứng, giảm `strength` xuống khoảng 0,7.
4. Theo dõi đầu ra `confidence` (nối vào một ô xem văn bản). Bằng 0 nghĩa là chưa đổi gì.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `image` | | Ảnh hoặc batch cần nắn thẳng. | |
| `mode` | `auto` | `auto` = làm các đường đứng song song nếu thấy tòa nhà, nếu không thì cân bằng đường chân trời. `level` = chỉ xoay. `vertical` = các đường đứng song song. `full` = đường đứng cộng các đường ngang chính. | `level` cho phong cảnh, cảnh biển. `full` cho mặt tiền chụp chính diện. |
| `strength` | 1.0 | 0 đến 1. Phần trăm mức chỉnh được áp. | Khoảng 0,7 giữ lại chút phối cảnh tự nhiên cho tòa nhà cao. |
| `crop` | `none` | `none` giữ nguyên mọi thứ, góc bị đen ở chỗ ảnh đã xoay. `valid` phóng to đến khi không còn góc trống. | `valid` cho ảnh xuất cuối. |
| `max_angle` (tùy chọn) | 25 | 0 đến 45 độ. Mức chỉnh lớn hơn sẽ dừng ở góc này. | Ảnh ngước rất dốc chỉ được chỉnh một phần thay vì bị kéo giãn quá đà. |

**Đầu ra**

| Đầu ra | Là gì |
|---|---|
| `image` | Ảnh đã nắn thẳng, cùng kích thước với ảnh vào. |
| `confidence` | 0 đến 1, lấy giá trị thấp nhất trong batch. Bằng 0 nghĩa là ảnh được giữ nguyên. |
| `matrix` | Danh sách văn bản (JSON) các phép biến đổi 3x3 theo pixel đã áp, mỗi ảnh một phép. |

## Mẹo khi chỉnh ảnh

- Chạy Auto Upright trước [Auto Tone](auto-tone.md) và các bước chỉnh màu, để phép đo thấy đúng khung hình cuối.
- Với `crop` là `none`, các góc sẽ bị đen. Dùng `valid`, hoặc tự crop, trước khi xuất ảnh.
- Méo góc rộng mạnh cũng làm đường thẳng bị cong. Nếu đường cong ở gần mép ảnh, hãy sửa trước bằng [Lens Distortion](lens-distortion.md).
- Dùng `level` khi cảnh chỉ có đường chân trời và không có tòa nhà. Nó sẽ không tự bịa ra phép đổi phối cảnh.
- `confidence` bằng 0 thường nghĩa là ảnh có quá ít đường thẳng dài, hoặc các đường đứng nghiêng quá 15 độ nên không được coi là tường nhà. Hãy nắn tay hoặc thử `level`.

??? note "Bên trong node hoạt động thế nào"
    Các đoạn thẳng được tìm bằng bộ dò đoạn thẳng (line segment detector) của OpenCV trên bản sao có cạnh dài tối đa 1024 px. Mã nguồn: `core/upright.py`, wrapper `nodes/upright_node.py`.

    **level**: góc nghiêng là trung vị có trọng số theo độ dài của các đường gần ngang và gần đứng, rồi ảnh được xoay quanh tâm.

    **vertical / full**: điểm tụ (vanishing point) được tìm bằng RANSAC rồi tinh chỉnh. Máy ảnh được xoay ảo (`H = K·R·K⁻¹`) để điểm tụ đứng ra vô cực, các đường đứng thành song song. `full` làm tương tự với điểm tụ ngang chính. Tiêu cự lấy từ hai điểm tụ khi chúng vuông góc, nếu không thì dùng mặc định tương đương ống kính khoảng 36 mm.

    **Cổng an toàn**: `level` cần ít nhất 3 đường, đủ tổng chiều dài và 60 % chiều dài đó đồng thuận. Một điểm tụ cần ít nhất 5 đường và các đường đồng thuận có tổng dài bằng nửa cạnh dài. Đường nghiêng quá 15 độ không được coi là tường. Nếu không đạt, ảnh được trả về nguyên với confidence 0.

    Phép warp chạy bằng Kornia trên thiết bị của ảnh. `crop` `valid` phóng đến hình chữ nhật lớn nhất cùng tỷ lệ gốc mà chỉ chứa pixel của ảnh.

## Node liên quan

- [Auto Tone](auto-tone.md): chỉnh exposure và cân bằng trắng sau khi nắn thẳng
- [Lens Distortion](lens-distortion.md): sửa méo thùng hoặc méo gối trước khi nắn thẳng
- [Auto Align](auto-align.md): căn chỉnh một loạt ảnh khớp với nhau

## Workflow mẫu

[Workflow mẫu Auto Upright](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-56-auto-upright.json): ảnh nhà thờ chụp ngước lên được nắn thẳng với `crop` là `valid`, rồi đi qua [Auto Tone](auto-tone.md). Workflow chỉ dùng node lõi của ComfyUI và pack này. Chép `workflows/sample_architecture.jpg` vào `ComfyUI/input/` (giấy phép Pexels).

![Kết quả: Workflow mẫu Auto Upright](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/N-56-auto-upright-preview.png)
