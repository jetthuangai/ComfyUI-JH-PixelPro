---
title: Palette Extract
description: Tìm các màu chính của ảnh, cho bạn dải swatch, danh sách mã màu và bản ảnh bệt màu.
---

# Palette Extract

Lấy ra các màu chủ đạo của ảnh, giống như bạn dùng ống hút màu (eyedropper) nhặt swatch. Node trả về một dải swatch, mã màu dạng văn bản và một bản ảnh posterize.

![Palette Extract — trước / sau](../assets/nodes/palette-extract.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → color → Color: Palette Extract` |
| **Dùng cho** | Lấy bảng màu từ ảnh tham chiếu, kiểm tra màu sau khi grade, tạo kiểu ảnh bệt màu (posterize), đưa mã hex cho công cụ khác |

## Khi nào dùng

Dùng khi bạn muốn biết "bức ảnh này thực sự gồm những màu nào?". Node gom các pixel giống nhau bằng không gian màu CIE LAB, một không gian được xây theo cách mắt người thấy sự khác biệt, nên mỗi nhóm khớp với thứ bạn gọi là "một màu". Cùng tông nhưng một bản tối, một bản sáng thì vẫn được tách riêng nếu nhìn khác nhau.

Các màu được xếp theo tỉ lệ (share), màu chiếm nhiều nhất đứng đầu. Share cho biết mỗi màu chiếm bao nhiêu phần ảnh (hoặc vùng mask).

## Bắt đầu nhanh

1. Load Image → **Color: Palette Extract** → Preview Image ở cổng `palette`.
2. Đặt `colors` là 5 hoặc 6 để bắt đầu.
3. Tùy chọn: nối mask chủ thể vào `mask` để phông nền không chiếm hết bảng màu.
4. Đưa `palette_json` vào node hiển thị hoặc lưu văn bản để giữ lại mã hex.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `colors` | 6 | Số màu cần tìm (2 đến 16). | 4 đến 8 màu hợp với chân dung. Nhiều màu hơn thì ra nhiều sắc thái hơn. |
| `mask` (tùy chọn) | không | Chỉ pixel nằm trong mask (giá trị trên 0.5) được tính vào bảng màu và tỉ lệ. | Dùng mask chủ thể hoặc mask da. Nếu một ảnh trong batch có mask rỗng, ảnh đó dùng toàn khung hình (có ghi cảnh báo) thay vì làm hỏng cả batch. |

**Đầu ra**

| Đầu ra | Là gì |
|---|---|
| `palette` | Dải swatch 1024 × 192, mỗi màu một dải rộng bằng nhau, màu chiếm nhiều nhất nằm bên trái. |
| `palette_json` | Danh sách văn bản như `[{"hex": "#c8a58b", "rgb": [200, 165, 139], "share": 0.31}, …]`, màu nhiều nhất đứng đầu. Với batch một ảnh đây là danh sách phẳng; với batch nhiều ảnh là danh sách các danh sách, mỗi ảnh một danh sách. |
| `quantized` | Ảnh của bạn với mỗi pixel được thay bằng màu gần nhất trong bảng (kiểu posterize, bệt màu). Luôn phủ toàn khung hình, kể cả khi dùng mask. |

## Mẹo khi chỉnh ảnh

- Đặt `colors` là 2 hoặc 3 để thấy sự chia màu chính của một bản grade, ví dụ da ấm so với bóng (shadow) lạnh.
- Nên mask chủ thể trước, nếu không bầu trời hay phông lớn sẽ chiếm hết các màu.
- Cùng một ảnh và cùng thiết lập thì kết quả luôn giống nhau.
- So sánh hai bản chỉnh bằng cách lấy bảng màu của cả hai rồi đối chiếu mã hex.
- Nối `quantized` vào [Color Matcher](color-matcher.md) hoặc đặt cạnh ảnh gốc để nhìn nhanh cấu trúc tông.

??? note "Bên trong node hoạt động thế nào"
    Ảnh được đổi sang CIE LAB (Kornia). K-means++ và 25 vòng Lloyd chạy trên mẫu ngẫu nhiên tối đa 200.000 pixel (seed cố định 0 nên kết quả lặp lại được). Sau đó mọi pixel (hoặc mọi pixel trong mask khi tính share) được gán vào tâm gần nhất theo từng khối khoảng một triệu pixel, nên bộ nhớ dùng rất ít. Các tâm được đổi về sRGB và làm tròn 8 bit; chính các giá trị đã làm tròn này dùng cho dải swatch, JSON và `quantized`. Nếu ảnh có ít màu khác nhau hơn `colors`, các mục dư có share bằng 0. Mã nguồn: `core/palette.py`.

## Node liên quan

- [Color Matcher](color-matcher.md) để kéo màu của ảnh này về gần ảnh khác
- [Skin Tone Tri-Region](skin-tone-tri-region.md) để chỉnh tông da theo vùng
- [Mask Combine](mask-combine.md) để dựng mask chủ thể

## Workflow mẫu

[Workflow mẫu Palette Extract](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-47-palette-extract.json) lấy 6 màu từ ảnh chân dung và hiển thị dải màu, danh sách HEX và ảnh lượng tử hóa. Workflow chỉ dùng node lõi của ComfyUI và pack này. Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/`.

![Kết quả: Workflow mẫu Palette Extract](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/N-47-palette-extract-preview.png)
