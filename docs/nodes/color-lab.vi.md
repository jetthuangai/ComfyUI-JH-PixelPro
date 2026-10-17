---
title: ColorLab
description: Bảng chỉnh kiểu Camera Raw gói trong một node, gồm Basic, HSL, Color Grading và Gray Mix.
---

# ColorLab

Một khâu hoàn thiện kiểu Camera Raw trong đúng một node: phơi sáng và tông, HSL theo từng màu, color grading ba vùng và bộ trộn đen trắng. Mọi thanh trượt đều bắt đầu ở mức trung tính.

![ColorLab — trước / sau](../assets/nodes/color-lab.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → color → Color: ColorLab (ACR)` |
| **Dùng để** | Grade nhanh toàn ảnh, tông teal-and-orange, split toning (nhuộm màu bóng/sáng), chuyển đen trắng, hoàn thiện cuối bằng một node |

## Khi nào dùng

Nếu bạn quen Lightroom hay Camera Raw thì đã biết cách dùng node này. Thay vì nối nhiều node nhỏ, bạn có đủ bốn bảng theo đúng thứ tự quen thuộc và chạy thành một chuỗi: **Basic → HSL → Color Grading → Gray Mix**. Nó hợp làm bước cuối sau khi retouch nặng xong, hoặc thay cho một lượt Camera Raw trên ảnh AI.

Đây là bản xấp xỉ viết riêng cho pack, không phải code của Adobe, nên con số sẽ không khớp từng chút với Camera Raw. Hãy đánh giá bằng mắt. Khi mọi thanh trượt ở 0 và Gray Mix tắt, ảnh đi qua không đổi.

Node có 55 thông số trên cùng một thân. Tên thông số cho biết bảng: `basic_*`, `hsl_*`, `grade_*`, `gray_*`.

## Bắt đầu nhanh

1. **Load Image** → ColorLab → **Preview Image**.
2. Bắt đầu ở Basic: chỉnh nhẹ `basic_exposure` và `basic_contrast`, rồi `basic_highlights` và `basic_shadows`.
3. Về màu, chuyển sang HSL hoặc Color Grading. Mỗi lần chỉ chỉnh một bảng và so với ảnh gốc.
4. Muốn đen trắng, bật `gray_enable` và tạo tông bằng các thanh `gray_*`.

## Thông số

Mọi thanh trượt mặc định là 0, trừ khi ghi khác. "±100" nghĩa là thanh chạy từ -100 đến 100.

### Basic

Tông toàn ảnh và độ "nổi".

| Thông số | Khoảng | Tác dụng |
|---|---|---|
| `basic_exposure` | -5 đến 5 (stop) | Độ sáng chung. Mỗi +1 là gấp đôi lượng sáng. |
| `exposure_mode` (tùy chọn) | `display` | `display` nhân giá trị ảnh với 2^EV (cách ColorLab hoạt động từ trước). `linear` đổi sang ánh sáng tuyến tính, nhân, rồi đổi ngược lại, giống máy ảnh hoặc Lightroom. |
| `basic_contrast` | ±100 | Đẩy các tông ra xa mức xám trung bình. |
| `basic_highlights` | ±100 | Làm sáng hoặc cứu lại nửa sáng của ảnh. |
| `basic_shadows` | ±100 | Mở hoặc làm sâu nửa tối. |
| `basic_whites` | ±100 | Chỉnh những tông sáng nhất. |
| `basic_blacks` | ±100 | Chỉnh những tông tối nhất. |
| `basic_texture` | ±100 | Độ nổi chi tiết nhỏ (bán kính nhỏ). |
| `basic_clarity` | ±100 | Local contrast ở midtone (bán kính lớn hơn), mạnh nhất ở vùng midtone. |
| `basic_dehaze` | ±100 | Thêm tương phản và hơi làm sâu tông để xua sương mù; giá trị âm tạo lớp sương nhẹ. |
| `basic_vibrance` | ±100 | Tăng các màu yếu nhiều hơn màu mạnh và bảo vệ màu da. |
| `basic_saturation` | ±100 | Tăng giảm mọi màu như nhau. |

### HSL

Tám dải màu: `red`, `orange`, `yellow`, `green`, `aqua`, `blue`, `purple`, `magenta`. Mỗi dải có ba thanh `hsl_<màu>_hue`, `hsl_<màu>_sat` và `hsl_<màu>_lum`, đều ±100 (tổng 24, ví dụ `hsl_orange_sat`).

| Thanh | Tác dụng |
|---|---|
| `hsl_<màu>_hue` | Xoay màu đó về phía các màu lân cận (tối đa khoảng 30 độ ở đầu thanh). |
| `hsl_<màu>_sat` | Làm màu đó đậm hơn hoặc nhạt hơn. |
| `hsl_<màu>_lum` | Làm màu đó sáng hơn hoặc tối hơn. |

Các dải chồng nhẹ lên nhau nên màu kế bên sẽ dịch chuyển một chút theo màu bạn chỉnh. Đây là bảng để làm teal-and-orange, lá cây xanh hơn hoặc da bớt đỏ.

### Color Grading

Ba vùng tông: `shadow`, `mid`, `highlight`. Mỗi vùng có bốn thanh `grade_<vùng>_hue`, `grade_<vùng>_sat`, `grade_<vùng>_lum` và `grade_<vùng>_bal` (tổng 12, ví dụ `grade_shadow_hue`).

| Thanh | Khoảng | Tác dụng |
|---|---|---|
| `grade_<vùng>_hue` | 0 đến 360 | Màu nhuộm, tính theo vị trí trên vòng màu. |
| `grade_<vùng>_sat` | 0 đến 100 | Độ đậm của màu nhuộm. Màu nhuộm chỉ hiện khi giá trị này lớn hơn 0. |
| `grade_<vùng>_lum` | ±100 | Làm vùng tông đó sáng hơn hoặc tối hơn. |
| `grade_<vùng>_bal` | ±100 | Coi pixel là sáng hơn (dương) hoặc tối hơn (âm) khi xếp vào vùng. Dương sẽ thu hẹp vùng shadow và nới rộng vùng highlight. |

Đặt `hue` trước, rồi tăng `sat` từ từ. Chút xanh lam ở vùng bóng và vàng ấm ở vùng sáng là split-tone kinh điển.

### Gray Mix

Chuyển đen trắng với điều khiển theo từng màu.

| Thông số | Mặc định | Khoảng | Tác dụng |
|---|---|---|---|
| `gray_enable` | tắt | bật / tắt | Chuyển ảnh sang xám. Khi tắt, không thanh nào trong bảng này có tác dụng. |
| `gray_red`, `gray_orange`, `gray_yellow`, `gray_green`, `gray_aqua`, `gray_blue`, `gray_purple` | 0 | -200 đến 300 | Màu đó sáng hay tối thế nào trong ảnh xám. Dương làm sáng, âm làm tối. |

Không có thanh `gray_magenta`: magenta đi theo các màu lân cận.

**Đầu ra:** `image`, ảnh đã chỉnh.

## Mẹo khi chỉnh ảnh

- Muốn chỉnh phơi sáng tự nhiên, đặt `exposure_mode` là `linear`. Ánh sáng thật cộng dồn trong không gian tuyến tính, nên +1 stop giống mở khẩu máy ảnh thêm một stop: xám trung bình 0.5 lên khoảng 0.69, vùng sáng chuyển mượt và màu giữ được cân bằng. `display` nhân thẳng giá trị đã lưu nên cùng +1 đẩy xám trung bình lên 1.0 và vùng sáng cháy rất nhanh. Workflow cũ vẫn dùng `display`.
- Làm từ trên xuống: tông ở Basic trước, màu ở HSL hoặc Grading sau, để chỉnh màu nằm trên nền ổn định.
- Chỉnh bước nhỏ. Nhiều thanh rất mạnh ở gần hai đầu; 10 đến 30 thường là đủ.
- Da chủ yếu nằm ở dải orange. Chỉnh `hsl_orange_*` nhẹ tay, đẩy mạnh hơn ở blue và green khi muốn tạo tông mà da không bị xanh xao.
- Đen trắng: hạ `gray_blue` để bầu trời sâu hơn, nâng `gray_orange` và `gray_red` để da mịn hơn.
- ColorLab tác động lên toàn khung. Muốn grade riêng một phần, trộn kết quả lại bằng mask (xem [Luminosity Masking](luminosity-masking.md)).
- Chỉ cần local contrast? [CLAHE Local Contrast](clahe-local-contrast.md) nâng vùng phẳng hoặc tối theo cách khác với `basic_clarity`.

??? note "Bên trong node hoạt động thế nào"
    Chuỗi xử lý nằm ở `core/color_lab.py`, dùng PyTorch thuần cùng box blur của Kornia. Ảnh vào được kẹp trong 0..1.

    - **Basic:** exposure nhân với `2^exposure` (`display`, trên giá trị sRGB) hoặc đổi sang RGB tuyến tính bằng Kornia, nhân với `2^exposure`, kẹp rồi đổi ngược lại (`linear`; xám 0.5 ở +1 EV thành khoảng 0.686); contrast xoay quanh 0.5; highlights, shadows, whites, blacks cộng độ lệch có trọng số smoothstep theo luma (hệ số Rec. 601) và có bảo vệ biên để không bị cháy; texture và clarity là kiểu unsharp dùng box blur 5 px và 9 px với biên được sao chép (clarity ưu tiên midtone); dehaze là thay đổi tương phản và offset; vibrance và saturation làm việc trong HSV, vibrance giảm tác động gần hue của da.
    - **HSL:** mỗi dải là một đường Gauss trên vòng hue (sigma 20 độ) quanh điểm neo cố định: red 0, orange 30, yellow 60, green 120, aqua 180, blue 240, purple 280, magenta 310.
    - **Color Grading:** mỗi vùng dùng một mask luma mượt (dịch bởi `bal`), màu nhuộm dựng từ `hue` và `sat`, còn `lum` là độ nâng.
    - **Gray Mix:** luma được điều biến theo trọng số của từng dải; magenta không phải một dải.

    Kết quả đi qua `nan_to_num` rồi kẹp 0..1. Nếu mọi thanh bằng 0 và `gray_enable` tắt, ảnh vào được trả về nguyên vẹn.

## Node liên quan

- [Tone Curve](tone-curve.md): tạo tương phản chính xác bằng đường cong
- [Hue/Saturation per Range](hue-saturation-range.md): một dải hue chỉnh được độ rộng
- [CLAHE Local Contrast](clahe-local-contrast.md): local contrast thích ứng chỉ trên độ sáng
- [Luminosity Masking](luminosity-masking.md): chỉ grade vùng bóng, midtone hoặc sáng
- [LUT Preset](lut-preset.md): áp một look dựng sẵn sau khi grade

## Workflow mẫu

Các workflow mẫu (chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/` trước):

- [Workflow mẫu ColorLab Basic](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-21-colorlab-basic-only.json)
- [Workflow mẫu ColorLab HSL teal-orange](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-22-colorlab-hsl-teal-orange.json)
- [Workflow mẫu ColorLab Color Grading](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-23-colorlab-color-grading-cinematic.json)
- [Workflow mẫu ColorLab Gray Mix](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-24-colorlab-gray-mix-bw.json)
- [Workflow mẫu ColorLab đầy đủ](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-25-colorlab-full-acr-preset.json)
