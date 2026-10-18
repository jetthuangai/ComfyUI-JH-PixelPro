---
title: Auto Tone
description: Chỉnh exposure, tông và cân bằng trắng chỉ với một cú nhấp, và trả về các thông số dưới dạng ColorLab để bạn tinh chỉnh tiếp.
---

# Auto Tone

Đo ảnh rồi áp một bản chỉnh sơ bộ: exposure, whites, blacks, highlights, shadows và cân bằng trắng. Node cũng trả lại đúng các thông số đã dùng, để bạn tiếp tục chỉnh trong ColorLab.

![Auto Tone — trước, Auto và Auto B&W](../assets/nodes/auto-tone.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → color → Auto: Tone & White Balance` |
| **Dùng cho** | Bản chỉnh khởi đầu nhanh cho cả loạt ảnh, cứu ảnh tối hoặc bị ám màu, chuyển đen trắng tự động nhanh |
| **Tốc độ** | Khoảng 1,5 giây cho một ảnh 4K, VRAM khoảng 1 GB |

## Khi nào dùng

Dùng như nút Auto trong Lightroom: bạn có một nền hợp lý rồi chỉnh tay tiếp. Kết quả không phải bản cuối. Đầu ra `params_json` chứa mọi thông số đã áp, dưới dạng khóa của ColorLab, nên bạn thấy chính xác node đã đổi gì và tinh chỉnh được.

Node cố ý thận trọng:

- Ảnh tối được làm sáng, nhưng ảnh high-key sáng hoặc nền trắng vẫn giữ sáng.
- Cân bằng trắng chỉ nhìn vào các pixel gần xám. Da, cây lá và hoàng hôn không bao giờ kéo lệch nó. Ảnh không có vùng xám nào thì được giữ nguyên.

!!! note
    Các hằng số được chỉnh tay trên ảnh thử. Chúng không được khớp với Auto của Adobe, và node không tuyên bố giống Lightroom.

## Bắt đầu nhanh

1. Load Image → **Auto: Tone & White Balance** → Preview Image.
2. Giữ mặc định. Giảm `strength` (thử 0,6) nếu thấy chỉnh quá tay.
3. Để tinh chỉnh, nối `params_json` vào đầu vào `params_json` của [ColorLab](color-lab.md). Các thanh trượt của ColorLab được cộng thêm lên trên thông số tự động.
4. Để dùng lại cùng thông số cho ảnh khác, nối `params_json` vào [Apply ColorLab Preset](apply-colorlab-preset.md).

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `image` | | Ảnh hoặc batch cần chỉnh. Mỗi ảnh trong batch được đo và chỉnh riêng. | |
| `strength` | 1.0 | 0 đến 1. Mức áp bản chỉnh tự động. 0 trả về ảnh gốc. | 0,5 đến 0,8 cho tay nhẹ hơn. |
| `exposure` (tùy chọn) | bật | Chỉnh độ sáng chung. | Tắt nếu độ sáng đã ổn. |
| `tone` (tùy chọn) | bật | Đặt Whites, Blacks, Highlights và Shadows. | |
| `white_balance` (tùy chọn) | bật | Khử ám màu dựa trên các vùng gần xám. | Tắt với cảnh dùng ánh sáng màu có chủ ý. |
| `black_and_white` (tùy chọn) | tắt | Chuyển đen trắng với bộ trộn xám tự động. | Dải màu sáng hơn thành xám sáng hơn, giữ tương phản màu thành tương phản tông. |
| `max_wb_shift` (tùy chọn) | 50 | 0 đến 100. Mức đổi Temp/Tint lớn nhất mà cân bằng trắng được phép. | Giảm xuống với hoàng hôn, ánh nến và ánh sáng ấm hoặc lạnh có chủ ý. |

**Đầu ra**

| Đầu ra | Là gì |
|---|---|
| `image` | Ảnh đã chỉnh, cùng kích thước với ảnh vào. |
| `params_json` | Các thông số ColorLab đã áp, gồm `tone_engine` = local và `exposure_mode` = linear. Chỉ của ảnh đầu tiên trong batch. |

## Mẹo khi chỉnh ảnh

- Chạy [Auto Upright](auto-upright.md) trước, rồi mới đến node này, để phép đo thấy đúng khung hình cuối.
- Auto Tone hợp làm bản grade khởi đầu cho một loạt ảnh giống nhau. Muốn đồng nhất look, hãy chỉnh một ảnh rồi dùng lại thông số bằng [Apply ColorLab Preset](apply-colorlab-preset.md).
- Ám màu ấm hoặc lạnh là một phần của không khí ảnh (golden hour, đèn tungsten) thì nên bảo vệ bằng `max_wb_shift` thấp hoặc tắt `white_balance`.
- Cảnh không có vật trung tính thì cân bằng trắng đổi rất ít hoặc không đổi. Đây là chủ ý; hãy tự đặt Temp và Tint trong [ColorLab](color-lab.md).
- Nếu vùng shadows vẫn nặng, thêm [Shadows/Highlights Pro](shadows-highlights-pro.md) hoặc [CLAHE Local Contrast](clahe-local-contrast.md).

??? note "Bên trong node hoạt động thế nào"
    Mọi phép đo chạy trên bản sao có cạnh dài tối đa 512 px, nên kết quả không phụ thuộc kích thước ảnh. Mã nguồn: `core/auto_tone.py`, wrapper `nodes/auto_tone_node.py`.

    **Exposure**: tính từ độ sáng trung bình theo log, hướng về mức 12 %, thấp hơn xám trung tính một chút để ảnh low-key, ảnh có không khí giữ được tâm trạng. Ảnh tối được làm sáng tối đa +1 EV, bị giới hạn để 0,5 % điểm sáng nhất không cháy quá nửa stop. Ảnh sáng chỉ bị làm tối khi thật sự thừa sáng (tối đa -1 EV). Khi từ 40 % khung trở lên là màu gần trắng (sản phẩm nền trắng, studio high-key), node chỉ xét phần còn lại của ảnh (chủ thể) và chỉ làm tối nếu chính chủ thể quá sáng; khung trắng hoàn toàn được giữ nguyên. Các chỉnh quá nhỏ bị bỏ qua.

    **Cân bằng trắng**: màu trung bình của các pixel gần xám trong ánh sáng tuyến tính, tìm qua vài vòng với ngưỡng dung sai nới dần. Mức chỉnh được nhân theo tỷ lệ pixel gần xám, giảm nhẹ, và bị giới hạn bởi `max_wb_shift`.

    **Tông**: Whites và Blacks đẩy 0,5 % pixel sáng nhất và tối nhất về mục tiêu cố định. Highlights và Shadows theo phần ảnh gần bị cháy hoặc đen.

    **Đen trắng**: mỗi dải màu được làm xám sáng hơn hoặc tối hơn mức trung bình của ảnh.

    Các thông số được áp bằng pipeline ColorLab, `strength` nhân chúng lên hoặc xuống, giống Apply ColorLab Preset. ComfyUI đưa ảnh vào trên CPU.

## Node liên quan

- [ColorLab](color-lab.md): tinh chỉnh thông số tự động bằng thanh trượt
- [Apply ColorLab Preset](apply-colorlab-preset.md): dùng lại thông số cho ảnh khác, có strength và mask
- [Auto Upright](auto-upright.md): nắn thẳng ảnh trước khi chỉnh
- [Shadows/Highlights Pro](shadows-highlights-pro.md): kiểm soát vùng tối và vùng sáng kỹ hơn

## Workflow mẫu

[Workflow mẫu Auto Tone](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-55-auto-tone.json): ảnh quán café tối đi qua Auto Tone, rồi `params_json` điều khiển một ColorLab với Vibrance +15 và được hiện trong một ô xem văn bản. Workflow chỉ dùng node lõi của ComfyUI và pack này. Chép `workflows/sample_cafe.jpg` vào `ComfyUI/input/` (giấy phép Pexels).

![Kết quả: Workflow mẫu Auto Tone](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/N-55-auto-tone-preview.png)
