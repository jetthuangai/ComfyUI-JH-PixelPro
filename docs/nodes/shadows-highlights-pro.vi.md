---
title: Shadows/Highlights Pro
description: Mở vùng tối và kéo lại vùng sáng bằng hai thanh Shadows và Highlights quen thuộc của Camera Raw hoặc Photoshop.
---

# Shadows/Highlights Pro

Nâng vùng tối (shadows) và cứu vùng sáng (highlights) của ảnh mà không bị bệt, xám kiểu "HDR rẻ tiền". Bạn chọn giữa mô hình kiểu Camera Raw không để lại quầng sáng (halo) và mô hình hộp thoại Shadows/Highlights cổ điển của Photoshop.

![Shadows/Highlights Pro — trước / sau](../assets/nodes/shadows-highlights-pro.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → color → Color: Shadows/Highlights Pro` |
| **Dùng cho** | Chân dung ngược sáng, mở mặt và tóc bị tối, ghìm trời và cửa sổ quá sáng, giảm bóng dầu trên da |
| **Tốc độ** | Khoảng 0,1 giây cho một ảnh 4K trên GPU CUDA (sau lần chạy đầu), VRAM đỉnh dưới 1 GB. Trên CPU khoảng 14 giây ở 4K. |

## Khi nào dùng

Dùng khi ánh sáng của ảnh đã ổn nhưng dải tông quá rộng: mặt nằm trong bóng râm trước phông sáng, áo tối mất chi tiết, hoặc bầu trời gần như trắng. Curve hay exposure thông thường đẩy mọi pixel như nhau; node này nhìn vào vùng xung quanh của từng chỗ, nên nó nâng những gì tối *so với lân cận* và dịu những gì sáng, mà vẫn giữ tương phản cục bộ.

Có hai mô hình:

- **`camera_raw`** hoạt động như hai thanh Shadows và Highlights trong Lightroom / Camera Raw. Cả hai thanh đều kéo được hai chiều, cạnh cứng không bị halo (viền sáng hoặc tối quanh vật thể), và màu được giữ vì node chỉ đổi độ sáng. Trên bảng thử màu xám, kết quả bám sát số đo đã công bố của Camera Raw trong Photoshop 25.4 (Shadows +100: 165,2 so với 165,3 trên 255; Highlights -100: 90,4 so với 89,6). Hãy xem đây là một mô hình gần đúng, không phải bản sao chính xác của Lightroom hay Photoshop trên mọi ảnh.
- **`photoshop`** theo hộp thoại *Image > Adjustments > Shadows/Highlights*, có các thông số Tone, Radius, Color và Midtone. Giống hộp thoại gốc, nó chỉ nâng được shadows và hạ được highlights.

## Bắt đầu nhanh

1. Load Image → **Color: Shadows/Highlights Pro** → Preview Image.
2. Để `mode` là `camera_raw` và bắt đầu từ mặc định (`shadows` 30, `highlights` -30).
3. Tăng `shadows` đến khi vùng tối hiện chi tiết; giảm `highlights` đến khi trời hoặc bóng dầu trên da trông vừa mắt.
4. Tùy chọn: nối một mask vào `mask` để chỉ chỉnh chủ thể hoặc bầu trời.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `shadows` | 30 | -100 đến 100. Dương thì nâng vùng tối so với xung quanh; âm thì làm vùng tối sâu hơn (chỉ ở `camera_raw`). | 20 đến 50 cho chân dung; đến 100 khi ngược sáng mạnh. |
| `highlights` | -30 | -100 đến 100. Âm thì cứu vùng sáng (trời, cửa sổ, bóng dầu trên da); dương thì làm chúng sáng thêm (chỉ ở `camera_raw`). | -20 đến -60 cho phần lớn ảnh. |
| `mode` | `camera_raw` | `camera_raw` = thanh trượt kiểu Lightroom / Camera Raw, không halo. `photoshop` = mô hình Image > Adjustments > Shadows/Highlights. | Bắt đầu với `camera_raw`; chọn `photoshop` khi bạn muốn các thông số của hộp thoại. |
| `mask` (tùy chọn) | không | Giới hạn vùng chịu tác dụng (1 = tác dụng đầy đủ, 0 = giữ nguyên, giá trị mềm thì hòa trộn). Phải cùng kích thước với ảnh; mask một ảnh sẽ áp cho cả batch. | Dùng mask chủ thể, mask trời hoặc mask từ [Luminosity Masking](luminosity-masking.md). |
| `ps_shadow_tone` | 50 | Chế độ Photoshop. Độ rộng dải tông của shadows (0–100 %): phần nâng tối lan lên tới tông nào. | Giảm xuống nếu chỉ muốn chạm vào vùng tối sâu nhất. |
| `ps_shadow_radius` | 30 | Chế độ Photoshop. Kích thước vùng lân cận của shadows, tính bằng pixel (0–2500). | Ảnh lớn thì dùng radius lớn; giá trị nhỏ cho cảm giác cục bộ, gắt hơn. |
| `ps_highlight_tone` | 50 | Chế độ Photoshop. Độ rộng dải tông của highlights (0–100 %). | Giảm xuống nếu chỉ muốn chạm vào vùng sáng nhất. |
| `ps_highlight_radius` | 30 | Chế độ Photoshop. Kích thước vùng lân cận của highlights, tính bằng pixel (0–2500). | Giống radius của shadows. |
| `ps_color` | 20 | Chế độ Photoshop. Độ bão hòa trong các tông được chỉnh (-100 đến 100). | Tăng nếu vùng tối sau khi nâng bị nhạt màu; giảm nếu màu quá rực. |
| `ps_midtone` | 0 | Chế độ Photoshop. Tương phản vùng trung tính (midtone) (-100 đến 100). | Giá trị dương nhỏ giúp ảnh có lực trở lại sau khi nâng tối mạnh. |

Các thông số `ps_` bị bỏ qua ở chế độ `camera_raw`.

**Đầu ra**

| Đầu ra | Là gì |
|---|---|
| `image` | Ảnh đã chỉnh, cùng kích thước với ảnh vào. Khi cả hai thanh bằng 0, đây chính là ảnh vào, không đổi. |

## Mẹo khi chỉnh ảnh

- Ở `camera_raw`, `highlights` cũng làm cả ảnh tối đi một chút, vì phần tối nhất được giữ cố định, giống Camera Raw. Nếu cần, bù lại một chút exposure bằng [ColorLab](color-lab.md).
- Ở chế độ `photoshop`, `shadows` âm và `highlights` dương không có tác dụng: hộp thoại gốc không làm tối shadows hay làm sáng highlights được. Khi `shadows` từ 0 trở xuống và `highlights` từ 0 trở lên, ảnh đi qua không đổi, nên chỉnh riêng `ps_color` hay `ps_midtone` cũng không có tác dụng.
- Node tắt phần kéo giãn Black Clip / White Clip của hộp thoại Photoshop, nên vùng tối đã nâng không bị đẩy ngược về đen.
- Giá trị quá mạnh sẽ làm ảnh phẳng. Lấy lại chiều sâu bằng [CLAHE Local Contrast](clahe-local-contrast.md) hoặc một đường S nhẹ trong [Tone Curve](tone-curve.md).
- Với chân dung ngược sáng, hãy mask chủ thể để phông sáng phía sau vẫn giữ được vẻ thoáng.

??? note "Bên trong node hoạt động thế nào"
    **camera_raw**: ảnh được đổi sang RGB tuyến tính, độ chói Rec. 709 được đổi sang log2 (đơn vị stop). Bộ lọc local Laplacian nhanh (Paris, Hasinoff & Kautz 2011, theo dạng nhanh của Aubry và cộng sự 2014, các giá trị tham chiếu cách nhau 1/8 stop) chạy trên một ảnh proxy có cạnh dài tối đa 1600 px. Highlights kéo những pixel sáng hơn xung quanh về gần xung quanh khi vượt một ngưỡng chi tiết nhỏ, rồi đặt lại phân vị 1 %; Shadows nâng những pixel tối hơn xung quanh, rồi đặt lại phân vị 99 %. Phần thay đổi được đưa về kích thước gốc bằng guided upsampling nhanh, rồi nhân như một hệ số khuếch đại trên RGB tuyến tính, nên sắc độ được giữ và kết quả không phụ thuộc kích thước ảnh.

    **photoshop**: độ sáng vùng lân cận là trung bình có giữ cạnh của `(max(R,G,B) + luma) / 2` (Gaussian sigma khoảng 1,14 × Radius). Mask theo tông là phần bù của hàm phân phối Beta, hình dạng thay đổi theo Tone; độ mạnh của đường cong theo Amount; Color cộng lại chênh lệch màu của pixel có nhân hệ số và giữ trong gamut; Midtone là một đường S toàn cục. Mô hình làm việc trên giá trị sRGB, như Photoshop làm trong tài liệu sRGB.

    Các điểm nút độ mạnh, bảng mask Beta, bảng Color và Midtone được chuyển từ PhotoCraft (giấy phép MIT, xem `THIRD_PARTY_NOTICES.md`); PhotoCraft khớp chúng với Photoshop 25.4 bằng cách đo hộp đen.

    ComfyUI đưa ảnh vào trên CPU; node tự chuyển phần tính toán sang GPU nếu có và quay về CPU khi hết bộ nhớ GPU. Mask tùy chọn hòa kết quả với ảnh gốc. Mã nguồn: `core/shadows_highlights.py`, `core/local_laplacian.py`, `core/shadows_highlights_ps.py`.

## Node liên quan

- [ColorLab](color-lab.md) để chỉnh exposure, cân bằng trắng và phần còn lại của bản grade cơ bản
- [Luminosity Masking](luminosity-masking.md) để dựng mask shadows hoặc highlights
- [CLAHE Local Contrast](clahe-local-contrast.md) để lấy lại độ gắt cục bộ
- [Tone Curve](tone-curve.md) để chỉnh tương phản toàn ảnh sau khi nâng tối

## Workflow mẫu

[Workflow mẫu Shadows/Highlights Pro](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-50-shadows-highlights-pro.json) mở vùng tối và cứu vùng sáng của ảnh chân dung mẫu. Workflow chỉ dùng node lõi của ComfyUI và pack này. Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/`.

![Kết quả: Workflow mẫu Shadows/Highlights Pro](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/N-50-shadows-highlights-pro-preview.png)
