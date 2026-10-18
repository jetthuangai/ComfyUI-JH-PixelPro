---
title: Lightroom Preset Contact Sheet
description: Hiển thị một ảnh với nhiều preset Lightroom được áp, xếp cạnh nhau thành lưới có nhãn.
---

# Lightroom Preset Contact Sheet

Bày ảnh của bạn ra với cả một bộ preset Lightroom được áp, giống tờ contact sheet: ảnh gốc đứng đầu, sau đó mỗi preset một ô có ghi tên. Cách nhanh để lọc ra vài look ưng ý trước khi chọn hẳn một cái.

![Lightroom Preset Contact Sheet — ví dụ contact sheet](../assets/nodes/lightroom-preset-contact-sheet.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → color → Color: Lightroom Preset Contact Sheet` |
| **Dùng cho** | Chọn preset cho một tấm ảnh cụ thể, xem thử một bộ preset mới mua, gửi khách xem nhiều look cùng lúc, thử lại các preset yêu thích trên buổi chụp mới |
| **Tốc độ** | 10 preset ở 320 px mất khoảng 0,14 giây trên GPU (sau lần chạy khởi động) |

## Khi nào dùng

Thumbnail trong trình duyệt của node [Lightroom Preset](lightroom-preset.md) khá nhỏ. Khi muốn đánh giá look kỹ hơn trên chính ảnh của mình, node này render các preset đó ở kích thước bạn chọn và gom vào một ảnh để xem, lưu hoặc gửi đi.

Node dùng chung thư mục preset, collection, danh sách yêu thích và Recent với node Lightroom Preset, và cùng cách chuyển sang ColorLab, nên mỗi ô ở đây trông giống node đầy đủ ở `strength` 1.0.

## Bắt đầu nhanh

1. Load Image → **Color: Lightroom Preset Contact Sheet** → Preview Image ở cổng `sheet`.
2. Chọn `collection`, ví dụ một thư mục con chứa preset của bạn hoặc ★ Favorites.
3. Chạy. Ghi lại tên look bạn thích (cổng `presets` liệt kê theo đúng thứ tự).
4. Chọn preset đó trong một node [Lightroom Preset](lightroom-preset.md) để áp ở kích thước đầy đủ.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `image` | — | Ảnh để xem thử preset. Chỉ dùng khung hình đầu tiên của batch. | |
| `collection` | All | Nhóm preset cần hiển thị: All, ★ Favorites, Recent, hoặc một thư mục con trong thư mục preset. | Yêu thích và Recent là những preset bạn đã đánh dấu/dùng trong trình duyệt Lightroom Preset. |
| `columns` | 4 | Số ô mỗi hàng (1 đến 12). | 4 đến 6 hợp với màn hình rộng. |
| `thumb_size` | 320 | Cạnh dài mỗi ô, tính bằng pixel (128 đến 768). Ảnh chỉ được thu nhỏ, không bao giờ phóng to. | 320 để xem tổng quan, từ 512 trở lên để soi tông da. |
| `max_presets` | 24 | Số preset tối đa trên sheet (1 đến 200). | Nên kết hợp với `search` thay vì render một sheet khổng lồ. |
| `search` (tùy chọn) | trống | Chỉ lấy preset có tên chứa mọi từ bạn gõ. | Ví dụ `film warm`. |
| `tone_engine` (tùy chọn) | `local` | `local` dùng công cụ tông giữ cạnh (edge-aware), gần Lightroom hơn; `legacy` dùng phép tính tông đơn giản của ColorLab. | Dùng cùng engine với node Lightroom Preset để các ô khớp với kết quả thật. |

**Đầu ra**

| Đầu ra | Là gì |
|---|---|
| `sheet` | Một ảnh dạng lưới có nhãn trên nền tối: "Original" đứng đầu, sau đó là từng preset. Preset chỉ được hỗ trợ một phần có thêm `[!]` sau tên. |
| `presets` | Tên các preset trên sheet, mỗi dòng một tên, cùng thứ tự với các ô. |

## Mẹo khi chỉnh ảnh

- Đánh sao các preset tốt nhất trong trình duyệt Lightroom Preset, rồi đặt `collection` là ★ Favorites để thử tất cả trên mỗi buổi chụp mới.
- Ô có `[!]` đang thiếu một phần look gốc (thường là creative profile). Xem report của preset đó trong node Lightroom Preset trước khi chọn.
- Dùng ảnh có da người, vùng trung tính và chút màu bão hòa; các preset trông na ná nhau trên ảnh phong cảnh có thể khác hẳn trên da.
- Preset không đọc được sẽ bị bỏ khỏi sheet, nên danh sách `presets` là bản ghi chính xác những gì bạn đang thấy.

??? note "Bên trong node hoạt động thế nào"
    Khung hình đầu tiên được thu nhỏ một lần bằng phép lấy trung bình vùng (area) sao cho cạnh dài không quá `thumb_size`. Mỗi preset được đọc và chuyển thành thiết lập ColorLab (preset đã đọc được lưu đệm theo file và thời điểm sửa), rồi áp với exposure trong không gian tuyến tính, đúng như node Lightroom Preset ở mức đầy đủ. Tone engine `local` chạy trên GPU khi có. Các ô được ghép bằng Pillow: khoảng cách 6 px, một dải nhãn phía trên mỗi ô, tên dài quá 48 ký tự bị cắt. Mã nguồn: `core/lr_preset_library.py`.

## Node liên quan

- [Lightroom Preset](lightroom-preset.md) để áp preset đã chọn ở kích thước đầy đủ
- [ColorLab](color-lab.md) để tinh chỉnh look thêm
- [Apply ColorLab Preset](apply-colorlab-preset.md) để dùng lại một look ColorLab đã lưu
- [LUT Preset](lut-preset.md) cho look lưu dạng LUT .cube

## Workflow mẫu

[Workflow mẫu Lightroom Preset Contact Sheet](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-53-lightroom-preset-contact-sheet.json) hiển thị ảnh chân dung với các preset trong thư mục preset xếp cạnh nhau. Workflow chỉ dùng node lõi của ComfyUI và pack này. Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/`.

![Kết quả: Workflow mẫu Lightroom Preset Contact Sheet](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/N-53-lightroom-preset-contact-sheet-preview.png)
