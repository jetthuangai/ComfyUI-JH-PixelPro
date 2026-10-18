---
title: HDR Merge
description: Trộn các ảnh bracket thành một ảnh giữ được chi tiết ở vùng tối lẫn vùng sáng.
---

# HDR Merge

Đưa vào một bộ bracket của cùng một cảnh, ví dụ tối, chuẩn và sáng. Node trả về một ảnh lấy phần phơi sáng tốt nhất của từng tấm.

![HDR Merge — ba mức phơi sáng và kết quả sau khi trộn](../assets/nodes/hdr-merge.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → color → Color: HDR Merge (bracket)` |
| **Dùng cho** | Nội thất có cửa sổ sáng, phong cảnh có bầu trời sáng, ảnh bất động sản, mọi cảnh tương phản cao chụp theo bracket |
| **Tốc độ** | Khoảng 1 giây cho ba khung 2 MP, chạy trên CPU |

## Khi nào dùng

Một tấm ảnh thường không giữ được cả ô cửa sáng và căn phòng tối. Bộ bracket (ví dụ -2, 0 và +2 EV) ghi lại được cả hai. Node chọn từng pixel, lấy phần phơi sáng tốt, nhiều tương phản và đậm màu của mỗi khung rồi hòa lại mượt.

Phương pháp này là exposure fusion. Không cần giá trị phơi sáng, không cần đường cong đáp ứng của máy ảnh, và kết quả xem được ngay. Không có bước tone mapping với hàng loạt thanh trượt cần chỉnh.

Kết quả là một ảnh thông thường để xem trên màn hình, không phải file HDR 32-bit. Hãy xem đây là điểm xuất phát, rồi hoàn thiện bằng [Auto Tone](auto-tone.md) hoặc [ColorLab](color-lab.md).

## Bắt đầu nhanh

1. Load các ảnh bracket và gộp thành một batch bằng Batch Images (hoặc Load Image Batch). Thứ tự không quan trọng, nhưng mọi khung phải cùng kích thước.
2. Nối batch vào **Color: HDR Merge (bracket)** rồi thêm Preview Image.
3. Giữ `align` và `deghost` ở trạng thái bật.
4. Muốn chỉnh look thì hoàn thiện bằng [Auto Tone](auto-tone.md) hoặc [ColorLab](color-lab.md).

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `images` | | Batch gồm từ 2 ảnh bracket trở lên, thứ tự bất kỳ, cùng kích thước. | Dùng Batch Images hoặc Load Image Batch để tạo batch. |
| `align` | bật | Căn chỉnh các ảnh chụp cầm tay bằng cách sửa lệch nhỏ (chỉ tịnh tiến). | Tắt nếu chụp bằng chân máy và muốn chạy nhanh nhất. |
| `deghost` | bật | Chỗ nào có vật thể di chuyển giữa các tấm (người, lá cây), dùng ảnh phơi sáng giữa đã khớp độ sáng, nên không bị nhân đôi. | Tắt với cảnh hoàn toàn tĩnh. |

**Đầu ra**

| Đầu ra | Là gì |
|---|---|
| `image` | Một ảnh đã trộn. |

## Mẹo khi chỉnh ảnh

- Chụp -2, 0 và +2 EV trên chân máy, hoặc bấm liên tiếp thật nhanh, để các khung khớp nhau nhất có thể.
- Chuyển động lớn hoặc xoay giữa các tấm sẽ không được căn chỉnh. Hãy chụp lại, hoặc sửa phần xoay trước bằng [Auto Align](auto-align.md) nếu các khung cho phép.
- Chủ thể chuyển động như người, xe, cành cây được `deghost` xử lý. Sau khi trộn, hãy phóng 100 % để kiểm tra.
- Ảnh sau khi trộn trông tự nhiên và hơi phẳng. Thêm contrast và màu sau đó bằng [Auto Tone](auto-tone.md), [ColorLab](color-lab.md) hoặc [Shadows/Highlights Pro](shadows-highlights-pro.md).
- Hai khung vẫn chạy được, nhưng từ ba khung trở lên cho chuyển tiếp mượt hơn.

??? note "Bên trong node hoạt động thế nào"
    Các khung được sắp theo độ sáng và khung giữa làm chuẩn. Với `align`, mỗi khung được dịch về khung chuẩn bằng thuật toán căn chỉnh median-threshold-bitmap (OpenCV AlignMTB). Độ lệch lớn hơn 5 % cạnh dài bị coi là căn chỉnh thất bại và bỏ qua.

    Với `deghost`, mỗi khung trước tiên được đưa về độ sáng của khung chuẩn. Chỗ nào vẫn lệch mạnh, tức là có chuyển động, khung sẽ lấy pixel của khung chuẩn đã khớp độ sáng, với viền mềm.

    Các khung được trộn bằng Mertens exposure fusion (Mertens, Kautz và Van Reeth, 2007): hòa nhiều mức phân giải, có trọng số theo contrast, saturation và mức độ phơi sáng tốt. Node chạy trên CPU bằng OpenCV. Mã nguồn: `core/hdr_merge.py`, wrapper `nodes/hdr_merge_node.py`.

## Node liên quan

- [Auto Tone](auto-tone.md): hoàn thiện tự động sau khi trộn
- [ColorLab](color-lab.md): kiểm soát đầy đủ màu và tông
- [Shadows/Highlights Pro](shadows-highlights-pro.md): cứu thêm vùng tối và vùng sáng
- [Auto Align](auto-align.md): căn chỉnh các khung bị xoay hoặc lệch nhiều

## Workflow mẫu

[Workflow mẫu HDR Merge](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-62-hdr-merge.json) biến một ảnh quán cà phê thành ba mức phơi sáng (-2, 0 và +2) bằng ColorLab, gộp bằng Batch Images rồi trộn. Chép `workflows/sample_cafe.jpg` vào `ComfyUI/input/` (Pexels License).
