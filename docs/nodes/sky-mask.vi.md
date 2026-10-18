---
title: Sky Mask
description: Tự tìm bầu trời trong ảnh và trả về mask mềm, không dùng model AI.
---

# Sky Mask

Chọn vùng bầu trời giúp bạn: trời xanh, trời âm u hay hoàng hôn. Dùng mask này để chỉnh màu riêng cho trời, thay trời, hoặc bảo vệ trời khỏi các chỉnh sửa khác.

![Sky Mask — ảnh gốc, mask bầu trời, trời đã chỉnh màu qua mask](../assets/nodes/sky-mask.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → mask → Mask: Sky` |
| **Dùng cho** | Chỉnh màu hoặc thay trời, hạ độ sáng của trời cháy, loại bầu trời khỏi chỉnh sửa da hoặc tiền cảnh |
| **Tốc độ** | Khoảng 0,2 giây mỗi ảnh |

## Khi nào dùng

Chọn trời bằng tay nghĩa là kéo gradient mask rồi dọn rất nhiều quanh cây và mái nhà. Node này làm việc đó từ chính nội dung ảnh, không cần model AI và không phải tải gì.

Node dựa trên ba tín hiệu: màu giống trời (xanh, mây trắng hoặc xám, ánh hoàng hôn ấm), bề mặt mịn (trời gần như không có chi tiết nhỏ), và vị trí hở lên mép trên khung hình. Lá cây, nhà cửa và cỏ không qua được bước kiểm tra độ mịn. Bức tường trắng nằm thấp trong khung không qua được bước kiểm tra vị trí.

Mask là mask mềm (soft mask). Viền quanh cành cây và mái nhà bám theo đúng biên thật, nên khi chỉnh màu qua mask sẽ không bị quầng sáng.

## Bắt đầu nhanh

1. Load Image → **Mask: Sky**.
2. Thêm [Mask Preview](mask-preview.md) để xem mask phủ lên ảnh.
3. Giữ `sensitivity` ở 0.5 và bật `refine_edges`.
4. Để chỉnh màu riêng cho trời, nối `sky_mask` vào ngõ mask của [Apply ColorLab Preset](apply-colorlab-preset.md).

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `image` | | Ảnh có bầu trời. | |
| `sensitivity` | 0.5 | 0 đến 1. Càng cao càng lấy cả trời mờ sương, có nhiều chi tiết hoặc trời hoàng hôn. Thấp thì chỉ lấy trời quang. | Tăng với mây mỏng hoặc hoàng hôn. Giảm nếu tường hay mặt nước bị lẹm vào. |
| `refine_edges` | bật | Bám theo cành cây và đường mái ở độ phân giải đầy đủ bằng colour guided filter. | Chỉ tắt khi bạn muốn một mask mềm đơn giản. |

**Đầu ra**

| Đầu ra | Là gì |
|---|---|
| `sky_mask` | Mask mềm của bầu trời. Trắng là trời. Mỗi ảnh trong batch có một mask. |

## Mẹo khi chỉnh ảnh

- Xem mask bằng [Mask Preview](mask-preview.md) trước khi chỉnh màu. Tăng `sensitivity` từng bước nhỏ cho đến khi vùng mờ sương hoặc hoàng hôn được phủ đủ.
- Mặt nước phẳng nối liền với trời ở đường chân trời có thể bị lấy vào. Tạo mask cho vùng nước rồi trừ đi bằng [Mask Combine](mask-combine.md).
- Tường trắng mịn chạm vào trời có thể lẹm một chút. Giảm `sensitivity`, hoặc dọn bằng [Mask Cleanup](mask-cleanup.md).
- Mây có nhiều chi tiết thì trông kém mịn, nên cần `sensitivity` cao hơn.
- Dùng mask đảo ngược để bảo vệ trời khi chỉnh phần còn lại, ví dụ với [Color Matcher](color-matcher.md) hoặc node Layer.

??? note "Bên trong node hoạt động thế nào"
    Ảnh được thu nhỏ còn tối đa 768 px ở cạnh dài. Trên bản thu nhỏ này, điểm số "giống trời" kết hợp màu (xanh theo hue và saturation, vùng sáng gần xám cho mây, vùng ấm sáng ở nửa trên với trọng số một nửa cho hoàng hôn) với phép kiểm tra độ mịn dựa trên biến thiên luma cục bộ. `sensitivity` cao hạ ngưỡng điểm và nới lỏng phép kiểm tra độ mịn.

    Chỉ giữ các vùng ứng viên nối với các hàng trên cùng của khung, hoặc vùng lớn nằm ở nửa trên. Khi có ít trời chạm mép trên (ví dụ nhìn lên qua tán cây), các vùng xanh rõ và đủ lớn ở bất kỳ đâu cũng được giữ. Những khe nhỏ như chim và dây điện được lấp trước.

    Với `refine_edges`, mask được phóng về kích thước gốc và tinh chỉnh bằng colour guided filter dựa trên ảnh gốc. Mã nguồn: `core/sky_mask.py`, wrapper `nodes/sky_mask_node.py`.

## Node liên quan

- [Apply ColorLab Preset](apply-colorlab-preset.md): chỉnh màu riêng cho vùng trời
- [Mask Combine](mask-combine.md): cộng hoặc trừ mask, ví dụ để loại mặt nước
- [Mask Cleanup](mask-cleanup.md): xóa đốm nhỏ và lấp lỗ
- [Mask Preview](mask-preview.md): xem mask phủ lên ảnh

## Workflow mẫu

[Workflow mẫu Sky Mask](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-61-sky-mask.json) tìm bầu trời của một ảnh kiến trúc, hiển thị bằng Mask Preview, rồi áp một preset ColorLab chỉ lên vùng trời qua mask. Chép `workflows/sample_architecture.jpg` vào `ComfyUI/input/` (Pexels License).
