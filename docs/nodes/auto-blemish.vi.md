---
title: Auto Blemish
description: Tìm khuôn mặt rồi xóa mụn, đốm và vết nhỏ trên da, vẫn giữ nguyên lỗ chân lông.
---

# Auto Blemish

Phát hiện khuôn mặt trong ảnh chân dung và xóa mụn, đốm, vết nhỏ trên da. Texture da được nối liền qua chỗ vừa xóa nên ảnh không bị mịn giả.

![Auto Blemish — trước, sau, các điểm đã được xóa](../assets/nodes/auto-blemish.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → face → Face: Auto Blemish` |
| **Dùng cho** | Lượt xóa mụn đầu tiên cho ảnh chân dung, ảnh nhóm, loạt ảnh headshot, da cơ thể khi bạn tự đưa mask |
| **Tốc độ** | Khoảng 5 giây trên CPU với ảnh chân dung 8 MP, dưới 1,1 GB VRAM |

## Khi nào dùng

Đây là bản tự động của Spot Healing Brush. Node làm phần việc nhàm chán nhất khi retouch chân dung: tìm từng vết nhỏ. Sau đó bạn kiểm tra kết quả và xử lý tay những gì còn sót.

!!! warning
    Node cũng xóa cả nốt ruồi tự nhiên nếu bạn không bảo vệ. Hãy tô mask lên nốt ruồi duyên, mảng tàn nhang hay hình xăm muốn giữ rồi nối vào `protect_mask`.

Mỗi khuôn mặt được xử lý trong crop riêng, với kích thước đốm riêng, nên ảnh nhóm vẫn dùng được. Mắt, lông mày và miệng không bao giờ bị đụng tới. Với vùng da không có mặt trong khung (lưng, tay, cánh tay), hãy đưa vào `mask` vùng da cần làm sạch.

## Bắt đầu nhanh

1. Load Image → **Face: Auto Blemish** → Preview Image.
2. Giữ mặc định. Tăng `sensitivity` nếu còn sót vết mờ. Giảm nếu tàn nhang bắt đầu biến mất.
3. Nối `spot_mask` vào [Mask Preview](mask-preview.md) để thấy chính xác chỗ nào đã được xóa.
4. Xem `face_count`. Nếu bằng 0 trên ảnh chân dung nghĩa là không tìm thấy mặt; hãy dùng `mask`.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `image` | | Ảnh chân dung cần chỉnh. | |
| `sensitivity` | 0.5 | 0 đến 1. Càng cao càng bắt được vết mờ. Rất cao sẽ xóa luôn tàn nhang. | 0.3 đến 0.6 cho đa số ảnh chân dung. |
| `spot_scale` | 1.0 | 0.25 đến 4. Vết lớn nhất được xóa, so với kích thước tự động (5 % bề ngang mỗi khuôn mặt). | Tăng cho mụn to. Giảm để giữ chi tiết nhỏ. |
| `mask` (tùy chọn) | | Vùng da cần làm sạch, thay cho khuôn mặt tự phát hiện. Kích thước đốm khi đó theo bề ngang mask. | Dùng cho thân người, lưng, tay. |
| `protect_mask` (tùy chọn) | | Vùng không bao giờ bị đụng tới. | Nốt ruồi duyên muốn giữ, hình xăm. |

**Đầu ra**

| Đầu ra | Là gì |
|---|---|
| `image` | Ảnh đã retouch. |
| `spot_mask` | Các pixel đã được xóa. Kiểm tra bằng Mask Preview. |
| `face_count` | Số khuôn mặt tìm thấy. Bằng 0 khi có `mask`. |

## Mẹo khi chỉnh ảnh

- Luôn xem `spot_mask` một lần. Nó cho biết node bắt đúng mụn hay bắt cả thứ bạn muốn giữ.
- Nốt ruồi tự nhiên và tàn nhang đậm dễ bị xóa khi `sensitivity` cao. Hãy bảo vệ bằng `protect_mask`.
- Sau đó dùng [Skin Smoother](edge-aware-smoother.md) hoặc [Frequency Separation](frequency-separation.md) (tách tần số) để đều màu da. Auto Blemish chỉ xóa vết.
- Vết còn sót thường to hơn giới hạn tự động. Tăng `spot_scale`, hoặc xóa bằng [Heal / Content-Aware Fill](heal.md).
- Ảnh lớn vẫn chạy tốt: mỗi khuôn mặt được crop riêng, nên chi phí phụ thuộc số khuôn mặt chứ không phải cả khung hình.

??? note "Bên trong node hoạt động thế nào"
    1. **Vùng xử lý.** [Face Detect Lite](face-detect-lite.md) (YuNet) tìm khuôn mặt. Trọng số model được Kornia tải về một lần. Mỗi khung mặt, nới thêm 25 %, được xử lý như một crop riêng. Chỉ giữ pixel có màu da (phép thử YCrCb mềm, phủ từ da sáng đến da tối), trừ các hình tròn quanh mắt, lông mày và miệng đặt theo 5 landmark. `mask` do bạn đưa vào sẽ thay bước tìm mặt.
    2. **Phát hiện.** Bộ dò đốm của [Dust & Spot Removal](dust-spot-removal.md) chạy trên tín hiệu gồm độ tối của luma cộng độ đỏ, vì mụn đỏ hơn là tối hơn. Đốm lớn nhất tỉ lệ theo kích thước mặt (khoảng 5 % bề ngang mặt).
    3. **Xóa.** Texture fill của Dust & Spot chép mảng da gần đó khớp nhất, so trên vòng quanh và hiệu chỉnh độ sáng, nên lỗ chân lông được nối liền.

    Mã nguồn: `core/blemish.py`, wrapper `nodes/blemish_node.py`.

## Node liên quan

- [Heal / Content-Aware Fill](heal.md): xóa vết bạn tự tô mask
- [Dust & Spot Removal](dust-spot-removal.md): bộ dò đốm được dùng ở đây, dành cho bụi cảm biến
- [Face Detect Lite](face-detect-lite.md): bộ nhận diện khuôn mặt được dùng ở đây
- [Skin Smoother](edge-aware-smoother.md): làm đều màu da sau khi làm sạch
- [Mask Preview](mask-preview.md): xem chỗ nào đã được xóa

## Workflow mẫu

[Workflow mẫu Auto Blemish](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-59-auto-blemish.json) cho ảnh chân dung mẫu qua Auto Blemish và hiển thị các preview cùng Mask Preview của `spot_mask`. Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/` (Pexels License).
