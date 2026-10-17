---
title: Frequency Separation
description: Tách ảnh thành lớp màu và tông mượt cùng lớp chi tiết mịn để retouch da và texture.
---

# Frequency Separation

Tách tần số (frequency separation): chia ảnh thành hai lớp là **low** (màu mềm, sáng tối) và **high** (chi tiết mịn như lỗ chân lông, tóc, texture vải). Nhờ vậy bạn sửa da loang màu ở lớp này mà không đụng đến texture ở lớp kia.

![Frequency Separation — trước / sau](../assets/nodes/frequency-separation.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → filters → GPU Frequency Separation` |
| **Dùng cho** | Retouch da, xóa vệt loang màu mà vẫn giữ lỗ chân lông, làm sắc hoặc làm mềm riêng phần texture |

## Khi nào dùng

Frequency separation là kỹ thuật retouch chân dung kinh điển:

- **Tần số thấp (low)** là những thay đổi lớn, chậm: màu da, sáng tối, tông trang điểm. Đây chính là ảnh đã làm mờ Gaussian.
- **Tần số cao (high)** là những thay đổi nhỏ, nhanh: lỗ chân lông, râu, lông mi, sợi vải. Đây là phần mà bước làm mờ đã loại bỏ.

Sửa lỗi màu trên `low`, sửa lỗi texture trên `high`, rồi ghép hai lớp lại. `radius` càng lớn thì càng nhiều chi tiết được đẩy sang `high`.

### Vì sao `high` có thể âm

`high` bằng ảnh gốc trừ ảnh đã làm mờ. Chỗ nào pixel sáng hơn vùng xung quanh (mép lỗ chân lông sáng, điểm sáng trên tóc) thì kết quả dương. Chỗ nào tối hơn (lỗ chân lông, bóng trong nếp nhăn) thì kết quả âm. Trung bình các giá trị nằm quanh 0. Ảnh chỉ lưu được giá trị từ 0 đến 1 nên giá trị âm không hiển thị như ảnh thường được: bản xem trước trông tối hoặc lệch màu. Điều này bình thường. Hai lớp vẫn cộng lại đúng nguyên vẹn: `low + high = ảnh gốc`.

!!! warning "Ghép lại thành ảnh"
    Hãy dùng [Frequency Recombine](frequency-recombine.md) để ghép hai lớp lại. Các node blend có giới hạn giá trị trong 0..1 (kể cả các node layer của pack này) sẽ cắt mất nửa âm của `high`, nên tổng không còn khớp với ảnh gốc.

## Bắt đầu nhanh

1. `Load Image` → **GPU Frequency Separation**.
2. Giữ `radius = 8`, `sigma = 0`, `precision = float32`.
3. Nối `low` vào `Preview Image` để xem lớp mượt. Bản xem trước của `high` sẽ tối, đó là bình thường.
4. Chỉnh `low` hoặc `high` bằng các node của bạn, rồi ghép lại bằng [Frequency Recombine](frequency-recombine.md). Với `detail = 1` node trả về đúng ảnh gốc.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `radius` | `8` (1 đến 128) | Độ lớn của vùng làm mờ, tính bằng pixel. Chi tiết nhỏ hơn mức này sang `high`. | Nhỏ (3 đến 6) cho texture mịn, lớn hơn cho vệt loang to. Không được vượt quá nửa cạnh ngắn của ảnh. |
| `sigma` | `0.0` (0 đến 50) | Độ mềm của blur. `0` là tự động: `radius / 2`. | Để 0 trừ khi cần khớp với công cụ khác. |
| `precision` | `float32` | `float32` ghép lại gần như chính xác tuyệt đối. `float16` nhanh gấp khoảng hai lần trên GPU đời mới, sai số ghép lại khoảng 0.001. | Dùng `float32` cho bản cuối. |

| Đầu ra | Kiểu | Ý nghĩa |
|---|---|---|
| `low` | `IMAGE` | Lớp mờ, mượt. |
| `high` | `IMAGE` | Lớp chi tiết. Có thể chứa giá trị âm. |

## Mẹo khi chỉnh ảnh

- Chọn `radius` theo kích thước ảnh. Cùng 8 px là chi tiết mịn trên ảnh 1 MP nhưng gần như không đáng kể trên ảnh 24 MP.
- Làm mượt tông da trên `low` bằng [Skin Smoother](edge-aware-smoother.md), giữ nguyên `high` để lỗ chân lông còn nguyên.
- Dùng [Detail Mask](high-freq-detail-masker.md) khi cần bảo vệ vùng nhiều texture.
- Tránh `float16` trên CPU. Có thể chậm và kém chính xác hơn.

??? note "Bên trong node hoạt động thế nào"
    `low` là `gaussian_blur2d` của Kornia với kernel `2 × radius + 1`, sigma `radius / 2` trừ khi bạn đặt khác, biên replicate. `high = working − low` theo độ chính xác đã chọn. Đầu vào phải là RGB 3 kênh trong 0..1; `radius` lớn hơn nửa cạnh ngắn sẽ báo lỗi. `sigma` lớn hơn `radius / 2` sẽ ghi cảnh báo vì kernel cắt cụt đuôi Gaussian. Nguồn: `core/frequency.py` và `nodes/frequency_node.py`.

## Node liên quan

- [Frequency Recombine](frequency-recombine.md)
- [Skin Smoother](edge-aware-smoother.md)
- [Smart Sharpen](smart-sharpen.md)
- [Detail Mask](high-freq-detail-masker.md)

## Workflow mẫu

[Workflow mẫu Frequency Separation](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-01-frequency-separation.json). Workflow tách ảnh chân dung, xem trước `low` và `high`, rồi ghép lại bằng Frequency Recombine với `detail` 1.6. Workflow chỉ dùng node lõi của ComfyUI và pack này. Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/`.
