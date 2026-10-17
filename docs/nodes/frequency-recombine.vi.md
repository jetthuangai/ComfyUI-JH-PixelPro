---
title: Frequency Recombine
description: Ghép lại hai lớp low và high từ Frequency Separation, có thanh trượt tăng hoặc giảm texture.
---

# Frequency Recombine

Dựng lại bức ảnh từ hai lớp do [Frequency Separation](frequency-separation.md) tạo ra. Bạn sửa màu trên `low`, sửa texture trên `high`, rồi ghép lại. Một thanh trượt cho phép đẩy lỗ chân lông và chi tiết mịn mạnh lên hoặc yếu đi cùng lúc.

![Frequency Recombine — trước / sau](../assets/nodes/frequency-recombine.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → filters → Frequency Recombine (low + high)` |
| **Dùng cho** | Hoàn tất một lượt retouch tách tần số (frequency separation), tăng hoặc giảm texture da mà không đụng đến màu |

## Khi nào dùng

[Frequency Separation](frequency-separation.md) tách ảnh thành lớp `low` mượt (màu và ánh sáng) và lớp `high` có dấu (texture). Lớp `high` chứa giá trị âm, nên các node blend hay node cộng thông thường không ghép lại được: chúng cắt mất nửa âm. **Frequency Recombine** cộng đúng cách và chỉ giới hạn kết quả cuối trong khoảng 0 đến 1.

Thông số `detail` nhân lớp texture trước khi cộng lại:

- `1.0` trả về đúng ảnh gốc nếu bạn chưa sửa lớp nào.
- Trên `1.0` làm texture mạnh lên: lỗ chân lông, tóc, sợi vải rõ hơn.
- Dưới `1.0` làm texture dịu đi. `0.0` chỉ còn lớp `low`.

## Bắt đầu nhanh

1. `Load Image` → [Frequency Separation](frequency-separation.md).
2. Nối `low` và `high` vào hai đầu vào cùng tên của **Frequency Recombine**.
3. Giữ `detail = 1.0` và xem trước: bạn sẽ thấy lại ảnh gốc.
4. Sửa `low` (ví dụ bằng [Skin Smoother](edge-aware-smoother.md)), rồi thử `detail = 1.2` đến `1.6` để texture rõ hơn.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `low` | — | Lớp màu và tông, đã sửa hoặc chưa. | Lấy từ Frequency Separation. |
| `high` | — | Lớp texture, giá trị có dấu. | Nối trực tiếp. Đừng cho đi qua node có cắt giá trị. |
| `detail` | `1.0` (0 đến 3, bước 0.05) | Mức texture được cộng lại. | 0.8 đến 1.0 cho da mịn hơn, 1.2 đến 1.6 cho texture rõ hơn. |

| Đầu ra | Kiểu | Ý nghĩa |
|---|---|---|
| `image` | `IMAGE` | `low + detail × high`, giới hạn trong 0..1. |

## Mẹo khi chỉnh ảnh

- Kiểm tra trước: chưa sửa gì và `detail = 1.0` thì kết quả phải giống ảnh gốc.
- Làm mượt `low` và để `detail` bằng hoặc hơi trên 1 để che vệt loang mà vẫn giữ lỗ chân lông.
- Giữ nguyên `radius` ở Frequency Separation trong cả một lượt chỉnh. Hai lớp chỉ khớp nhau khi đến từ cùng một lần tách.
- `detail` quá lớn trên ảnh nhiều nhiễu sẽ làm nhiễu rõ lên. Hãy tăng từng bước nhỏ.
- Hai lớp phải cùng chiều rộng và chiều cao.

??? note "Bên trong node hoạt động thế nào"
    Kết quả là `low + detail × high` tính bằng float32, giới hạn đúng một lần ở cuối. Hai đầu vào phải là RGB cùng chiều cao và chiều rộng. Batch phải bằng nhau, hoặc một bên bằng 1 (sẽ được nhân rộng). `detail` ngoài khoảng 0 đến 3 sẽ báo lỗi. Nguồn: `frequency_recombine` trong `core/frequency.py` và node trong `nodes/frequency_node.py`.

## Node liên quan

- [Frequency Separation](frequency-separation.md)
- [Skin Smoother](edge-aware-smoother.md)
- [Smart Sharpen](smart-sharpen.md)
- [Detail Mask](high-freq-detail-masker.md)

## Workflow mẫu

[Workflow mẫu Frequency Separation](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-01-frequency-separation.json) tách ảnh chân dung rồi ghép lại bằng Frequency Recombine với `detail` 1.6. Workflow chỉ dùng node lõi của ComfyUI và pack này. Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/`.
