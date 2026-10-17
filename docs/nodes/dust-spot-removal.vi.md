---
title: Dust & Spot Removal
description: Tìm các hạt bụi, đốm nhỏ và vết tì rồi xóa chúng bằng màu của vùng xung quanh.
---

# Dust & Spot Removal

Dọn bụi tự động, giống như chạy Spot Healing Brush qua mọi đốm cùng lúc. Node chỉ đụng tới các đốm nhỏ, còn mắt, tóc và đường viền thì giữ nguyên.

![Dust & Spot Removal — trước / sau](../assets/nodes/dust-spot-removal.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → filters → Filters: Dust & Spot Removal` |
| **Dùng cho** | Bụi cảm biến trên bầu trời và phông nền, đốm nhỏ trên da, bụi phim scan, vết tì nhỏ |

## Khi nào dùng

Bụi trên cảm biến hiện ra thành các chấm tối nhỏ, dễ thấy nhất trên vùng mịn như bầu trời, phông studio và da. Node tìm các đốm nhỏ tối hơn (hoặc sáng hơn) vùng xung quanh, rồi lấp từng đốm bằng màu của chính vùng quanh nó.

Một đốm chỉ được tính nếu nằm gọn trong `spot_size` pixel. Hình lớn hơn như mắt, nốt ruồi bạn muốn giữ, hoặc vật mảnh dài như sợi tóc sẽ không bao giờ bị đụng tới.

## Bắt đầu nhanh

1. Load Image → **Filters: Dust & Spot Removal** → Preview Image.
2. Giữ `mode` ở `dark` cho bụi cảm biến. Dùng `bright` cho đốm trắng, `both` cho cả hai.
3. Đặt `spot_size` lớn hơn đốm to nhất bạn muốn xóa một chút.
4. Xem đầu ra `spot_mask` (bằng [Mask Preview](mask-preview.md)) để biết đã xóa những gì. Tăng hoặc giảm `sensitivity` cho đến khi chỉ có các đốm được chọn.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `spot_size` | 12 | Đường kính đốm lớn nhất tính bằng pixel (2 đến 64). Thứ gì lớn hơn hoặc dài hơn sẽ bị bỏ qua. | Giữ nhỏ trên da để tàn nhang và lỗ chân lông còn nguyên. |
| `sensitivity` | 0.5 | Đốm mờ đến đâu vẫn bị bắt (0 đến 1). Cao hơn thì bắt được đốm mờ hơn. | Tăng từ từ; quá cao sẽ bắt luôn kết cấu da. |
| `mode` | `dark` | `dark`, `bright` hoặc `both`. | `dark` cho bụi cảm biến. |
| `mask` (tùy chọn) | không | Chỉ xóa đốm ở nơi mask lớn hơn 0.5. | Dùng mask da hoặc mask trời. |
| `protect_mask` (tùy chọn) | không | Không bao giờ xóa đốm ở nơi mask này lớn hơn 0.5. | Tô lên tàn nhang hay nốt ruồi bạn muốn giữ. |

**Đầu ra**

| Đầu ra | Là gì |
|---|---|
| `image` | Ảnh đã làm sạch. |
| `spot_mask` | Trắng (1) ở nơi pixel bị thay, đen (0) ở nơi khác. Mọi thứ ngoài vùng này giữ nguyên. |

## Mẹo khi chỉnh ảnh

- Với ảnh chân dung, nối mask da vào `mask` (và mắt, tóc vào `protect_mask`). Node đã bỏ qua các hình mảnh, dài và các đốm nằm trong vùng nhiều chi tiết nên mi mắt, tóc và các đường viền được giữ nguyên, nhưng dùng mask vẫn là cách an toàn nhất để giữ chi tiết thật như nốt ruồi hay điểm sáng trong mắt.
- Xem `spot_mask` trước. Nếu thấy các đường dài hoặc rất nhiều lỗ chân lông, hãy giảm `sensitivity` hoặc `spot_size`.
- Giới hạn node vào vùng da bằng `mask`, hoặc bảo vệ mắt và tóc bằng `protect_mask`.
- Với việc xử lý da nặng hơn, nối thêm [Skin Smoother](edge-aware-smoother.md) hoặc làm trên layer riêng với [Frequency Separation](frequency-separation.md) (tách tần số).
- Chạy node này trước khi làm nét, nếu không [Smart Sharpen](smart-sharpen.md) sẽ làm bụi rõ hơn.
- Những mảng bụi rất lớn (trên 64 px) cần clone thủ công.

??? note "Bên trong node hoạt động thế nào"
    Độ sáng (luminance, trọng số Rec. 709) được so với nền trung vị (median) tính trên bản thu nhỏ (lấy trung bình vùng, rồi median 9 × 9 theo từng khối hàng), nên không bao giờ có median kernel lớn chạy ở độ phân giải gốc. Độ chênh được lấy ngưỡng (`0.01 + 0.2 × (1 − sensitivity)²`), sau đó các vùng liên thông chỉ được giữ nếu gọn: khung bao không quá `spot_size` pixel, tỉ lệ dài/rộng tối đa 2, độ lấp đầy tối thiểu 0,5 và dày khoảng 3 px trở lên (đỉnh distance transform từ 1,3 px), nên sợi tóc và lông mi bị loại. Vùng đó còn phải nằm trong môi trường yên tĩnh: độ lệch độ sáng trong cửa sổ quanh nó (không tính chính các đốm) phải nhỏ hơn 0,25 × độ tương phản của đốm, nhờ vậy đốm nằm trên cạnh hay trong vùng chi tiết bị loại. Trên ảnh chân dung sạch mẫu, node chỉ đổi khoảng 0,006 % số pixel. Các vùng này được nở thêm `max(1, spot_size // 6)` pixel và bị giới hạn bởi các mask. Phần lấp dùng tích chập Gauss chuẩn hóa của các pixel nằm ngoài đốm, tính ở kích thước thu nhỏ rồi phóng lại, sau đó hòa vào với viền mềm 1 px. Ảnh 2048 × 2048 dùng đỉnh dưới 0,5 GB bộ nhớ GPU. Nếu không tìm thấy đốm nào, ảnh gốc được trả về nguyên vẹn. Mã nguồn: `core/dust_spot.py`.

## Node liên quan

- [Mask Preview](mask-preview.md) để xem `spot_mask` phủ lên ảnh
- [Skin Smoother](edge-aware-smoother.md) để làm mịn da tổng thể
- [Frequency Separation](frequency-separation.md) để xử lý vết tì thủ công
- [Mask Combine](mask-combine.md) để dựng `mask` hoặc `protect_mask`

## Workflow mẫu

[Workflow mẫu Dust & Spot Removal](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-48-dust-spot-removal.json) xóa 70 vết bụi cảm biến giả lập trên ảnh chân dung với thông số mặc định và tô đỏ các vết đã tìm thấy. Workflow chỉ dùng node lõi của ComfyUI và pack này. Chép `workflows/sample_dust.jpg` vào `ComfyUI/input/`.

![Kết quả: Workflow mẫu Dust & Spot Removal](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/N-48-dust-spot-removal-preview.png)
