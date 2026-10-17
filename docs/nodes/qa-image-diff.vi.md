---
title: QA Image Diff
description: So sánh hai ảnh và cho thấy khác nhau ở đâu, bao nhiêu, kèm các chỉ số SSIM, PSNR và MAE.
---

# QA Image Diff

Đặt ảnh trước và sau của bạn cạnh nhau cho máy xem. Node vẽ bản đồ những gì đã thay đổi và đưa ra con số, để bạn kiểm tra việc retouch chỉ chạm vào đúng chỗ bạn muốn.

![QA Image Diff — trước / sau](../assets/nodes/qa-image-diff.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → qa → QA: Image Diff (SSIM/PSNR)` |
| **Phù hợp để** | Kiểm tra retouch đã đổi pixel ở đâu, so sánh hai bộ thông số hoặc hai pipeline, xác nhận chỉnh sửa nằm trong mask |

## Khi nào dùng

Mắt người bỏ sót những thay đổi nhỏ. Node này đo chúng:

- **Ảnh diff:** cho thấy hai ảnh khác nhau ở đâu. Càng sáng, thay đổi càng lớn.
- **SSIM** (độ tương đồng cấu trúc): khoảng từ 0 đến 1. 1 nghĩa là cấu trúc giống hệt. Chỉ số này nhạy với thay đổi texture và cạnh hơn là với việc sáng lên tối đi nhẹ trên toàn ảnh.
- **PSNR** (đơn vị dB): càng cao càng gần nhau. Hai ảnh giống hệt cho giá trị trần, 100 dB.
- **Report:** JSON gồm `ssim`, `psnr`, `mae` (sai khác tuyệt đối trung bình, 0 đến 1), `max_abs` (chênh lệch lớn nhất của một pixel) và `changed_pct` (phần trăm pixel thay đổi hơn 1/255).

Dùng nó để chứng minh một bước frequency separation hay làm mịn không chạm vào mắt và tóc, hoặc để thấy chính xác vùng nào bị thay đổi khi đổi thông số.

## Bắt đầu nhanh

1. Nối ảnh gốc vào `image_a` và ảnh đã retouch vào `image_b`. Hai ảnh phải cùng chiều rộng và chiều cao.
2. Giữ `diff_mode` là `heatmap` và `amplify` là 4.
3. Nối `diff` vào node `Preview Image`. Nối `report` vào node xem văn bản để đọc các con số.
4. Tùy chọn nối thêm một mask để chỉ đo trong một vùng.

## Thông số

| Thông số | Mặc định | Tác dụng | Gợi ý |
|---|---|---|---|
| `diff_mode` | `heatmap` | `heatmap` là thang màu của chênh lệch mạnh nhất giữa các kênh. `abs` là chênh lệch tuyệt đối theo từng kênh. `ssim_map` cho thấy cấu trúc thay đổi ở đâu (sáng = đã đổi). | Bắt đầu với `heatmap`. |
| `amplify` | 4.0 | Độ khuếch đại chênh lệch để thay đổi nhỏ hiện rõ. `ssim_map` không dùng. | 1 đến 100. Tăng để lộ ra các chỉnh sửa tinh tế. |
| `window_size` | 11 | Kích thước cửa sổ SSIM (số lẻ). Lớn hơn = đánh giá cấu trúc thô hơn. | 3 đến 31. Tự giảm với ảnh nhỏ. |
| `mask` (tùy chọn) | không có | Giới hạn mọi con số trong vùng này, có trọng số theo mask. | Bản thân ảnh diff không bị mask che. |

**Đầu ra:** `diff` (IMAGE), `ssim` (FLOAT), `psnr` (FLOAT), `report` (STRING).

!!! note "Batch"
    Đầu ra `ssim` và `psnr` chỉ tính cho ảnh đầu tiên của batch. `report` liệt kê mọi ảnh.

## Mẹo khi chỉnh ảnh

- `amplify` quá cao sẽ làm sai số làm tròn nhỏ xíu cũng sáng lên. Hãy đối chiếu với các con số, đừng chỉ nhìn hình.
- So sánh cùng với mask bạn đã dùng để chỉnh. Nếu `changed_pct` trong mask cao mà ngoài mask gần bằng 0, chỉnh sửa đã nằm đúng chỗ.
- Nếu mask hoàn toàn đen, các con số trả về giá trị trung tính (SSIM 1, PSNR 100, còn lại 0).
- Dùng `ssim_map` để bỏ qua việc sáng tối đổi đều và chỉ thấy thay đổi về cấu trúc.

??? note "Bên trong node hoạt động thế nào"
    SSIM được tính bằng `kornia.metrics.ssim` trên cửa sổ đã chọn, rồi lấy trung bình qua các kênh. PSNR là `-10 log10(MSE)`, giới hạn trần 100 dB. `heatmap` áp colormap `turbo` của Kornia lên chênh lệch kênh lớn nhất nhân với `amplify`. Khi có mask, mọi chỉ số là trung bình có trọng số theo mask. Hai ảnh phải cùng kích thước (không tự resize); batch 1 được nhân ra. Mã nguồn: `core/qa_diff.py`.

## Node liên quan

- [Mask Preview](mask-preview.md) để kiểm tra mask bằng mắt
- [Smart Sharpen](smart-sharpen.md) và [Skin Smoother](edge-aware-smoother.md), các bước chỉnh điển hình nên kiểm tra
- [Frequency Separation](frequency-separation.md) cho các chỉnh sửa mà việc so sánh trước/sau rất đáng làm

## Workflow mẫu

[Workflow mẫu QA Image Diff](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-44-qa-image-diff.json). Hãy chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/` trước.
