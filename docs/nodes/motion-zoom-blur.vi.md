---
title: Motion / Zoom Blur
description: Thêm hiệu ứng mờ chuyển động theo đường thẳng hoặc zoom burst, có thể chỉ áp lên vùng trắng của mask.
---

# Motion / Zoom Blur

Giả lập máy ảnh đang chuyển động hoặc đang zoom trong lúc phơi sáng. Chỉ làm mờ phông để có cảm giác panning, hoặc kéo vệt sáng từ rìa khung về một điểm trung tâm.

![Motion / Zoom Blur — trước, motion 60 px chỉ trên phông, zoom 0.15](../assets/nodes/motion-zoom-blur.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → filters → Filters: Motion / Zoom Blur` |
| **Dùng cho** | Ảnh panning, hiệu ứng tốc độ, zoom burst, thêm năng lượng cho phông tĩnh |
| **Tốc độ** | Ở 4K trên GPU: motion 200 px khoảng 1,4 giây, zoom khoảng 0,1 giây, dưới 1 GB VRAM. |

## Khi nào dùng

- **`motion`** lấy trung bình ảnh dọc theo một đường thẳng, như máy ảnh hoặc chủ thể di chuyển trong lúc phơi sáng. `angle` đặt hướng, `distance` đặt độ dài.
- **`zoom`** lấy trung bình dọc theo các tia hướng về điểm trung tâm, như xoay vòng zoom khi đang phơi sáng. Gần tâm gần như không đổi; rìa khung bị kéo vệt mạnh nhất.

Muốn kiểu panning, hãy chỉ làm mờ phông. Nối mask chủ thể đã đảo ngược vào `mask`: hiệu ứng chỉ áp ở nơi mask trắng.

## Bắt đầu nhanh

1. Load Image → **Filters: Motion / Zoom Blur** → Preview Image.
2. Motion: đặt `mode` là `motion`, `angle` 0 để vệt ngang, thử `distance` 60.
3. Zoom: đặt `mode` là `zoom` và giữ `zoom` ở 0.15, tâm ở 0.5 / 0.5.
4. Tùy chọn: nối mask chủ thể đã đảo ngược vào `mask` để chủ thể vẫn nét.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `mode` | `motion` | `motion` = mờ theo một hướng. `zoom` = vệt hướng về điểm trung tâm. | |
| `angle` | 0 | -180 đến 180 độ. 0 = ngang, 90 = dọc. Chỉ dùng cho `motion`. | Chọn theo hướng chủ thể di chuyển. |
| `distance` | 20 | 0 đến 1000 pixel. Độ dài vệt mờ của `motion`. | Tăng theo kích thước ảnh: 60 px ở 2K là nhẹ, 200 px ở 4K là mạnh. |
| `zoom` | 0.15 | 0 đến 0.9. Rìa khung bị kéo vệt trên đoạn bằng phần này của khoảng cách tới tâm. Chỉ dùng cho `zoom`. | 0.1 đến 0.3 trông tự nhiên. |
| `center_x`, `center_y` | 0.5, 0.5 | 0 đến 1. Điểm mà các vệt zoom hướng tới, theo chiều ngang và dọc khung. Chỉ dùng cho `zoom`. | Đặt lên mặt hoặc mắt chủ thể. |
| `mask` (tùy chọn) | không | Chỉ làm mờ ở nơi mask trắng; chỗ khác giữ nguyên. | Dùng mask chủ thể đã đảo ngược. |

**Đầu ra**

| Đầu ra | Là gì |
|---|---|
| `image` | Ảnh đã làm mờ, cùng kích thước ảnh vào. `motion` với `distance` dưới 0.5, hoặc `zoom` bằng 0, trả về ảnh vào không đổi. |

## Mẹo khi chỉnh ảnh

- Đảo ngược mask chủ thể bằng [Mask Combine](mask-combine.md) và làm mềm viền bằng [Mask Edge Smoother](mask-edge-smoother.md) để chủ thể nét không bị viền cứng.
- Làm mờ trước, rồi mới làm nét bằng [Smart Sharpen](smart-sharpen.md) chỉ trên chủ thể.
- `distance` rất dài trên phông nhiều chi tiết sẽ thành một vệt nhòe. Tăng từ từ.
- Đặt `center_x` và `center_y` lên chủ thể để zoom burst kéo mắt người xem vào.

??? note "Bên trong node hoạt động thế nào"
    Cả hai chế độ đều là box blur dọc theo một đường một chiều. Một box `n1 × n2` điểm lấy mẫu tương đương một box `n1` điểm cách nhau 1 rồi một box `n2` điểm cách nhau `n1`, nên vệt mờ dài D pixel chỉ cần khoảng 2·√D lần warp thay vì D. Zoom dùng cùng mẹo này trong không gian log của bán kính, nơi các phép scale cộng dồn. Mỗi lần warp là một `grid_sample` bilinear với phản xạ (reflection) ở mép ảnh, và phép lấy trung bình giữ nguyên độ sáng chung. Bộ nhớ tỉ lệ với số pixel. ComfyUI đưa ảnh vào trên CPU; node tự chuyển phần tính toán sang GPU nếu có và quay về CPU khi hết bộ nhớ GPU. Mask tùy chọn hòa kết quả với ảnh gốc. Mã nguồn: `core/directional_blur.py`.

## Node liên quan

- [Smart Sharpen](smart-sharpen.md) để làm nét chủ thể sau đó
- [Mask Combine](mask-combine.md) để đảo ngược hoặc kết hợp mask
- [Mask Edge Smoother](mask-edge-smoother.md) để làm mềm viền mask
- [Skin Smoother](edge-aware-smoother.md) cho một kiểu làm mềm khác

## Workflow mẫu

[Workflow mẫu Motion / Zoom Blur](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-57-motion-zoom-blur.json) minh họa motion blur chỉ trên phông và zoom burst. Chép `workflows/sample_portrait.jpg` và `workflows/sample_binary_mask.png` vào `ComfyUI/input/`.
