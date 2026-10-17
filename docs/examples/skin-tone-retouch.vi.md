---
title: Retouch tông da
description: Tách ảnh chân dung thành mask shadow, midtone và highlight rồi retouch từng dải tông riêng.
---

# Retouch tông da

Da hiếm khi cần cùng một cách sửa ở mọi chỗ. Ví dụ này cho thấy cách tạo ba mask theo tông từ ảnh chân dung bằng [Skin Tone Tri-Region](../nodes/skin-tone-tri-region.md), rồi dùng chúng để xử lý bóng dầu, tông da chính và vùng tối một cách riêng biệt.

## Chuỗi node

```text
Load Image ──────────────────────────────┐
                                         ▼
Mask người / da (tùy chọn) ─────► Skin Tone Tri-Region
                                     ├─ shadow_mask
                                     ├─ midtone_mask
                                     └─ highlight_mask
```

Mask da có thể đến từ bất kỳ node phân vùng nào. Workflow mẫu dùng một node tách người từ pack custom node khác. Nếu không có mask da, cả khung hình sẽ được tách theo độ sáng, kể cả tóc, quần áo và hậu cảnh.

## Thử ngay

1. Tải [workflow mẫu](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-35-skin-tone-tri-region.json) và kéo thả vào canvas ComfyUI.
2. Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/` rồi chọn ảnh trong **Load Image**.
3. Chạy và xem ba bản xem trước mask. Vùng trắng là nơi mỗi dải tông được áp dụng.

## Dùng ba mask

- **`highlight_mask`:** làm dịu bóng dầu hoặc kéo lại màu vùng sáng. Đưa vào một bước chỉnh màu, hoặc vào [Face Beauty Blend](../nodes/face-beauty-blend.md) cùng một bản ảnh đã làm mịn.
- **`midtone_mask`:** chỉnh tông da chính. Khớp màu với ảnh tham chiếu bằng [Color Matcher Region](../nodes/color-matcher-region.md), hoặc áp một LUT qua [Face Beauty Blend](../nodes/face-beauty-blend.md).
- **`shadow_mask`:** nâng sáng hoặc làm ấm vùng tối, hoặc giới hạn khử noise chỉ ở vùng tối.

Ba mask luôn cộng lại bằng mask da của bạn, nên các chỉnh sửa qua chúng không chồng lấn và không để sót vùng nào.

## Thông số nên thử trước

| Thông số | Mặc định | Khi nào đổi |
|---|---|---|
| `shadow_cutoff` | `0.33` | Hạ xuống nếu mask shadow chiếm quá nhiều khuôn mặt. |
| `highlight_cutoff` | `0.66` | Nâng lên để chỉ nhắm vào vùng bóng sáng nhất. |
| `soft_sigma` | `1.0` | Tăng để chuyển tiếp giữa các dải êm hơn. |

Xem chi tiết từng thông số tại [Skin Tone Tri-Region](../nodes/skin-tone-tri-region.md). Để kiểm tra mask trước khi dùng, hãy thêm [Mask Preview](../nodes/mask-preview.md).
