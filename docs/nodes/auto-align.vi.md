---
title: Auto Align
description: Tự động căn một ảnh khớp với ảnh khác bằng cách so khớp đặc trưng rồi warp vào khung tham chiếu.
---

# Auto Align

Tìm những điểm giống nhau giữa hai ảnh, tính xem ảnh này phải dịch, xoay hay nghiêng thế nào để khớp ảnh kia, rồi warp vào đúng chỗ. Không cần đánh dấu thủ công.

![Auto Align — trước / sau](../assets/nodes/auto-align.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → geometry → Geometry: Auto Align` |
| **Dùng cho** | Căn các ảnh bracket phơi sáng, cặp ảnh trước/sau retouch, và các ảnh phải khớp ảnh tham chiếu trước khi match màu hoặc blend layer |

## Khi nào dùng

Dùng khi hai ảnh cùng một cảnh nhưng không khớp từng pixel, ví dụ ảnh chụp cầm tay hoặc một ảnh crop và ảnh gốc. Hãy căn trước rồi mới so sánh hoặc blend: [Color Matcher](color-matcher.md) và [Layer Add](layer-add.md) hoạt động tốt hơn nhiều khi hai ảnh cùng nội dung ở cùng vị trí.

## Bắt đầu nhanh

1. Nối ảnh cần di chuyển vào `image` và khung đích vào `reference`.
2. Giữ `method = sift` và `model = homography`.
3. Xem `aligned` trong `Preview Image`, và kiểm tra `inliers` cao hơn `min_inliers` một cách thoải mái.
4. Dùng `valid_mask` để bỏ qua phần viền trống.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `image` | — | Ảnh cần warp. | |
| `reference` | — | Khung đích. Đầu ra có kích thước này. | Một ảnh, hoặc cùng số lượng batch với `image`. |
| `method` | `sift` | `sift`: điểm đặc trưng cổ điển, không cần tải thêm. `loftr`: bộ so khớp học máy cho cặp ảnh khó. | LoFTR tải khoảng 45 MB ở lần dùng đầu. |
| `model` | `homography` | `homography` xử lý thay đổi phối cảnh. `affine` chỉ cho xoay, co giãn, nghiêng và dịch. | `affine` ổn định hơn với cảnh phẳng hoặc ít điểm khớp. |
| `max_side` | `1024` (256 đến 2048) | Cạnh dài nhất dùng để so khớp. Bước warp cuối dùng độ phân giải đầy đủ. | Thấp hơn thì nhanh hơn. |
| `ransac_threshold` | `3.0` (0.5 đến 20) | Một điểm khớp được phép lệch bao nhiêu pixel (ở kích thước so khớp) mà vẫn được tính. | Tăng với cặp ảnh hơi mờ, giảm khi cần khớp chặt. |
| `min_inliers` | `12` (4 đến 500) | Ít điểm khớp đồng thuận hơn mức này thì kích hoạt phương án dự phòng. | |

| Đầu ra | Kiểu | Ý nghĩa |
|---|---|---|
| `aligned` | `IMAGE` | `image` đã warp theo kích thước ảnh tham chiếu. |
| `valid_mask` | `MASK` | Bằng 1 nơi ảnh warp có pixel thật, 0 nơi khung còn trống. |
| `matrix_json` | `STRING` | Ma trận 3×3 (mỗi ảnh trong batch một ma trận) tính theo pixel độ phân giải đầy đủ. |
| `inliers` | `INT` | Số điểm khớp đồng thuận của ảnh đầu tiên trong batch. |

## Mẹo khi chỉnh ảnh

- **sift hay loftr:** bắt đầu với `sift`. Thử `loftr` khi SIFT tìm được ít điểm khớp (ánh sáng đổi nhiều, ảnh mất nét, texture lặp lại).
- **valid_mask:** sau khi xoay hoặc dịch, mép của `aligned` bị đen. Dùng mask này để crop, feather, hoặc giới hạn vùng mà bước match màu lấy thống kê.
- Nếu node không tìm đủ điểm khớp đồng thuận, nó không dừng lại. Nó ghi cảnh báo, co ảnh về kích thước ảnh tham chiếu và trả `valid_mask` toàn trắng. Luôn kiểm tra `inliers`.
- Chủ thể chuyển động (người, lá cây) bị bỏ qua miễn là nền có đủ texture.
- Với batch, `inliers` chỉ báo ảnh đầu tiên; hãy kiểm tra các ảnh còn lại trong `matrix_json` (ma trận chỉ có co giãn nghĩa là ảnh đó đã dùng phương án dự phòng).

??? note "Bên trong node hoạt động thế nào"
    Cả hai ảnh được chuyển thành ảnh xám thu nhỏ về `max_side`. SIFT trích xuất keypoint và giữ các cặp khớp là láng giềng gần nhất hai chiều; LoFTR chạy một mạng neural. RANSAC khớp mô hình đã chọn trên các tập con ngẫu nhiên và giữ phép khớp được nhiều điểm đồng thuận nhất, sau đó các điểm inlier được tinh chỉnh bằng bình phương tối thiểu và ma trận được quy về độ phân giải đầy đủ. Bước warp là bilinear (`warp_perspective` của Kornia). `valid_mask` là cùng phép warp áp dụng lên ảnh toàn 1. Nguồn: `core/auto_align.py` và `nodes/auto_align_node.py`.

## Node liên quan

- [Color Matcher](color-matcher.md)
- [Layer Add](layer-add.md)
- [Lens Distortion](lens-distortion.md)
- [QA Image Diff](qa-image-diff.md)

## Workflow mẫu

[Workflow mẫu Auto Align](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-39-auto-align.json). Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/`.

![Kết quả Auto Align](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/N-39-auto-align-preview.png)
