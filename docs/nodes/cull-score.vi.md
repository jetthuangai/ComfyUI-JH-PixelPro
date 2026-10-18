---
title: Cull Score
description: Chấm điểm một batch ảnh theo độ nét, phơi sáng, mắt nhắm và ảnh trùng, rồi giữ lại những tấm tốt nhất.
---

# Cull Score

Thả cả buổi chụp vào. Node chấm điểm từng tấm, bỏ ảnh mờ, phơi sáng sai và ảnh gần giống nhau, rồi trả về các tấm nên giữ, tấm tốt nhất đứng đầu.

![Cull Score — một batch với tấm nét được giữ, tấm mờ và tấm cháy sáng bị loại](../assets/nodes/cull-score.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → qa → QA: Cull Score` |
| **Dùng cho** | Chọn nhanh từ một loạt ảnh chụp liên tiếp hoặc buổi chụp chân dung, loại ảnh mờ và cháy sáng, giữ một tấm trong nhiều tấm gần giống nhau |
| **Tốc độ** | Mỗi ảnh được đo trên bản 1024 px, nên chi phí không phụ thuộc kích thước ảnh gốc |

## Khi nào dùng

Sau buổi chụp, bạn thường có hàng chục khung hình gần giống nhau. Node này là công cụ hỗ trợ chọn lọc (culling) để lập danh sách rút gọn. Nó không đánh giá được biểu cảm hay bố cục, nên bạn vẫn cần tự xem lại các tấm được chọn trước khi giao ảnh.

Node kiểm tra bốn thứ: độ nét, phơi sáng, mắt nhắm (khi tìm thấy khuôn mặt) và ảnh trùng. Độ nét được so với tấm nét nhất của cùng một cảnh, nên một bức chân dung mịn không bị thua một ảnh đường phố nhiều chi tiết chỉ vì nằm chung batch.

## Bắt đầu nhanh

1. Load một batch ảnh cùng kích thước (ví dụ bằng Load Image Batch).
2. Nối vào **QA: Cull Score**.
3. Đưa `selected` vào Preview Image hoặc Save Image. Muốn xem số liệu thì đọc `report` bằng một node hiển thị văn bản.
4. Đặt `top_k` bằng số ảnh bạn muốn, hoặc để 0 để giữ mọi ảnh đạt `min_score`.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `images` | | Batch ảnh, tất cả cùng kích thước. | |
| `top_k` | 0 | 0 đến 1000. Giữ tối đa chừng này ảnh. 0 là giữ mọi ảnh đạt yêu cầu. | Đặt 1 nếu chỉ muốn tấm tốt nhất. |
| `min_score` | 0.5 | 0 đến 1. Loại ảnh có điểm thấp hơn mức này. | Tăng để lọc khắt khe hơn, giảm để giữ nhiều hơn. |
| `check_eyes` | bật | Trừ điểm ảnh có khuôn mặt nhắm mắt. Cần MediaPipe, nếu thiếu thì bước này được bỏ qua. | Tắt với cảnh không có người. |

**Đầu ra**

| Đầu ra | Là gì |
|---|---|
| `selected` | Các ảnh được giữ, tấm tốt nhất đứng đầu. Không bao giờ rỗng: nếu không tấm nào đạt thì trả về tấm tốt nhất. |
| `report` | JSON, mỗi ảnh đầu vào một mục: `index`, `score`, `sharpness`, `exposure`, `eyes_closed`, `faces`, `duplicate_of`, `keep` và `energy`. |
| `kept` | Số ảnh được giữ (INT). |

## Mẹo khi chỉnh ảnh

- Điểm gồm 0.6 cho độ nét và 0.4 cho phơi sáng, trừ 0.4 nếu nhắm mắt. Xem `report` để biết vì sao một khung bị loại.
- Trong `report`, `duplicate_of` cho biết ảnh gần giống đã được gộp vào khung nào. Mỗi nhóm chỉ giữ tấm tốt nhất.
- Ảnh trong một batch phải cùng kích thước. Nếu buổi chụp lẫn nhiều kích thước, hãy resize trước.
- Nền mờ với chủ thể nét vẫn được tính là nét. Khi tìm thấy khuôn mặt, node đo độ nét trên vùng mặt.
- Đưa `selected` vào chuỗi chỉnh ảnh, ví dụ [Auto Tone](auto-tone.md), để bạn chỉ phải chỉnh những tấm đã chọn.
- `eyes_closed` là `null` trong report khi tắt kiểm tra mắt hoặc không có MediaPipe.

??? note "Bên trong node hoạt động thế nào"
    Mỗi ảnh được đo trên bản 1024 px. Độ nét là năng lượng Laplacian theo từng ô của lưới 8 x 8, lấy phân vị thứ 90, nên chủ thể nét trên nền mờ vẫn được điểm cao. Khi tìm thấy khuôn mặt (YuNet), node đo trong các khung mặt. Giá trị được đưa lên thang log và so với tấm nét nhất của cùng một cảnh, nhận biết bằng difference hash gần giống nhau. Tấm mờ hơn tấm đó 10 lần thì phần độ nét tương đối bằng 0.

    Phơi sáng bị trừ điểm khi có vùng highlight cháy, vùng shadow bị nén đen, và độ sáng trung bình lệch xa mức xám trung tính. Mắt nhắm dùng eye aspect ratio (Soukupova và Cech, 2016) tính từ landmark của MediaPipe, ngưỡng 0.18.

    Ảnh trùng được gom nhóm bằng difference hash 64-bit. Các ảnh cách nhau không quá 10 bit thuộc cùng nhóm và chỉ giữ tấm tốt nhất. Mã nguồn: `core/cull.py`, wrapper `nodes/cull_node.py`.

## Node liên quan

- [QA Image Diff](qa-image-diff.md): đo hai ảnh khác nhau bao nhiêu
- [Face Detect Lite](face-detect-lite.md): bộ tìm khuôn mặt dùng để đo độ nét trên mặt
- [Auto Tone](auto-tone.md): chỉnh các ảnh đã chọn
- [Contact Sheet](lightroom-preset-contact-sheet.md): xem nhiều kết quả cạnh nhau

## Workflow mẫu

[Workflow mẫu Cull Score](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-63-cull-score.json) dựng một batch nhỏ từ ảnh chân dung mẫu (bản nét, bản mờ và bản cháy sáng) rồi hiển thị ảnh được chọn và report. Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/` (Pexels License).
