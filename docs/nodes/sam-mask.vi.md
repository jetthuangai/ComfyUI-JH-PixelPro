---
title: SAM One-Click
description: Tách bất kỳ vật thể nào bằng Segment Anything chỉ từ vài điểm "click", khung khuôn mặt hoặc một mask thô.
---

# SAM One-Click

Chọn một người, một sản phẩm hay bất kỳ vật thể nào chỉ trong một bước. Bạn chỉ cho node biết vật thể nằm ở đâu, Segment Anything sẽ vẽ vùng chọn sát viền, giống công cụ Object Selection trong Photoshop.

![SAM One-Click — trước / sau](../assets/nodes/sam-mask.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → mask → Mask: SAM One-Click` |
| **Dùng cho** | Chọn chủ thể để thay nền, tách sản phẩm, tạo mask đầu tiên rồi tinh chỉnh phần tóc |
| **Tốc độ** | Ảnh 2048 × 1365 trên RTX 5090: `vit_b` khoảng 0,2 giây và 2,9 GB VRAM, `mobile_sam` khoảng 0,04 giây và 0,35 GB. `mobile_sam` trên CPU mất khoảng 1,5 giây. Lần chạy đầu còn tốn thời gian nạp model. |

## Khi nào dùng

Segment Anything (SAM) là model của Meta đã học được "một vật thể trông như thế nào". Bạn không cần tô vùng chọn. Chỉ cần gợi ý, model tự tìm viền.

ComfyUI không có nút "click lên ảnh", nên gợi ý được nhập bằng chữ. Có ba cách:

- **Điểm (points).** Gõ vị trí một pixel nằm trên vật thể, ví dụ `[[1200, 700]]`. Điểm cần loại ra thì nhập vào `negative_points_json`.
- **Khung khuôn mặt.** Nối `bbox_json` từ [Face Detect](face-detect.md) hoặc [Face Detect Lite](face-detect-lite.md).
- **Mask thô.** Nối bất kỳ MASK nào vào `mask_hint`. Node dùng khung bao và một điểm nằm bên trong mask đó.

Nếu bạn không đưa gì, node sẽ click vào giữa ảnh.

## Bắt đầu nhanh

1. Load Image → **Mask: SAM One-Click**.
2. Ở `points_json` nhập một điểm trên chủ thể, ví dụ `[[0.5, 0.45]]` (xem hai cách ghi tọa độ bên dưới).
3. Nối `preview` vào Preview Image. Vùng chọn được phủ màu đỏ và có viền.
4. Chưa đúng? Thêm điểm vào phần còn thiếu ở `points_json`, hoặc thêm điểm vào phần bị chọn lẹm ở `negative_points_json`.

### Cách "click"

**Điểm theo pixel.** `[[x, y], [x, y]]`, gốc tọa độ ở góc trên bên trái. Bạn đọc vị trí từ bất kỳ trình xem ảnh nào. Giá trị hợp lệ từ 0 đến rộng − 1 và 0 đến cao − 1; điểm nằm đúng mép hoặc ngoài ảnh (ví dụ `x` bằng đúng chiều rộng) sẽ bị từ chối.

**Điểm theo tỉ lệ ảnh.** Nếu mọi số đều nằm trong khoảng 0 đến 1 và có ít nhất một số là phân số thật (như 0.5), node hiểu đó là tỉ lệ: `[[0.5, 0.4]]` là giữa ảnh theo chiều ngang và cách đỉnh 40 %. Cách này tiện vì dùng được cho mọi độ phân giải. Danh sách chỉ gồm 0 và 1, như `[[1, 1]]` hay `[[0, 0]]`, được hiểu là pixel vì không phân biệt được với pixel 1 hoặc pixel 0.

**Chỉ có điểm âm.** Được phép, nhưng khi đó SAM không có gì để chọn nên kết quả không đáng tin. Node ghi cảnh báo vào log. Hãy thêm một điểm dương, một khung hoặc mask hint.

**Từ khuôn mặt.** Nối `bbox_json` từ Face Detect. Node đọc danh sách khung `{x, y, w, h, ...}`. Khi chạy nhiều ảnh, node chọn khung có `batch_index` trùng với từng ảnh. Nếu các khung không có `batch_index` thì lấy khung đầu tiên.

**Từ mask thô.** Vẽ mask sơ sài (hình chữ nhật cũng được) hoặc lấy từ node khác rồi nối vào `mask_hint`. SAM sẽ làm nó khớp sát viền thật.

Bạn có thể kết hợp. Khung mặt cộng một điểm trên vai là cách hay dùng.

## Thông số

| Thông số | Mặc định | Tác dụng | Mẹo |
|---|---|---|---|
| `model` | `vit_b` | Chọn model Segment Anything. | `mobile_sam` cho nhanh, `vit_h` cho viền sạch nhất. |
| `points_json` | trống | Các điểm nằm trên vật thể. | Thường một đến ba điểm là đủ. |
| `negative_points_json` | trống | Các điểm nằm trên thứ không được chọn. | Dùng để loại cái ghế hay bàn tay khỏi mask. |
| `multimask` | bật | SAM đề xuất 3 mask và giữ mask có điểm dự đoán cao nhất. | Giữ bật khi chỉ click một điểm. Tắt khi đã có khung hoặc nhiều điểm. |
| `mask_threshold` | 0.0 | Ngưỡng cắt mask cuối. | Nhỏ hơn 0 làm mask nở ra, lớn hơn 0 làm mask co lại. Thử ±1. |
| `bbox_json` (tùy chọn) | trống | Khung mặt từ Face Detect / Face Detect Lite. | Nối dây, đừng gõ tay. |
| `mask_hint` (tùy chọn) | không | Một mask thô. | Mask hình chữ nhật là đủ. |
| `keep_model_loaded` (tùy chọn) | bật | Giữ model SAM trong VRAM giữa các lần chạy. | Tắt để giải phóng 0,4–7 GB ngay sau khi chạy; nút *Free memory* của ComfyUI không giải phóng được model này. |

### Đầu ra

| Đầu ra | Kiểu | Ý nghĩa |
|---|---|---|
| `mask` | MASK | Vùng chọn: 1 bên trong, 0 bên ngoài. Mỗi ảnh một mask. |
| `score` | FLOAT | Điểm tự đánh giá chất lượng của SAM (khoảng 0 đến 1). Nếu chạy nhiều ảnh thì lấy trung bình. Dưới 0,7 nên thêm điểm. |
| `preview` | IMAGE | Ảnh gốc phủ màu và viền mask để kiểm tra. |

### Các model

File nằm trong `ComfyUI/models/sams/`. File nào chưa có sẽ được tải ở lần dùng đầu tiên.

| Model | File | Dung lượng | VRAM (khoảng) | Ghi chú |
|---|---|---|---|---|
| `vit_b` | `sam_vit_b_01ec64.pth` | 375 MB | 3 GB | Mặc định, cân bằng. |
| `mobile_sam` | `mobile_sam.pt` | 39 MB | 0,4 GB | Nhanh nhất, chạy tốt trên CPU. Kém chính xác hơn khi nền rối. |
| `vit_l` | `sam_vit_l_0b3195.pth` | 1,2 GB | 5 GB (ước tính) | File tải về lớn. |
| `vit_h` | `sam_vit_h_4b8939.pth` | 2,5 GB | 7 GB (ước tính) | File tải về lớn. Chất lượng tốt nhất. |

Chỉ một model được giữ trong bộ nhớ. Đổi `model` sẽ giải phóng model trước khỏi VRAM. Thư mục theo `extra_model_paths.yaml` nếu có mục `sams` ở đó. Việc tải có giới hạn thời gian chờ, ghi vào file `.part` trước, và `vit_b`, `mobile_sam` được kiểm tra SHA256 (không có mã băm chính thức cho `vit_l` và `vit_h` nên chỉ kiểm tra dung lượng).

Segment Anything và MobileSAM dùng giấy phép Apache-2.0, cho phép dùng thương mại.

## Mẹo khi chỉnh ảnh

- Click một điểm trên chủ thể lớn đôi khi chỉ chọn một phần (chọn mặt mà không chọn thân). Hãy thêm điểm thứ hai trên phần còn lại, hoặc tắt `multimask` và thêm khung.
- Với chân dung, khung mặt là điểm khởi đầu tốt nhưng chỉ bao quanh khuôn mặt. Thêm một điểm trên thân, hoặc đưa mask thô của cả người vào `mask_hint`.
- Viền của SAM sạch nhưng cứng (nhị phân). Với tóc và lông, hãy nối tiếp vào [Sub-Pixel Mask Refiner](sub-pixel-mask-refiner.md), hoặc tạo [Trimap](trimap-builder.md) rồi đưa vào [Alpha Matte](alpha-matte.md) để có tóc mềm thật sự.
- Dùng `preview` khi đặt điểm, sau đó lấy `mask` để làm việc.
- Ảnh rất lớn được thu nhỏ trong model (cạnh dài 1024 px) rồi mask được phóng lại. Chi tiết tóc mảnh không thể chỉ nhờ SAM.

??? note "Bên trong node hoạt động thế nào"
    Node dùng `VisualPrompter` của Kornia với model `Sam`. Ảnh (giá trị 0 đến 1) được resize để cạnh dài là 1024, chuẩn hóa rồi đi qua image encoder một lần cho mỗi ảnh. Điểm được đưa vào dưới dạng keypoint có nhãn (1 là dương, 0 là âm), khung là box `xyxy`, đều theo tọa độ pixel gốc; Kornia tự quy đổi sang khung hình của model.

    Decoder trả về tối đa 3 bản đồ logit độ phân giải thấp, mỗi bản có một điểm IoU dự đoán. Bản có IoU dự đoán cao nhất được phóng về kích thước gốc rồi cắt tại `mask_threshold` (là logit, 0 là mặc định của SAM).

    Mask hint đóng góp khung bao (trừ khi `bbox_json` đã có khung) và một điểm dương tại trọng tâm, được dời đến pixel mask gần nhất nếu trọng tâm rơi ngoài mask. Khi bật multimask, các mask ứng viên phủ hơn 90 % ảnh bị bỏ qua, trừ khi mọi ứng viên đều như vậy. Chỉ một model được giữ trong bộ nhớ (đổi model hoặc thiết bị sẽ loại model cũ); model chạy trên CUDA khi có, nếu không thì CPU. Mỗi ảnh trong batch được encode riêng.

## Node liên quan

- [Face Detect Lite](face-detect-lite.md) — cho khung khuôn mặt.
- [Sub-Pixel Mask Refiner](sub-pixel-mask-refiner.md) — làm sạch và mượt viền SAM.
- [Trimap Builder](trimap-builder.md) và [Alpha Matte](alpha-matte.md) — tóc và lông mềm.
- [Mask Preview](mask-preview.md) — thêm các kiểu xem trước.

## Workflow mẫu

[Workflow mẫu SAM One-Click Mask](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-46-sam-one-click-mask.json) đưa khung mặt của Face Detect Lite vào SAM, rồi lấp lỗ bằng Mask Cleanup và hiển thị ảnh tách nền. Workflow chỉ dùng node lõi của ComfyUI và pack này. Chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/`.

![Kết quả: Workflow mẫu SAM One-Click Mask](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/N-46-sam-one-click-mask-preview.png)
