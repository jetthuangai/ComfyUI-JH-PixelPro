---
title: Face Detect Lite
description: Bộ nhận diện khuôn mặt nhanh chạy trên GPU, cho 5 điểm mốc, điểm tin cậy thật và mask khuôn mặt. Không cần MediaPipe.
---

# Face Detect Lite

Bộ tìm khuôn mặt nhanh chạy trên GPU. Node trả về cùng 5 điểm mốc như [Face Detect](face-detect.md), kèm một mask khuôn mặt làm sẵn.

![Face Detect Lite — trước / sau](../assets/nodes/face-detect-lite.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → face → Face: Detect Lite (YuNet GPU)` |
| **Dùng để** | Xử lý hàng loạt và xem trước nhanh; đưa dữ liệu cho [Facial Aligner](facial-aligner.md); giới hạn bước chỉnh sau trong vùng mặt |
| **Tốc độ** | Khoảng 31 ms trên RTX 5090, so với khoảng 84 ms của MediaPipe trên CPU (ảnh chân dung mẫu) |

## Khi nào dùng

Dùng Lite khi bạn cần biết vị trí mặt thật nhanh và không cần lưới 468 điểm. Node dùng mạng YuNet đi kèm Kornia, nên vẫn chạy được khi máy chưa cài MediaPipe. Có GPU thì chạy GPU, không thì tự chuyển sang CPU.

Hãy ở lại với [Face Detect](face-detect.md) và [Face Landmarks](face-landmarks.md) khi bạn cần lưới mặt dày (warp, chỉnh dáng) hoặc cần xử lý tốt hơn các khuôn mặt nghiêng, bị che một phần.

| | Detect Lite (YuNet) | Face Detect (MediaPipe) |
|---|---|---|
| Cần MediaPipe | Không | Có |
| `conf` trong `bbox_json` | Điểm thật của bộ nhận diện | Chính ngưỡng bạn đặt |
| Đầu ra mask khuôn mặt | Có (theo khung mặt) | Không |
| Điểm mắt | Tâm mắt | Khóe ngoài của mắt |
| Dùng cho các node 468 điểm | Không | Dùng Face Landmarks |

Cả hai đều dùng được với node căn chỉnh. Điểm mắt lệch nhau khoảng 10–20 % khoảng cách hai mắt, không ảnh hưởng đến việc căn chỉnh.

## Bắt đầu nhanh

1. Nối **Load Image** vào **Face: Detect Lite (YuNet GPU)**.
2. Nối `landmarks_json` vào ô `landmarks` của [Facial Aligner](facial-aligner.md). Aligner đọc khuôn mặt đầu tiên.
3. Tùy chọn: dùng `face_mask` để giới hạn bước chỉnh sau trong vùng mặt.

## Thông số

| Thông số | Mặc định | Tác dụng | Gợi ý |
|---|---|---|---|
| `mode` | `single_largest` | Mỗi ảnh một mặt lớn nhất, hoặc nhiều mặt xếp theo kích thước khung. | Chân dung: giữ mặc định. |
| `max_faces` | `1` (1–10) | Số mặt mỗi ảnh khi dùng `multi_top_k`. | Ảnh nhóm: tăng lên. |
| `confidence_threshold` | `0.6` (0.1–0.95) | Điểm tối thiểu của bộ nhận diện. | Tăng để loại nhận nhầm. |
| `nms_threshold` | `0.3` (0.1–0.9) | Mức chồng lấp mà trên đó các khung trùng được gộp lại. | Hạ xuống nếu một mặt bị ra nhiều khung. |
| `detect_size` | `640` (160–1024) | Cạnh dài của bản sao dùng để nhận diện. Không bao giờ phóng to. | Chân dung: giữ 640. Chỉ tăng với ảnh nhóm có mặt nhỏ. |

Đầu ra:

| Đầu ra | Kiểu | Ý nghĩa |
|---|---|---|
| `landmarks_json` | `STRING` | Mỗi mặt: `[mắt trái, mắt phải, đầu mũi, khóe miệng trái, khóe miệng phải]` theo pixel. |
| `bbox_json` | `STRING` | Mỗi mặt: `x, y, w, h` (pixel), `conf` (điểm thật) và `batch_index`. |
| `face_count` | `INT` | Số mặt được trả về. |
| `face_mask` | `MASK` | Bằng 1 bên trong các khung mặt được chọn, bằng 0 ở ngoài. |

## Mẹo khi chỉnh ảnh

- `face_mask` là hình khung chữ nhật chứ không phải đường viền khuôn mặt. Hãy làm mềm viền (feather) nó, ví dụ bằng [Mask Edge Smoother](mask-edge-smoother.md), trước khi dùng làm mask chỉnh sửa.
- `detect_size` lớn hơn không phải lúc nào cũng tốt hơn. Ở giá trị rất cao, bộ nhận diện có thể tách đôi hoặc đặt sai khung mặt lớn, và nhận nhầm trên ảnh cận mặt.
- Một khung hình đen hoàn toàn có thể gây nhận diện nhầm. Hãy dùng ảnh thật hoặc tăng ngưỡng.
- Cần chạy cả loạt ảnh chân dung? Lite là lựa chọn hợp lý: nhanh hơn và không phải cài thêm gì.

??? note "Bên trong node hoạt động thế nào"
    Ảnh được thu nhỏ để cạnh dài bằng `detect_size` (ảnh nhỏ hơn giữ nguyên), chạy qua YuNet bằng `kornia.contrib.FaceDetector`, rồi khung và điểm mốc được đổi ngược về pixel của ảnh gốc. Các mặt được sắp theo diện tích khung. Ảnh nhỏ hơn 64 px được đệm bằng màu trung bình, chỉ để nhận diện. Bộ nhận diện được cache theo từng thiết bị. Kornia tải trọng số một lần vào cache torch hub; nếu tải lỗi, node báo lỗi rõ ràng. Mã nguồn nằm ở `core/face_detect_lite.py`.

## Node liên quan

- [Face Detect](face-detect.md) — phiên bản MediaPipe.
- [Facial Aligner](facial-aligner.md) — bước kế tiếp trong chuỗi.
- [Mask Preview](mask-preview.md) — xem `face_mask` dưới dạng lớp phủ.
- [Tổng quan nhóm Khuôn mặt](../categories/face.md)

## Workflow mẫu

[Workflow mẫu Face Detect Lite](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-38-face-detect-lite.json). Hãy chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/` trước.
