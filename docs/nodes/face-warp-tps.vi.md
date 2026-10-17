---
title: Face Warp TPS
description: Chỉnh dáng khuôn mặt bằng một trường biến dạng thin-plate-spline mượt, không có đường nối tam giác.
---

# Face Warp TPS

Dịch chuyển khuôn mặt từ landmark nguồn sang landmark đích bằng một biến dạng liền mạch, mềm như tấm cao su. Dùng khi bạn chỉnh dáng nhẹ nhàng và không muốn lộ đường nối hay chỗ gấp khúc của tam giác.

![Face Warp TPS — trước / sau](../assets/nodes/face-warp-tps.jpg)

| | |
|---|---|
| **Trong ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → face → Face Warp (TPS smooth)` |
| **Dùng để** | Thon mặt, chỉnh mắt hoặc hàm; đưa khuôn mặt về dáng của ảnh khác; những chỉnh sửa cần trông tự nhiên |
| **Tốc độ** | Khoảng 19 ms trên CUDA và khoảng 0,85 s trên CPU với ảnh 1536×2048, 468 điểm (RTX 5090 / CPU máy bàn) |

## Khi nào dùng

Thin-plate spline (TPS) là một mặt cong mượt, uốn như tấm kim loại mỏng: đi qua các điểm điều khiển của bạn và phẳng nhất có thể ở giữa chúng. Áp lên ảnh, nó tạo ra một biến dạng liên tục, không có cạnh tam giác nào lộ ra.

Chọn TPS khi cần thay đổi nhẹ nhàng, tự nhiên và muốn tốc độ GPU. Chọn [Face Warp](face-warp.md) (Delaunay) khi thay đổi phải nằm gọn trong lưới mặt, hoặc khi cả 468 điểm đều phải được tôn trọng ở mức cục bộ.

| | Delaunay ([Face Warp](face-warp.md)) | TPS (node này) |
|---|---|---|
| Trường biến dạng | Mỗi tam giác một mảnh affine | Một trường mượt toàn cục |
| Đường nối / chỗ gấp | Có thể lộ ở cạnh tam giác khi dịch chuyển lớn | Không có |
| Vùng ngoài khuôn mặt | Giữ nguyên | Mờ dần một cách mượt (được ghim bởi `anchor_border`) |
| Chạy trên | CPU (OpenCV) | GPU nếu có |

## Bắt đầu nhanh

1. Chạy [Face Landmarks](face-landmarks.md) trên ảnh của bạn và nối `landmarks` vào `src_landmarks`.
2. Chạy [Face Landmarks](face-landmarks.md) trên ảnh đích (ảnh thứ hai, hoặc bản zoom) và nối vào `dst_landmarks`.
3. Giữ giá trị mặc định và xem `warped` trong **Preview Image**.
4. Nếu vẫn thấy gợn sóng, hãy tăng `smoothness`.

## Thông số

| Thông số | Mặc định | Tác dụng | Gợi ý |
|---|---|---|---|
| `max_points` | `468` (16–468) | Số cặp landmark tối đa dùng cho phép warp. Các cặp có dịch chuyển luôn được giữ lại. | Giữ 468. Chỉ hạ khi cần tốc độ, vì ít điểm ghim hơn khiến các vùng lân cận bị trôi. |
| `anchor_border` | bật | Ghim các điểm dọc viền ảnh để hậu cảnh không bị trôi. | Chỉ tắt khi cả khung hình cần di chuyển cùng khuôn mặt (chỉ dịch hoặc co giãn). |
| `smoothness` | `1.0` (0–20) | `0` cho mọi landmark rơi đúng vị trí. Giá trị cao hơn hấp thụ nhiễu khi nhận diện. | Dùng `0` với điểm chỉnh tay. Giữ từ 1 trở lên khi điểm đích lấy từ ảnh khác. Tăng nếu vẫn còn gợn sóng. |

Đầu ra: `warped` (`IMAGE`), ảnh đã chỉnh dáng.

## Mẹo khi chỉnh ảnh

- Landmark từ lần nhận diện thứ hai khác nguồn không chỉ do thay đổi dáng thật mà còn do nhiễu của bộ nhận diện. Trên ảnh chân dung mẫu, nhiễu này khoảng 13–17 px (trung vị) với bản zoom 3–6 %. Ép khớp chính xác sẽ làm ảnh bị gập, nên hãy giữ `smoothness` từ 1 trở lên cho loại điểm đích này.
- Di chuyển cả nhóm landmark lân cận cùng nhau. Kéo vài điểm đơn lẻ sẽ tạo hiệu ứng cao su loang rộng.
- Thiếu khuôn mặt (điểm NaN) hoặc có ít hơn 3 cặp hợp lệ thì ảnh được trả về nguyên vẹn.
- Warp trước, làm da sau, để các bước xử lý texture theo đúng hình dáng cuối cùng.

??? note "Bên trong node hoạt động thế nào"
    Spline (kernel `r² log r` cộng một phần affine) được fit trên ánh xạ từ đầu ra về đầu vào, nên mỗi pixel đầu ra lấy mẫu từ ảnh nguồn đúng chỗ spline chỉ tới. Trường biến dạng được tính chính xác trên lưới bước 4 px, phóng lên kích thước đầy đủ bằng bicubic, rồi dùng để lấy mẫu lại ảnh bằng bilinear. Với `smoothness = 0`, các điểm điều khiển rơi cách đích khoảng 0,6 px trên ảnh chân dung thật 1536×2048. Các cặp có dịch chuyển (hơn 0,5 px) không bao giờ bị loại khi hạ `max_points`; phần còn lại được lấp một nửa bằng các cặp không dịch chuyển gần vùng chỉnh nhất, một nửa bằng các cặp cách đều. Khi bật `anchor_border`, các cặp phụ có nguồn trùng đích (4 góc cộng 3 điểm mỗi cạnh) ghim viền ảnh; `smoothness` cao hơn nới lỏng chúng đôi chút (dưới 1 px ở giá trị mặc định). Bộ nhớ đỉnh khoảng 480 MiB ở kích thước đó, phần lớn là buffer ảnh. Mã nguồn nằm ở `core/face_warp_tps.py`.

## Node liên quan

- [Face Warp](face-warp.md) — lựa chọn dựa trên tam giác.
- [Face Landmarks](face-landmarks.md) — cung cấp cả hai bộ điểm.
- [Face Beauty Blend](face-beauty-blend.md) — giới hạn phép warp trong vùng mặt bằng mask.

## Workflow mẫu

[Workflow mẫu Face Warp TPS](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-42-face-warp-tps.json). Hãy chép `workflows/sample_portrait.jpg` vào `ComfyUI/input/` trước.
