# Khuôn mặt

Nhóm Face gồm mọi thứ một lần retouch chân dung cần quanh khuôn mặt: tìm mặt, chỉnh dáng, dán bản đã chỉnh về chỗ cũ và tách da theo dải tông. Phần lớn các chuỗi đều theo một mẫu chung (mô tả bên dưới), giúp bạn đưa một vùng mặt nhỏ, thẳng cho AI hoặc bước retouch xử lý, rồi trả về ảnh đầy đủ mà không đụng đến phần còn lại.

<div class="grid cards" markdown>

-   ![](../assets/nodes/face-detect.jpg)

    **[Face Detect](../nodes/face-detect.md)**

    Tìm khuôn mặt bằng MediaPipe, trả về 5 điểm mốc cho mỗi mặt.

-   ![](../assets/nodes/face-detect-lite.jpg)

    **[Face Detect Lite](../nodes/face-detect-lite.md)**

    Bộ nhận diện YuNet nhanh, cho 5 điểm mốc, điểm tin cậy thật và mask khuôn mặt.

-   ![](../assets/nodes/face-landmarks.jpg)

    **[Face Landmarks](../nodes/face-landmarks.md)**

    Bản đồ 468 điểm dày đặc của khuôn mặt, kèm ảnh xem trước.

-   ![](../assets/nodes/face-warp.jpg)

    **[Face Warp](../nodes/face-warp.md)**

    Chỉnh dáng mặt từng tam giác một, chỉ trong phạm vi lưới mặt.

-   ![](../assets/nodes/face-warp-tps.jpg)

    **[Face Warp TPS](../nodes/face-warp-tps.md)**

    Chỉnh dáng mặt bằng một trường biến dạng mượt, không có đường nối tam giác.

-   ![](../assets/nodes/unwrap-face.jpg)

    **[Unwrap Face](../nodes/unwrap-face.md)**

    Dán vùng mặt đã chỉnh (đã căn thẳng) trở lại ảnh gốc.

-   ![](../assets/nodes/face-beauty-blend.jpg)

    **[Face Beauty Blend](../nodes/face-beauty-blend.md)**

    Trộn ảnh đã retouch lên ảnh gốc qua mask và thanh cường độ.

-   ![](../assets/nodes/skin-tone-tri-region.jpg)

    **[Skin Tone Tri-Region](../nodes/skin-tone-tri-region.md)**

    Tách da thành mask shadow, midtone và highlight.

-   ![](../assets/nodes/auto-blemish.jpg)

    **[Auto Blemish](../nodes/auto-blemish.md)**

    Tìm khuôn mặt và xóa mụn, vết nhỏ trên da mà vẫn giữ vân da.

</div>

## Chuỗi căn chỉnh → chỉnh sửa → dán lại

AI làm da, phục hồi khuôn mặt và các bộ tăng chi tiết hoạt động tốt nhất trên một khuôn mặt nhỏ và thẳng. Chuỗi chuẩn đưa cho chúng đúng thứ đó, rồi đặt kết quả về chỗ cũ:

1. **Nhận diện.** [Face Detect](../nodes/face-detect.md) hoặc [Face Detect Lite](../nodes/face-detect-lite.md) tìm khuôn mặt và xuất 5 landmark dưới dạng `landmarks_json`.
2. **Căn chỉnh.** [Facial Aligner](../nodes/facial-aligner.md) (thuộc nhóm [Hình học](geometry.md)) biến khuôn mặt thành vùng cắt vuông, thẳng (`image_aligned`) và xuất thêm `inverse_matrix_json`, công thức để hoàn tác phép biến đổi.
3. **Chỉnh sửa.** Chạy bước AI hoặc retouch của bạn trên vùng mặt đã căn chỉnh.
4. **Dán lại.** [Unwrap Face](../nodes/unwrap-face.md) nhận vùng mặt đã chỉnh, ảnh gốc và `inverse_matrix_json`, rồi trả về ảnh đầy đủ mà chỉ vùng mặt thay đổi.
5. **Hoàn thiện (tùy chọn).** [Face Beauty Blend](../nodes/face-beauty-blend.md) trộn kết quả lên ảnh gốc qua mask, có thanh strength, để phần retouch trông tự nhiên.

Hãy nối ảnh gốc vào cả Unwrap Face và Face Beauty Blend. Các pixel không bị động tới ngoài vùng mặt lấy từ đó.

## Nên chọn node nào?

**Face Detect (MediaPipe) hay Face Detect Lite (YuNet)?**

| Chọn | Khi |
|---|---|
| [Face Detect Lite](../nodes/face-detect-lite.md) | Bạn cần tốc độ (xử lý hàng loạt, xem trước), mask khuôn mặt, điểm tin cậy thật, hoặc máy chưa cài MediaPipe. Dùng GPU khi có. |
| [Face Detect](../nodes/face-detect.md) | Bạn đã dùng MediaPipe, muốn điểm mắt ở khóe ngoài, hoặc còn chạy [Face Landmarks](../nodes/face-landmarks.md) và muốn hai node thống nhất với nhau. |

Cả hai cho cùng đầu ra 5 điểm và đều dùng được với aligner. Không node nào nối trực tiếp vào các node warp: warp cần đầu ra 468 điểm của Face Landmarks.

**Face Warp Delaunay hay TPS?**

| Chọn | Khi |
|---|---|
| [Face Warp TPS](../nodes/face-warp-tps.md) | Bạn muốn chỉnh dáng nhẹ nhàng, tự nhiên (thon mặt, chỉnh mắt hoặc hàm) mà không có đường nối tam giác, muốn tốc độ GPU, hoặc đang khớp landmark từ ảnh khác (có thông số `smoothness` để hấp thụ nhiễu nhận diện). |
| [Face Warp](../nodes/face-warp.md) | Thay đổi phải nằm gọn trong lưới mặt, hoặc cả 468 điểm đều phải được tôn trọng ở mức cục bộ. Chạy trên CPU. |

## Chuỗi node thường dùng

- **AI làm da, dán lại:** Load Image → [Face Detect Lite](../nodes/face-detect-lite.md) → [Facial Aligner](../nodes/facial-aligner.md) → bước AI của bạn → [Unwrap Face](../nodes/unwrap-face.md) → [Face Beauty Blend](../nodes/face-beauty-blend.md).
- **Chỉnh dáng tinh tế:** Load Image → [Face Landmarks](../nodes/face-landmarks.md) (nguồn) và một [Face Landmarks](../nodes/face-landmarks.md) thứ hai (đích) → [Face Warp TPS](../nodes/face-warp-tps.md) → Preview Image.
- **Retouch theo dải tông:** Load Image + mask da → [Skin Tone Tri-Region](../nodes/skin-tone-tri-region.md) → một mask đưa vào [Face Beauty Blend](../nodes/face-beauty-blend.md). Xem [ví dụ Retouch tông da](../examples/skin-tone-retouch.md).
