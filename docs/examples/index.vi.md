# Ví dụ

Những lối vào ngắn gọn, thực dụng để làm quen với pack. Mỗi trang hay workflow đưa ra một chuỗi cụ thể để bạn nạp, chạy rồi điều chỉnh cho ảnh của mình.

## Các trang ví dụ

- [LUT Preset Gallery](lut-preset-gallery.md): so sánh trước/sau cho cả sáu LUT preset đi kèm.
- [Skin Tone Retouch Chain](skin-tone-retouch.md): tách chân dung thành mask shadow, midtone và highlight để retouch có mục tiêu.

## Workflow mẫu

Danh mục đầy đủ, kèm mô tả từng graph và ghi nguồn ảnh, nằm trong [workflows README](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/README.md). Vài chỗ nên bắt đầu:

- [Frequency Separation](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-01-frequency-separation.json)
- [Skin Smoother](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-03-edge-aware-smoother.json)
- [Layer compositing: chỉnh màu 5 layer kiểu cinematic](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-27-layer-compositing-5layer-cinematic.json)
- [LUT Preset](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-34-lut-preset.json)
- [Skin Tone Tri-Region](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-35-skin-tone-tri-region.json)
- [Pyramid Blend](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-36-pyramid-blend.json)

## Cách chạy một workflow mẫu

1. Chép `workflows/sample_portrait.jpg` từ repository vào `ComfyUI/input/`. Node Load Image chỉ tìm ảnh trong thư mục đó.
2. Trong ComfyUI, mở menu **Load** và chọn file workflow `.json`, hoặc kéo thả file lên canvas.
3. Nhấn **Queue Prompt**.

## Lộ trình nên theo lúc đầu

1. Bắt đầu với [Quickstart](../quickstart.md).
2. Chạy workflow LUT Preset một lần để chắc chắn đầu vào và đầu ra ảnh đã nối đúng.
3. Chuyển sang workflow Skin Tone Tri-Region nếu công việc của bạn là retouch chân dung.
4. Thử các workflow layer compositing khi muốn có một bước chỉnh màu hoàn thiện lặp lại được.
