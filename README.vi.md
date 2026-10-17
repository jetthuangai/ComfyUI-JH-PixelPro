<div align="center">

# PixelPro cho ComfyUI

**Bộ công cụ retouch, màu và mask chuyên nghiệp cho ComfyUI — 50 node chạy GPU, xây dựng trên PyTorch + Kornia.**

[![CI](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/actions/workflows/ci.yml/badge.svg)](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-EN%20%7C%20VI-0A84FF?logo=materialformkdocs&logoColor=white)](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/)
[![Release](https://img.shields.io/github/v/release/jetthuangai/ComfyUI-JH-PixelPro)](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/releases/latest)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-555555)](LICENSE)

[English](README.md) · **Tiếng Việt** · [Tài liệu](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/)

![Một chân dung qua tám node PixelPro](docs/assets/hero.jpg)

</div>

## PixelPro là gì

PixelPro đưa các kỹ thuật của bàn hậu kỳ chuyên nghiệp vào ComfyUI dưới dạng node: **tách tần số (frequency separation), luminosity mask, khớp màu trong LAB, LUT, chồng layer kiểu Photoshop, tách alpha chuẩn từng sợi tóc, căn khuôn mặt rồi dán lại**, kèm các chỉ số đo chất lượng để kiểm tra kết quả.

- **Kỹ thuật retouch thật**, không phải bộ lọc đặt tên kêu: mỗi node làm đúng việc mà người làm hậu kỳ nghĩ tới khi đọc tên nó.
- **Ưu tiên GPU, vẫn an toàn trên CPU.** Bộ lọc nặng chạy CUDA với bộ nhớ có giới hạn; máy chỉ có CPU vẫn chạy được.
- **Ăn khớp với quy trình AI.** Căn mặt, chạy inpaint hay ControlNet, dán lại liền mạch. Mask dùng được với mọi công cụ phân vùng.
- **Kiểm thử và tài liệu đầy đủ.** Hơn 800 test tự động, workflow mẫu, và [hướng dẫn song ngữ](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/) cho từng node.

<table>
<tr>
<td><img src="docs/assets/nodes/frequency-separation.jpg" alt="Frequency Separation"></td>
<td><img src="docs/assets/nodes/alpha-matte.jpg" alt="Alpha Matte"></td>
</tr>
<tr>
<td align="center"><b>Frequency Separation</b> — màu và chi tiết trên hai lớp riêng</td>
<td align="center"><b>Alpha Matte</b> — từ trimap ra alpha chuẩn từng sợi tóc</td>
</tr>
<tr>
<td><img src="docs/assets/nodes/color-matcher-region.jpg" alt="Color Matcher Region"></td>
<td><img src="docs/assets/nodes/pyramid-blend.jpg" alt="Pyramid Blend"></td>
</tr>
<tr>
<td align="center"><b>Color Matcher Region</b> — khớp màu da với da, trời với trời</td>
<td align="center"><b>Pyramid Blend</b> — ghép ảnh không lộ đường nối</td>
</tr>
</table>

## Cài đặt

**ComfyUI Manager:** tìm `ComfyUI-JH-PixelPro`, bấm Install, khởi động lại ComfyUI.

**Thủ công:**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jetthuangai/ComfyUI-JH-PixelPro.git
cd ComfyUI-JH-PixelPro
pip install -r requirements.txt   # dùng đúng Python mà ComfyUI đang chạy
```

Khởi động lại ComfyUI. Các node nằm trong **Add Node → ComfyUI-JH-PixelPro**.

Yêu cầu: Python 3.10+, PyTorch (đi kèm ComfyUI), Kornia ≥ 0.7, MediaPipe ≥ 0.10, OpenCV ≥ 4.8, SciPy ≥ 1.10. Tùy chọn: `onnxruntime-gpu` (hoặc `onnxruntime`) cho Depth Anything và Depth Lens Blur. Nên có GPU NVIDIA từ 8 GB VRAM; CPU vẫn chạy được.

## Bắt đầu nhanh

1. Chép `workflows/sample_portrait.jpg` (và `sample_binary_mask.png`) vào `ComfyUI/input/`.
2. Kéo một workflow trong [`workflows/`](workflows/README.md) thả vào canvas, ví dụ CLAHE, Face Detect Lite hoặc Mask Cleanup.
3. Bấm Queue Prompt, mỗi lần chỉ đổi một thông số.

Hướng dẫn đầy đủ: [Hướng dẫn nhanh](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/quickstart/).

## 50 node

Bấm vào tên node để xem hướng dẫn (thông số, mẹo, ảnh trước/sau).

### Màu & tông · 14

| Node | Công dụng |
|---|---|
| [ColorLab](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/color-lab/) | Bảng chỉnh kiểu Camera Raw: exposure, contrast, highlights/shadows, texture, clarity, dehaze, HSL, color grading, trộn đen trắng. |
| [Tone Curve](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/tone-curve/) | Đường cong tông mượt cho kênh tổng hoặc từng kênh RGB, có preset. |
| [Color Matcher](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/color-matcher/) | Chuyển màu từ ảnh mẫu sang ảnh của bạn trong không gian LAB. |
| [Color Matcher Region](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/color-matcher-region/) | Khớp màu giữa hai vùng có mask, ảnh khác kích thước vẫn được. |
| [CLAHE Local Contrast](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/clahe-local-contrast/) | Tăng tương phản cục bộ chỉ trên độ sáng, giữ nguyên màu. |
| [Hue/Saturation per Range](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/hue-saturation-range/) | Đổi sắc độ hoặc độ bão hòa của một dải màu. |
| [Luminosity Masking](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/luminosity-masking/) | Mask vùng tối, trung tính và vùng sáng. |
| [Saturation Mask](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/saturation-mask/) | Chọn vùng theo độ đậm màu. |
| [LUT Preset](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/lut-preset/) | Các look .cube có sẵn, chọn từ danh sách. |
| [LUT Import](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/lut-import/) | Áp dụng file LUT .cube bất kỳ, có mask và cường độ. |
| [LUT Export](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/lut-export/) | Xuất HALD đã chỉnh màu thành file .cube. |
| [HALD Identity](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/hald-identity/) | Ảnh HALD trung tính để chỉnh màu rồi xuất LUT. |
| [Tone Match LUT](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/tone-match-lut/) | Tạo file .cube từ một khung hình mẫu đã chỉnh màu. |
| [Palette Extract](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/palette-extract/) | Màu chủ đạo thành ô màu + mã HEX, kèm ảnh màu phẳng. |

### Mask · 11

| Node | Công dụng |
|---|---|
| [Sub-Pixel Mask Refiner](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/sub-pixel-mask-refiner/) | Biến mask cứng thành viền alpha mềm, giữ nguyên lõi trong và ngoài. |
| [Edge-Aware Mask Refiner](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/edge-aware-mask-refiner/) | Kéo viền mask bám theo viền thật của ảnh (guided filter). |
| [Trimap Builder](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/trimap-builder/) | Bản đồ tiền cảnh / vùng chưa rõ / hậu cảnh để tách alpha. |
| [Alpha Matte](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/alpha-matte/) | Tách alpha closed-form: chuẩn từng sợi tóc từ trimap. |
| [Mask Morphology](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/mask-morphology/) | Nở, co, open, close và các phép biến đổi hình dạng mask khác. |
| [Mask Combine](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/mask-combine/) | Cộng, trừ, giao, hợp hoặc nhân hai mask. |
| [Mask Edge Smoother](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/mask-edge-smoother/) | Làm mượt viền mask răng cưa bằng lọc bilateral (CPU hoặc GPU). |
| [Mask Cleanup](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/mask-cleanup/) | Xóa chấm lẻ, lấp lỗ, làm mềm viền theo khoảng cách. |
| [SAM One-Click Mask](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/sam-mask/) | Segment Anything từ một điểm, một khung mặt hoặc một mask thô. |
| [Mask Preview](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/mask-preview/) | Xem mask dạng phủ màu, bản đồ nhiệt hoặc cắt nền. |
| [Detail Mask](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/high-freq-detail-masker/) | Mask vùng chi tiết nhỏ (tóc, mắt, texture) để bảo vệ hoặc làm nét. |

### Khuôn mặt · 8

| Node | Công dụng |
|---|---|
| [Face Detect](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/face-detect/) | Khung mặt và 5 điểm landmark bằng MediaPipe. |
| [Face Detect Lite](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/face-detect-lite/) | Nhận diện mặt nhanh bằng YuNet trên GPU, có mask mặt và độ tin cậy thật. |
| [Face Landmarks](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/face-landmarks/) | Lưới 468 điểm trên khuôn mặt để chỉnh dáng. |
| [Face Warp](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/face-warp/) | Chỉnh dáng mặt bằng cách dịch các landmark (theo từng tam giác). |
| [Face Warp TPS](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/face-warp-tps/) | Cùng ý tưởng nhưng dùng một trường thin-plate spline mượt, không đường nối. |
| [Unwrap Face](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/unwrap-face/) | Dán khuôn mặt đã căn và chỉnh sửa trở lại ảnh gốc. |
| [Face Beauty Blend](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/face-beauty-blend/) | Hòa bản retouch mặt vào ảnh với mask và cường độ. |
| [Skin Tone Tri-Region](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/skin-tone-tri-region/) | Mask vùng tối / trung tính / sáng của làn da. |

### Ghép ảnh · 5

| Node | Công dụng |
|---|---|
| [Layer Stack Start](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/layer-stack-start/) | Bắt đầu một chồng layer từ ảnh nền. |
| [Layer Add](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/layer-add/) | Thêm layer với blend mode, opacity, fill, mask và clipping. |
| [Layer Group](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/layer-group/) | Gộp một chồng con thành một layer nhóm. |
| [Layer Flatten](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/layer-flatten/) | Gộp chồng layer thành ảnh. |
| [Pyramid Blend](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/pyramid-blend/) | Hòa trộn đa tần số để ghép ảnh liền mạch. |

### Bộ lọc · 5 · Độ sâu · 2 · Hình học · 3 · Look màu · 1 · Kiểm tra · 1

| Node | Công dụng |
|---|---|
| [Frequency Separation](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/frequency-separation/) | Tách màu/khối (low) khỏi texture (high). |
| [Frequency Recombine](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/frequency-recombine/) | Ghép lại ảnh từ low + high, chỉnh được lượng texture. |
| [Skin Smoother](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/edge-aware-smoother/) | Làm mịn da giữ viền, có mask và cường độ. |
| [Smart Sharpen](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/smart-sharpen/) | Unsharp mask trên độ sáng, có ngưỡng và mask. |
| [Dust & Spot Removal](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/dust-spot-removal/) | Xóa đốm nhỏ và bụi, giữ nguyên chi tiết lớn. |
| [Depth Anything V2](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/depth-anything/) | Bản đồ độ sâu từ một ảnh (bản Small: Apache-2.0). |
| [Depth Lens Blur](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/depth-lens-blur/) | Bokeh theo độ sâu, lấy nét theo mask mặt hoặc chủ thể. |
| [Facial Aligner](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/facial-aligner/) | Căn mặt về khung vuông chuẩn; trả về ma trận để Unwrap Face dán lại. |
| [Lens Distortion](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/lens-distortion/) | Sửa hoặc tạo méo thùng/méo gối của ống kính. |
| [Auto Align](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/auto-align/) | Căn ảnh chụp thứ hai khớp ảnh chuẩn (SIFT hoặc LoFTR). |
| [Look Select](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/look-select/) | Các look màu hoàn chỉnh, có cường độ. |
| [QA Image Diff](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/nodes/qa-image-diff/) | SSIM, PSNR và bản đồ nhiệt chênh lệch giữa hai ảnh. |

## Tài liệu

- **Hướng dẫn (EN / VI):** <https://jetthuangai.github.io/ComfyUI-JH-PixelPro/vi/>
- **Workflow mẫu:** [`workflows/README.md`](workflows/README.md)
- **Lịch sử thay đổi:** [CHANGELOG.md](CHANGELOG.md)

## Giấy phép

Apache-2.0. Xem [LICENSE](LICENSE). Ảnh chân dung mẫu của [cottonbro studio](https://www.pexels.com/@cottonbro/) trên Pexels (Pexels License).

Người duy trì: **JH** ([@jetthuangai](https://github.com/jetthuangai)).
