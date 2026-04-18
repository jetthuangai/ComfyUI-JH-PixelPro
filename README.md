# ComfyUI-JH-PixelPro

> GPU-powered pro-grade image suite cho ComfyUI. Kornia làm lõi. Tensor thuần, không rời VRAM.

**Status:** 🎉 **v0.1.0 alpha** (2026-04-18) — 2 node live: `JHPixelProFrequencySeparation` + `JHPixelProSubPixelMaskRefiner`. Xem [CHANGELOG](./CHANGELOG.md) cho chi tiết. Phase 1 M1 còn 3 node (N-03..N-05), roadmap ở `.agent-hub/10_plan/roadmap.md` (internal).

## Vì sao có pack này

ComfyUI mạnh ở pipeline generative nhưng thiếu các thao tác retouch chuyên nghiệp vận hành trực tiếp trên tensor GPU:

- Mất chi tiết da sau VAE / inpaint.
- Mask cứng, lẹm viền (halo) từ SAM / YOLO.
- Sai lệch tông màu da sau khi sinh ảnh.

Pack này đóng gói 9 node Kornia-powered cho Phase 1–3, sau đó mở rộng sang các tác vụ CV khác (segmentation, tracking, 3D, color science).

## Scope

| Phase | Nhóm node | Phạm vi |
|---|---|---|
| 1 | filters + morphology | Frequency separation, mask refiner, edge-aware smoother, detail masker, luminosity masking |
| 2 | geometry | Facial aligner, lens distortion corrector |
| 3 | color | RAW-space color matcher, tone curve & color balance |
| sau v1.0 | *(sẽ quyết sau)* | Segmentation, tracking, depth, advanced color |

## Install

Copy/clone thư mục này vào `ComfyUI/custom_nodes/`:

```bash
cd ComfyUI/custom_nodes
git clone <repo-url> ComfyUI-JH-PixelPro
cd ComfyUI-JH-PixelPro
pip install -r requirements.txt
```

Khởi động lại ComfyUI. Các node sẽ xuất hiện dưới menu `image/pixelpro/<group>`.

## Requirements

- ComfyUI ≥ 0.43.x
- Python ≥ 3.10
- PyTorch (đã có qua ComfyUI)
- Kornia ≥ 0.7.0
- GPU NVIDIA ≥ 8 GB VRAM (primary); **CPU fallback được support** (correctness, không bảo đảm tốc độ).

## Node list *(cập nhật theo tiến độ Phase)*

Xem chi tiết tại `.agent-hub/10_plan/master-plan.md` (tạm thời bên trong ComfyUI, sẽ dời ra umbrella folder).

- [ ] N-01 GPU Frequency Separation *(đang làm — S-01 spec READY)*
- [ ] N-02 Sub-Pixel Mask Refiner
- [ ] N-03 Edge-Aware Skin Smoother
- [ ] N-04 High-Frequency Detail Masker
- [ ] N-05 Luminosity Masking
- [ ] N-06 Landmark-Based Facial Aligner
- [ ] N-07 Lens Distortion Corrector
- [ ] N-08 RAW-Space Color Matcher
- [ ] N-09 GPU Tone Curve & Color Balance

## Triết lý kỹ thuật

1. **Pure tensor in / pure tensor out** — không đụng file, PIL, NumPy trong core math.
2. **Invariant test là primary acceptance** — không chỉ "output có vẻ đúng".
3. **Device awareness** — tự động `cpu` và `cuda:N`; không hard-code.
4. **Channel convention BCHW** trong core, convert ở integration layer với ComfyUI.
5. **Không catch exception im lặng.**

## N-01 GPU Frequency Separation

Tách 1 ảnh thành 2 layer: `low` (Gaussian blur — màu + form mềm) và `high` (chi tiết tần số cao — texture, viền, nốt). Đây là kỹ thuật retouch chuẩn của ngành: làm da mịn ở `low` mà không phá texture ở `high`. Math invariant: `low + high = original` (trước clamp), reconstruct lossless với `precision=float32`.

**Inputs:**

| Tên | Type | Default | Mô tả |
|---|---|---|---|
| `image` | IMAGE | — | ComfyUI IMAGE tensor (BHWC, float32 0..1). |
| `radius` | INT | `8` | Bán kính Gaussian blur (pixel). Range 1..128. |
| `sigma` | FLOAT | `0.0` | Sigma override. `0.0` = auto `radius/2` (Photoshop convention). |
| `precision` | COMBO | `float32` | `float32` = lossless reconstruct (atol 1e-5). `float16` = ~2× nhanh trên GPU, reconstruct error ~1e-3. |

**Outputs:**

| Tên | Type | Mô tả |
|---|---|---|
| `low` | IMAGE | Low-frequency layer. Range `[0, 1]`. |
| `high` | IMAGE | High-frequency layer. **⚠️ Có thể có giá trị âm** (mean ≈ 0). PreviewImage hiển thị sai màu — bình thường, không phải bug. Cần `ImageAdd` PURE (không clamp) để reconstruct. |

**Sample workflow:** [workflows/S-01-frequency-separation.json](workflows/S-01-frequency-separation.json)

**Chạy sample:** copy `workflows/sample_portrait.jpg` → `ComfyUI/input/sample_portrait.jpg` trước, rồi Load workflow + Queue Prompt. Ảnh mẫu là [Photo by cottonbro studio on Pexels](https://www.pexels.com/photo/close-up-photo-of-woman-s-beautiful-face-6567969/) (license redistribute free).

![screenshot](workflows/S-01-frequency-separation-screenshot.png)
<!-- screenshot: JH chụp sau khi test workflow với sample_portrait.jpg trong ComfyUI -->

### Limitations

- **Reconstruct branch không nằm trong v0.1.** ComfyUI core chỉ có `ImageBlend` clamp `[0, 1]` → phá invariant khi `high` âm. Dùng external `ImageAdd` pack hoặc chờ `JHPixelProImageAdd` ở v0.2.
- Xem Note node trong [workflows/S-01-frequency-separation.json](workflows/S-01-frequency-separation.json) để hiểu invariant chi tiết.
- `precision=float16` chỉ khuyên dùng trên GPU; trên CPU sẽ warn (chậm hơn float32).

## N-02 Sub-Pixel Mask Refiner

Feather một MASK nhị phân (từ SAM / YOLO / rembg upstream) thành alpha mask sub-pixel: phần "definitely inside" pin về `1.0`, phần "definitely outside" pin về `0.0`, vùng band uncertain ở mép được Gaussian-feather mượt. Dùng cho cutout, composite, alpha matting trong pipeline retouch chuyên nghiệp.

**Inputs:**

| Tên | Type | Default | Mô tả |
|---|---|---|---|
| `mask` | MASK | — | ComfyUI MASK tensor (BHW float32 `[0, 1]`). Binary-ish — có thể có midtone nhưng sẽ bị threshold trước morphology. |
| `erosion_radius` | INT | `2` | Pixel radius của "definitely inside" core. Range 0..64. `0` = không có inside protection. |
| `dilation_radius` | INT | `4` | Pixel radius của "definitely outside" core. Set ≥ `erosion_radius` để có band feather ổn định. Range 0..64. |
| `feather_sigma` | FLOAT | `2.0` | Gaussian sigma (pixel) feather vùng band uncertain. Range 0.1..32.0 (step 0.1). |
| `threshold` | FLOAT | `0.5` | Binarization threshold (`mask > threshold`) áp trước morphology. Range 0.0..1.0 (step 0.01). |

**Outputs:**

| Tên | Type | Mô tả |
|---|---|---|
| `refined_mask` | MASK | Sub-pixel alpha mask. Range `[0, 1]`. Inside core = `1.0` exact, outside core = `0.0` exact, band feather giữa hai. |

**Sample workflow:** [workflows/S-02-subpixel-mask-refiner.json](workflows/S-02-subpixel-mask-refiner.json)

**Chạy sample:** copy `workflows/sample_binary_mask.png` → `ComfyUI/input/sample_binary_mask.png` trước (sẽ có sau T-08b artifact task), rồi Load workflow + Queue Prompt.

![screenshot](workflows/S-02-subpixel-mask-refiner-screenshot.png)
<!-- screenshot fill bởi T-08b artifact task -->

### Limitations

- **Square kernel (Chebyshev metric L∞).** Erosion/dilation dùng kernel hình vuông, không phải disk Euclidean — với `radius > 16`, mép mask sẽ hơi vuông góc thay vì tròn mượt. Disk kernel option defer v0.2.
- **V1 chỉ float32.** Không expose `precision` pin như N-01 — `feather_sigma > 0` cần float precision đủ cho Gaussian. float16 defer v0.2 sau khi có lesson learn từ N-01.
- Xem Note node trong [workflows/S-02-subpixel-mask-refiner.json](workflows/S-02-subpixel-mask-refiner.json) để hiểu invariant + edge case `er=dr=0`.

## License

Apache-2.0 — xem [LICENSE](./LICENSE).

## Author

JH (Product Owner). Pack được build với sự hỗ trợ của team agent (Cowork architect, Codex core math, Claude Code integration).
