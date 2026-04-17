# ComfyUI-JH-PixelPro

> GPU-powered pro-grade image suite cho ComfyUI. Kornia làm lõi. Tensor thuần, không rời VRAM.

**Status:** 🚧 Phase 1 — đang xây N-01 Frequency Separation. Chưa phát hành.

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

## License

Apache-2.0 — xem [LICENSE](./LICENSE).

## Author

JH (Product Owner). Pack được build với sự hỗ trợ của team agent (Cowork architect, Codex core math, Claude Code integration).
