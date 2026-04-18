# Changelog

All notable changes to this pack are recorded here. Format follows [Keep a Changelog 1.1](https://keepachangelog.com/en/1.1.0/) and [SemVer](https://semver.org/).

## [Unreleased]

(nothing yet)

## [0.3.0] — 2026-04-19

Batch-1 face-retouch trio ships (N-04 + N-05 + N-06) and the 3 pre-existing nodes are retrofit under the pack-branded namespace so all 6 live nodes appear under a single category in the ComfyUI Add Node menu.

### Added

- **N-04 `JHPixelProHighFreqDetailMasker`** (`ComfyUI-JH-PixelPro/mask`): high-frequency detail mask generator for texture protection during AI passes. Inputs: `IMAGE` + `kernel_type` (`laplacian` / `sobel` / `fs_gaussian`, default `laplacian`) + `sensitivity` (0..1, default 0.5) + `threshold_mode` (`relative_percentile` / `absolute`, default `relative_percentile`) + optional `MASK` pre-gate. Output: a binary `MASK` (BHW float32). Feeds `SetLatentNoiseMask` / `MaskCompose` / `ImageBlend` to keep hair, eyebrows, fabric weave and pore texture intact through inpaint / denoise / style-transfer.
- **N-05 `JHPixelProLuminosityMasking`** (`ComfyUI-JH-PixelPro/filters`): Photoshop-style luminosity masks with partition-of-unity guarantee (`shadows + midtones + highlights ≈ 1.0` per pixel). Inputs: `IMAGE` + `luminance_source` (`lab_l` / `ycbcr_y` / `hsv_v`, default `lab_l`) + `shadow_end` (0..0.5, default 0.33) + `highlight_start` (0.5..1.0, default 0.67) + `soft_edge` (0.01..0.3, default 0.1). Outputs: 3 `MASK` (`mask_shadows`, `mask_midtones`, `mask_highlights`). Band-limited grading, denoise, dodge/burn.
- **N-06 `JHPixelProFacialAligner`** (`ComfyUI-JH-PixelPro/geometry`): landmark-based face alignment to an FFHQ-like canonical frame (eyes @ Y=0.40, nose @ Y=0.55, mouth @ Y=0.70) via a similarity transform (rotation + uniform scale + translation). Inputs: `IMAGE` + `landmarks` STRING 5-point JSON + `target_size` (512 / 768 / 1024, default 1024) + `padding` (0..0.5, default 0.2). Outputs: `IMAGE` aligned + `STRING` JSON-serialized `B × 3 × 3` inverse matrix for unwrap. Pre-processing step for SDXL face refine / InstantID / IPAdapter FaceID pipelines.
- 3 sample workflows (`S-04-hf-detail-masker.json` / `S-05-luminosity-masking.json` / `S-06-facial-aligner.json`) + inline screenshots (JH manual smoke test verify).

### Changed

- **Q-NS** — the 3 pre-existing nodes (N-01 FS, N-02 MR, N-03 ES) retrofit category from `image/pixelpro/*` to `ComfyUI-JH-PixelPro/*`, unifying the 6-node pack under a single pack-branded namespace in the ComfyUI Add Node menu. Category UI-only change — existing workflow JSON links are **not** affected because ComfyUI references nodes by class name, not by category path. Saved workflows from v0.1 / v0.2 continue to load and run unchanged.

### Known limitations

- **CUDA benchmarks not evaluated** for N-04 / N-05 / N-06 (batch-1 was benchmarked on a CPU-only runner). JH GPU follow-up is non-blocking — the integration path is validated end-to-end on CPU with a 2K synthetic tensor.
- **N-05 `lab_l` perceptual cost**: ~172 ms @ 2K CPU (miss the < 80 ms stretch target). Users who need realtime preview on CPU should switch to `ycbcr_y` (~7 ms @ 1024 CPU) at the cost of a slightly less perceptual luminance split. `lab_l` is kept as the default because it matches the Photoshop convention.
- **N-06 is a standalone pre-processor**: it does not auto-detect landmarks. For production, feed landmarks from an upstream face-detector (community `ComfyUI_FaceAnalysis`, or the forthcoming batch-2 **N-10 `JHPixelProFaceDetect`** + **N-11 `JHPixelProUnwrapFace`** which will close the chain internally).
- **Roundtrip bilinear smoothing** (N-06): align + unwrap puts the image through two bilinear resamples, which softens the result by ~34/255 in uint8 — fine for a single retouch pass, not near-lossless.

### Dependencies

- `kornia >= 0.7.0` (unchanged from v0.2; verified on 0.8.1 CPU + CUDA).
- Python `>= 3.10` (unchanged).

## [0.2.0] — 2026-04-18

Third retouch node lands + v1.1 UX hotfix after JH feedback from v0.1.0 usage.

### Added

- **N-03 `JHPixelProEdgeAwareSmoother`** (`image/pixelpro/filters`): edge-preserving skin smoother using Kornia bilateral blur with tone-aware blend. Inputs: `IMAGE` + `strength` (0..1, default 0.4 — blend dose at full 40%) + `sigma_color` (0.01..0.5, default 0.1 — tone similarity σ) + `sigma_space` (0.5..8.0, default 6.0 — spatial σ in pixels) + `device` (auto/cpu/cuda, default `auto`) + `tile_mode` (bool, default false — enable for ≥ 2K images to cap VRAM) + optional `MASK`. Output: smoothed `IMAGE`. Invariant: `output = lerp(original, bilateral(original), strength)` — when `strength=0` returns original bit-exact. 14 tests (core + node wrap) + CPU/GPU bench module (CPU 1K B=1 baseline measured; GPU 2K budget chưa verified — see Known limitations).
- Sample workflow `workflows/S-03-edge-aware-smoother.json` + inline screenshot (PNG 2326×1513 RGBA) showing 5-widget UI + A/B Preview.

### Changed (v1.1 hotfix over pre-release N-03 v1.0 draft)

- **N-03 tile processing**: added tile 512×512 + overlap `k//2+1` with hard-crop for images ≥ 2K, gated by `tile_mode` pin. Prevents OOM on 4K inputs with `sigma_space` up to 8.0.
- **N-03 device pin**: explicit `device` dropdown (auto/cpu/cuda) — replaces silent `.to(input.device)` auto-detect. Pro-tool convention: user stays in control.
- **Display name**: stripped `(Kornia)` suffix across N-01/N-02/N-03 — cleaner node title on the canvas for end-users who don't need the implementation detail.
- **Docs**: README pack + `workflows/` docs fully English (mixed VN/EN cleanup pre-v0.2).

### Known limitations

- GPU 2K B=1 median budget (< 400 ms) not verified on JH's GPU machine (NOT EVALUATED — deferred non-blocking follow-up).
- `tile_mode` seam detection is qualitative only (visual A/B); no pixel-level seam test matrix yet.
- N-03 is float32 only; float16 is deferred to a later batch (shared with the N-02 float16 plan).

### Dependencies

- `kornia >= 0.7.0` (verified on 0.8.1 across CPU and CUDA GPU).
- Python `>= 3.10` (unchanged).

## [0.1.0] — 2026-04-18

First public alpha. Two MVP nodes for professional portrait retouching on GPU, with Kornia at the core and pure tensors that never leave VRAM.

### Added

- **N-01 `JHPixelProFrequencySeparation`** (`image/pixelpro/filters`): split an image into low-freq (Gaussian blur) + high-freq (detail) layers. Inputs: `IMAGE` + `radius` (1..128) + `sigma` (0 = auto `radius/2`, else manual) + `precision` (`float32` lossless / `float16` GPU-fast). Outputs: 2 × `IMAGE`. Invariant: `low + high = original` (pre-clamp) for `float32` with atol 1e-5. 21 tests + CPU/GPU bench module.
- **N-02 `JHPixelProSubPixelMaskRefiner`** (`image/pixelpro/morphology`): feather a binary mask into a sub-pixel alpha mask. Inputs: `MASK` + `erosion_radius` (0..64) + `dilation_radius` (0..64) + `feather_sigma` (0.1..32) + `threshold` (0..1). Output: a feathered `MASK` with inside core = 1.0 exact, outside core = 0.0 exact, and a Gaussian-feathered band at the edge. Kernel is Chebyshev (L∞ / square); a disk option is deferred to v0.2. 15 tests + bench module.
- Two sample workflow JSONs, a Pexels sample image (cottonbro), and inline screenshots in the README.
- Apache-2.0 license, pytest suite, bench module, ruff config.

### Dependencies

- `kornia >= 0.7.0` (verified on 0.8.1 across CPU and CUDA GPU).
- Python `>= 3.10`.

### Known limitations

- The N-01 reconstruct branch (`low + high → image`) is deferred to v0.2, pending `JHPixelProImageAdd` (a 2-pin non-clamping add); ComfyUI's core `ImageBlend` clamps to `[0, 1]`, which breaks the invariant when `high` contains negative values.
- The N-02 kernel is a square Chebyshev kernel — at `radius > 16`, mask edges look slightly boxy. A disk-kernel option is deferred to v0.2.
- N-02 is float32 only in v1; float16 is deferred to v0.2.
- GPU benchmark verdicts for N-01 and N-02 rest on a single real-world run (FS at 1.680 s for a 3610×5416 float32 image on a single GPU); there is no CI benchmark matrix yet.

### Contributors

- **JH** ([@jetthuangai](https://github.com/jetthuangai)) — maintainer & product owner.
- Built with AI pair-programming assistance.

[Unreleased]: https://github.com/jetthuangai/ComfyUI-JH-PixelPro/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/jetthuangai/ComfyUI-JH-PixelPro/releases/tag/v0.3.0
[0.2.0]: https://github.com/jetthuangai/ComfyUI-JH-PixelPro/releases/tag/v0.2.0
[0.1.0]: https://github.com/jetthuangai/ComfyUI-JH-PixelPro/releases/tag/v0.1.0
