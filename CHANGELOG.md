# Changelog

Tất cả thay đổi đáng kể của pack này ghi ở đây. Format theo [Keep a Changelog 1.1](https://keepachangelog.com/en/1.1.0/), [SemVer](https://semver.org/lang/vi/).

## [Unreleased]

(chưa có)

## [0.1.0] — 2026-04-18

First public alpha. 2 node MVP cho retouch chân dung chuyên nghiệp trên GPU, Kornia làm lõi, tensor thuần không rời VRAM.

### Added

- **N-01 `JHPixelProFrequencySeparation`** (`image/pixelpro/filters`): split ảnh thành low-freq (Gaussian blur) + high-freq (detail). Input `IMAGE` + `radius` (1..128) + `sigma` (0=auto `radius/2`, else manual) + `precision` (`float32` lossless / `float16` GPU-fast). Output 2 × `IMAGE`. Invariant `low + high = original` trước clamp với `float32` (atol 1e-5). Tests 21 + bench module CPU/GPU.
- **N-02 `JHPixelProSubPixelMaskRefiner`** (`image/pixelpro/morphology`): feather binary mask thành sub-pixel alpha mask. Input `MASK` + `erosion_radius` (0..64) + `dilation_radius` (0..64) + `feather_sigma` (0.1..32) + `threshold` (0..1). Output `MASK` feather với invariant inside core = 1.0 exact, outside core = 0.0 exact, band mép Gaussian. Kernel Chebyshev (L∞) — square, disk option defer v0.2. Tests 15 + bench module.
- 2 sample workflow JSON demo + ảnh Pexels (cottonbro) + screenshot render inline trong README.
- Apache-2.0 license, tests pytest, bench module, ruff config.

### Dependencies

- `kornia >= 0.7.0` (verified trên 0.8.1 CPU + GPU CUDA).
- Python `>= 3.10`.

### Known limitations

- Reconstruct branch của N-01 (low + high → image) defer v0.2 chờ `JHPixelProImageAdd` (2-pin pure add, không clamp); `ImageBlend` core ComfyUI clamp `[0,1]` → phá invariant khi `high` có giá trị âm.
- N-02 kernel square (Chebyshev), radius > 16 mép hơi vuông góc. Disk kernel option defer v0.2.
- N-02 chỉ float32 v1; float16 defer v0.2.
- GPU verdict cho benchmark T-02 + T-06 dựa trên 1 run JH thực tế (FS 1.680s cho ảnh 3610×5416 float32 single GPU) — chưa có matrix benchmark CI.

### Contributors

- **JH** — Product Owner, retoucher chuyên nghiệp.
- **Cowork** — architect (specs, task coordination, DoD).
- **Codex** — core math (Kornia integration, tests, benchmark).
- **Claude Code** — ComfyUI node integration (wrap, workflows, README).

[Unreleased]: https://github.com/jetthuangai/ComfyUI-JH-PixelPro/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jetthuangai/ComfyUI-JH-PixelPro/releases/tag/v0.1.0
