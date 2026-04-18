# Changelog

All notable changes to this pack are recorded here. Format follows [Keep a Changelog 1.1](https://keepachangelog.com/en/1.1.0/) and [SemVer](https://semver.org/).

## [Unreleased]

(nothing yet)

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

[Unreleased]: https://github.com/jetthuangai/ComfyUI-JH-PixelPro/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jetthuangai/ComfyUI-JH-PixelPro/releases/tag/v0.1.0
