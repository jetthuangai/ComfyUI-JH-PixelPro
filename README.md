# ComfyUI-JH-PixelPro

> GPU-powered pro-grade image suite for ComfyUI. Kornia at the core. Pure tensor, never leaves VRAM.

**Status:** 🎉 **v0.2.0** (2026-04-18) — 3 nodes live: `JHPixelProFrequencySeparation` + `JHPixelProSubPixelMaskRefiner` + `JHPixelProEdgeAwareSmoother`. See [CHANGELOG](./CHANGELOG.md) for details. Phase 1 has 6 more nodes queued (N-04..N-09) — batch mode plan post-v0.2.

## Why this pack exists

ComfyUI is strong at generative pipelines but lacks the professional retouching operations that work directly on GPU tensors:

- Skin detail loss after VAE / inpaint.
- Hard, chipped mask edges (halo) from SAM / YOLO.
- Skin tone drift after image generation.

This pack packages 9 Kornia-powered nodes for Phases 1–3, then expands into other CV tasks (segmentation, tracking, 3D, color science).

## Scope

| Phase | Node group | Coverage |
|---|---|---|
| 1 | filters + morphology | Frequency separation, mask refiner, edge-aware smoother, detail masker, luminosity masking |
| 2 | geometry | Facial aligner, lens distortion corrector |
| 3 | color | RAW-space color matcher, tone curve & color balance |
| after v1.0 | *(TBD)* | Segmentation, tracking, depth, advanced color |

## Install

Copy or clone this folder into `ComfyUI/custom_nodes/`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jetthuangai/ComfyUI-JH-PixelPro.git
cd ComfyUI-JH-PixelPro
pip install -r requirements.txt
```

Restart ComfyUI. The nodes appear under the `image/pixelpro/<group>` menu.

## Requirements

- ComfyUI ≥ 0.43.x
- Python ≥ 3.10
- PyTorch (installed alongside ComfyUI)
- Kornia ≥ 0.7.0
- NVIDIA GPU with ≥ 8 GB VRAM (primary target); **CPU fallback is supported** (correctness only — no speed guarantee).

## Node list *(updated per Phase progress)*

- [x] N-01 GPU Frequency Separation
- [x] N-02 Sub-Pixel Mask Refiner
- [x] N-03 Edge-Aware Skin Smoother
- [x] N-04 High-Frequency Detail Masker
- [x] N-05 Luminosity Masking
- [x] N-06 Landmark Facial Aligner
- [ ] N-07 Lens Distortion Corrector
- [ ] N-08 RAW-Space Color Matcher
- [ ] N-09 GPU Tone Curve & Color Balance

## Engineering principles

1. **Pure tensor in, pure tensor out** — no file I/O, no PIL, no NumPy in the core math.
2. **Invariant tests as primary acceptance** — not just "output looks right".
3. **Device awareness** — automatic `cpu` and `cuda:N`, never hard-coded.
4. **BCHW channel convention** inside the core; convert at the ComfyUI integration boundary.
5. **No silent exceptions.**

## N-01 GPU Frequency Separation

Splits an image into two layers: `low` (Gaussian blur — color + soft form) and `high` (high-frequency detail — texture, edges, pores). This is the industry-standard retouch technique: smooth skin on `low` without destroying texture on `high`. Math invariant: `low + high = original` (pre-clamp), lossless reconstruction with `precision=float32`.

**Inputs:**

| Name | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | ComfyUI IMAGE tensor (BHWC, float32 `[0, 1]`). |
| `radius` | INT | `8` | Gaussian blur radius in pixels. Range 1..128. |
| `sigma` | FLOAT | `0.0` | Sigma override. `0.0` = auto `radius/2` (Photoshop convention). |
| `precision` | COMBO | `float32` | `float32` = lossless reconstruction (atol 1e-5). `float16` = ~2× faster on GPU, reconstruction error ~1e-3. |

**Outputs:**

| Name | Type | Description |
|---|---|---|
| `low` | IMAGE | Low-frequency layer. Range `[0, 1]`. |
| `high` | IMAGE | High-frequency layer. **⚠️ May contain negative values** (mean ≈ 0). `PreviewImage` will display miscoloured output — this is expected, not a bug. A pure (non-clamping) `ImageAdd` node is required to reconstruct. |

**Sample workflow:** [workflows/S-01-frequency-separation.json](workflows/S-01-frequency-separation.json)

**Run the sample:** copy `workflows/sample_portrait.jpg` → `ComfyUI/input/sample_portrait.jpg`, then Load the workflow and press Queue Prompt. The sample image is [Photo by cottonbro studio on Pexels](https://www.pexels.com/photo/close-up-photo-of-woman-s-beautiful-face-6567969/) (redistribution free under the Pexels Content License).

![screenshot](workflows/S-01-frequency-separation-screenshot.png)

### Limitations

- **The reconstruct branch is not included in v0.1.** ComfyUI core ships only `ImageBlend`, which clamps to `[0, 1]` and breaks the invariant when `high` contains negative values. Use an external `ImageAdd` pack, or wait for `JHPixelProImageAdd` in v0.2.
- See the Note node inside [workflows/S-01-frequency-separation.json](workflows/S-01-frequency-separation.json) for a detailed invariant explanation.
- `precision=float16` is recommended on GPU only; on CPU it will warn (slower than float32).

## N-02 Sub-Pixel Mask Refiner

Feathers a binary MASK (from SAM / YOLO / rembg upstream) into a sub-pixel alpha mask: "definitely inside" pixels pin to `1.0`, "definitely outside" pixels pin to `0.0`, and the uncertain band near the edge is Gaussian-feathered. Used for cutout, compositing, and alpha matting in professional retouch pipelines.

**Inputs:**

| Name | Type | Default | Description |
|---|---|---|---|
| `mask` | MASK | — | ComfyUI MASK tensor (BHW float32 `[0, 1]`). Binary-ish — midtones are allowed but will be thresholded before morphology. |
| `erosion_radius` | INT | `2` | Pixel radius of the "definitely inside" core. Range 0..64. `0` = no inside protection. |
| `dilation_radius` | INT | `4` | Pixel radius of the "definitely outside" core. Set ≥ `erosion_radius` for a stable feather band. Range 0..64. |
| `feather_sigma` | FLOAT | `2.0` | Gaussian sigma (pixels) used to feather the uncertain band. Range 0.1..32.0 (step 0.1). |
| `threshold` | FLOAT | `0.5` | Strict binarization threshold (`mask > threshold`) applied before morphology. Range 0.0..1.0 (step 0.01). |

**Outputs:**

| Name | Type | Description |
|---|---|---|
| `refined_mask` | MASK | Sub-pixel alpha mask. Range `[0, 1]`. Inside core = `1.0` exact, outside core = `0.0` exact, feather band in between. |

**Sample workflow:** [workflows/S-02-subpixel-mask-refiner.json](workflows/S-02-subpixel-mask-refiner.json)

**Run the sample:** copy `workflows/sample_binary_mask.png` → `ComfyUI/input/sample_binary_mask.png`, then Load the workflow and press Queue Prompt.

![screenshot](workflows/S-02-subpixel-mask-refiner-screenshot.png)

### Limitations

- **Square kernel (Chebyshev / L∞ metric).** Erosion and dilation use a square kernel, not a Euclidean disk — with `radius > 16`, mask edges look slightly boxy rather than rounded. Disk-kernel option deferred to v0.2.
- **v1 is float32 only.** Unlike N-01, the mask refiner does not expose a `precision` pin — `feather_sigma > 0` requires enough floating-point precision for the Gaussian. Float16 deferred to v0.2 once lessons from N-01 settle.
- See the Note node inside [workflows/S-02-subpixel-mask-refiner.json](workflows/S-02-subpixel-mask-refiner.json) for the full invariant description plus the `er=dr=0` edge case.

## N-03 Edge-Aware Skin Smoother

Edge-preserving bilateral smoothing for portrait skin retouch. Smooths flat regions (cheeks, forehead) while preserving sharp edges (eyes, lips, hair). Typical pro dose is 30–50% (`strength=0.4`). An optional `mask` input gates smoothing to a specific region — connect the N-02 refined mask upstream to smooth skin only, leaving eyes and hair untouched.

**Inputs:**

| Name | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | ComfyUI IMAGE tensor (BHWC, float32 `[0, 1]`). RGB only. |
| `strength` | FLOAT | `0.4` | Blend between smoothed and original. `0.0` = identity (bypass), `1.0` = full smoothing. Typical pro dose 0.3–0.5. Range 0.0..1.0 (step 0.01). |
| `sigma_color` | FLOAT | `0.1` | Intensity sigma on the `[0, 1]` image scale — **not** the 8-bit 10–50 range from OpenCV docs. Small values preserve edges; large values smooth across weak edges. Range 0.01..0.5 (step 0.01). |
| `sigma_space` | FLOAT | `6.0` | Spatial sigma in pixels. Larger = wider spatial influence = stronger smoothing. Kernel size auto-sized to `2*ceil(3*sigma_space)+1`. Range 1.0..8.0 (v1.1 cap). For wider smoothing, downsample the image first with an upstream Resize node. |
| `device` | COMBO | `auto` | Compute device. `auto` picks CUDA if available, else CPU. Explicit `cuda` raises if CUDA is unavailable. `cpu` forces CPU (slow but deterministic). |
| `tile_mode` | BOOLEAN | `False` | Enable 512×512 tile processing to avoid OOM on large images. Required for 4K+ or `sigma_space > 4` on most GPUs. Leave off for small images (≤1K) for max speed. |
| `mask` | MASK | *(optional)* | Optional region gate (BHW float32 `[0, 1]`). Where `mask=0` the output equals the input pixel-exact; where `mask=1` full smoothing applies; intermediate values blend. |

**Outputs:**

| Name | Type | Description |
|---|---|---|
| `image` | IMAGE | Smoothed image. Range `[0, 1]`. Same shape and dtype as the input. |

**Sample workflow:** [workflows/S-03-edge-aware-smoother.json](workflows/S-03-edge-aware-smoother.json)

**Run the sample:** copy `workflows/sample_portrait.jpg` → `ComfyUI/input/sample_portrait.jpg`, then Load the workflow and press Queue Prompt. The two PreviewImage nodes render the original and the smoothed result side by side.

![screenshot](workflows/S-03-edge-aware-smoother-screenshot.png)

### Performance & device options (v1.1)

- **`device` pin.** `auto` picks CUDA if a GPU is present, otherwise CPU. Use `cuda` to fail loudly if a GPU is required (raises `RuntimeError` if CUDA is unavailable). Use `cpu` to pin the run to CPU regardless of hardware.
- **`tile_mode` pin.** Off by default. Enable it for 4K+ images or any run with `sigma_space > 4` — the kernel is processed in 512×512 tiles and stitched, trading a small speed hit for an OOM-proof path. Leave it off for images ≤1K for max throughput.
- **`sigma_space` cap = 8.0.** Wider smoothing is deliberately blocked: it blows up the kernel and the memory budget. To smooth wider, downsample the image first (Resize node upstream), run the smoother, and upscale back.
- **2 GB memory guardrail.** A non-tile run whose projected peak memory exceeds ≈2 GB raises a `RuntimeError` early with an actionable message (`enable tile_mode`, `reduce sigma_space`, or `downsample`), instead of crashing with CUDA OOM halfway through.

### Limitations

- **`sigma_color` is on the `[0, 1]` image scale.** OpenCV's `cv2.bilateralFilter` uses 8-bit sigmas in the 10–50 range; they do not port over. Start around `0.05–0.3` for natural skin retouch.
- **CPU path is correctness-only above 1K.** On CPU a `1×3×1024×1024` run takes tens of seconds. For production-sized images prefer GPU, and enable `tile_mode` for anything ≥2K. The memory guardrail will surface the problem early if you forget.
- **Guided-filter mode and float16 are deferred to v2.** The v1 kernel is bilateral-only and float32-only.

## N-04 High-Frequency Detail Masker

Generate a binary detail-preservation mask from high-frequency energy in the image. Feeds downstream `ImageBlend` / `MaskCompose` / `SetLatentNoiseMask` nodes so AI passes (inpaint, upscale, style-transfer) can keep the hair, eyebrows, fabric weave and pore texture intact while the rest of the face is freely repainted. Three high-pass operators are selectable and the threshold is either adaptive (per-image percentile) or deterministic (per-image max-normalized).

**Inputs:**

| Name | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | ComfyUI IMAGE tensor (BHWC, float32 `[0, 1]`). RGB only. |
| `kernel_type` | COMBO | `laplacian` | High-pass operator. `laplacian` is scale-invariant and isotropic (default). `sobel` is directional (emphasizes edges). `fs_gaussian` reuses the N-01 high-pass path. |
| `sensitivity` | FLOAT | `0.5` | Fraction of pixels kept as detail. Higher = more pixels pass. `0.0` → empty mask, `1.0` → full mask. Range 0.0..1.0 (step 0.01). |
| `threshold_mode` | COMBO | `relative_percentile` | `relative_percentile` adapts per image (robust cross-image). `absolute` normalizes by per-image max (deterministic but more sensitive to outliers). |
| `mask` | MASK | *(optional)* | Pre-gate region (BHW float32 `[0, 1]`). Output detail is zeroed where this mask is 0 — useful for restricting detail to the skin region returned by SAM / rembg. |

**Outputs:**

| Name | Type | Description |
|---|---|---|
| `mask_detail` | MASK | Binary detail mask, BHW float32 `[0, 1]`. |

**Sample workflow:** [workflows/S-04-hf-detail-masker.json](workflows/S-04-hf-detail-masker.json)

**Run the sample:** copy `workflows/sample_portrait.jpg` → `ComfyUI/input/sample_portrait.jpg`, then Load the workflow and press Queue Prompt.

### Use cases

- **Post-AI texture protection.** Compose the detail mask into a `SetLatentNoiseMask` so denoise keeps the high-frequency regions unchanged.
- **Hair/eyelash preservation** during inpaint — blend original hair back on top of the inpainted face using the detail mask as alpha.

### Limitations

- **Output is a MASK, not an IMAGE.** Pipe through `MaskCompose`, `ImageBlend` or `SetLatentNoiseMask` to apply it — there is no built-in visualization beyond `MaskPreview`.
- **`sensitivity` is fraction-of-pixels, not a hard luma threshold.** Two images with different noise floors will give different absolute thresholds even at the same `sensitivity` — that is the point of `relative_percentile`.

## N-05 Luminosity Masking

Split an image into three smooth luminosity masks (shadows / midtones / highlights), Photoshop-style, with a partition-of-unity guarantee (`shadows + midtones + highlights ≈ 1.0` per pixel). Use each band as an alpha mask to target local contrast, color grading, dodge/burn or AI denoise only in the bright or dark regions — selection by luminance band, not by shape.

**Inputs:**

| Name | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | ComfyUI IMAGE tensor (BHWC, float32 `[0, 1]`). RGB only. |
| `luminance_source` | COMBO | `lab_l` | Luminance channel. `lab_l` is perceptual (Photoshop default, ~120 ms @ 2K CPU). `ycbcr_y` is the fast path (~7 ms @ 1024 CPU) — use it for realtime preview or CPU-bound pipelines. `hsv_v` is simple max-RGB and less perceptual. |
| `shadow_end` | FLOAT | `0.33` | Upper bound of the shadow band (luminance `[0, 0.5]`). |
| `highlight_start` | FLOAT | `0.67` | Lower bound of the highlight band (luminance `[0.5, 1.0]`). Must be > `shadow_end` (node raises otherwise). |
| `soft_edge` | FLOAT | `0.1` | Smoothstep transition width at both band edges. Smaller = sharper bands, larger = smoother blend. Range 0.01..0.3 (step 0.01). |

**Outputs:**

| Name | Type | Description |
|---|---|---|
| `mask_shadows` | MASK | Shadow band mask, BHW float32 `[0, 1]`. |
| `mask_midtones` | MASK | Midtone band mask, BHW float32 `[0, 1]`. |
| `mask_highlights` | MASK | Highlight band mask, BHW float32 `[0, 1]`. |

**Sample workflow:** [workflows/S-05-luminosity-masking.json](workflows/S-05-luminosity-masking.json)

**Run the sample:** copy `workflows/sample_portrait.jpg` → `ComfyUI/input/sample_portrait.jpg`, then Load the workflow and press Queue Prompt. Three `MaskPreview` nodes render the shadow, midtone and highlight masks separately.

### Use cases

- **Luminosity grading.** Multiply a color LUT only through `mask_midtones` to split-tone without touching shadows and highlights.
- **Band-limited denoise** — restrict denoise to `mask_shadows` so shadow noise is cleaned without softening highlight detail.
- **Local dodge/burn** — apply exposure lift through `mask_shadows` and crush through `mask_highlights`.

### Limitations

- **Performance tradeoff on CPU.** `lab_l` is perceptual but costs ~120 ms @ 2K CPU vs ~7 ms @ 1024 CPU for `ycbcr_y`. Switch to `ycbcr_y` when driving a realtime preview on CPU; stay on `lab_l` for final renders.
- **Partition is approximate near band edges.** With small `soft_edge` (< 0.03) the normalize step cannot preserve exact unity at transition pixels — the sum is rescaled to 1.0, so you will see a ~`soft_edge`-wide blend zone.

## N-06 Landmark Facial Aligner

Align a face to a canonical FFHQ-like frame via 5 landmarks using a similarity transform (rotation + uniform scale + translation — no shear), and return both the aligned image and the inverse transform so you can unwrap the result back onto the original canvas. This is the canonical pre-processing step in front of ControlNet, face-detail and inpaint passes — it gives every output a consistent eye / nose / mouth position so batch operations stay stable.

**Inputs:**

| Name | Type | Default | Description |
|---|---|---|---|
| `image` | IMAGE | — | ComfyUI IMAGE tensor (BHWC, float32 `[0, 1]`). RGB only. |
| `landmarks` | STRING | *(5-point JSON)* | 5-point landmark JSON in order `[L-eye, R-eye, nose, L-mouth, R-mouth]`. Pixel-absolute or normalized — values ≤ 1.5 are auto-treated as normalized. Shape `5x2` for single image or `Bx5x2` for batch. |
| `target_size` | INT | `1024` | Square output size in pixels. Accepts `512 / 768 / 1024` (step 256). `1024` is SDXL-friendly. |
| `padding` | FLOAT | `0.2` | Ratio of canonical frame reserved around the face (hair/chin room). `0.0` = tight crop, `0.5` = half-frame padding. |

**Outputs:**

| Name | Type | Description |
|---|---|---|
| `image_aligned` | IMAGE | Aligned face image at `target_size × target_size`. Range `[0, 1]`. |
| `inverse_matrix_json` | STRING | JSON-serialized `B × 3 × 3` inverse affine matrix (list of 3×3 per batch item). Use it to unwrap the edited aligned face back onto the original canvas. |

**Canonical frame (FFHQ-like).** In normalized coordinates: eyes at `Y=0.40`, nose at `Y=0.55`, mouth at `Y=0.70`, face centered horizontally. Pulled in by `padding` (default `0.2`).

**Sample workflow:** [workflows/S-06-facial-aligner.json](workflows/S-06-facial-aligner.json)

**Run the sample:** copy `workflows/sample_portrait.jpg` → `ComfyUI/input/sample_portrait.jpg`, then Load the workflow and press Queue Prompt. Two `PreviewImage` nodes render the original and the aligned result.

### Use cases

- **Consistent ControlNet / inpaint pipeline.** Align → run `ControlNet` / `KSampler` → unwrap via `inverse_matrix_json` so the edited face lands back in the original composition.
- **Batch portrait grading** — everyone gets the same eye/mouth position before global filters are applied.

### Limitations

- **Manual landmarks are a stop-gap.** In production, feed landmarks from an upstream face detector (InsightFace, MediaPipe). Wrong landmarks = wrong alignment — this node does not sanity-check face geometry beyond the 5×2 shape.
- **Roundtrip bilinear smoothing.** Align + unwrap puts the image through two bilinear resamples, which softens the result by ~34/255 in uint8. Good enough for a retouch chain, not near-lossless — do not chain more than one round.
- **Mediapipe dependency for landmark detection is optional.** The core ships a fallback 5-point JSON so the pack loads without `mediapipe` installed. For automatic landmark detection, pair with a separate face-detect custom node upstream.

## License

Apache-2.0 — see [LICENSE](./LICENSE).

## Contributors

- **JH** ([@jetthuangai](https://github.com/jetthuangai)) — maintainer & product owner.
- Built with AI pair-programming assistance.
