# Changelog

All notable changes to this pack are recorded here. Format follows [Keep a Changelog 1.1](https://keepachangelog.com/en/1.1.0/) and [SemVer](https://semver.org/).

## [Unreleased]

### Fixed

- **N-21 `JHPixelProFaceBeautyBlend`**: auto-resizes masks to image height/width instead of raising `ValueError` on spatial shape mismatch, resolving the S-18 crash when placeholder masks are `64×64`.

### Changed

- **N-17 `JHPixelProToneMatchLUT`**: upgrades generated LUTs from Reinhard LAB mean/std transfer to MKL (Monge-Kantorovich Linear) covariance transfer, capturing cross-channel color correlations with a mean-only fallback for singular covariance references.

### Sample workflows

- **S-18 Face Pipeline v2**: rewires the scaffold mask path to core `ImageToMask` from the loaded image red channel instead of `LoadImage.MASK`, keeping the demo self-contained with image-sized masks.

## [0.8.1] — 2026-04-21

Batch-6 smoke-test patch release for two issues found by JH in real ComfyUI runs after v0.8.0: N-20 Face Warp edge-landmark crashes and N-17 Tone Match LUT wrong-color bias.

### Fixed

- **N-20 `JHPixelProFaceWarp`**: clamps Delaunay triangle bounds to the image rect before OpenCV affine slicing/compositing, preventing `cv2.warpAffine` crashes when MediaPipe landmarks land on or just outside frame edges.
- **N-17 `JHPixelProToneMatchLUT`**: fixes the v0.8.0 identity-HALD LAB histogram-match path that could bake heavy sepia/desaturated bias into generated LUTs.

### Changed

- **N-17 algorithm**: switches generated LUTs to Reinhard-style LAB mean/std transfer with a neutral-gray near-identity guard, producing a more predictable reference cast direction for natural images.

### Dependencies

**Unchanged from v0.8.0 — zero new pinned dependency.** The patch reuses the existing Kornia, MediaPipe, SciPy and OpenCV runtime stack.

## [0.8.0] — 2026-04-21

Batch-6 mega ships **6 new nodes** and closes two roadmap vectors at once: selective color tooling on `/color` (N-15 / N-16 / N-17) and face-pipeline v2 on `/face` (N-19 / N-20 / N-21). Pack now ships **20 live nodes** under the unified `ComfyUI-JH-PixelPro/*` namespace. `/color` expands to **9 nodes** and `/face` expands to **5 nodes**. Batch-6 also validates the one-time Codex full-stack execution path: core math, wrappers, workflows, docs and release bundled in a single autonomous thread.

### Added

- **N-15 `JHPixelProHueSaturationRange`** (`ComfyUI-JH-PixelPro/color`): builds a soft HSV hue-band mask and applies hue rotation + saturation changes only inside that band. Inputs: `IMAGE` + `hue_center` + `band_width` + `hue_shift` + `sat_mult` + `sat_add`. Output: selectively adjusted `IMAGE`.
- **N-16 `JHPixelProSaturationMask`** (`ComfyUI-JH-PixelPro/color`): extracts a saturation-range `MASK` from the HLS S-channel with optional threshold feather. Inputs: `IMAGE` + `sat_min` + `sat_max` + `feather`. Output: `MASK` `(B, H, W)` float32 `[0, 1]`.
- **N-17 `JHPixelProToneMatchLUT`** (`ComfyUI-JH-PixelPro/color`): auto-generates an Adobe Cube 1.0 `.cube` file by histogram-matching an identity HALD to a graded reference frame in LAB space, then exporting the graded HALD. Inputs: `IMAGE` reference + `level` + `filename` + `title`. Output: absolute `STRING` path. `OUTPUT_NODE = True`.
- **N-19 `JHPixelProFaceLandmarks`** (`ComfyUI-JH-PixelPro/face`): dense MediaPipe landmark extraction returning a custom `LANDMARKS` tensor `(B, F, 468, 2)` plus an overlay `IMAGE` with landmark dots. Inputs: `IMAGE` + `max_num_faces` + `min_detection_confidence` + `refine_landmarks` + `draw_overlay`. Outputs: `LANDMARKS`, `IMAGE`.
- **N-20 `JHPixelProFaceWarp`** (`ComfyUI-JH-PixelPro/face`): per-triangle Delaunay face warp using SciPy triangulation + OpenCV affine warps. Inputs: `IMAGE` + `LANDMARKS` source + `LANDMARKS` destination. Output: warped `IMAGE`. Identity invariant covered in test suite.
- **N-21 `JHPixelProFaceBeautyBlend`** (`ComfyUI-JH-PixelPro/face`): mask-aware beauty blend with optional Gaussian feather and global `strength`. Inputs: `IMAGE` base + `IMAGE` retouched + `MASK` + `strength` + `feather`. Output: blended `IMAGE`.
- **3 sample workflows + 6 README sections**: `S-16-selective-color.json`, `S-17-tone-match-lut.json`, `S-18-face-pipeline-v2.json`, plus README coverage for N-15 / N-16 / N-17 / N-19 / N-20 / N-21. The new `/color` subgroup is now 9 nodes and the `/face` subgroup is now 5 nodes, with the custom `LANDMARKS` type introduced for batch-6 chaining.

### Known limitations

- **Face warp is CPU-only and triangle-heavy.** N-20 uses SciPy Delaunay + OpenCV affine compositing on CPU; expect roughly `~100–300 ms` per 1K face depending on landmark spread and triangle coverage.
- **GPU parity NOT EVALUATED for batch-6 nodes.** Functional correctness is covered by the new test suite (`254 passed / 10 skipped` at release), but no CUDA benchmark numbers were recorded on this runner.
- **MediaPipe dense landmarks rely on the tasks backend.** The installed backend emits 478 landmarks internally; the pack truncates to the canonical first 468 to keep shape compatibility stable for N-19 / N-20.
- **Selective-color + tone-match tools are color-statistical, not semantic.** N-15/N-16 operate on hue / saturation bands only, and N-17 transfers LAB histograms rather than scene understanding. Strong inputs can produce aggressive looks.

### Dependencies

**Unchanged from v0.7.0 — zero new pinned dependency.** Batch-6 reuses the existing pack stack: `kornia >= 0.7.0`, `mediapipe >= 0.10.0`, plus the already-available SciPy/OpenCV runtime on this runner for N-20. `scipy>=1.10` was verified present during pre-flight and did not require an explicit new pin for this release.

## [0.7.0] — 2026-04-20

Batch-5 closes the LUT round-trip loop. The new **N-14 `JHPixelProLUTImport`** pairs with N-13 `JHPixelProLUTExport` (v0.6.0) so a color-grade chain developed inside ComfyUI can be exported as a portable Adobe Cube 1.0 `.cube` file, re-applied on fresh sources via N-14, or shipped downstream to DaVinci Resolve 18+ / Premiere Pro 2023+ / OBS Studio 29+ / OCIO 2.2+ and any Cube-1.0-reading tool. The `/color` subgroup consolidates to **6 nodes** (N-05 + N-08 + N-09 + N-12 + N-13 + N-14). Pack now ships **14 live nodes** and the **M4-tail milestone ships**.

### Added

- **N-14 `JHPixelProLUTImport`** (`ComfyUI-JH-PixelPro/color`): Reads a portable Adobe Cube 1.0 (`.cube`) 3D LUT from disk and applies it to a BHWC image via trilinear 3D `grid_sample`. Inputs: `IMAGE` + `filename` STRING (relative to ComfyUI `input/` dir or absolute, `.cube` extension NOT auto-appended on read for explicit intent) + `strength` FLOAT `[0, 1]` default `1.0` (linear blend `in * (1 - s) + lut(in) * s`) + optional `MASK` (spatial gating, `(B, H, W)` or `(B, 1, H, W)` shape both accepted transparently). Output: `IMAGE` LUT-applied (same shape as input). `RETURN_TYPES = ("IMAGE",)`, `FUNCTION = "apply"`. Honors `.cube` `DOMAIN_MIN` / `DOMAIN_MAX` header (default `[0,0,0]` / `[1,1,1]`) with clamp-no-extrapolation policy.
- **Sample workflow `S-15-lut-import.json`** — demo chain `LoadImage → JHPixelProLUTImport(filename="S-14 Demo.cube", strength=0.8) → NH_ImageCompare → PreviewImage` + Note block documenting the round-trip demo, tunables, and caveats. JH smoke test in ComfyUI confirmed trilinear LUT apply round-trip against the `S-14 Demo.cube` file previously exported from the S-14 workflow (closing the Export → Import loop concretely). The demo JSON was extended by JH with an `NH_ImageCompare` node (from the `nh-nodes` community pack) for split-vertical before/after preview; the pack's core N-14 node remains zero-community-dep.
- **Workflow screenshot for S-15 LUT Import**, so README §N-14 now renders its reference image alongside §N-01..§N-13 (consolidated from the `[Unreleased]` buffer cleared at release).

### Known limitations

- **CPU 2K bench throughput**: trilinear 3D `grid_sample` on `2048 × 2048` image with `N=65` LUT runs at roughly `~547 ms` per invocation on the CPU-only runner (measured in `tests/R-20260420-bench-S-15.md`). Acceptable for single-frame preview and batch exports; for realtime monitor grading on 2K+ input, recommend GPU.
- **GPU path NOT EVALUATED** on this runner (`torch.cuda.is_available() == False` during T-24-a bench authorship). JH's RTX 4090 should be well within headroom, but no formal parity number is recorded yet.
- **3-channel RGB only**: `(B, H, W, 3)` input required. No alpha LUT, no multi-channel, no depth. Documented in N-14 wrapper tooltip + README §N-14 Caveats.
- **Trilinear interpolation only**: tetrahedral interpolation (slightly smoother at LUT vertex boundaries) is deferred to v2. Trilinear matches the DaVinci / OBS / OCIO default and most reference implementations.

### Dependencies

**Unchanged from v0.6.0 — zero new dependency.** `core/lut.py`'s new `parse_cube` + `apply_lut_3d` functions use pure `torch.nn.functional.grid_sample` + `pathlib` + `re` stdlib (§2b one-time single-agent exception NOT invoked — canonical 2-agent Codex core + Claude Code wrapper split restored). `kornia >= 0.7.0`, `mediapipe >= 0.10.0`, `opencv-python-headless >= 4.5`, plus core `torch` and `numpy` — all preserved from v0.6.0.

## [0.6.0] — 2026-04-20

Batch-4 ships the LUT portability layer. The new **N-12 / N-13** pair generates an identity HALD tensor, runs it through the existing color-grade chain (tone curve, color match), and exports the result as a valid Adobe Cube 1.0 `.cube` file — letting users bake their ComfyUI grade into a portable artifact readable by DaVinci Resolve 18+, Premiere Pro 2023+, OBS Studio 29+, and any OCIO-compatible tool. `/color` subgroup consolidates to 5 nodes (N-05 + N-08 + N-09 + N-12 + N-13). Pack now ships **13 live nodes** and the **M4-head milestone ships**.

### Added

- **N-12 `JHPixelProHALDIdentity`** (`ComfyUI-JH-PixelPro/color`): Generates an identity HALD image encoding a cube of `N = L²` color samples as a single `L³ × L³` RGB image (ImageMagick HALD convention). Inputs: `level` COMBO `{4, 6, 8, 10, 12}` default `"8"` (L=8 → N=64, image 512×512, cube 262144 samples — industry standard). Outputs: `IMAGE` identity HALD tensor `(1, L³, L³, 3)` float32 `[0, 1]` + `INT` pass-through `level` (wire to N-13 to match cube size).
- **N-13 `JHPixelProLUTExport`** (`ComfyUI-JH-PixelPro/color`): Writes a graded HALD image as an Adobe Cube 1.0 (`.cube`) 3D-LUT file. Inputs: `IMAGE` graded HALD + `INT` level (validates `H == W == L³`) + `STRING` filename (relative to ComfyUI `output/` or absolute; `.cube` extension auto-appended) + `STRING` title (Adobe Cube `TITLE` header). Output: `STRING` absolute resolved path. `OUTPUT_NODE = True`. File body is N³ float lines blue-outer / green-middle / red-innermost, Unix LF separator, `%.6f` precision — readable by DaVinci Resolve / Premiere / OBS / OCIO / LUTCalc / CubeLUT / pillow-lut.
- **Sample workflow `S-14-lut-export.json`** — 3-node chain `N-12 HALD Identity(level=8) → N-09 Tone Curve(s_curve_mild) → N-13 LUT Export(filename="s14_demo.cube")` + Note block documenting the color-only-chain constraint between N-12 and N-13. JH smoke test in ComfyUI real produced a valid Adobe Cube 1.0 `.cube` file (~7 MB = 262144 body lines for level 8). The demo workflow JSON was extended by JH with a `ShowText|pysssss` node wired to `N-13.path` so the resolved output path is visible in the UI — pack's core nodes (`JHPixelProHALDIdentity` / `JHPixelProLUTExport`) remain zero-community-dep.
- **3 workflow screenshots** consolidated from the `[Unreleased]` buffer: S-08 Color Matcher (batch-3 follow-up) + S-09 Tone Curve (batch-3 follow-up) + S-14 LUT Export (batch-4 follow-up). README §N-08 + §N-09 + §N-12 + §N-13 now render reference images like §N-01..§N-07. Workflow JSONs for S-08 / S-09 were extended by JH with optional community-pack utility nodes (`ImageResize+` from `comfyui_essentials`, `NH_ImageCompare` from `nh-nodes`) for richer interactive demos — the pack's core nodes remain zero-community-dep.

### Known limitations

- **Level 12 intermediate VRAM**: `identity_hald(level=12)` produces a `(1, 1728, 1728, 3)` float32 tensor (~35.8 MB) before the color-grade chain runs; downstream operations may briefly 2–3× that footprint. JH's RTX 4090 (24 GB) has no issue; CPU runners with 8 GB+ RAM are fine; low-memory setups should stay at level ≤ 10.
- **GPU path NOT EVALUATED for N-12 / N-13**: `core/lut.py` is pure `torch.arange` + stdlib file I/O, so there is no CUDA/CPU parity math to validate. `identity_hald` trivially runs on any device; `export_cube` moves the tensor to CPU for the file write. No perf-critical path — functional correctness verified via 8 `tests/test_lut.py` AC pass (214 passed / 10 CUDA-skip at release).
- **N-13 is file-I/O bound, not compute-bound**: writing 262144 float triplets is dominated by `open()` + `write()` latency, not tensor math. No CPU/GPU bench applicable by design — no `bench_lut.py` shipped.
- **LUT export validates only 3-channel images**: `(B, H, W, 3)` input required. No alpha / no depth / no multi-channel LUT variant. Users needing ACES / LogC / HLG should bake the transform upstream of N-12. Documented in N-13 wrapper tooltip + README §N-13 Caveats.

### Dependencies

Unchanged from v0.5.0 — zero new dependency. `core/lut.py` is pure `torch.arange` + stdlib file I/O (§2b one-time single-agent exception ratified in T-20 pack precedent for trivial-math nodes with no Kornia / no bench-critical GPU parity). `kornia >= 0.7.0`, `mediapipe >= 0.10.0`, `opencv-python-headless >= 4.5`, plus core `torch` and `numpy` — all preserved from v0.5.0.

## [0.5.0] — 2026-04-19

Batch-3 ships the color-grade trio (N-08 + N-09) and consolidates the new `/color` subgroup for Add Node menu discoverability via a Q-L4 retrofit on N-05. Pack now ships **11 live nodes** and the M3 milestone (color-grade layer) closes.

### Added

- **N-08 `JHPixelProColorMatcher`** (`ComfyUI-JH-PixelPro/color`): Reinhard 2001 statistical color transfer in LAB. Inputs: `IMAGE` target + `IMAGE` reference (must match H × W) + `channels` (`ab` default — preserve target luminance / `lab` — full tone match) + `strength` (0..1, default 1.0) + optional `MASK` stat-gate. Output: `IMAGE` matched. Drop-in for re-anchoring AI-generated skin tones to a pre-AI source plate. Pair downstream of any AI generation step before final composite.
- **N-09 `JHPixelProToneCurve`** (`ComfyUI-JH-PixelPro/color`): 1024-step float LUT from Catmull-Rom cubic interpolation over 8 control points. Inputs: `IMAGE` + `preset` (`linear` / `s_curve_mild` / `s_curve_strong` / `lift_shadows` / `crush_blacks` / `custom`) + `channel` (`rgb_master` / `r` / `g` / `b`) + `points_json` STRING (used only when preset = `custom`) + `strength` (0..1, default 1.0). Output: `IMAGE` toned. Photoshop-style global contrast curves and per-channel color balance — Catmull-Rom guarantees no overshoot at endpoints.
- 2 sample workflows (`S-08-color-matcher.json` 7-node A/B/matched + `S-09-tone-curve.json` 5-node A/B). Screenshots not bundled in this release — workflows are functional drag-drop ready; `JH` to chụp + commit later (pattern T-10 post-release).

### Changed

- **N-05 `JHPixelProLuminosityMasking` category retrofit**: `ComfyUI-JH-PixelPro/filters` → `ComfyUI-JH-PixelPro/color`. **UI-only change** — existing workflow JSON is **not affected** because ComfyUI references nodes by class name (`JHPixelProLuminosityMasking`), not by category path. Pattern precedent: T-14 v0.3.0 Q-NS retrofit across 3 nodes (zero JSON breakage validated). **Rationale**: consolidate luminance / chroma-grade operations under the new `/color` subgroup alongside N-08 + N-09 for Add Node menu discoverability. Post-v0.5.0 `/color` subgroup = N-05 Luminosity + N-08 Color Matcher + N-09 Tone Curve.

### Known limitations

- **N-08 CPU 2K miss**: `channels=ab` ~549 ms / `channels=lab` ~456 ms vs aspirational `<100 ms` bound. Root cause: Kornia `rgb_to_lab` / `lab_to_rgb` round-trip alone takes ~254 ms at 2K (library throughput ceiling, not the masked-stat math). CPU 1K passes the bound (`ab ~111 ms` / `lab ~82 ms`). See README §N-08 §Performance H3 for honest disclosure table.
- **N-09 CPU 2K miss**: `rgb_master` ~225 ms / single-channel ~96 ms vs `<30 ms` bound. Root cause: LUT sampling + 3-channel blend hits the CPU memory-bandwidth ceiling at 2K. CPU 1K passes (`rgb_master ~26 ms` / `r ~18 ms`). See README §N-09 §Performance H3.
- **CUDA benchmarks not evaluated** for N-08 / N-09 (CPU-only runner). Both paths are well-structured for GPU acceleration — Kornia LAB conversion supports CUDA natively, and LUT apply is embarrassingly parallel.

### Dependencies

- **No new dependencies for batch-3**. Kornia ≥ 0.7.0 (used by N-08 LAB round-trip) + torch (LUT Catmull-Rom hand-rolled) + numpy. MediaPipe ≥ 0.10.0 + OpenCV ≥ 4.5 unchanged from v0.4.0.

## [0.4.0] — 2026-04-19

Batch-2 face pipeline ships (N-07 + N-10 + N-11) and closes the community-pack dependency gap — the full face chain `LoadImage → S-07 Lens → S-10 FaceDetect → S-06 Aligner → [AI block] → S-11 Unwrap → Composite` now runs entirely inside `ComfyUI-JH-PixelPro/*`. Pack now ships **9 live nodes**.

### Added

- **N-07 `JHPixelProLensDistortion`** (`ComfyUI-JH-PixelPro/geometry`): Brown–Conrady 5-coefficient lens distortion corrector / simulator. Inputs: `IMAGE` + `preset` COMBO (4 calibrated presets — `canon_24mm_wide` / `sony_85mm_tele` / `gopro_fisheye` / `no_op_identity` — plus `custom`) + `direction` COMBO (`inverse` = rectify, default; `forward` = simulate) + 5 FLOAT widgets `k1..p2`. Output: `IMAGE` rectified. Drop in upstream of S-10 FaceDetect to clean up wide-angle portrait shots before landmark detection.
- **N-10 `JHPixelProFaceDetect`** (`ComfyUI-JH-PixelPro/face`): MediaPipe `FaceLandmarker` (tasks API) wrapper. Inputs: `IMAGE` + `mode` (`single_largest` default / `multi_top_k`) + `max_faces` (1..10) + `confidence_threshold` (0.1..0.95, default 0.5). Outputs: `STRING` `landmarks_json` (list[face][5][2] pixel-abs, S-06-compatible — paste `[0]` into S-06 `landmarks` widget for single-face chains) + `STRING` `bbox_json` + `INT` `face_count`. Auto-downloads `face_landmarker.task` (~5 MB) to `ComfyUI/models/mediapipe/` on first call. Zero community-pack dependency for face detection.
- **N-11 `JHPixelProUnwrapFace`** (`ComfyUI-JH-PixelPro/face`): pair node for S-06 FacialAligner. Consumes the `inverse_matrix_json` from S-06, warps an edited aligned crop back onto the original canvas via `kornia.warp_affine`, and alpha-composites with a feathered face mask. Inputs: `IMAGE` `edited_aligned` + `IMAGE` `original_image` + `STRING` `inverse_matrix_json` (default identity 1×3×3, safe pass-through) + `FLOAT` `feather_radius` (0..128, default 16) + optional `MASK` `mask_override`. Outputs: `IMAGE` `image_composited` + `MASK` `mask_used`. Closes the face-edit chain so ControlNet / IPAdapter / inpaint passes on the canonical-frame face land back in the original composition without losing the rest of the scene.
- 3 sample workflows (`S-07-lens-distortion.json` 5-node A/B preview + `S-10-face-detect.json` 4-node single-largest detect + `S-11-unwrap-face.json` 7-node chain demo with widget-to-input conversion on `inverse_matrix_json`) + inline screenshots (JH manual smoke test verify).

### Changed

- **Display name convention for batch-2**: N-07 uses a descriptive name (`"Lens Distortion Corrector"`); N-10 and N-11 use the pack-branded literal class names (`"JHPixelProFaceDetect"` / `"JHPixelProUnwrapFace"`) per the Q-NS retrofit precedent established in batch-1. The `ComfyUI-JH-PixelPro/face` category is new.

### Known limitations

- **N-11 CPU 2K is warp-dominated (~217 ms)** — misses the aspirational `< 60 ms` target because `kornia.warp_affine` over the full canvas takes ~110 ms even with `feather_radius = 0`. CUDA recommended for production 2K+. A bbox-crop fast path (warp only the affected canvas region instead of the full canvas) is a candidate optimization for v0.5+. See README §N-11 §Performance H3 honest disclosure table for full numbers; bench report at `R-20260419-bench-S-11.md`.
- **N-10 `bbox_json[*].conf` is the threshold-gate metadata** that admitted the detection (it echoes `confidence_threshold`), **NOT** MediaPipe's per-face detector probability — the `FaceLandmarker` tasks API does not expose a per-face score. Use `mode = multi_top_k` + bbox area for ranking subjects, not the `conf` field.
- **N-10 `confidence_threshold > 0.85` may miss faces on typical portraits** (the `sample_portrait.jpg` fixture tested ceiling ~0.85). Default `0.5` is balanced; raise only for strict crowd-filtering scenarios.
- **CUDA benchmarks not evaluated** for N-07 / N-10 / N-11 (CPU-only runner). JH GPU follow-up is non-blocking — the integration path is validated end-to-end on CPU with `sample_portrait.jpg` (chain S-10 → S-06 produces aligned `(1, 1024, 1024, 3)`; N-11 round-trip identity drift `0.0064`, well under the spec budget `~0.04–0.13`).

### Dependencies

- **New**: `mediapipe >= 0.10.0` for N-10 (tasks API, Python 3.12+ compatible).
- Unchanged: Kornia ≥ 0.7.0, OpenCV ≥ 4.5.

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
