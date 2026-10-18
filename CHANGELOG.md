# Changelog

All notable changes to this pack are recorded here. Format follows [Keep a Changelog 1.1](https://keepachangelog.com/en/1.1.0/) and [SemVer](https://semver.org/).

## [Unreleased]

### Added

- Nothing yet.

## [1.10.0] — 2026-10-18

### Added

- **Mask: Sky** (`JHPixelProSkyMask`): soft sky mask without an AI model, from sky colours (blue, overcast, sunset at half weight), smooth texture and openness to the top of the frame; in a blue-sky photo warm whites (sunlit walls) are left out; edges refined on the full image with a colour guided filter. About 0.2 s per photo.
- **Color: HDR Merge (bracket)** (`JHPixelProHDRMerge`): fuses a bracketed batch (any order) by exposure fusion (Mertens et al. 2007), with MTB alignment for hand-held shifts and deghosting against the middle exposure. Display-ready output, no tone mapping step.
- **QA: Cull Score** (`JHPixelProCullScore`): ranks a batch by sharpness (tile Laplacian, faces when found, relative to the sharpest shot of the same scene), exposure, closed eyes (MediaPipe, optional) and near-duplicates (dHash); outputs the keepers best first, a JSON report and the count.
- Sample workflows N-61 … N-63. Pack now registers **64 nodes** (color 21, mask 13, face 9, filters 7, compositing 5, geometry 4, depth 2, qa 2, looks 1).

## [1.9.0] — 2026-10-18

### Added

- **Filters: Heal / Content-Aware Fill** (`JHPixelProHeal`): paint a mask over what should go and it is rebuilt from the rest of the photo. `content_aware` is exemplar-based completion (Wexler et al. 2007) with a parallel PatchMatch search over an image pyramid, so texture and structure continue into the hole; `smooth` is a soft colour fill. `grow`, `feather` and `context`; parts of a mask far apart are healed separately. GPU 1–3 s for typical masks on an 8 MP photo, under 1.1 GB at 4K.
- **Face: Auto Blemish** (`JHPixelProAutoBlemish`): finds faces (YuNet) and heals pimples, spots and small marks on their skin with the Dust & Spot texture fill, so pores continue. Eyes, brows and mouth are protected from the landmarks; each face is processed in its own crop with a spot size from its own width. Optional skin `mask` (body, hands) and `protect_mask`. Outputs the image, the healed-spot mask and the face count.
- **Color: Defringe** (`JHPixelProDefringe`): removes purple (and optionally green) fringes along high-contrast edges by moving their colour toward the nearby non-fringe colour, lightness kept, and corrects lateral chromatic aberration (red/blue plane scale measured automatically, applied only when it clearly helps).
- Sample workflows N-58 … N-60 and `sample_heal_mask.png`. Pack now registers **61 nodes** (color 20, mask 12, face 9, filters 7, compositing 5, geometry 4, depth 2, looks 1, qa 1).

### Changed

- Dust & Spot Removal is faster on large spots (separable dilation); results are unchanged (checked against the v1.7.0 golden).

## [1.8.1] — 2026-10-18

### Changed

- Auto Tone brightens more gently: dark photos are lifted toward a 12 % key (was 18 %) and by at most +1 EV (was +1.5), so moody, low-key photos keep their mood. Found on real test photos where full strength lit up an intentionally dark fashion shot.

## [1.8.0] — 2026-10-18

### Added

- **Auto: Tone & White Balance** (`JHPixelProAutoTone`, Color): one-click exposure, Whites/Blacks, Highlights/Shadows and colour-cast correction, measured on a 512 px copy (same result at any size). Bright photos are darkened only when really overexposed; on a mostly white frame (product on white, high-key) only the subject is judged. White balance uses only near-grey pixels, so skin, foliage or a sunset never pull it, and a photo without any is left alone. Optional automatic B&W mix. Outputs the image and `params_json` (ColorLab settings) to fine-tune in ColorLab or reuse with Apply ColorLab Preset. Constants are hand-tuned on test photos, not fitted to Adobe's Auto.
- **Auto: Upright** (`JHPixelProAutoUpright`, Geometry): levels horizons (`level`) and makes converging verticals parallel (`vertical`, `full`; `auto` picks) from straight lines (OpenCV LSD, vanishing points by RANSAC, virtual camera rotation). A confidence gate leaves photos without enough straight lines untouched (`confidence` 0); `max_angle` caps steep corrections; `crop = valid` zooms until no empty corners remain. Also outputs the applied matrices.
- **Mask: Decontaminate Colors** (`JHPixelProDecontaminateColors`, Mask): removes the background colour (green screen, sky) from a cut-out's soft edges by solving the compositing equation from nearby subject and background colours; alpha 0 and 1 pixels are untouched.
- **Filters: Motion / Zoom Blur** (`JHPixelProMotionZoomBlur`, Filters): motion blur along an angle or a zoom burst toward a point, optional mask; about 2·√length warps, runs on the GPU when there is one.
- Edge-Aware Mask Refiner: optional `guide_mode = rgb` (colour guide; also follows edges between colours of similar brightness). Default `luma` is unchanged.
- Dust & Spot Removal: optional `fill_mode = texture` (copies the best-matching nearby patch and matches its brightness, so skin pores and grain continue through the spot). Default `blur` is unchanged.
- Sample workflows N-54 … N-57 and two Pexels sample photos (`sample_cafe.jpg`, `sample_architecture.jpg`). Pack now registers **58 nodes** (color 19, mask 12, face 8, filters 6, compositing 5, geometry 4, depth 2, looks 1, qa 1).

### Notes

- Old workflows render exactly as in v1.7.0: the Refiner and Dust & Spot defaults are checked against a golden made from the v1.7.0 code.
- Peak VRAM at 4K: Auto Tone about 1 GB, Decontaminate about 0.9 GB, Motion / Zoom under 0.9 GB, Upright under 0.3 GB, Dust texture fill under 0.2 GB.

## [1.7.0] — 2026-10-18

### Added

- **Shadows/Highlights Pro** (`JHPixelProShadowsHighlights`, Color): lift shadows and recover highlights without the flat "cheap HDR" look. `camera_raw` mode is a local Laplacian filter on log luminance (big edges keep no halo); `photoshop` mode follows Image › Adjustments › Shadows/Highlights (Tone, Radius, Color, Midtone in the optional `ps_*` inputs). Optional mask; runs on the GPU when there is one (4K ≈ 0.1 s, < 1 GB).
- **ColorLab, all optional and off by default** (old workflows render exactly as in v1.6.0, checked against a golden made from the v1.6.0 code):
  - `tone_engine = local`: edge-aware Highlights/Shadows, Clarity from a guided filter, dark-channel Dehaze and a Contrast curve that keeps black and white in place.
  - `wb_temperature` / `wb_tint` (relative, luminance kept).
  - A second output, `params_json`, with every changed setting, and a `params_json` input whose numbers add onto the sliders, so one ColorLab can feed another.
  - Preset-only settings carried in `params_json`: point curves (master and R/G/B), camera calibration, film grain, post-crop vignette and a magenta B&W mixer.
  - Tabbed layout: Basic / HSL / Grading / B&W / Options, edited-tab dots, per-tab Reset, an 8-colour HSL picker and three Lightroom-style colour-grading wheels. Display only: saved workflows and API prompts are unchanged.
- **Apply ColorLab Preset** (`JHPixelProColorLabApply`, Color): put a ColorLab look on any image or batch, with `strength` (0–2) and an optional mask.
- **Lightroom Preset (XMP)** (`JHPixelProLightroomPreset`, Color): applies Lightroom / Camera Raw presets (`.xmp`, and classic `.lrtemplate` read as data only) from `ComfyUI/models/lightroom_presets/`. Translates Basic, presence, white balance (Kelvin approximated against the as-shot white), HSL, colour grading, B&W, point and parametric curves, calibration, grain and vignette; outputs the image, `params_json` and a report of anything not supported (creative profiles, local masks, lens and detail settings). Thumbnail browser on the node: collections from sub-folders plus Favorites and Recent, search, paging, favourite stars, thumbnails of your own photo after the first run. Two sample presets ship in `presets/lightroom/`.
- **Lightroom Preset Contact Sheet** (`JHPixelProLightroomPresetSheet`, Color): a labelled grid of a whole preset collection on one photo.
- Sample workflows N-50 … N-53 with real result previews. Pack now registers **54 nodes** (color 18, mask 11, face 8, filters 5, compositing 5, geometry 3, depth 2, looks 1, qa 1).

### Changed

- `mediapipe>=0.10.33` (the face nodes need the Tasks API of that release; `0.10.0` was too loose).

### Notes

- Shadows/Highlights and the Camera Raw tone tools translate constants from PhotoCraft (MIT); see `THIRD_PARTY_NOTICES.md`. Matches to Lightroom/Photoshop are approximations measured on test charts, not guarantees.

## [1.6.0] — 2026-10-18

### Added

- **Depth Anything V2** (`JHPixelProDepthAnything`, Depth): relative depth map from one photo (1 = nearest), run with ONNX Runtime (CUDA when available). The `small` model (Apache-2.0, ~100 MB) is downloaded on first use to `ComfyUI/models/depth_anything/` from a pinned Hugging Face revision and checked by SHA256; `base` and `large` are CC-BY-NC-4.0 (non-commercial) and flagged in the tooltip.
- **Depth Lens Blur** (`JHPixelProDepthLensBlur`, Depth): photographic background/foreground blur from depth with disc or gaussian bokeh, `focus_depth` / `depth_of_field` / `max_blur`, and a `focus_mask` that focuses on a subject (for example a face mask). Without a `depth` input it estimates depth itself with Depth Anything. 2048² on CUDA: ~35 ms, ~0.4 GB.
- **SAM One-Click Mask** (`JHPixelProSAMMask`, Mask): Segment Anything from positive/negative points, the `bbox_json` of Face Detect / Face Detect Lite, or a rough `mask_hint`. Models: `vit_b` (default, ~3 GB VRAM), `mobile_sam` (~0.4 GB), `vit_l`, `vit_h`; checkpoints live in `ComfyUI/models/sams/` and are downloaded on first use with SHA256 checks.
- **Palette Extract** (`JHPixelProPaletteExtract`, Color): dominant colours by k-means++ in LAB, sorted by share: a swatch strip, a HEX/RGB/share JSON and a quantized image; optional mask.
- **Dust & Spot Removal** (`JHPixelProDustSpotRemoval`, Filters): finds small compact spots (dark, bright or both) in calm areas, skips lashes, hair and edges, and fills them from the surroundings; outputs the cleaned image and the spot mask. Optional `mask` / `protect_mask`.
- **Color Matcher / Color Matcher Region `method`**: `stats` (default, unchanged output) or `histogram` (full per-channel distribution matching).
- **ColorLab `exposure_mode`**: `display` (default, unchanged) or `linear` (exposure in linear light, like a camera stop).
- `keep_model_loaded` on SAM One-Click, Depth Anything and Depth Lens Blur (default on). Turn it off to free the model's VRAM after the run; ComfyUI's *Free memory* button does not reach these models.
- Sample workflows N-45 … N-49 (lens blur, SAM, palette, dust, stats vs histogram) with real result previews, and `workflows/sample_dust.jpg` (the sample portrait with 70 synthetic dust specks).
- New docs category **Depth**; pack now registers **50 nodes** (color 14, mask 11, face 8, filters 5, compositing 5, geometry 3, depth 2, looks 1, qa 1).

### Notes

- `onnxruntime` (or `onnxruntime-gpu`) is an optional dependency, needed only for Depth Anything and for Depth Lens Blur without a `depth` input. The node raises a clear error if it is missing.

## [1.5.4] — 2026-10-18

### Added

- **Frequency Recombine** (`JHPixelProFrequencyRecombine`, Filters): rebuilds an image from Frequency Separation's `low` and `high` layers (`low + detail × high`, clamped once at the end). `detail = 1` restores the original exactly, above 1 boosts texture, below 1 softens it. ComfyUI has no non-clamping image add, so the separation could not be put back together before; the old tooltip pointed to a node that did not exist. Pack now registers **45 nodes** (`/filters` 4).

### Fixed

- **Layer Group:** `group_opacity` was written to both opacity and fill, which multiply, so 0.5 acted like 0.25. It is now applied once.
- **Blend mode `dissolve`:** behaved exactly like `normal` (a smooth mix). It now scatters whole pixels like Photoshop: each pixel is either the layer or the base, with probability opacity × fill × mask, using a fixed seed so renders repeat.
- **Sample workflows are self-contained.** 14 workflows that needed other node packs (A Person Mask Generator, Essentials, LayerStyle, image-compare and show-text nodes) or images missing from the repo were rebuilt with core ComfyUI and PixelPro nodes only, run on a live ComfyUI, and given a real `-preview.png` result. Three more now load `sample_portrait.jpg` instead of a missing `sample.jpg`. `N-35-skin-tone-tri-region.json` still uses the external person-mask node.

## [1.5.3] — 2026-10-18

### Changed

- **Documentation rewritten, English and Vietnamese.** New README (+ `README.vi.md`) and docs site with an EN/VI language switch (mkdocs-static-i18n): every node page now has a plain-language guide, quick start, settings table, retouching tips, an "under the hood" section and a real before/after image (all 44 generated by running the nodes on the sample portrait). Nodes are referred to by name; internal IDs are gone from user-facing text, tooltips and messages.
- Display names: `JHPixelProFaceDetect` → "Face Detect (MediaPipe)", `JHPixelProUnwrapFace` → "Unwrap Face (paste back)". Saved workflows are unaffected (they reference the class name).
- Skin Smoother `tile_mode` tooltip describes the v1.5.2 budget-sized tiles.

### Fixed

- **Face Landmarks overlay:** landmarks that MediaPipe places outside the left/top edge (cropped faces) painted full-height green stripes; out-of-frame points are now skipped.
- **Unwrap Face:** the default feather straddled the edge of the warped crop and blended in the zero padding outside it, leaving a dark seam around pasted-back faces. The mask is now inset by 2 × `feather_radius` before blurring, so the soft edge stays inside the crop.

## [1.5.2] — 2026-10-18

### Fixed

- **N-03 Edge-Aware Skin Smoother `tile_mode`:** tiles were a fixed 512 px, but Kornia's bilateral filter keeps ~4x the unfold size in temporaries, so a full-size portrait (2333x3500, sigma 6) peaked at ~20 GB VRAM. With other models resident this spilled into shared system memory and froze or crashed ComfyUI (reported with the N-44 sample workflow). Tile size is now derived from the 2 GB budget: peak 1.8 GB, and 20x faster on the same image (53.7 s -> 2.7 s). Output is unchanged (tiles overlap by the kernel radius).

## [1.5.1] — 2026-10-17

### Added

- Sample workflows for N-36..N-44 (`workflows/N-36-pyramid-blend.json` … `N-44-qa-image-diff.json`), built from core ComfyUI nodes and this pack only, each executed on a live ComfyUI; `-preview.png` files are the real outputs. Linked from the docs pages and `workflows/README.md`.
- **N-42 Face Warp (TPS smooth) `smoothness`** widget (default `1.0`, `0` = exact interpolation). Spline regularisation absorbs landmark detection noise between two photos, which folded/rippled the image with exact interpolation (found while building the N-42 sample).

### Fixed

- **N-19 Face Landmarks / N-10 Face Detect:** crashed in ComfyUI with "incompatible constructor arguments" when the image came from ImageScale/ImageCrop (channel-planar strides); the frame is now passed to MediaPipe as a C-contiguous array.

### Changed

- **N-42:** default output is now regularised (`smoothness = 1`); set `0` to restore the v1.5.0 exact behaviour. Clean edits move by < 1 px (median 0.01 px, p95 0.9 px for a 6% zoom on the sample portrait). `smoothness` is an optional input, so v1.5.0 API-format prompts still queue.

## [1.5.0] — 2026-10-17

### Added

- **N-40 `JHPixelProSmartSharpen`** (Filters: Smart Sharpen, `/filters`): unsharp mask with amount, radius, soft threshold, luminance-only mode (BT.601 luma delta, chroma untouched) and an optional mask gate. Matches `kornia.filters.unsharp_mask` at amount 1 / threshold 0.
- **N-41 `JHPixelProMaskCleanup`** (Mask: Cleanup, `/mask`): removes islands below `min_island_area`, fills enclosed holes below `max_hole_area`, and feathers with a signed-distance ramp (`2 × feather_px` wide, centred on the edge). 4- or 8-connectivity for the foreground; holes use the complementary connectivity so diagonal outlines still enclose their holes.
- **N-42 `JHPixelProFaceWarpTPS`** (Face Warp (TPS smooth), `/face`): thin-plate-spline alternative to N-20 with the same `LANDMARKS` inputs and fixed border anchors; one smooth field, no triangle seams. Uses all 468 points by default; with a smaller `max_points` budget every moved landmark is still kept. Control points land within ~0.5 px at 2048 px (~18 ms CUDA, ~0.8 s CPU).
- **N-43 `JHPixelProMaskPreview`** (Mask: Preview Overlay, `/mask`): overlay, heatmap (`kornia.color.apply_colormap`, all Kornia colormaps), mask-only and checkerboard cutout views with optional contour; image input optional.
- **N-44 `JHPixelProQADiff`** (QA: Image Diff (SSIM/PSNR), new `/qa` category): SSIM (`kornia.metrics.ssim`), PSNR (capped at 100 dB), MAE, max abs and changed-pixel % with optional mask, plus heatmap / abs / SSIM-map diff image and a JSON report.
- Docs pages for N-40..N-44 and a QA category page.
- Pack now registers **44 nodes**: `/color` 13, `/compositing` 5, `/filters` 3, `/mask` 10, `/geometry` 3, `/face` 8, `/looks` 1, `/qa` 1.

### Fixed

- **N-20 Face Warp (Delaunay):** returns the image unchanged instead of crashing when one side has no detected face (all-NaN landmarks).
- **N-39 Auto Align:** featureless (flat) inputs now always fall back to a plain resize. On Linux CPU, SIFT returned keypoints on rounding noise of a constant image and RANSAC accepted them (CI failure on v1.4.0).

## [1.4.0] — 2026-10-17

### Added

- **N-36 `JHPixelProPyramidBlend`** (Compositing: Pyramid Blend, `/compositing`): Burt–Adelson Laplacian pyramid blending of two images through a mask, built on `kornia.geometry.transform.pyrdown` / `pyrup`. Any image size; the level count is clamped so the coarsest level stays ≥ 8 px.
- **N-37 `JHPixelProCLAHE`** (Color: CLAHE Local Contrast, `/color`): `kornia.enhance.equalize_clahe` on LAB lightness only, with clip limit, grid size, strength and optional mask. a*/b* are preserved.
- **N-38 `JHPixelProFaceDetectLite`** (Face: Detect Lite (YuNet GPU), `/face`): `kornia.contrib.FaceDetector` (YuNet) with N-10-compatible `landmarks_json` / `bbox_json`, real per-face confidence and a face-box `face_mask` output; `detect_size` up to 1024 (640 is best for portraits), tiny inputs are padded instead of crashing. ~31 ms on an RTX 5090 vs ~84 ms for N-10 MediaPipe on the sample portrait.
- **N-39 `JHPixelProAutoAlign`** (Geometry: Auto Align, `/geometry`): SIFT (`kornia.feature.SIFTFeature` + `match_smnn`) or LoFTR matching, Kornia RANSAC homography with a least-squares refit on the inliers or a seeded robust affine fit, `warp_perspective`, valid-coverage mask and full-resolution matrix JSON. Corner error ≤ 0.4 px on exact-shift tests up to 3500 px. Featureless pairs fall back to a plain resize.
- **N-33 Mask Edge Smoother `backend`** optional widget: `gpu` runs a torch joint-bilateral filter (same square window and L1 colour distance as `kornia.filters.joint_bilateral_blur`, verified to 1e-5) on CUDA with O(H·W) memory (~290 MB at 2048²), and always honours the guide image, even without opencv-contrib. About 2x faster than OpenCV from sigma 16 up, slightly slower at small sigma. Default `opencv` is unchanged.
- Docs page for Color Matcher Region; docs pages for N-36..N-39.
- CUDA baseline rows for `tests/bench_mask_refiner.py` (the bench failed with "baseline missing" on GPU hosts).
- Pack now registers **39 nodes**: `/color` 13, `/compositing` 5, `/filters` 2, `/mask` 8, `/geometry` 3, `/face` 7, `/looks` 1.

### Changed

- **N-02 Sub-Pixel Mask Refiner:** the feather Gaussian kernel now covers ±3 sigma (`2·ceil(3σ)+1`, never smaller than the band, capped at twice the image size). It used to be sized by the radii only, so a large `feather_sigma` with small radii was truncated into a narrow box. Edge bands are slightly softer than before at the default settings (radius 4, sigma 2: 9 → 13 taps) and noticeably softer when sigma is large relative to the radii.

## [1.3.1] — 2026-10-17

### Added

- **`JHPixelProColorMatcherRegion`** (Color Matcher Region) (`ComfyUI-JH-PixelPro/color`): region-aware LAB color matching — transfers statistics from an optional reference-mask region into an optional target-mask region. Pack now registers **35 nodes** (`/color` 12).
- Post-v1.3.0 documentation polish: quickstart guide, homepage milestone hero, examples section, and generated N-34 LUT preset visual gallery assets.
- Border and memory regression tests for N-28, N-02 and N-23 ColorLab.

### Fixed

- **N-28 Edge-Aware Mask Refiner:** the guided filter's box filter now uses replicate padding with two separable average pools. The previous zero-padded box filter biased every window touching the frame, so a full mask came back as low as ~0.26 at the corners (radius 16); it now stays exactly 1.0. Interior pixels match the previous output within ~5e-6, and CPU runtime drops ~4x (1024², radius 8).
- **N-02 Sub-Pixel Mask Refiner:** erosion/dilation now use separable `max_pool2d` instead of `kornia.morphology` (unfold engine). Output is bit-identical, but peak GPU memory drops from ~5.6 GB to ~30 MB at radius 16 on a 1024² mask, radius 64 on 2048² (multi-GB to hundreds of GB before, unusable) runs in ~110 MB, and CPU runtime drops 3–16x.
- **N-23 ColorLab Texture / Clarity:** local-contrast blur uses `kornia.filters.box_blur` with replicate padding, removing the bright halo (0.5 → ~0.62) along the image frame. Interior pixels are unchanged.

## [1.3.0] — 2026-04-24

### Added

- **N-34 `JHPixelProLUTPreset`** (`ComfyUI-JH-PixelPro/color`): dropdown-based bundled Adobe Cube LUT preset node with six generic 17-cube presets (`neutral-identity`, `warm-portrait`, `cool-portrait`, `cinematic-teal-orange`, `soft-pastel`, `high-contrast`) and an `intensity` blend slider. Reuses the existing N-14 trilinear LUT engine with zero new runtime dependency.
- **N-35 `JHPixelProSkinToneTriRegion`** (`ComfyUI-JH-PixelPro/face`): Rec.601 luminance split that outputs `shadow_mask`, `midtone_mask`, and `highlight_mask`, with optional `skin_mask` gating and Gaussian-soft normalized boundaries preserving the selected region.
- Two new workflow scaffolds (`workflows/N-34-lut-preset.json`, `workflows/N-35-skin-tone-tri-region.json`) plus unit, node-wrapper, bench, and multi-snapshot baseline coverage for both new nodes.
- Full 34-node public documentation coverage on the mkdocs-material site, including the Batch-14 scaffold, Batch-15 29-node rollout, and the new N-34 / N-35 reference pages.
- CI bench guardrail framework with `multi_snapshot_v1` baseline schema and report-only hosted-runner policy, preserving the Batch-13 L-19 runner-match clause while keeping main CI feasible.

### Changed

- Pack node count expands from 32 to 34. `/color` expands from 10 to 11 nodes, and `/face` expands from 5 to 6 nodes.
- CI architecture is split: `.github/workflows/ci.yml` runs lint + unit tests on main/PR, while `.github/workflows/bench.yml` runs the expanded bench suite on manual dispatch and release tags.
- N-29 Alpha Matte Extractor now exposes `compute_device` and uses the Levin 2008 closed-form matting Laplacian with GPU acceleration support via sparse PyTorch operations.
- README and project metadata now identify `v1.3.0` as the feature-complete 34-node milestone.

### Fixed

- GitHub Actions MediaPipe runtime dependencies are installed in CI (`libgl1`, `libglib2.0-0`, `libgles2`, `libegl1`) so face detection tests can load the TFLite backend on hosted Ubuntu runners.
- Hosted-runner bench cancellations on heavy 4K / iterative-solver cases are no longer main-CI blockers because the expanded bench suite moved to the dedicated non-blocking bench workflow.
- N-29 hair/fur matte aliasing from the older edge-weighted diffusion solver is replaced by the closed-form matting solver.

### Dependencies

- No new runtime dependencies for N-34 or N-35. The release reuses the existing Torch/Kornia/MediaPipe/OpenCV/SciPy stack.

### Migration

- v1.2.1 -> v1.3.0 has no breaking changes. Existing v1.x workflows remain valid.

## [1.1.0] — 2026-04-21

### Added

- **M10 Mask Refinement Pack:** six new mask nodes under `ComfyUI-JH-PixelPro/mask`, expanding mask coverage from 2 to 8 nodes and pack count from 26 to 32 live nodes.
- **N-28 `JHPixelProEdgeAwareMaskRefiner`:** guided-filter style mask refinement that aligns alpha edges to IMAGE detail with optional feathering.
- **N-29 `JHPixelProAlphaMatteExtractor`:** sparse classical alpha-matte extraction from the pack's 3-value trimap convention.
- **N-30 `JHPixelProTrimapBuilder`:** converts binary or soft masks into explicit `0.0` background / `0.5` unknown / `1.0` foreground trimaps, with exported `validate_trimap`.
- **N-31 `JHPixelProMaskMorphology`:** dilate, erode, open, close, gradient, tophat, and blackhat operations with elliptical kernels.
- **N-32 `JHPixelProMaskCombine`:** add, subtract, intersect, union, difference, xor, and multiply mask composition with hard or soft-feather mode.
- **N-33 `JHPixelProMaskEdgeSmoother`:** bilateral mask edge smoothing with optional guide IMAGE and OpenCV contrib joint-bilateral path when available.
- **Benchmark coverage:** six new CPU benchmark files and baselines for N-28..N-33.
- **Workflow scaffolds:** six new sample workflows for N-28..N-33.

### Changed

- Pack node count: 26 → 32.
- `/mask` category expands from 2 → 8 nodes.
- README status, node inventory, requirements, and pyproject metadata now reflect v1.1.0.

### Dependencies

- Added `scipy>=1.10.0` for sparse alpha matte solving.
- Added `opencv-python-headless>=4.8.0` for trimap, morphology, and bilateral mask smoothing helpers.
- Added optional `opencv-contrib-python>=4.8.0` extra for environments that want `cv2.ximgproc` joint-bilateral filtering in N-33.

### Migration

- v1.0.0 → v1.1.0 has no breaking changes. Existing v1.0.0 workflows remain valid.

## [1.0.0] — 2026-04-21

### Added

- **Pre-v1.0 benchmark retrofit:** 5 CPU-only benchmark files for `selective_color`, `tone_match`, `face_pipeline`, `color_lab`, and `blend_modes`, closing bench coverage across N-15/N-16/N-17/N-18/N-20/N-21/N-23/N-24/N-25/N-26/N-27.
- **Benchmark baselines:** committed `tests/bench_baselines/*.json` per-case snapshots with device, shape, mean, and stdev fields generated from the Batch-10 CPU runner.
- **Benchmark guardrail marker:** registered `@pytest.mark.bench_guardrail` plus a 10% threshold constant for future CI enforcement wiring.
- **Smoke-test matrix:** new `docs/smoke-test-matrix.md` 27-row tracker for workflow scaffold, JH smoke status, screenshot coverage, and batch origin.
- **Workflow screenshots:** migrated the post-v0.11.0 batch-9 screenshot update into this release: S-21 ColorLab Basic plus S-26/S-27/S-28 Layer Compositing 2-layer/5-layer/group-clipping, all JH manual smoke-test verified.

### Changed

- **API consistency audit:** completed type annotations for all remaining `core/*.py` and `nodes/*.py` function signatures and added missing `JHPixelPro*` class docstrings.
- **v1.0 release documentation:** README status and pyproject metadata now identify the pack as the first production-ready stable release.
- **CHANGELOG structure:** reset `[Unreleased]` after bundling post-v0.11.0 documentation into the v1.0.0 milestone.

### Fixed

- None. Batch-10 audit surfaced style and coverage gaps only.

### Deprecated

- None.

### Removed

- None.

### Dependencies

- Unchanged from v0.11.0.

### Migration

- v0.11.0 → v1.0.0 has no breaking changes. Existing v0.11.0 workflows remain valid.

## [0.11.0] — 2026-04-21

### Added

**M7 ColorLab (ACR-level Color module):**

- **N-23 `JHPixelProColorLab`** (`ComfyUI-JH-PixelPro/color`): monolithic Adobe Camera Raw-style color suite with **55 parameters** across Basic 11, HSL 24, Color Grading 12, and Gray Mix 8. Pipeline order is fixed: Basic → HSL → Color Grading → Gray Mix. Identity at all-zero plus `gray_enable=False` preserves input at `atol=1e-5`.

**M8 Layer Compositing (Photoshop-style 27 blend modes):**

- **N-24 `JHPixelProLayerStackStart`** introduces the custom `LAYER_STACK` pass-through type from a background `IMAGE`.
- **N-25 `JHPixelProLayerAdd`** appends immutable stack layers with 27 fixed Photoshop-style blend modes, opacity, fill, optional mask, and `clip_to_below`.
- **N-26 `JHPixelProLayerGroup`** eagerly flattens a sub-stack into a grouped parent layer for nested compositing.
- **N-27 `JHPixelProLayerFlatten`** renders `LAYER_STACK` back to `IMAGE`; empty stacks raise `ValueError`.

**Workflow screenshots:**

- Workflow screenshots for batch-8 S-19 Look Select Single and S-20 Look Select Compare 6-Up (post-v0.10.0, JH manual smoke test verified). README §N-22 `JHPixelProLookSelect` now renders reference images for single-node dropdown UX + 6-up parametrized compare.

### Changed

- Pack node count: 21 → 26 (M7 ColorLab +1, M8 Layer Compositing +4).
- Categories: 6 → 7 with new `ComfyUI-JH-PixelPro/compositing`; `/color` expands from 9 to 10 nodes.
- Workflow scaffolds: added S-21..S-28 (5 ColorLab presets + 3 Layer Compositing demos).

### Dependencies

- Zero new dependency. ColorLab and all 27 blend modes are pure PyTorch / `torch.nn.functional`; existing pack dependency set is unchanged from v0.10.0.

## [0.10.0] — 2026-04-21

### ⚠️ BREAKING CHANGES

- **M6 Looks refactor from 6 per-preset node classes to 1 dropdown-based node**. Workflow JSON saved from v0.9.0 **WILL NOT LOAD** — re-create from new scaffolds `S-19-look-select-single.json` or `S-20-look-select-compare-6up.json`.
- Removed classes: `JHPixelProLookCinematicTealOrange`, `JHPixelProLookWarmSkinTone`, `JHPixelProLookMoodyGreen`, `JHPixelProLookFadedFilm`, `JHPixelProLookGoldenHour`, `JHPixelProLookDesaturatedPop`.
- Added class: `JHPixelProLookSelect` with `preset` COMBO input (6 option dropdown).

### Changed

- Pack node count: 26 → 21 (`/looks` category reduced from 6 to 1 node).
- Workflow scaffolds: 7 removed (S-19..S-25 per-preset), 2 added (S-19 single + S-20 compare-6up).

### Rationale

UX feedback from JH smoke-test 2026-04-21 12:48: 6 per-preset nodes cluttered `/looks` menu (scaling concern for 20+ presets future) + rotate-preset required delete/reconnect instead of dropdown toggle. 1-node dropdown matches ComfyUI KSampler/sampler_name convention. Preset JSON framework + `nodes/look_base.py` dispatch unchanged — zero new core math, pure UX refactor.

## [0.9.0] — 2026-04-21

Batch-7 ships the **M6 Looks preset palette**: 6 JSON-driven one-slider look nodes under the new `/looks` category. Pack now ships **26 live nodes** across 6 categories.

### Added

- **N-22 `JHPixelProLookCinematicTealOrange`**, **N-23 `JHPixelProLookWarmSkinTone`**, **N-24 `JHPixelProLookMoodyGreen`**, **N-25 `JHPixelProLookFadedFilm`**, **N-26 `JHPixelProLookGoldenHour`**, and **N-27 `JHPixelProLookDesaturatedPop`**: preset wrapper nodes with `IMAGE` input, continuous `intensity` slider, optional `protect_skin`, and `IMAGE` output.
- **JSON preset framework**: `presets/*.json` schema_version=1 plus `nodes/look_base.py` dispatch routing to existing batch-3/batch-6 primitives.

### Changed

- Pack category count expands from 5 to 6 with new `ComfyUI-JH-PixelPro/looks`; public node count expands from 20 to 26.

### Sample workflows

- Added 7 batch-7 workflow scaffolds: S-19..S-24 individual Look preset demos plus S-25 six-preset chain preview. Also bundles the post-v0.8.2 batch-6 workflow screenshots and JH smoke-test JSON expansions from the former `[Unreleased]` buffer.

## [0.8.2] — 2026-04-21

Batch-6 follow-up smoke-test patch for S-18 mask-shape robustness and higher-fidelity N-17 tone-match LUT generation.

### Fixed

- **N-21 `JHPixelProFaceBeautyBlend`**: auto-resizes masks to image height/width instead of raising `ValueError` on spatial shape mismatch, resolving the S-18 crash when placeholder masks are `64×64`.

### Changed

- **N-17 `JHPixelProToneMatchLUT`**: upgrades generated LUTs from Reinhard LAB mean/std transfer to MKL (Monge-Kantorovich Linear) covariance transfer, capturing cross-channel color correlations with a mean-only fallback for singular covariance references.

### Dependencies

**Unchanged from v0.8.1 — zero new pinned dependency.** The patch uses existing PyTorch linear algebra plus the existing Kornia, MediaPipe, SciPy and OpenCV runtime stack.

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

[Unreleased]: https://github.com/jetthuangai/ComfyUI-JH-PixelPro/compare/v1.5.4...HEAD
[1.5.4]: https://github.com/jetthuangai/ComfyUI-JH-PixelPro/releases/tag/v1.5.4
[1.5.3]: https://github.com/jetthuangai/ComfyUI-JH-PixelPro/releases/tag/v1.5.3
[1.5.2]: https://github.com/jetthuangai/ComfyUI-JH-PixelPro/releases/tag/v1.5.2
[1.5.1]: https://github.com/jetthuangai/ComfyUI-JH-PixelPro/releases/tag/v1.5.1
[1.5.0]: https://github.com/jetthuangai/ComfyUI-JH-PixelPro/releases/tag/v1.5.0
[1.4.0]: https://github.com/jetthuangai/ComfyUI-JH-PixelPro/releases/tag/v1.4.0
[1.3.1]: https://github.com/jetthuangai/ComfyUI-JH-PixelPro/releases/tag/v1.3.1
[1.3.0]: https://github.com/jetthuangai/ComfyUI-JH-PixelPro/releases/tag/v1.3.0
[0.3.0]: https://github.com/jetthuangai/ComfyUI-JH-PixelPro/releases/tag/v0.3.0
[0.2.0]: https://github.com/jetthuangai/ComfyUI-JH-PixelPro/releases/tag/v0.2.0
[0.1.0]: https://github.com/jetthuangai/ComfyUI-JH-PixelPro/releases/tag/v0.1.0
