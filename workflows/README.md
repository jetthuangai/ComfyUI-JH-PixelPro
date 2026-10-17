# Workflows

Sample workflows for the `ComfyUI-JH-PixelPro` pack. Every `.json` file is a graph you can load straight into ComfyUI (menu **Load**, then pick the file) or drag onto the canvas.

Tiếng Việt: mỗi file `.json` là một graph mẫu, nạp bằng menu **Load** hoặc kéo thả vào canvas ComfyUI.

Unless noted, a workflow uses only core ComfyUI nodes and this pack. `-preview.png` images are the real result of running the graph; `-screenshot` images are canvas screenshots and may show older settings.

## Usage

ComfyUI's `LoadImage` node only scans `ComfyUI/input/`.

1. Copy `workflows/sample_portrait.jpg` into `ComfyUI/input/`.
2. For Dust & Spot Removal, also copy `workflows/sample_dust.jpg`.
3. For the mask workflows (Edge-Aware Mask Refiner, Alpha Matte, Trimap Builder, Mask Morphology, Mask Combine, Mask Edge Smoother, Mask Cleanup, Sub-Pixel Mask Refiner), also copy `workflows/sample_binary_mask.png` into `ComfyUI/input/`.
4. In ComfyUI, open **Load** and pick a `.json` file, or drag it onto the canvas.
5. Press **Queue Prompt**.

Tiếng Việt: chép `sample_portrait.jpg` (và `sample_binary_mask.png` cho các workflow mask, `sample_dust.jpg` cho Dust & Spot Removal) vào `ComfyUI/input/`, nạp file `.json`, rồi bấm **Queue Prompt**.

One workflow needs another pack: [Skin Tone Tri-Region](N-35-skin-tone-tri-region.json) uses the person-mask node of **A Person Mask Generator** (install it separately).


## Filters

*Bộ lọc*

| Workflow | What it shows | Result |
|---|---|---|
| [Frequency Separation and Recombine](S-01-frequency-separation.json) | Split the portrait into smooth `low` and texture `high` layers, then rebuild it with Frequency Recombine at detail 1.6. | [image](S-01-frequency-separation-preview.png) |
| [Skin Smoother](S-03-edge-aware-smoother.json) | Edge-preserving smoothing with a before/after preview (strength 0.4). | [image](S-03-edge-aware-smoother-screenshot.png) |
| [Smart Sharpen](N-40-smart-sharpen.json) | Luminance-only sharpening gated by the Detail Mask, with a QA heatmap of changed pixels. | [image](N-40-smart-sharpen-preview.png) |
| [Dust & Spot Removal](N-48-dust-spot-removal.json) | Clears 70 synthetic sensor-dust specks from `sample_dust.jpg` at default settings and marks the spots it found. | [image](N-48-dust-spot-removal-preview.png) |

## Color and tone

*Màu và tông*

| Workflow | What it shows | Result |
|---|---|---|
| [Color Matcher](S-08-color-matcher.json) | A cool-graded copy is matched to a golden-hour reference in LAB. | [image](S-08-color-matcher-preview.png) |
| [Color Matcher: stats vs histogram](N-49-color-matcher-histogram.json) | The same pair matched with `method = stats` and `method = histogram`, side by side. | [image](N-49-color-matcher-histogram-preview.png) |
| [Palette Extract](N-47-palette-extract.json) | Six dominant colours as a strip, a HEX list and a quantized image. | [image](N-47-palette-extract-preview.png) |
| [Tone Curve](S-09-tone-curve.json) | The `s_curve_mild` preset on the portrait. | [image](S-09-tone-curve-preview.png) |
| [LUT Preset](N-34-lut-preset.json) | One bundled look on the portrait; switch the `preset` dropdown to try the others. | — |
| [LUT Export](S-14-lut-export.json) | HALD Identity, Tone Curve and LUT Export write a `.cube` file to `ComfyUI/output/`; no input image needed. | [image](S-14-lut-export-preview.png) |
| [LUT Import](S-15-lut-import.json) | Builds its own LUT and wires the file path into LUT Import, so it runs without extra files. | [image](S-15-lut-import-preview.png) |
| [Selective color (Hue/Saturation per Range and Saturation Mask)](S-16-selective-color.json) | Shift and boost one colour band, with the matching saturation mask as a heatmap. | [image](S-16-selective-color-preview.png) |
| [Tone Match LUT](S-17-tone-match-lut.json) | A golden-hour reference becomes a reusable `.cube` that LUT Import applies to the original. | [image](S-17-tone-match-lut-preview.png) |
| [Luminosity Masking](S-05-luminosity-masking.json) | Shadow, midtone and highlight masks from the portrait. | [image](S-05-luminosity-masking-screenshot.png) |
| [CLAHE Local Contrast](N-37-clahe-local-contrast.json) | Adaptive contrast on LAB lightness only, colours untouched. | [image](N-37-clahe-local-contrast-preview.png) |
| [ColorLab: basic only](S-21-colorlab-basic-only.json) | ColorLab with only the Basic panel adjustments. | [image](S-21-colorlab-basic-only-screenshot.png) |
| [ColorLab: teal and orange](S-22-colorlab-hsl-teal-orange.json) | ColorLab HSL panel used for a teal-and-orange look. | — |
| [ColorLab: cinematic colour grading](S-23-colorlab-color-grading-cinematic.json) | ColorLab colour-grading wheels for a cinematic grade. | — |
| [ColorLab: black and white mix](S-24-colorlab-gray-mix-bw.json) | ColorLab gray mix for a black-and-white conversion. | — |
| [ColorLab: full preset](S-25-colorlab-full-acr-preset.json) | ColorLab with every panel set, like a full Adobe Camera Raw preset. | — |
| [Look Select (single)](S-19-look-select-single.json) | One look applied to one image. | [image](S-19-look-select-single-screenshot.png) |
| [Look Select (six-look compare)](S-20-look-select-compare-6up.json) | All six looks side by side. | [image](S-20-look-select-compare-6up-screenshot.png) |

## Compositing

*Ghép ảnh*

| Workflow | What it shows | Result |
|---|---|---|
| [Layer compositing: 2-layer overlay](S-26-layer-compositing-2layer-overlay.json) | Background plus a graded copy on top with `soft_light` at 80%, rendered by Layer Flatten. | [image](S-26-layer-compositing-2layer-overlay-preview.png) |
| [Layer compositing: 5-layer cinematic grade](S-27-layer-compositing-5layer-cinematic.json) | A five-layer stack for a cinematic finishing grade. | [image](S-27-layer-compositing-5layer-cinematic-screenshot.png) |
| [Layer compositing: group and clipping](S-28-layer-compositing-group-clipping.json) | A nested Layer Group and `clip_to_below` clipping. | [image](S-28-layer-compositing-group-clipping-screenshot.png) |
| [Pyramid Blend](N-36-pyramid-blend.json) | A LUT-graded face blended back through a face box; the pyramid hides the rectangle edge. | [image](N-36-pyramid-blend-preview.png) |

## Masks

*Mask*

| Workflow | What it shows | Result |
|---|---|---|
| [SAM One-Click Mask](N-46-sam-one-click-mask.json) | Face Detect Lite's box prompts Segment Anything (vit_b); Mask Cleanup fills the holes. The checkpoint downloads to `ComfyUI/models/sams/` on first use. | [image](N-46-sam-one-click-mask-preview.png) |
| [Sub-Pixel Mask Refiner](S-02-subpixel-mask-refiner.json) | Feather a binary mask into a sub-pixel alpha mask. | [image](S-02-subpixel-mask-refiner-screenshot.png) |
| [Edge-Aware Mask Refiner](N-28-edge-aware-mask-refiner.json) | A soft mask is snapped onto the real hair and skin edge, guided by the portrait. | [image](N-28-edge-aware-mask-refiner-preview.png) |
| [Alpha Matte](N-29-alpha-matte-extractor.json) | Trimap, then closed-form matting at 25% size. `compute_device = auto` stays on CPU for CPU images; pick `cuda` for roughly 5x speed. | [image](N-29-alpha-matte-extractor-preview.png) |
| [Trimap Builder](N-30-trimap-builder.json) | Sure-subject, unknown and sure-background bands from a subject mask. | [image](N-30-trimap-builder-preview.png) |
| [Mask Morphology](N-31-mask-morphology.json) | `close` at radius 64 seals the small shadow gap by the nose. | [image](N-31-mask-morphology-preview.png) |
| [Mask Combine](N-32-mask-combine.json) | Subject mask intersected with Luminosity Masking highlights, with `soft_feather`. | [image](N-32-mask-combine-preview.png) |
| [Mask Edge Smoother](N-33-mask-edge-smoother.json) | Guided smoothing of stair-stepped mask edges (`backend = gpu` needs an NVIDIA GPU). | [image](N-33-mask-edge-smoother-preview.png) |
| [Mask Cleanup](N-41-mask-cleanup.json) | Remove specks, fill the hole by the nose, add a 6 px feather. | [image](N-41-mask-cleanup-preview.png) |
| [Mask Preview](N-43-mask-preview.json) | A saturation mask shown as a heatmap and as an overlay with contour. | [image](N-43-mask-preview-preview.png) |
| [Detail Mask (High-Frequency Detail Masker)](S-04-hf-detail-masker.json) | A mask of textured areas such as hair, eyes and fabric. | [image](S-04-hf-detail-masker-screenshot.png) |

## Depth

*Độ sâu*

| Workflow | What it shows | Result |
|---|---|---|
| [Depth Lens Blur](N-45-depth-lens-blur.json) | Depth from the built-in Depth Anything V2 Small (downloads ~100 MB on first use, needs `onnxruntime`), focus on the face, crops of the arm before and after. | [image](N-45-depth-lens-blur-preview.png) |

## Face

*Khuôn mặt*

| Workflow | What it shows | Result |
|---|---|---|
| [Face Detect](S-10-face-detect.json) | MediaPipe detection with landmarks and boxes as JSON. | [image](S-10-face-detect-screenshot.png) |
| [Face Detect Lite](N-38-face-detect-lite.json) | YuNet detection with a face-box mask and landmarks fed to the Facial Aligner. | [image](N-38-face-detect-lite-preview.png) |
| [Face pipeline (Face Landmarks, Face Warp, Face Beauty Blend)](S-18-face-pipeline-v2.json) | Dense landmarks, a Delaunay warp, then a blend back onto the base. | [image](S-18-face-pipeline-v2-screenshot.png) |
| [Face Warp TPS](N-42-face-warp-tps.json) | A thin-plate-spline warp that enlarges the face about 6% while the frame edges stay fixed. | [image](N-42-face-warp-tps-preview.png) |
| [Facial Aligner](S-06-facial-aligner.json) | Align a face to a canonical crop from five landmarks. | [image](S-06-facial-aligner-screenshot.png) |
| [Unwrap Face](S-11-unwrap-face.json) | Align, edit, then paste the edited face back onto the original canvas. | [image](S-11-unwrap-face-screenshot.png) |
| [Skin Tone Tri-Region](N-35-skin-tone-tri-region.json) | Highlight, midtone and shadow masks of the skin. Needs the external A Person Mask Generator pack. | [image](N-35-skin-tone-tri-region-screenshot.png) |

## Geometry and QA

*Hình học và kiểm tra*

| Workflow | What it shows | Result |
|---|---|---|
| [Lens Distortion](S-07-lens-distortion.json) | Correct barrel distortion with a lens preset. | [image](S-07-lens-distortion-screenshot.png) |
| [Auto Align](N-39-auto-align.json) | Align a second shot to a reference with SIFT and RANSAC. | [image](N-39-auto-align-preview.png) |
| [QA Image Diff](N-44-qa-image-diff.json) | SSIM, PSNR and MAE report plus a heatmap of where an edit changed the image. | [image](N-44-qa-image-diff-preview.png) |

## Image credits

- `sample_portrait.jpg` — Photo by [cottonbro studio](https://www.pexels.com/@cottonbro/) on Pexels. Source: <https://www.pexels.com/photo/close-up-photo-of-woman-s-beautiful-face-6567969/>. License: [Pexels Content License](https://www.pexels.com/license/) (free for commercial + non-commercial redistribution; attribution is not required but recommended as best practice).
- `sample_binary_mask.png` — derivative work: a SAM / rembg cutout (subject mask) produced from `sample_portrait.jpg`. Original image: Photo by [cottonbro studio](https://www.pexels.com/@cottonbro/) on Pexels (<https://www.pexels.com/photo/close-up-photo-of-woman-s-beautiful-face-6567969/>). License: [Pexels Content License](https://www.pexels.com/license/) (derivative redistribution permitted).
- `sample_dust.jpg` — derivative work: `sample_portrait.jpg` with 70 synthetic soft dark specks added to imitate sensor dust. Same photo, author and [Pexels Content License](https://www.pexels.com/license/).
