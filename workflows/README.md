# Workflows

Sample workflows that demo each node in the `ComfyUI-JH-PixelPro` pack. Every `.json` file is a graph you can load straight into ComfyUI (menu **Load** → pick the file) or drag-and-drop onto the canvas.

## Catalog

| File | Main node | Description |
|---|---|---|
| [S-01-frequency-separation.json](S-01-frequency-separation.json) | `JHPixelProFrequencySeparation` | Split an image into low-freq (smooth) and high-freq (detail) layers. Professional retouch demo. An in-canvas Note explains the invariant `low + high = original` and why the `high` pin can contain negative values. |
| [S-02-subpixel-mask-refiner.json](S-02-subpixel-mask-refiner.json) | `JHPixelProSubPixelMaskRefiner` | Feather a binary mask into a sub-pixel alpha mask for cutout / compositing. Graph: LoadImage → ImageToMask → SubPixelMaskRefiner → MaskPreview. The Note covers the invariant and the Chebyshev-kernel quirk. |
| [S-03-edge-aware-smoother.json](S-03-edge-aware-smoother.json) | `JHPixelProEdgeAwareSmoother` | Edge-preserving bilateral smoothing for portrait retouch. A/B preview: one PreviewImage shows the original, the other shows the smoothed result (default dose 40%). Mask pin unplugged in this v1 demo; connect the N-02 refined mask upstream to gate by skin region. |

### v1.4 / v1.5 nodes (N-36..N-44)

Built only from core ComfyUI nodes and this pack, and executed on a live ComfyUI before release. The `-preview.png` files are the real outputs of each graph (not canvas screenshots).

| File | Main node | Description |
|---|---|---|
| [N-36-pyramid-blend.json](N-36-pyramid-blend.json) | `JHPixelProPyramidBlend` | Grade the face with a LUT (N-34) and blend it back through the N-38 face box: the pyramid hides the rectangular mask edge. Result: [N-36-pyramid-blend-preview.png](N-36-pyramid-blend-preview.png). |
| [N-37-clahe-local-contrast.json](N-37-clahe-local-contrast.json) | `JHPixelProCLAHE` | CLAHE on LAB lightness at strength 0.6: flat areas gain local contrast, colours stay put. Result: [N-37-clahe-local-contrast-preview.png](N-37-clahe-local-contrast-preview.png). |
| [N-38-face-detect-lite.json](N-38-face-detect-lite.json) | `JHPixelProFaceDetectLite` | YuNet detection → face mask overlay (N-43) + landmarks straight into the Facial Aligner (N-06); bbox JSON with real confidence in a PreviewAny. Result: [N-38-face-detect-lite-preview.png](N-38-face-detect-lite-preview.png). |
| [N-39-auto-align.json](N-39-auto-align.json) | `JHPixelProAutoAlign` | A crop + rescale stands in for a second shot; SIFT + RANSAC aligns it back, N-44 measures the residual inside valid_mask. Result: [N-39-auto-align-preview.png](N-39-auto-align-preview.png). |
| [N-40-smart-sharpen.json](N-40-smart-sharpen.json) | `JHPixelProSmartSharpen` | Luminance-only sharpening gated by the N-04 detail mask, with an N-44 heatmap of where pixels changed. Result: [N-40-smart-sharpen-preview.png](N-40-smart-sharpen-preview.png). |
| [N-41-mask-cleanup.json](N-41-mask-cleanup.json) | `JHPixelProMaskCleanup` | Clean the rough sample mask (specks out, nose-side hole filled, 6 px feather); before/after via N-43 overlay and cutout. Result: [N-41-mask-cleanup-preview.png](N-41-mask-cleanup-preview.png). |
| [N-42-face-warp-tps.json](N-42-face-warp-tps.json) | `JHPixelProFaceWarpTPS` | Target landmarks from a zoomed copy of the portrait: the TPS warp enlarges the face while border anchors hold the frame. Result: [N-42-face-warp-tps-preview.png](N-42-face-warp-tps-preview.png). |
| [N-43-mask-preview.json](N-43-mask-preview.json) | `JHPixelProMaskPreview` | The N-16 saturation mask shown as a turbo heatmap and as a magenta overlay with contour. Result: [N-43-mask-preview-preview.png](N-43-mask-preview-preview.png). |
| [N-44-qa-image-diff.json](N-44-qa-image-diff.json) | `JHPixelProQADiff` | Original vs N-03 skin smoothing: SSIM / PSNR / MAE report plus a diff heatmap. Result: [N-44-qa-image-diff-preview.png](N-44-qa-image-diff-preview.png). |

## Usage

ComfyUI's `LoadImage` node only scans `ComfyUI/input/`. To run a sample workflow:

1. Copy `workflows/sample_portrait.jpg` → `ComfyUI/input/sample_portrait.jpg`.
2. Open the **Load** menu in ComfyUI and pick `S-01-frequency-separation.json` (or drag the JSON onto the canvas).
3. Press **Queue Prompt** — the two PreviewImage nodes render `low` (smooth) and `high` (detail — dark because values are near zero).

## Image credits

- `sample_portrait.jpg` — Photo by [cottonbro studio](https://www.pexels.com/@cottonbro/) on Pexels. Source: <https://www.pexels.com/photo/close-up-photo-of-woman-s-beautiful-face-6567969/>. License: [Pexels Content License](https://www.pexels.com/license/) (free for commercial + non-commercial redistribution; attribution is not required but recommended as best practice).
- `sample_binary_mask.png` — derivative work: a SAM / rembg cutout (subject mask) produced from `sample_portrait.jpg`. Original image: Photo by [cottonbro studio](https://www.pexels.com/@cottonbro/) on Pexels (<https://www.pexels.com/photo/close-up-photo-of-woman-s-beautiful-face-6567969/>). License: [Pexels Content License](https://www.pexels.com/license/) (derivative redistribution permitted).
