# Workflows

Sample workflows that demo each node in the `ComfyUI-JH-PixelPro` pack. Every `.json` file is a graph you can load straight into ComfyUI (menu **Load** → pick the file) or drag-and-drop onto the canvas.

## Catalog

| File | Main node | Description |
|---|---|---|
| [S-01-frequency-separation.json](S-01-frequency-separation.json) | `JHPixelProFrequencySeparation` | Split an image into low-freq (smooth) and high-freq (detail) layers. Professional retouch demo. An in-canvas Note explains the invariant `low + high = original` and why the `high` pin can contain negative values. |
| [S-02-subpixel-mask-refiner.json](S-02-subpixel-mask-refiner.json) | `JHPixelProSubPixelMaskRefiner` | Feather a binary mask into a sub-pixel alpha mask for cutout / compositing. Graph: LoadImage → ImageToMask → SubPixelMaskRefiner → MaskPreview. The Note covers the invariant and the Chebyshev-kernel quirk. |

## Usage

ComfyUI's `LoadImage` node only scans `ComfyUI/input/`. To run a sample workflow:

1. Copy `workflows/sample_portrait.jpg` → `ComfyUI/input/sample_portrait.jpg`.
2. Open the **Load** menu in ComfyUI and pick `S-01-frequency-separation.json` (or drag the JSON onto the canvas).
3. Press **Queue Prompt** — the two PreviewImage nodes render `low` (smooth) and `high` (detail — dark because values are near zero).

## Image credits

- `sample_portrait.jpg` — Photo by [cottonbro studio](https://www.pexels.com/@cottonbro/) on Pexels. Source: <https://www.pexels.com/photo/close-up-photo-of-woman-s-beautiful-face-6567969/>. License: [Pexels Content License](https://www.pexels.com/license/) (free for commercial + non-commercial redistribution; attribution is not required but recommended as best practice).
- `sample_binary_mask.png` — derivative work: a SAM / rembg cutout (subject mask) produced from `sample_portrait.jpg`. Original image: Photo by [cottonbro studio](https://www.pexels.com/@cottonbro/) on Pexels (<https://www.pexels.com/photo/close-up-photo-of-woman-s-beautiful-face-6567969/>). License: [Pexels Content License](https://www.pexels.com/license/) (derivative redistribution permitted).
