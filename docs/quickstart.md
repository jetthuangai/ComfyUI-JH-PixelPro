# Get Started in 5 Minutes

This guide gets `ComfyUI-JH-PixelPro` installed, visible in ComfyUI, and running a first workflow with the v1.3.0 node pack.

## 1. Install ComfyUI Manager

Install [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) if it is not already present. Restart ComfyUI after installing Manager so the manager panel is available.

## 2. Install the pack

Open **Manager**, choose **Custom Nodes Manager**, and search for `ComfyUI-JH-PixelPro`. Install the pack and let Manager fetch the Python dependencies.

Manual install is also supported:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jetthuangai/ComfyUI-JH-PixelPro.git
cd ComfyUI-JH-PixelPro
pip install -e .
```

## 3. Restart ComfyUI

Restart ComfyUI after installation. The nodes appear under:

```text
ComfyUI-JH-PixelPro/<category>
```

The v1.3.0 release exposes 34 nodes across color, mask, face, geometry, filters, looks, and compositing.

## 4. Load a starter workflow

Use one of the bundled workflows:

- [N-34 LUT Preset](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-34-lut-preset.json) for a one-node color preset pass.
- [N-35 Skin Tone Tri-Region](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-35-skin-tone-tri-region.json) for portrait tonal masks.
- [N-10 Face Detect](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-10-face-detect.json) for the face pipeline entry point.

Download or drag the workflow JSON into the ComfyUI canvas.

## 5. Replace the placeholder image and run

Replace the `LoadImage` placeholder with your portrait or source image, then queue the prompt. Start with these simple checks:

- For N-34, switch the `preset` dropdown and compare the output image.
- For N-35, preview the three returned masks: `shadow_mask`, `midtone_mask`, and `highlight_mask`.
- For mask work, open [N-29 Alpha Matte Extractor](nodes/n29-alpha-matte-extractor.md) and [N-33 Mask Edge Smoother](nodes/n33-mask-edge-smoother.md) after you have a usable source mask.

## Next links

- [Examples](examples/index.md)
- [LUT preset gallery](examples/lut-preset-gallery.md)
- [N-34 Preset Pack LUT](nodes/lut-preset.md)
- [N-35 Skin Tone Tri-Region](nodes/skin-tone-tri-region.md)
- [Latest release](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/releases/latest)
