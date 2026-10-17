# Examples

Short, practical ways into the pack. Each page or workflow gives a concrete chain you can load, run and then adapt to your own photos.

## Example pages

- [LUT Preset Gallery](lut-preset-gallery.md): before/after comparisons for all six bundled LUT presets.
- [Skin Tone Retouch Chain](skin-tone-retouch.md): splitting a portrait into shadow, midtone and highlight masks for targeted retouching.

## Sample workflows

The full catalogue, with a description of each graph and the image credits, is in the [workflows README](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/README.md). Some good places to start:

- [Frequency Separation and Recombine](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-01-frequency-separation.json)
- [Skin Smoother](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-03-edge-aware-smoother.json)
- [Layer compositing: 5-layer cinematic grade](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-27-layer-compositing-5layer-cinematic.json)
- [LUT Preset](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-34-lut-preset.json)
- [Skin Tone Tri-Region](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-35-skin-tone-tri-region.json) (needs the external A Person Mask Generator pack)
- [Pyramid Blend](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-36-pyramid-blend.json)

## How to run a sample workflow

1. Copy `workflows/sample_portrait.jpg` from the repository into `ComfyUI/input/`. The Load Image node only looks in that folder. The mask workflows also need `workflows/sample_binary_mask.png` there.
2. In ComfyUI, open the **Load** menu and pick the workflow `.json` file, or drag it onto the canvas.
3. Press **Queue Prompt**.

## Recommended first path

1. Start with the [Quickstart](../quickstart.md).
2. Run the LUT Preset workflow once to confirm that image input and output are wired correctly.
3. Move to the Skin Tone Tri-Region workflow if your work is portrait retouching. It needs the external A Person Mask Generator pack; most other sample workflows use only core ComfyUI nodes and PixelPro.
4. Try the layer compositing workflows when you want a repeatable finishing grade.
