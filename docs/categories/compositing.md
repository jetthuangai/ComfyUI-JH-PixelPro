# Compositing

These nodes turn ComfyUI into a Photoshop-style layer stack: a background, layers on top with blend modes, opacity, fill, masks and clipping, groups, and a final flatten. They are meant for deterministic finishing passes after generation or retouching, where you want to adjust one layer without redoing the rest. Pyramid Blend is the odd one out: it joins two images through a mask so the seam disappears.

<div class="grid cards" markdown>

-   ![](../assets/nodes/layer-stack-start.jpg)

    **[Layer Stack Start](../nodes/layer-stack-start.md)**

    Makes the background layer of a stack from an image.

-   ![](../assets/nodes/layer-add.jpg)

    **[Layer Add](../nodes/layer-add.md)**

    Adds a layer with blend mode, opacity, fill, mask and clipping.

-   ![](../assets/nodes/layer-group.jpg)

    **[Layer Group](../nodes/layer-group.md)**

    Merges a sub-stack into one layer, like a Photoshop group.

-   ![](../assets/nodes/layer-flatten.jpg)

    **[Layer Flatten](../nodes/layer-flatten.md)**

    Renders the stack back into a normal image.

-   ![](../assets/nodes/pyramid-blend.jpg)

    **[Pyramid Blend](../nodes/pyramid-blend.md)**

    Seamless two-image composite through a mask using multi-band blending.

</div>

## Typical chains

- **Colour grade as layers:** Load Image → [Layer Stack Start](../nodes/layer-stack-start.md) → [Layer Add](../nodes/layer-add.md) (`soft_light`) → [Layer Add](../nodes/layer-add.md) (`screen`) → [Layer Flatten](../nodes/layer-flatten.md) → Preview Image
- **A look as one group:** two [Layer Stack Start](../nodes/layer-stack-start.md) chains → [Layer Group](../nodes/layer-group.md) (`overlay`, opacity 0.8) → [Layer Flatten](../nodes/layer-flatten.md)
- **Paste a retouched face back:** [LUT Preset](../nodes/lut-preset.md) on the face → [Pyramid Blend](../nodes/pyramid-blend.md) through a face mask → Preview Image
