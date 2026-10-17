# Filters

Filters work on the picture itself: texture, edges and smoothness. Split detail from tone with frequency separation, soften skin without losing edges, and sharpen at the end. Together they cover the core of a portrait retouch.

<div class="grid cards" markdown>

-   ![](../assets/nodes/frequency-separation.jpg)

    **[Frequency Separation](../nodes/frequency-separation.md)**

    Splits an image into a smooth colour layer and a fine-detail layer.

-   ![](../assets/nodes/frequency-recombine.jpg)

    **[Frequency Recombine](../nodes/frequency-recombine.md)**

    Puts the low and high layers back together, with a slider to boost or soften texture.

-   ![](../assets/nodes/edge-aware-smoother.jpg)

    **[Skin Smoother](../nodes/edge-aware-smoother.md)**

    Smooths skin with an edge-preserving blur, with optional mask and tiled processing.

-   ![](../assets/nodes/smart-sharpen.jpg)

    **[Smart Sharpen](../nodes/smart-sharpen.md)**

    Unsharp mask with a noise threshold, luminance-only mode and a mask.

-   ![](../assets/nodes/dust-spot-removal.jpg)

    **[Dust & Spot Removal](../nodes/dust-spot-removal.md)**

    Find and heal small dark or bright specks (sensor dust, tiny blemishes), leaving larger shapes alone.

</div>

## Typical chains

- **Natural skin:** Load Image → [Skin Smoother](../nodes/edge-aware-smoother.md) (strength 0.3 to 0.5, with a skin mask) → [Smart Sharpen](../nodes/smart-sharpen.md) → Preview Image
- **Fix colour, keep pores:** Load Image → [Frequency Separation](../nodes/frequency-separation.md) → edit `low` only → [Frequency Recombine](../nodes/frequency-recombine.md)
- **Sharpen only the detail:** [Detail Mask](../nodes/high-freq-detail-masker.md) → `mask` of [Smart Sharpen](../nodes/smart-sharpen.md)
