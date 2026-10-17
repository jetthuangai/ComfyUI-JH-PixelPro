# Masks

Masks decide *where* an edit happens: which pixels get smoothed, graded, replaced or cut out. This group covers the whole life of a mask, from cleaning up a rough AI selection, to softening and tightening its edge, to extracting a hair-quality cutout, to checking the result.

## Mask basics in plain language

- **Binary mask:** each pixel is either fully selected (1) or not at all (0). Fast and simple, but edges look hard and cut-out.
- **Soft mask / alpha matte:** pixels along the edge hold in-between values, such as 0.5 for a pixel that is half subject, half background. This is what makes hair and fur blend into a new background.
- **Feather:** a controlled soft transition across the edge. It hides the seam, but a plain blur can also eat into the subject.
- **Trimap:** a hint map with three values, `0` = sure background, `0.5` = unknown, `1` = sure subject. Matting only has to solve the grey band, which is why hair edges come out so much better than with a simple threshold.
- **Closed-form matting:** a method that solves the whole soft alpha at once, using the photo's own colours as a guide. It powers [Alpha Matte](../nodes/alpha-matte.md).
- **Guide image:** the original photo, given to a mask node so the edge can follow real boundaries in the picture.

## Pick the right mask tool

| Your problem | Use this node |
|---|---|
| The mask has small floating specks or tiny holes | [Mask Cleanup](../nodes/mask-cleanup.md) |
| The edge is hard and looks like a sticker | [Sub-Pixel Mask Refiner](../nodes/sub-pixel-mask-refiner.md) |
| The edge is jagged or wobbly but the shape is right | [Mask Edge Smoother](../nodes/mask-edge-smoother.md) |
| The mask edge sits a few pixels off the real subject edge | [Edge-Aware Mask Refiner](../nodes/edge-aware-mask-refiner.md) |
| Hair, fur or glass needs real transparency | [Trimap Builder](../nodes/trimap-builder.md), then [Alpha Matte](../nodes/alpha-matte.md) |
| The mask is too big or too small by a few pixels | [Mask Morphology](../nodes/mask-morphology.md) (`dilate` or `erode`) |
| You need only an outline band around the edge | [Mask Morphology](../nodes/mask-morphology.md) (`gradient`) |
| You want to add, subtract or overlap two masks | [Mask Combine](../nodes/mask-combine.md) |
| You want a mask of only the sharp, textured areas | [Detail Mask](../nodes/high-freq-detail-masker.md) |
| You want a mask of shadows, midtones or highlights | [Luminosity Masking](../nodes/luminosity-masking.md) |
| You want to see what the mask actually covers | [Mask Preview](../nodes/mask-preview.md) |
| You want to measure how much an edit changed the image | [QA Image Diff](../nodes/qa-image-diff.md) |

<div class="grid cards" markdown>

-   ![](../assets/nodes/mask-cleanup.jpg)

    **[Mask Cleanup](../nodes/mask-cleanup.md)**

    Removes specks, fills small holes and feathers by distance.

-   ![](../assets/nodes/sub-pixel-mask-refiner.jpg)

    **[Sub-Pixel Mask Refiner](../nodes/sub-pixel-mask-refiner.md)**

    Turns a hard mask into a feathered one that stays solid inside.

-   ![](../assets/nodes/edge-aware-mask-refiner.jpg)

    **[Edge-Aware Mask Refiner](../nodes/edge-aware-mask-refiner.md)**

    Snaps the mask edge onto the real edges of your photo.

-   ![](../assets/nodes/mask-edge-smoother.jpg)

    **[Mask Edge Smoother](../nodes/mask-edge-smoother.md)**

    Smooths jagged edges with a bilateral filter, optionally photo-guided.

-   ![](../assets/nodes/trimap-builder.jpg)

    **[Trimap Builder](../nodes/trimap-builder.md)**

    Makes the sure-subject, unknown and sure-background map for matting.

-   ![](../assets/nodes/alpha-matte.jpg)

    **[Alpha Matte](../nodes/alpha-matte.md)**

    Hair-quality soft alpha from a trimap and the photo.

-   ![](../assets/nodes/mask-morphology.jpg)

    **[Mask Morphology](../nodes/mask-morphology.md)**

    Grows, shrinks, closes or outlines a mask.

-   ![](../assets/nodes/mask-combine.jpg)

    **[Mask Combine](../nodes/mask-combine.md)**

    Adds, subtracts, intersects or multiplies two masks.

-   ![](../assets/nodes/high-freq-detail-masker.jpg)

    **[Detail Mask](../nodes/high-freq-detail-masker.md)**

    Selects the sharp, detailed parts of a photo.

-   ![](../assets/nodes/mask-preview.jpg)

    **[Mask Preview](../nodes/mask-preview.md)**

    Shows a mask as an overlay, heatmap, black-and-white view or cutout.

</div>

## Typical chains

- **Clean AI cutout (product or portrait):** [Mask Cleanup](../nodes/mask-cleanup.md) → [Edge-Aware Mask Refiner](../nodes/edge-aware-mask-refiner.md) → [Sub-Pixel Mask Refiner](../nodes/sub-pixel-mask-refiner.md) → [Mask Preview](../nodes/mask-preview.md)
- **Hair-quality cutout:** rough mask → [Mask Cleanup](../nodes/mask-cleanup.md) → [Trimap Builder](../nodes/trimap-builder.md) → [Alpha Matte](../nodes/alpha-matte.md) (with the photo as `guide`) → [Mask Preview](../nodes/mask-preview.md) in `cutout` mode
- **Edit only the textured areas:** [Detail Mask](../nodes/high-freq-detail-masker.md) → [Mask Combine](../nodes/mask-combine.md) with a subject mask (`intersect`) → use as the gate for your sharpen or smooth step
