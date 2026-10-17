# Depth

Depth nodes estimate how far every pixel is from the camera and use that map like a photographer uses a lens: to put the background (or foreground) out of focus after the shot.

<div class="grid cards" markdown>

-   ![](../assets/nodes/depth-anything.jpg)

    **[Depth Anything V2](../nodes/depth-anything.md)**

    Estimate a depth map from a single photo (near = white). Small model is Apache-2.0; Base/Large are non-commercial.

-   ![](../assets/nodes/depth-lens-blur.jpg)

    **[Depth Lens Blur](../nodes/depth-lens-blur.md)**

    Bokeh from depth: choose the focus plane (or let a face/subject mask pick it) and the depth of field.

</div>

## Typical chains

- **Portrait background blur:** Load Image → [Face Detect Lite](../nodes/face-detect-lite.md) (face_mask) → [Depth Lens Blur](../nodes/depth-lens-blur.md) with the face mask as `focus_mask`. Depth is estimated inside the node.
- **Product shot:** [SAM One-Click Mask](../nodes/sam-mask.md) on the product → `focus_mask` of [Depth Lens Blur](../nodes/depth-lens-blur.md).
- **Reuse the depth map:** [Depth Anything V2](../nodes/depth-anything.md) once → its `depth` into several Lens Blur variants, or into [Mask Preview](../nodes/mask-preview.md) to inspect it.
