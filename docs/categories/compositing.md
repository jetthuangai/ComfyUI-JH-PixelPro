# Compositing

The compositing category turns ComfyUI into a Photoshop-style layer stack with blend modes, grouping, and flattening. These nodes are meant for deterministic finishing passes after generation or retouch work is done.

- **N-24 Layer Stack Start** — Starts a `LAYER_STACK` from a background image.
- **N-25 Layer Add** — Adds a masked layer with opacity, fill, blend mode, and clipping options.
- **N-26 Layer Group** — Flattens a sub-stack into a grouped layer for nested compositing.
- **N-27 Layer Flatten** — Renders the layer stack back to a final image.

Lean-batch flagship pages: [N-29 Alpha Matte Extractor](../nodes/n29-alpha-matte-extractor.md), [N-33 Mask Edge Smoother](../nodes/n33-mask-edge-smoother.md), and [N-10 Face Detect](../nodes/n10-face-detect.md).
