# Mask

The mask category is the pack's largest finishing domain. It starts with mask cleanup and extends all the way to trimap generation, closed-form alpha matting, and edge-aware smoothing for production composites.

- **N-02 Sub-Pixel Mask Refiner** — Converts hard masks into feathered alpha masks with protected inside and outside cores.
- **N-04 High-Frequency Detail Masker** — Builds binary detail masks from image high-frequency energy.
- **N-28 Edge-Aware Mask Refiner** — Refines a mask against an image guide so edges follow subject detail.
- **N-29 Alpha Matte Extractor** — Solves a Levin 2008 closed-form matte from a trimap and guide image.
- **N-30 Trimap Builder** — Expands binary masks into strict 0.0 / 0.5 / 1.0 trimaps.
- **N-31 Mask Morphology** — Runs elliptical-kernel dilate, erode, open, close, and related morphology ops.
- **N-32 Mask Combine** — Performs add, subtract, union, intersect, xor, difference, and multiply blends.
- **N-33 Mask Edge Smoother** — Smooths mask edges with bilateral filtering and optional image guidance.

Flagship mask pages: [N-29 Alpha Matte Extractor](../nodes/n29-alpha-matte-extractor.md) and [N-33 Mask Edge Smoother](../nodes/n33-mask-edge-smoother.md).
