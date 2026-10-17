---
title: ComfyUI-JH-PixelPro
description: Pro-grade retouch, colour and mask tools for ComfyUI, GPU-accelerated with Kornia.
hide:
  - navigation
  - toc
---

<div class="pp-hero" markdown>

# PixelPro for ComfyUI

<p class="pp-lead">The retouching desk you already know (frequency separation, luminosity masks, LUTs, layer stacks, hair-accurate mattes, face alignment) rebuilt as 50 GPU nodes for ComfyUI.</p>

[Get started :material-arrow-right:](quickstart.md){ .md-button .md-button--primary }
[Browse the nodes](categories/color.md){ .md-button }
[GitHub](https://github.com/jetthuangai/ComfyUI-JH-PixelPro){ .md-button }

</div>

![One portrait through eight PixelPro nodes](assets/hero.jpg)

## Why PixelPro

<div class="grid cards" markdown>

-   :material-palette-swatch-outline:{ .lg } **Real retouch techniques**

    Not filters with fancy names. Frequency separation, luminosity masking, LAB colour matching, Laplacian-pyramid blending and closed-form matting work the way a retoucher expects.

-   :material-chip:{ .lg } **GPU first, CPU safe**

    Built on PyTorch and [Kornia](https://kornia.github.io/). Heavy filters run on CUDA and keep their memory bounded. The same nodes still run, slower, on a CPU-only machine.

-   :material-vector-difference:{ .lg } **Works with your AI pipeline**

    Align a face, run your inpaint or ControlNet pass, paste it back seamlessly. Build masks for any upstream segmenter, and check results with SSIM/PSNR.

-   :material-test-tube:{ .lg } **Tested and documented**

    800+ automated tests, a sample workflow for every recent node, and this guide in English and Vietnamese.

</div>

## Explore by task

<div class="grid cards" markdown>

-   ![](assets/nodes/color-lab.jpg)

    **[Color & Tone](categories/color.md)**

    ACR-style grading, tone curves, colour matching, LUT import/export and CLAHE local contrast.

-   ![](assets/nodes/alpha-matte.jpg)

    **[Masks](categories/mask.md)**

    From a rough segmentation to a clean, hair-accurate alpha: refine, matte, clean up and preview.

-   ![](assets/nodes/face-landmarks.jpg)

    **[Face](categories/face.md)**

    Detect, map 468 landmarks, reshape, align and paste back, plus skin-tone masks.

-   ![](assets/nodes/frequency-separation.jpg)

    **[Filters](categories/filters.md)**

    Frequency separation, edge-aware skin smoothing and halo-free sharpening.

-   ![](assets/nodes/pyramid-blend.jpg)

    **[Compositing](categories/compositing.md)**

    A Photoshop-style layer stack with 27 blend modes, and seamless multi-band blending.

-   ![](assets/nodes/auto-align.jpg)

    **[Geometry](categories/geometry.md)**

    Face alignment, lens distortion correction and automatic alignment of a second shot.

-   ![](assets/nodes/look-select.jpg)

    **[Looks](categories/looks.md)**

    Finished colour looks from one dropdown, with an intensity control.

-   ![](assets/nodes/qa-image-diff.jpg)

    **[QA](categories/qa.md)**

    Measure what an edit changed: SSIM, PSNR, and a heatmap of where it changed.

</div>

## A typical portrait chain

```text
Load Image
  → Face Detect Lite ──→ Facial Aligner ──→ (your AI or retouch pass) ──→ Unwrap Face
  → Frequency Separation / Skin Smoother  (skin)
  → Smart Sharpen + Detail Mask            (eyes, hair)
  → ColorLab or LUT Preset                 (colour)
  → QA Image Diff                          (check the result)
```

Each step is one node. Start with [Quickstart](quickstart.md), or open the [examples](examples/index.md).

## At a glance

| Group | Nodes | Highlights |
|---|---:|---|
| [Color & Tone](categories/color.md) | 14 | ColorLab (ACR-style), Tone Curve, Color Matcher (global and region), LUT Preset / Import / Export, HALD, Tone Match LUT, CLAHE |
| [Masks](categories/mask.md) | 11 | Sub-Pixel and Edge-Aware refiners, Alpha Matte, Trimap, Morphology, Combine, Edge Smoother, Cleanup, Preview, Detail Mask |
| [Face](categories/face.md) | 8 | Face Detect (MediaPipe) and Detect Lite (YuNet GPU), 468 landmarks, Delaunay and TPS warp, Unwrap, Beauty Blend, skin-tone masks |
| [Compositing](categories/compositing.md) | 5 | Layer stack (start, add, group, flatten), Pyramid Blend |
| [Filters](categories/filters.md) | 5 | Frequency Separation and Recombine, Skin Smoother, Smart Sharpen, Dust & Spot Removal |
| [Depth](categories/depth.md) | 2 | Depth Anything V2, Depth Lens Blur (bokeh) |
| [Geometry](categories/geometry.md) | 3 | Facial Aligner, Lens Distortion, Auto Align |
| [Looks](categories/looks.md) | 1 | Look Select |
| [QA](categories/qa.md) | 1 | QA Image Diff |
| **Total** | **50** | v1.5.x · Apache-2.0 |
