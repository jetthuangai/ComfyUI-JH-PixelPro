# ComfyUI-JH-PixelPro

GPU-powered pro-grade image suite for ComfyUI. The pack ships 44 production-ready nodes across eight categories for retouch, color science, mask finishing, face workflows, and Photoshop-style compositing without leaving the tensor pipeline.

[![CI](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/actions/workflows/ci.yml/badge.svg)](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-github_pages-0A84FF?logo=materialformkdocs&logoColor=white)](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-555555)](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/LICENSE)

## v1.5.0 Utilities

Version 1.5.0 adds five everyday tools: [Smart Sharpen](nodes/smart-sharpen.md), [Mask Cleanup](nodes/mask-cleanup.md), [Face Warp (TPS smooth)](nodes/face-warp-tps.md), [Mask Preview Overlay](nodes/mask-preview.md) and [QA Image Diff](nodes/qa-image-diff.md) in the new QA category.

## v1.4.0 Kornia expansion

Version 1.4.0 adds four GPU nodes built on Kornia: [Pyramid Blend](nodes/pyramid-blend.md) for seamless composites, [CLAHE Local Contrast](nodes/clahe-local-contrast.md), [Face Detect Lite](nodes/face-detect-lite.md) (YuNet on the GPU) and [Auto Align](nodes/auto-align.md) (SIFT or LoFTR feature matching). [N-33 Mask Edge Smoother](nodes/n33-mask-edge-smoother.md) gains a `gpu` backend that always honours the guide image.

## v1.3.0 Pack Feature Complete

Version 1.3.0 is the first feature-complete public milestone: 34 live nodes, full public documentation coverage, a deployed docs site, and a released pack workflow for fast onboarding.

Highlights:

- 44 pro-retouch nodes across color, mask, face, geometry, filters, looks, compositing, and QA.
- Levin 2008 closed-form alpha matting with CUDA-capable sparse PyTorch acceleration and CPU fallback.
- Zero-config creative color starts through N-34 bundled LUT presets.
- Portrait tone targeting through N-35 skin-tone shadow, midtone, and highlight masks.
- Active CI, docs deployment, and non-blocking expanded benchmark workflow.

## Quick Links

- [Get Started in 5 Minutes](quickstart.md)
- [Examples](examples/index.md)
- [LUT Preset Gallery](examples/lut-preset-gallery.md)
- [N-29 Alpha Matte Extractor](nodes/n29-alpha-matte-extractor.md)
- [N-33 Mask Edge Smoother](nodes/n33-mask-edge-smoother.md)
- [N-10 Face Detect](nodes/n10-face-detect.md)
- [GitHub repository](https://github.com/jetthuangai/ComfyUI-JH-PixelPro)
- [Latest release](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/releases/latest)

## Install via ComfyUI Manager

1. Open **ComfyUI Manager** and search for `ComfyUI-JH-PixelPro`.
2. Install the pack and let ComfyUI fetch the Python dependencies.
3. Restart ComfyUI. The nodes appear under the `ComfyUI-JH-PixelPro/*` menu.

If you prefer a manual install, clone the repository into `ComfyUI/custom_nodes/` and run `pip install -e .`.

## 44-node pack at a glance

| Category | Nodes | Highlights |
|---|---:|---|
| Color | 13 | Color matching (global and region-aware), tone curves, LUT import/export, preset LUTs, selective color, ColorLab, CLAHE local contrast |
| Compositing | 5 | Photoshop-style layer stack, blend modes, seamless pyramid blending |
| Filters | 3 | Frequency separation, edge-aware smoothing, smart sharpen |
| Mask | 10 | Sub-pixel refinement, alpha matte extraction, morphology, combine, edge smoothing, cleanup, preview overlay |
| Geometry | 3 | Facial alignment, lens distortion correction, automatic image alignment |
| Face | 8 | Face detection (MediaPipe and YuNet GPU), landmarks, Delaunay and TPS warp, unwrap, beauty blend, skin-tone masks |
| Looks | 1 | Preset-based creative look selection |
| QA | 1 | SSIM / PSNR image comparison with visual diff |
| **Total** | **44** | v1.5.0 |

## Lean-batch entry points

- [N-29 Alpha Matte Extractor](nodes/n29-alpha-matte-extractor.md) for quality-first Levin 2008 closed-form matting with CUDA acceleration and CPU fallback.
- [N-33 Mask Edge Smoother](nodes/n33-mask-edge-smoother.md) for bilateral or guide-aware cleanup after person or object mask generation.
- [N-10 Face Detect](nodes/n10-face-detect.md) for MediaPipe-powered 5-point landmarks that feed the pack's face alignment chain.

## Browse by category

- [Color](categories/color.md)
- [Compositing](categories/compositing.md)
- [Filters](categories/filters.md)
- [Mask](categories/mask.md)
- [Geometry](categories/geometry.md)
- [Face](categories/face.md)
- [Looks](categories/looks.md)
- [QA](categories/qa.md)
