# ComfyUI-JH-PixelPro

GPU-powered pro-grade image suite for ComfyUI. The pack ships 34 production-ready nodes across seven categories for retouch, color science, mask finishing, face workflows, and Photoshop-style compositing without leaving the tensor pipeline.

[![CI](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/actions/workflows/ci.yml/badge.svg)](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-github_pages-0A84FF?logo=materialformkdocs&logoColor=white)](https://jetthuangai.github.io/ComfyUI-JH-PixelPro/)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-555555)](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/LICENSE)

## Install via ComfyUI Manager

1. Open **ComfyUI Manager** and search for `ComfyUI-JH-PixelPro`.
2. Install the pack and let ComfyUI fetch the Python dependencies.
3. Restart ComfyUI. The nodes appear under the `ComfyUI-JH-PixelPro/*` menu.

If you prefer a manual install, clone the repository into `ComfyUI/custom_nodes/` and run `pip install -e .`.

## 34-node pack at a glance

| Category | Nodes | Highlights |
|---|---:|---|
| Color | 11 | Color matching, tone curves, LUT import/export, preset LUTs, selective color, ColorLab |
| Compositing | 4 | Photoshop-style layer stack and blend modes |
| Filters | 2 | Frequency separation and edge-aware smoothing |
| Mask | 8 | Sub-pixel refinement, alpha matte extraction, morphology, combine, edge smoothing |
| Geometry | 2 | Facial alignment and lens distortion correction |
| Face | 6 | Face detection, landmarks, warp, unwrap, beauty blend, skin-tone masks |
| Looks | 1 | Preset-based creative look selection |
| **Total** | **34** | Feature-complete pack v1.3.0 |

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
