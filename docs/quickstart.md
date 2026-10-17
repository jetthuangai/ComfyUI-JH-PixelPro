---
title: Quickstart
description: Install PixelPro, find the nodes and run your first workflow in five minutes.
---

# Quickstart

Five minutes from install to your first result.

## 1. Install

=== "ComfyUI Manager (recommended)"

    1. Open **Manager → Custom Nodes Manager**.
    2. Search for `ComfyUI-JH-PixelPro` and click **Install**.
    3. Restart ComfyUI.

=== "Manual (git)"

    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/jetthuangai/ComfyUI-JH-PixelPro.git
    cd ComfyUI-JH-PixelPro
    pip install -r requirements.txt
    ```

    Use the same Python that runs ComfyUI (for a portable or venv install, its own `python`). Then restart ComfyUI.

!!! info "Requirements"
    Python 3.10+, the PyTorch that comes with ComfyUI, Kornia ≥ 0.7, MediaPipe ≥ 0.10, OpenCV ≥ 4.8 and SciPy ≥ 1.10. Optional: `onnxruntime-gpu` (or `onnxruntime`) for Depth Anything and Depth Lens Blur.
    An NVIDIA GPU with 8 GB VRAM or more is recommended. Everything also runs on CPU, just slower.

## 2. Find the nodes

Right-click the canvas (or double-click and search), then open **Add Node → ComfyUI-JH-PixelPro**. Nodes are grouped the same way as this site:

`color` · `mask` · `face` · `compositing` · `filters` · `geometry` · `looks` · `qa`

!!! tip
    Type part of a node name in the search box, for example `Frequency`, `Matte` or `CLAHE`.

## 3. Run your first workflow

1. Copy `workflows/sample_portrait.jpg` from the pack folder into `ComfyUI/input/`.
2. Drag a sample workflow onto the canvas. Good first picks:
    - [CLAHE Local Contrast example](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-37-clahe-local-contrast.json): one node, instant result.
    - [Face Detect Lite example](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-38-face-detect-lite.json): face box, landmarks and alignment.
    - [Mask Cleanup example](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-41-mask-cleanup.json): also copy `workflows/sample_binary_mask.png` into `ComfyUI/input/`.
3. Press **Queue Prompt**. Change one setting at a time and compare.

All sample workflows are listed in [Examples](examples/index.md).

## 4. Build your own chain

A good way to learn the pack is one classic retouch step per group:

| Goal | Try |
|---|---|
| Even out skin but keep pores | [Frequency Separation](nodes/frequency-separation.md) or [Skin Smoother](nodes/edge-aware-smoother.md) |
| Sharpen eyes and hair only | [Detail Mask](nodes/high-freq-detail-masker.md) → [Smart Sharpen](nodes/smart-sharpen.md) |
| Grade colour like Camera Raw | [ColorLab](nodes/color-lab.md), [Tone Curve](nodes/tone-curve.md) |
| Match one photo to another | [Color Matcher](nodes/color-matcher.md) or [Color Matcher Region](nodes/color-matcher-region.md) |
| Cut out hair cleanly | [Trimap Builder](nodes/trimap-builder.md) → [Alpha Matte](nodes/alpha-matte.md) |
| Run AI on a face, paste it back | [Face Detect Lite](nodes/face-detect-lite.md) → [Facial Aligner](nodes/facial-aligner.md) → … → [Unwrap Face](nodes/unwrap-face.md) |
| Check what changed | [QA Image Diff](nodes/qa-image-diff.md) |

## Troubleshooting

??? question "The nodes do not show up"
    Check the ComfyUI console for an import error from `ComfyUI-JH-PixelPro`. Usually a dependency is missing: run `pip install -r requirements.txt` with ComfyUI's own Python, then restart.

??? question "Face Detect Lite or Auto Align (LoFTR) downloads something"
    Yes, on first use only. Face Detect Lite downloads its YuNet weights (about 0.4 MB) and Auto Align with `method = loftr` downloads LoFTR weights (about 45 MB) into the torch cache. Face Detect (MediaPipe) downloads its model (about 5 MB) into `ComfyUI/models/mediapipe`.

??? question "The machine slows down on a big image"
    Turn on `tile_mode` in [Skin Smoother](nodes/edge-aware-smoother.md) for anything larger than a preview. Since v1.5.2 tiles are sized to keep memory near 2 GB. Close other heavy workflows if VRAM is already full.

??? question "I changed the pack but nothing changed in ComfyUI"
    ComfyUI loads custom-node code once at start-up. Restart ComfyUI after updating the pack.
