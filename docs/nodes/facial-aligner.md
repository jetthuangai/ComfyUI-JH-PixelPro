---
title: Facial Aligner
description: Rotate, scale and crop a face to a standard upright frame from five landmark points.
---

# Facial Aligner

Straightens and crops a face into a standard square frame (the same layout used by face datasets such as FFHQ), using five landmark points. It also gives you the matrix needed to paste the edited face back later.

![Facial Aligner — before / after](../assets/nodes/facial-aligner.jpg)

| | |
|---|---|
| **In ComfyUI** | `Add Node → ComfyUI-JH-PixelPro → geometry → Landmark Facial Aligner` |
| **Good for** | Preparing a face for AI face editing, upscaling or inpainting; consistent face crops for batches |

## When to use it

Face models work best when the eyes, nose and mouth are always in the same place. This node rotates the picture so the eyes are level, scales it so the face has a fixed size, and crops a square. Hair and chin room is controlled by `padding`.

It is the first half of a round trip: align the face, edit the crop, then return it to the original photo with [Unwrap Face](unwrap-face.md) using `inverse_matrix_json`.

## Quick start

1. `Load Image` → [Face Detect Lite](face-detect-lite.md) (or [Face Detect](face-detect.md)).
2. Connect its `landmarks_json` output to the `landmarks` input of **Landmark Facial Aligner**.
3. Choose `target_size` (1024 is a good start) and `padding = 0.2`.
4. Send `image_aligned` to `Preview Image`.

## Settings

| Setting | Default | What it does | Tip |
|---|---|---|---|
| `image` | — | The portrait to align. | |
| `landmarks` | sample text for the sample portrait | Five points as JSON: left eye, right eye, nose, left mouth corner, right mouth corner. Pixel values or 0 to 1 values both work. | Connect it from a face-detection node instead of typing. |
| `target_size` | `1024` | Side of the square output: 512, 768 or 1024. | Smaller is faster; 1024 suits SDXL. |
| `padding` | `0.2` (0 to 0.5) | Space kept around the face. 0 is a tight crop, higher values show more hair and chin. | 0.2 to 0.3 for most portraits. |

| Output | Type | Meaning |
|---|---|---|
| `image_aligned` | `IMAGE` | The aligned square face crop. |
| `inverse_matrix_json` | `STRING` | The matrix that maps the crop back onto the original photo. Feed it to [Unwrap Face](unwrap-face.md). |

## Tips from retouching

- Values of 1.5 or less are treated as 0 to 1 coordinates; larger values are treated as pixels.
- Use a wide-angle correction first ([Lens Distortion](lens-distortion.md)) if the face is near the edge of a wide shot; stretched faces give less accurate landmarks.
- For a batch, give one set of landmarks per image (a list of five-point lists). One set is reused for every image.

??? note "Under the hood"
    The five points are matched to a canonical layout with a similarity transform (rotation, uniform scale, shift; Umeyama method), then applied with Kornia `warp_affine` to a `target_size × target_size` output. The inverse is returned as a 3×3 matrix. Landmarks must be non-negative; the node raises an error if they fall outside the image after normalisation. Only 512, 768 and 1024 are accepted for `target_size`. Source: `core/facial_aligner.py` and `nodes/facial_aligner_node.py`.

## Related nodes

- [Face Detect Lite](face-detect-lite.md)
- [Face Detect](face-detect.md)
- [Unwrap Face](unwrap-face.md)
- [Lens Distortion](lens-distortion.md)

## Example workflow

[Facial Aligner example workflow](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/S-06-facial-aligner.json). Copy `workflows/sample_portrait.jpg` into `ComfyUI/input/`. The default landmark text is set for that sample portrait.
