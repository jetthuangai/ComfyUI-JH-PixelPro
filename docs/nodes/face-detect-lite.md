---
title: N-38 Face Detect Lite (YuNet GPU)
description: Fast GPU face detector with 5-point landmarks, real confidence and a face mask, no MediaPipe needed.
---

# N-38 Face: Detect Lite (YuNet GPU)

`JHPixelProFaceDetectLite` finds faces with the YuNet network shipped in Kornia. It runs on the GPU when one is available (CPU otherwise) and returns the same JSON shape as [N-10 Face Detect](n10-face-detect.md), plus a ready-made face mask. Landmarks are the 5 points the Facial Aligner needs, so it can replace N-10 in most face chains.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image` | Input | `IMAGE` | Source image batch. |
| `mode` | Widget | `COMBO`, default `single_largest` | One largest face per image, or the top faces ranked by box area. |
| `max_faces` | Widget | `INT`, default `1` (1-10) | Faces per image when `mode = multi_top_k`. |
| `confidence_threshold` | Widget | `FLOAT`, default `0.6` (0.1-0.95) | Minimum YuNet score. |
| `nms_threshold` | Widget | `FLOAT`, default `0.3` (0.1-0.9) | Overlap level above which duplicate boxes are merged. |
| `detect_size` | Widget | `INT`, default `640` (160-1024, step 32) | Long side of the copy used for detection. Never upscales. 640 is best for portraits. |
| `landmarks_json` | Output | `STRING` | Per face: `[L-eye, R-eye, nose-tip, L-mouth, R-mouth]` as pixel `[x, y]` (image-left first). |
| `bbox_json` | Output | `STRING` | Per face: `x, y, w, h` (pixels), `conf` (real YuNet score), `batch_index`. |
| `face_count` | Output | `INT` | Faces returned. |
| `face_mask` | Output | `MASK` | 1 inside the selected face boxes, 0 elsewhere. |

## When to use Lite or N-10

| | Lite (YuNet) | N-10 (MediaPipe) |
|---|---|---|
| Speed (3500x2333 portrait) | about 30 ms on GPU, about 90 ms on CPU | about 80 ms on CPU |
| Needs MediaPipe | No | Yes |
| `conf` in bbox | Real detector score | The threshold you set |
| Landmarks | 5 points only | 5 points taken from a 468-point mesh |
| Face mask | Yes | No |
| Feeds 468-point nodes (Landmarks, Warp, Unwrap) | No | Yes (use Face Landmarks) |

Pick Lite for batches, quick previews, masks and plain alignment. Stay with N-10 or Face Landmarks when you need the full mesh or sharper handling of tilted or partial faces.

Eye points are the eye centres (YuNet), while N-10 uses the outer eye corners, so the two differ by roughly 10-20 % of the eye distance. Both work with the aligner.

## Chaining into Facial Aligner

1. Connect `image` to Face: Detect Lite.
2. Send `landmarks_json` to the Landmark Facial Aligner (it reads the first face).
3. Optionally use `face_mask` to limit a later edit to the face area.

## How it works

The image is shrunk so its long side equals `detect_size` (smaller images are left alone), run through YuNet, and boxes and points are scaled back to original pixels. Faces are sorted by box area; `single_largest` keeps one per image. Images smaller than 64 px are padded with their mean colour for detection only. The detector is cached per device, and the weights (`yunet_final.pth`) are downloaded once by Kornia into the torch hub cache. If loading fails the node raises a clear error.

## Tips

- Keep the default 640 for portraits. Raise `detect_size` (up to 1024) only for group shots with small faces. YuNet anchors are size-limited, so a larger value can split or misplace big faces and add false hits on close-ups.
- Too many false hits: raise `confidence_threshold`; duplicate boxes: lower `nms_threshold`.
- A pure black frame can trigger a false detection in YuNet; use a real image or a higher threshold.
