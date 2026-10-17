---
title: N-44 QA Image Diff
description: Compare two images with SSIM, PSNR and MAE plus a visual diff.
---

# N-44 QA: Image Diff (SSIM/PSNR)

`JHPixelProQADiff` compares two images (for example before and after a retouch) numerically and visually. It outputs a diff image, the SSIM and PSNR of the first batch item, and a JSON report with the full per-item metrics.

## Schema

| Name | Kind | Type / default | Description |
|---|---|---|---|
| `image_a` | Input | `IMAGE` | Reference image (before). |
| `image_b` | Input | `IMAGE` | Image to compare (after). Same width/height as `image_a`; batch 1 broadcasts. |
| `diff_mode` | Widget | `COMBO`, default `heatmap` | `heatmap`, `abs` or `ssim_map`. |
| `amplify` | Widget | `FLOAT`, default `4.0` (1..100) | Gain on the difference (`heatmap`/`abs`). |
| `window_size` | Widget | `INT`, default `11` (3..31, odd) | SSIM window size. |
| `mask` | Optional input | `MASK` | Restricts all metrics to the masked region (weighted). |
| `diff` | Output | `IMAGE` | Visual difference. |
| `ssim` | Output | `FLOAT` | SSIM of batch item 0 only (1.0 = identical). |
| `psnr` | Output | `FLOAT` | PSNR in dB of the first batch item, capped at 100.0. |
| `report` | Output | `STRING` | JSON list (indent 2) with `ssim`, `psnr`, `mae`, `max_abs`, `changed_pct` for every batch item. |

## How it works

- **SSIM** is computed per channel with a Gaussian window (`max_val` 1.0, same-size map) and averaged. With a mask, the SSIM map is averaged inside the mask.
- **PSNR** = 10 * log10(1 / MSE). Identical images would be infinite, so the value is capped at 100.0 dB to keep the output JSON-safe.
- **MAE** is the mean absolute difference (0..1); `max_abs` the largest single-channel difference; `changed_pct` the percentage of pixels whose largest channel difference exceeds 1/255.
- **Diff modes**: `abs` = amplified per-channel |a-b|; `heatmap` = turbo colormap of the amplified max-channel difference; `ssim_map` = 1 - SSIM map as grey (bright = structural change, `amplify` ignored).
- `window_size` is reduced to the largest odd number that fits the image (e.g. 5 on a 5x5 image); images smaller than 3x3 px raise an error.
- `image_b` and `mask` are moved to the device of `image_a`.
- Images of different size are rejected, never resized silently. The diff image is not masked; only the metrics are.
- An all-zero mask returns neutral metrics (ssim 1.0, psnr 100.0, rest 0).

## Tips

- **Retouch strength regression**: compare the original with the retouched result. Keep a reference run and watch `ssim`/`changed_pct` across settings or versions; a sudden jump means the retouch got stronger (or broke).
- **Comparing two pipelines**: feed each pipeline's output as `image_a` and `image_b` and use `heatmap` to see where they disagree.
- **Check only the face or skin**: pass a mask so background pixels do not dilute the numbers.
- **Rules of thumb**: SSIM >= 0.98 and PSNR >= 40 dB are visually near-identical; SSIM 0.90-0.98 / PSNR 30-40 dB is a visible but moderate change; SSIM < 0.80 or PSNR < 25 dB is a heavy change. A uniform 0.1 offset is exactly 20 dB.
- Use a larger `window_size` to judge coarse structure, a smaller one for fine texture such as skin grain.

## Workflow JSON

[workflows/N-44-qa-image-diff.json](https://github.com/jetthuangai/ComfyUI-JH-PixelPro/blob/main/workflows/N-44-qa-image-diff.json) — load it in ComfyUI after copying `workflows/sample_portrait.jpg` into `ComfyUI/input/`.

![N-44-qa-image-diff result](https://raw.githubusercontent.com/jetthuangai/ComfyUI-JH-PixelPro/main/workflows/N-44-qa-image-diff-preview.png)
