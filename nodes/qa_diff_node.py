"""ComfyUI wrapper for the QA image diff (SSIM/PSNR)."""

from __future__ import annotations

import json

import torch

from ..core.qa_diff import qa_diff


class JHPixelProQADiff:
    """Compare two images: diff visualisation plus SSIM / PSNR / MAE report."""

    CATEGORY = "ComfyUI-JH-PixelPro/qa"
    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("diff", "ssim", "psnr", "report")
    FUNCTION = "run"
    OUTPUT_TOOLTIPS = (
        "Visual difference image (all batch items).",
        "SSIM of batch item 0 only (1.0 = identical); see report for other items.",
        "PSNR in dB of batch item 0 only (capped at 100); see report for other items.",
        "JSON list with ssim, psnr, mae, max_abs, changed_pct for every batch item.",
    )

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "image_a": ("IMAGE", {"tooltip": "Reference image (e.g. before retouch)."}),
                "image_b": ("IMAGE", {"tooltip": "Image to compare (e.g. after retouch)."}),
                "diff_mode": (
                    ["heatmap", "abs", "ssim_map"],
                    {
                        "default": "heatmap",
                        "tooltip": (
                            "heatmap = turbo colormap of the amplified difference. "
                            "abs = amplified per-channel |a-b|. "
                            "ssim_map = 1 - SSIM map (bright = structural change)."
                        ),
                    },
                ),
                "amplify": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 1.0,
                        "max": 100.0,
                        "step": 0.5,
                        "tooltip": "Gain applied to the difference (heatmap/abs only).",
                    },
                ),
                "window_size": (
                    "INT",
                    {
                        "default": 11,
                        "min": 3,
                        "max": 31,
                        "step": 2,
                        "tooltip": "Odd SSIM window size. Larger = coarser structure. "
                        "Reduced automatically for small images.",
                    },
                ),
            },
            "optional": {
                "mask": (
                    "MASK",
                    {"tooltip": "Restrict all metrics to this region (weighted)."},
                ),
            },
        }

    def run(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        diff_mode: str,
        amplify: float,
        window_size: int,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float, float, str]:
        with torch.no_grad():
            diff, metrics = qa_diff(
                image_a,
                image_b,
                window_size=window_size,
                amplify=amplify,
                diff_mode=diff_mode,
                mask=mask,
            )
        report = json.dumps(metrics, indent=2)
        return (diff, float(metrics[0]["ssim"]), float(metrics[0]["psnr"]), report)
