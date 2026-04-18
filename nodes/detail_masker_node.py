"""ComfyUI wrapper for ``core.detail_masker.high_freq_detail_mask``."""

from __future__ import annotations

import torch

from ..core import high_freq_detail_mask


class JHPixelProHighFreqDetailMasker:
    """Generate a binary detail-preservation mask from high-frequency energy."""

    CATEGORY = "ComfyUI-JH-PixelPro/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask_detail",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802 — ComfyUI node contract mandates UPPER_CASE.
        return {
            "required": {
                "image": ("IMAGE",),
                "kernel_type": (
                    ["laplacian", "sobel", "fs_gaussian"],
                    {
                        "default": "laplacian",
                        "tooltip": (
                            "High-pass operator. 'laplacian' is scale-invariant "
                            "and isotropic (default). 'sobel' is directional "
                            "(emphasizes edges). 'fs_gaussian' reuses the N-01 "
                            "Frequency Separation high-pass path."
                        ),
                    },
                ),
                "sensitivity": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Fraction of pixels kept as detail. Higher = more "
                            "pixels pass. 0 → empty mask, 1 → full mask. "
                            "Typical retoucher dose 0.3–0.6."
                        ),
                    },
                ),
                "threshold_mode": (
                    ["relative_percentile", "absolute"],
                    {
                        "default": "relative_percentile",
                        "tooltip": (
                            "'relative_percentile' adapts per image "
                            "(robust cross-image). 'absolute' normalizes by "
                            "per-image max (deterministic but more sensitive "
                            "to outliers)."
                        ),
                    },
                ),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    def run(
        self,
        image: torch.Tensor,
        kernel_type: str,
        sensitivity: float,
        threshold_mode: str,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            img_bchw = image.permute(0, 3, 1, 2).contiguous()
            mask_bc1hw = mask.unsqueeze(1).contiguous() if mask is not None else None
            out_bhw = high_freq_detail_mask(
                img_bchw,
                sensitivity=sensitivity,
                kernel_type=kernel_type,
                threshold_mode=threshold_mode,
                mask_bchw=mask_bc1hw,
            )
        return (out_bhw,)
