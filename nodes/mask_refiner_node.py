"""ComfyUI wrapper for ``core.mask_refiner.subpixel_mask_refine``."""

from __future__ import annotations

import torch

from ..core import subpixel_mask_refine


class JHPixelProSubPixelMaskRefiner:
    """Refine a binary-ish ComfyUI MASK into a sub-pixel alpha mask."""

    CATEGORY = "ComfyUI-JH-PixelPro/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("refined_mask",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802 — ComfyUI node contract mandates UPPER_CASE.
        return {
            "required": {
                "mask": ("MASK",),
                "erosion_radius": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 64,
                        "tooltip": (
                            "Pixel radius eroded into the protected 'definitely "
                            "inside' core. 0 = no inside protection (output = "
                            "clamped gaussian feather)."
                        ),
                    },
                ),
                "dilation_radius": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 64,
                        "tooltip": (
                            "Pixel radius dilated into the protected 'definitely "
                            "outside' core. Set ≥ erosion_radius for a stable "
                            "feather band. 0 = no outside protection."
                        ),
                    },
                ),
                "feather_sigma": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.1,
                        "max": 32.0,
                        "step": 0.1,
                        "tooltip": (
                            "Gaussian sigma (pixel) used to feather the uncertain "
                            "band between inside and outside cores."
                        ),
                    },
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Strict binarization threshold (`mask > threshold`) "
                            "applied before morphology."
                        ),
                    },
                ),
            },
        }

    def run(
        self,
        mask: torch.Tensor,
        erosion_radius: int,
        dilation_radius: int,
        feather_sigma: float,
        threshold: float,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            refined = subpixel_mask_refine(
                mask,
                erosion_radius=erosion_radius,
                dilation_radius=dilation_radius,
                feather_sigma=feather_sigma,
                threshold=threshold,
            )
        return (refined,)
