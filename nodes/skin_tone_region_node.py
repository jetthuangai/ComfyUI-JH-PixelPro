"""ComfyUI wrapper for skin tone tri-region masks."""

from __future__ import annotations

import torch

from ..core.skin_tone_region import skin_tone_tri_region


class JHPixelProSkinToneTriRegion:
    """Split skin tones into shadow, midtone, and highlight masks."""

    CATEGORY = "ComfyUI-JH-PixelPro/face"
    RETURN_TYPES = ("MASK", "MASK", "MASK")
    RETURN_NAMES = ("shadow_mask", "midtone_mask", "highlight_mask")
    FUNCTION = "split"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802 — ComfyUI node contract mandates UPPER_CASE.
        return {
            "required": {
                "image": ("IMAGE",),
                "shadow_cutoff": (
                    "FLOAT",
                    {
                        "default": 0.33,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": "Luminance values below this are treated as shadow tones.",
                    },
                ),
                "highlight_cutoff": (
                    "FLOAT",
                    {
                        "default": 0.66,
                        "min": 0.5,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Luminance values above this are treated as highlight tones.",
                    },
                ),
                "soft_sigma": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": "Internal Gaussian sigma for smoother tone-region boundaries.",
                    },
                ),
            },
            "optional": {
                "skin_mask": ("MASK",),
            },
        }

    def split(
        self,
        image: torch.Tensor,
        shadow_cutoff: float,
        highlight_cutoff: float,
        soft_sigma: float,
        skin_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            image_bchw = image.permute(0, 3, 1, 2).contiguous()
            return skin_tone_tri_region(
                image_bchw,
                skin_mask=skin_mask,
                shadow_cutoff=float(shadow_cutoff),
                highlight_cutoff=float(highlight_cutoff),
                soft_sigma=float(soft_sigma),
            )
