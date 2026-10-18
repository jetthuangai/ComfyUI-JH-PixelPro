"""ComfyUI wrapper for the classic (no-AI) sky mask."""

from __future__ import annotations

import torch

from ..core.sky_mask import sky_mask


class JHPixelProSkyMask:
    """Soft mask of the sky: blue, overcast or sunset, open to the top of the frame."""

    CATEGORY = "ComfyUI-JH-PixelPro/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("sky_mask",)
    FUNCTION = "mask"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Photo with sky."}),
                "sensitivity": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Higher also takes hazy, textured or sunset sky; lower "
                        "keeps only clear sky.",
                    },
                ),
                "refine_edges": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Follow branches and roof lines at full resolution.",
                    },
                ),
            },
        }

    def mask(
        self, image: torch.Tensor, sensitivity: float, refine_edges: bool
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            out = sky_mask(image, sensitivity=float(sensitivity), refine_edges=bool(refine_edges))
        return (out.to(image.device),)
