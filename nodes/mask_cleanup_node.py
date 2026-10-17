"""ComfyUI wrapper for mask cleanup (specks, holes, feather)."""

from __future__ import annotations

import torch

from ..core.mask_cleanup import mask_cleanup


class JHPixelProMaskCleanup:
    """Remove specks, fill small holes and optionally feather a mask by distance."""

    CATEGORY = "ComfyUI-JH-PixelPro/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "cleanup"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Mask to clean, e.g. from SAM / rembg."}),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Pixels above this value count as foreground.",
                    },
                ),
                "min_island_area": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 1_000_000,
                        "step": 1,
                        "tooltip": "Remove foreground blobs under this many pixels. 0 = off.",
                    },
                ),
                "max_hole_area": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 1_000_000,
                        "step": 1,
                        "tooltip": (
                            "Fill enclosed holes smaller than this many pixels "
                            "(background touching the image border is never filled). 0 = off."
                        ),
                    },
                ),
                "feather_px": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 256.0,
                        "step": 0.5,
                        "tooltip": (
                            "Soft edge distance in pixels, measured from the cleaned edge. "
                            "0 = hard black/white mask."
                        ),
                    },
                ),
                "connectivity": (
                    ["8", "4"],
                    {
                        "default": "8",
                        "tooltip": (
                            "Foreground connectivity: 8 = diagonal pixels are connected, "
                            "4 = only side neighbours. Holes use the complement (8 -> 4)."
                        ),
                    },
                ),
            },
        }

    def cleanup(
        self,
        mask: torch.Tensor,
        threshold: float,
        min_island_area: int,
        max_hole_area: int,
        feather_px: float,
        connectivity: str,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            out = mask_cleanup(
                mask,
                threshold=threshold,
                min_island_area=min_island_area,
                max_hole_area=max_hole_area,
                feather_px=feather_px,
                connectivity=int(connectivity),
            )
        return (out,)
