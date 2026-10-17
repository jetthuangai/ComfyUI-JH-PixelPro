"""ComfyUI wrapper for Smart Sharpen."""

from __future__ import annotations

import torch

from ..core.sharpen import smart_sharpen


class JHPixelProSmartSharpen:
    """Unsharp-mask sharpening with threshold, luminance-only mode and optional mask."""

    CATEGORY = "ComfyUI-JH-PixelPro/filters"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sharpen"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.05,
                        "tooltip": "Sharpening strength. 0 leaves the image unchanged.",
                    },
                ),
                "radius": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.3,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Blur radius (sigma, pixels) of the detail layer.",
                    },
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.005,
                        "tooltip": "Detail weaker than this is not sharpened (suppresses noise).",
                    },
                ),
                "luminance_only": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Sharpen brightness only; colours stay untouched (no fringes).",
                    },
                ),
            },
            "optional": {
                "mask": (
                    "MASK",
                    {"tooltip": "Where the mask is 1 sharpening applies fully; 0 leaves it."},
                ),
            },
        }

    def sharpen(
        self,
        image: torch.Tensor,
        amount: float,
        radius: float,
        threshold: float,
        luminance_only: bool,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            out = smart_sharpen(
                image,
                amount=amount,
                radius=radius,
                threshold=threshold,
                luminance_only=luminance_only,
                mask=mask,
            )
        return (out,)
