"""ComfyUI wrapper for CLAHE local contrast on L* (LAB)."""

from __future__ import annotations

import torch

from ..core.clahe import clahe_local_contrast


class JHPixelProCLAHE:
    """Local contrast via CLAHE on the LAB lightness channel; colours stay intact."""

    CATEGORY = "ComfyUI-JH-PixelPro/color"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image (B,H,W,3)."}),
                "clip_limit": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.1,
                        "max": 40.0,
                        "step": 0.1,
                        "tooltip": "Histogram clip limit. Higher = stronger local contrast.",
                    },
                ),
                "grid_size": (
                    "INT",
                    {
                        "default": 8,
                        "min": 2,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Tiles per axis. Smaller = broader, larger = finer detail.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Blend between original (0) and CLAHE result (1).",
                    },
                ),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask limiting where the effect applies."}),
            },
        }

    def run(
        self,
        image: torch.Tensor,
        clip_limit: float,
        grid_size: int,
        strength: float,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            out = clahe_local_contrast(
                image,
                clip_limit=clip_limit,
                grid_size=grid_size,
                strength=strength,
                mask=mask,
            )
        return (out,)
