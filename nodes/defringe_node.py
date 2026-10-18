"""ComfyUI wrapper for Defringe (colour fringes and lateral chromatic aberration)."""

from __future__ import annotations

import torch

from ..core.defringe import run


class JHPixelProDefringe:
    """Remove purple/green fringes along high-contrast edges and lateral CA."""

    CATEGORY = "ComfyUI-JH-PixelPro/color"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "defringe"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Photo with coloured edges (backlight, "
                        "branches against the sky, chrome)."
                    },
                ),
                "purple": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Remove purple / magenta fringes next to strong edges.",
                    },
                ),
                "green": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Remove green fringes. Off by default: it can also grey "
                        "out leaves seen against a bright sky.",
                    },
                ),
                "lateral_ca": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Also line up the red and blue planes with green "
                        "(red/cyan outlines toward the corners). Only applied when it "
                        "clearly helps.",
                    },
                ),
            },
            "optional": {
                "radius": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Fringe width in pixels; 0 = automatic (0.15 % of the "
                        "long edge).",
                    },
                ),
            },
        }

    def defringe(
        self,
        image: torch.Tensor,
        purple: float,
        green: float,
        lateral_ca: bool,
        radius: int = 0,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            out, _ = run(
                image,
                purple=float(purple),
                green=float(green),
                radius=int(radius),
                lateral_ca=bool(lateral_ca),
            )
        return (out.to(image.device),)
