"""ComfyUI wrapper for Auto Upright (level horizons, parallel verticals)."""

from __future__ import annotations

import torch

from ..core.upright import CROPS, MODES, auto_upright, matrices_to_json


class JHPixelProAutoUpright:
    """Straighten tilted horizons and converging verticals from the lines in the photo."""

    CATEGORY = "ComfyUI-JH-PixelPro/geometry"
    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("image", "confidence", "matrix")
    OUTPUT_TOOLTIPS = (
        "The straightened image (same size).",
        "How sure the correction is, 0..1 (lowest of a batch). 0 = image left unchanged.",
        "JSON list of the 3x3 pixel homographies applied, one per image.",
    )
    FUNCTION = "upright"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Photo to straighten."}),
                "mode": (
                    list(MODES),
                    {
                        "default": "auto",
                        "tooltip": "auto = parallel verticals when the photo has buildings, "
                        "else a level horizon. level = rotate only. vertical = parallel "
                        "verticals. full = verticals and the main horizontal lines too.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Fraction of the correction. Around 0.7 keeps a little "
                        "natural perspective in tall buildings.",
                    },
                ),
                "crop": (
                    list(CROPS),
                    {
                        "default": "none",
                        "tooltip": "none = keep everything, corners may be black. valid = "
                        "zoom in until no empty corners remain.",
                    },
                ),
            },
            "optional": {
                "max_angle": (
                    "FLOAT",
                    {
                        "default": 25.0,
                        "min": 0.0,
                        "max": 45.0,
                        "step": 0.5,
                        "tooltip": "Largest camera turn in degrees; bigger corrections stop there.",
                    },
                ),
            },
        }

    def upright(
        self,
        image: torch.Tensor,
        mode: str,
        strength: float,
        crop: str,
        max_angle: float = 25.0,
    ) -> tuple[torch.Tensor, float, str]:
        with torch.no_grad():
            out, confidences, matrices = auto_upright(
                image,
                mode=mode,
                strength=float(strength),
                max_angle=float(max_angle),
                crop=crop,
            )
        return out.to(image.device), float(min(confidences)), matrices_to_json(matrices)
