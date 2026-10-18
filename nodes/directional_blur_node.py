"""ComfyUI wrapper for motion and zoom blur."""

from __future__ import annotations

import torch

from ..core.directional_blur import MODES, run


class JHPixelProMotionZoomBlur:
    """Motion blur along a direction, or zoom blur toward a centre point."""

    CATEGORY = "ComfyUI-JH-PixelPro/filters"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blur"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to blur."}),
                "mode": (
                    list(MODES),
                    {
                        "default": "motion",
                        "tooltip": "motion = movement in one direction. zoom = streaks "
                        "toward a centre point, like zooming during the exposure.",
                    },
                ),
                "angle": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -180.0,
                        "max": 180.0,
                        "step": 1.0,
                        "tooltip": "motion: direction in degrees (0 = horizontal, 90 = vertical).",
                    },
                ),
                "distance": (
                    "FLOAT",
                    {
                        "default": 20.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 1.0,
                        "tooltip": "motion: blur length in pixels.",
                    },
                ),
                "zoom": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.0,
                        "max": 0.9,
                        "step": 0.01,
                        "tooltip": "zoom: strength; the frame edge streaks over this "
                        "fraction of its distance to the centre.",
                    },
                ),
                "center_x": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "zoom: centre across the image (0 = left, 1 = right).",
                    },
                ),
                "center_y": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "zoom: centre down the image (0 = top, 1 = bottom).",
                    },
                ),
            },
            "optional": {
                "mask": (
                    "MASK",
                    {"tooltip": "Blur only where the mask is white (e.g. the background)."},
                ),
            },
        }

    def blur(
        self,
        image: torch.Tensor,
        mode: str,
        angle: float,
        distance: float,
        zoom: float,
        center_x: float,
        center_y: float,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            out = run(
                image,
                mode=mode,
                angle=float(angle),
                distance=float(distance),
                zoom=float(zoom),
                center_x=float(center_x),
                center_y=float(center_y),
                mask=mask,
            )
        return (out.to(image.device),)
