"""ComfyUI wrapper for ``core.lens_distortion.lens_distortion``."""

from __future__ import annotations

import torch

from ..core import lens_distortion

_PRESETS: dict[str, tuple[float, float, float, float, float]] = {
    "canon_24mm_wide": (-0.18, 0.08, -0.02, 0.0, 0.0),
    "sony_85mm_tele": (0.03, -0.01, 0.0, 0.0, 0.0),
    "gopro_fisheye": (-0.35, 0.12, -0.04, 0.0, 0.0),
    "no_op_identity": (0.0, 0.0, 0.0, 0.0, 0.0),
}

_PRESET_CHOICES = [*_PRESETS.keys(), "custom"]
_DIRECTION_CHOICES = ["inverse", "forward"]


class JHPixelProLensDistortion:
    """Apply Brown-Conrady lens distortion correction or simulation to a ComfyUI IMAGE."""

    CATEGORY = "ComfyUI-JH-PixelPro/geometry"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_rectified",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802 — ComfyUI node contract mandates UPPER_CASE.
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (
                    _PRESET_CHOICES,
                    {
                        "default": "no_op_identity",
                        "tooltip": (
                            "Lens preset — auto-fills k1..p2 with calibrated approximations. "
                            "canon_24mm_wide: barrel correction for wide portrait lenses. "
                            "sony_85mm_tele: mild pincushion correction for short-tele. "
                            "gopro_fisheye: aggressive barrel removal for action cams. "
                            "no_op_identity: pass-through (default, safe). "
                            "Choose 'custom' to use the 5 widget values below verbatim."
                        ),
                    },
                ),
                "direction": (
                    _DIRECTION_CHOICES,
                    {
                        "default": "inverse",
                        "tooltip": (
                            "inverse = rectify a distorted source (default; the retouch use case). "
                            "forward = simulate distortion on a clean source (creative effect)."
                        ),
                    },
                ),
                "k1": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": (
                            "Radial coefficient k1 (Brown-Conrady). "
                            "Used only when preset = custom."
                        ),
                    },
                ),
                "k2": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Radial coefficient k2. Used only when preset = custom.",
                    },
                ),
                "k3": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Radial coefficient k3. Used only when preset = custom.",
                    },
                ),
                "p1": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -0.1,
                        "max": 0.1,
                        "step": 0.0001,
                        "tooltip": "Tangential coefficient p1. Used only when preset = custom.",
                    },
                ),
                "p2": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -0.1,
                        "max": 0.1,
                        "step": 0.0001,
                        "tooltip": "Tangential coefficient p2. Used only when preset = custom.",
                    },
                ),
            },
        }

    def run(
        self,
        image: torch.Tensor,
        preset: str,
        direction: str,
        k1: float,
        k2: float,
        k3: float,
        p1: float,
        p2: float,
    ) -> tuple[torch.Tensor]:
        if preset != "custom":
            k1, k2, k3, p1, p2 = _PRESETS[preset]
        with torch.no_grad():
            img_bchw = image.permute(0, 3, 1, 2).contiguous()
            out_bchw = lens_distortion(
                img_bchw,
                k1=float(k1),
                k2=float(k2),
                k3=float(k3),
                p1=float(p1),
                p2=float(p2),
                direction=direction,
            )
            out = out_bchw.permute(0, 2, 3, 1).contiguous()
        return (out,)
