"""ComfyUI wrapper for ``core.luminosity.luminosity_masks``."""

from __future__ import annotations

import torch

from ..core import luminosity_masks


class JHPixelProLuminosityMasking:
    """Split an image into three smooth luminosity masks (shadows / midtones / highlights)."""

    CATEGORY = "ComfyUI-JH-PixelPro/color"
    RETURN_TYPES = ("MASK", "MASK", "MASK")
    RETURN_NAMES = ("mask_shadows", "mask_midtones", "mask_highlights")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802 — ComfyUI node contract mandates UPPER_CASE.
        return {
            "required": {
                "image": ("IMAGE",),
                "luminance_source": (
                    ["lab_l", "ycbcr_y", "hsv_v"],
                    {
                        "default": "lab_l",
                        "tooltip": (
                            "Luminance channel. 'lab_l' is perceptual (Photoshop "
                            "default, ~120ms @ 2K CPU). 'ycbcr_y' is fast "
                            "(~7ms @ 1024 CPU) — use for realtime preview. "
                            "'hsv_v' is simple max-RGB and less perceptual."
                        ),
                    },
                ),
                "shadow_end": (
                    "FLOAT",
                    {
                        "default": 0.33,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": (
                            "Upper bound of the shadow band. Luminance "
                            "values below this (minus half of soft_edge) "
                            "count fully as shadow."
                        ),
                    },
                ),
                "highlight_start": (
                    "FLOAT",
                    {
                        "default": 0.67,
                        "min": 0.5,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Lower bound of the highlight band. Luminance "
                            "values above this (plus half of soft_edge) "
                            "count fully as highlight. Must be > shadow_end."
                        ),
                    },
                ),
                "soft_edge": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.01,
                        "max": 0.3,
                        "step": 0.01,
                        "tooltip": (
                            "Smoothstep transition width at both band edges. "
                            "Smaller = sharper bands, larger = smoother "
                            "blend. Partition-of-unity preserved: "
                            "shadows + midtones + highlights ≈ 1."
                        ),
                    },
                ),
            },
        }

    def run(
        self,
        image: torch.Tensor,
        luminance_source: str,
        shadow_end: float,
        highlight_start: float,
        soft_edge: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            img_bchw = image.permute(0, 3, 1, 2).contiguous()
            shadows, midtones, highlights = luminosity_masks(
                img_bchw,
                luminance_source=luminance_source,
                shadow_end=shadow_end,
                highlight_start=highlight_start,
                soft_edge=soft_edge,
            )
        return (shadows, midtones, highlights)
