"""ComfyUI wrapper for Shadows/Highlights Pro (Camera Raw / Photoshop models)."""

from __future__ import annotations

import torch

from ..core.shadows_highlights import MODES, shadows_highlights


def _ps_float(default: float, low: float, high: float, step: float, tip: str) -> tuple[str, dict]:
    return (
        "FLOAT",
        {
            "default": default,
            "min": low,
            "max": high,
            "step": step,
            "tooltip": "Photoshop mode. " + tip,
        },
    )


class JHPixelProShadowsHighlights:
    """Edge-aware shadow lift and highlight recovery without the flat 'cheap HDR' look."""

    CATEGORY = "ComfyUI-JH-PixelPro/color"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image (B,H,W,3)."}),
                "shadows": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 1.0,
                        "tooltip": "Positive lifts dark areas relative to their surroundings.",
                    },
                ),
                "highlights": (
                    "FLOAT",
                    {
                        "default": -30.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 1.0,
                        "tooltip": "Negative recovers bright areas (skies, windows, skin shine).",
                    },
                ),
                "mode": (
                    list(MODES),
                    {
                        "default": "camera_raw",
                        "tooltip": (
                            "camera_raw = Lightroom/Camera Raw sliders (no halos). "
                            "photoshop = Image > Adjustments > Shadows/Highlights; only lifting "
                            "shadows and darkening highlights."
                        ),
                    },
                ),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask limiting where the effect applies."}),
                "ps_shadow_tone": _ps_float(
                    50.0, 0.0, 100.0, 1.0, "Tonal width of the shadows (%)."
                ),
                "ps_shadow_radius": _ps_float(
                    30.0, 0.0, 2500.0, 1.0, "Shadows neighbourhood (px)."
                ),
                "ps_highlight_tone": _ps_float(
                    50.0, 0.0, 100.0, 1.0, "Tonal width of the highlights (%)."
                ),
                "ps_highlight_radius": _ps_float(
                    30.0, 0.0, 2500.0, 1.0, "Highlights neighbourhood (px)."
                ),
                "ps_color": _ps_float(
                    20.0, -100.0, 100.0, 1.0, "Saturation in the adjusted tones."
                ),
                "ps_midtone": _ps_float(0.0, -100.0, 100.0, 1.0, "Midtone contrast."),
            },
        }

    def run(
        self,
        image: torch.Tensor,
        shadows: float,
        highlights: float,
        mode: str,
        mask: torch.Tensor | None = None,
        **ps: float,
    ) -> tuple[torch.Tensor]:
        options = {
            key.removeprefix("ps_"): float(value)
            for key, value in ps.items()
            if key.startswith("ps_")
        }
        with torch.no_grad():
            out = shadows_highlights(
                image,
                mode=mode,
                shadows=shadows,
                highlights=highlights,
                mask=mask,
                ps_options=options,
            )
        return (out,)
