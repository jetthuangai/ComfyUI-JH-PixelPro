"""ComfyUI wrapper nodes for JSON-driven Look presets."""

from __future__ import annotations

import torch

from .look_base import apply_preset


class _BaseLookPreset:
    CATEGORY = "ComfyUI-JH-PixelPro/looks"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    PRESET_ID = ""

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "Preset opacity. 0 = unchanged input, 1 = full look.",
                    },
                ),
            },
            "optional": {
                "protect_skin": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Reduce each preset operation over orange-pink skin-tone pixels."
                        ),
                    },
                ),
            },
        }

    def apply(
        self,
        image: torch.Tensor,
        intensity: float,
        protect_skin: bool = False,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            output = apply_preset(
                image,
                self.PRESET_ID,
                intensity=float(intensity),
                protect_skin=bool(protect_skin),
            )
        return (output,)


class JHPixelProLookCinematicTealOrange(_BaseLookPreset):
    """N-22: Teal shadows, warm highlights, and contrast S-curve."""

    PRESET_ID = "cinematic-teal-orange"


class JHPixelProLookWarmSkinTone(_BaseLookPreset):
    """N-23: Portrait-oriented skin warmth with optional skin protection."""

    PRESET_ID = "warm-skin-tone"


class JHPixelProLookMoodyGreen(_BaseLookPreset):
    """N-24: Crushed shadows, green cast, and muted saturation."""

    PRESET_ID = "moody-green"


class JHPixelProLookFadedFilm(_BaseLookPreset):
    """N-25: Lifted blacks, warm mids, and soft desaturation."""

    PRESET_ID = "faded-film"


class JHPixelProLookGoldenHour(_BaseLookPreset):
    """N-26: Global warmth, boosted yellow-orange, and protected blues."""

    PRESET_ID = "golden-hour"


class JHPixelProLookDesaturatedPop(_BaseLookPreset):
    """N-27: Muted backgrounds with protected orange-red tones."""

    PRESET_ID = "desaturated-pop"
