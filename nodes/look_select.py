"""JHPixelProLookSelect — 1-node dropdown look picker for M6 preset palette."""

from __future__ import annotations

from .look_base import apply_preset

PRESET_OPTIONS = [
    "cinematic-teal-orange",
    "warm-skin-tone",
    "moody-green",
    "faded-film",
    "golden-hour",
    "desaturated-pop",
]


class JHPixelProLookSelect:
    CATEGORY = "ComfyUI-JH-PixelPro/looks"

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (PRESET_OPTIONS, {"default": "cinematic-teal-orange"}),
                "intensity": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "protect_skin": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"

    def apply(self, image, preset, intensity, protect_skin):
        output = apply_preset(image, preset, intensity, protect_skin)
        return (output,)
