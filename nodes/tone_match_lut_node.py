"""ComfyUI wrapper for automatic tone-match LUT generation."""

from __future__ import annotations

import torch

from ..core.tone_match import tone_match_lut

_LEVEL_CHOICES = ["4", "6", "8", "10", "12"]


class JHPixelProToneMatchLUT:
    """Generate a .cube LUT by matching a HALD to a reference image."""

    CATEGORY = "ComfyUI-JH-PixelPro/color"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lut_path",)
    FUNCTION = "generate"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "reference": ("IMAGE",),
                "level": (
                    _LEVEL_CHOICES,
                    {
                        "default": "8",
                        "tooltip": (
                            "HALD LUT level. 8 -> 64^3 entries (industry-standard "
                            "creative-look resolution)."
                        ),
                    },
                ),
                "filename": (
                    "STRING",
                    {
                        "default": "tone_match.cube",
                        "tooltip": (
                            "Output .cube filename. Relative paths resolve against "
                            "ComfyUI's output/ directory; absolute paths are honored."
                        ),
                    },
                ),
                "title": (
                    "STRING",
                    {
                        "default": "Tone Match LUT",
                        "tooltip": "Adobe Cube TITLE header embedded into the exported file.",
                    },
                ),
            },
        }

    def generate(
        self,
        reference: torch.Tensor,
        level: str,
        filename: str,
        title: str,
    ) -> tuple[str]:
        path = tone_match_lut(
            reference,
            int(level),
            filename,
            title,
            device=reference.device,
        )
        return (path,)
