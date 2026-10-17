"""ComfyUI wrapper for dominant-colour palette extraction."""

from __future__ import annotations

import json

import torch

from ..core.palette import palette_extract


class JHPixelProPaletteExtract:
    """Extract the dominant colours of an image as a swatch strip, JSON and flat-colour image.

    ``palette_json`` shape: batch of 1 -> a list of ``{"hex", "rgb", "share"}`` dicts;
    batch > 1 -> a list (one per image) of such lists.
    """

    CATEGORY = "ComfyUI-JH-PixelPro/color"
    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("palette", "palette_json", "quantized")
    FUNCTION = "extract"
    OUTPUT_TOOLTIPS = (
        "Swatch strip, one equal-width band per colour, largest share first.",
        (
            "JSON text. Batch of 1: list of {hex, rgb, share}. Batch > 1: list of such "
            "lists, one per image."
        ),
        "The image with every pixel replaced by its nearest palette colour.",
    )

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to read the colours from."}),
                "colors": (
                    "INT",
                    {
                        "default": 6,
                        "min": 2,
                        "max": 16,
                        "step": 1,
                        "tooltip": "How many palette colours to find.",
                    },
                ),
            },
            "optional": {
                "mask": (
                    "MASK",
                    {
                        "tooltip": (
                            "Only pixels inside this mask count towards the palette "
                            "(e.g. the subject). The flat-colour image still covers the "
                            "whole frame. If a batch item's mask is empty, that item uses the "
                            "full image (a warning is logged)."
                        )
                    },
                ),
            },
        }

    def extract(
        self,
        image: torch.Tensor,
        colors: int,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, str, torch.Tensor]:
        with torch.no_grad():
            strip, palettes, quantized = palette_extract(image, mask, colors=colors)
        payload = palettes[0] if len(palettes) == 1 else palettes
        return (strip, json.dumps(payload), quantized)
