"""ComfyUI wrapper for Heal / Content-Aware Fill."""

from __future__ import annotations

import torch

from ..core.heal import MODES, run


class JHPixelProHeal:
    """Remove what the mask covers and rebuild it from the rest of the photo."""

    CATEGORY = "ComfyUI-JH-PixelPro/filters"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "heal"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Photo to retouch."}),
                "mask": (
                    "MASK",
                    {"tooltip": "White = remove and fill (wires, objects, blemishes, logos)."},
                ),
                "mode": (
                    list(MODES),
                    {
                        "default": "content_aware",
                        "tooltip": "content_aware = rebuild texture and structure from nearby "
                        "patches (like Photoshop Content-Aware Fill). smooth = soft colour "
                        "fill, for flat areas.",
                    },
                ),
                "grow": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Pixels added around the mask so halos and shadows of the "
                        "object go too.",
                    },
                ),
                "feather": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 32.0,
                        "step": 0.5,
                        "tooltip": "Soft blend at the edge of the filled area, in pixels.",
                    },
                ),
            },
            "optional": {
                "context": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.25,
                        "max": 8.0,
                        "step": 0.25,
                        "tooltip": "How far around the mask to look for material, as a "
                        "multiple of the mask size. Raise it when the right texture is far "
                        "away.",
                    },
                ),
            },
        }

    def heal(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        mode: str,
        grow: int,
        feather: float,
        context: float = 1.0,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            out = run(
                image,
                mask,
                mode=mode,
                grow=int(grow),
                feather=float(feather),
                context=float(context),
            )
        return (out.to(image.device),)
