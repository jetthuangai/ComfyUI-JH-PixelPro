"""ComfyUI wrapper for dust and spot removal."""

from __future__ import annotations

import torch

from ..core.dust_spot import dust_spot_removal


class JHPixelProDustSpotRemoval:
    """Detect small dust specks and blemishes and fill them from their surroundings."""

    CATEGORY = "ComfyUI-JH-PixelPro/filters"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "spot_mask")
    FUNCTION = "remove"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to clean."}),
                "spot_size": (
                    "INT",
                    {
                        "default": 12,
                        "min": 2,
                        "max": 64,
                        "step": 1,
                        "tooltip": (
                            "Largest spot diameter in pixels. Anything bigger or longer "
                            "(eyes, hair, edges) is left alone."
                        ),
                    },
                ),
                "sensitivity": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Higher catches fainter spots (and more skin texture).",
                    },
                ),
                "mode": (
                    ["dark", "bright", "both"],
                    {
                        "default": "dark",
                        "tooltip": (
                            "dark = dust on the sensor, bright = white specks, both = either."
                        ),
                    },
                ),
            },
            "optional": {
                "mask": (
                    "MASK",
                    {"tooltip": "Only remove spots inside this mask (e.g. skin only)."},
                ),
                "protect_mask": (
                    "MASK",
                    {"tooltip": "Never touch spots inside this mask (e.g. freckles, eyes)."},
                ),
            },
        }

    def remove(
        self,
        image: torch.Tensor,
        spot_size: int,
        sensitivity: float,
        mode: str,
        mask: torch.Tensor | None = None,
        protect_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            cleaned, spot_mask = dust_spot_removal(
                image,
                mask,
                protect_mask,
                spot_size=spot_size,
                sensitivity=sensitivity,
                mode=mode,
            )
        return (cleaned, spot_mask)
