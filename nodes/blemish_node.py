"""ComfyUI wrapper for Auto Blemish."""

from __future__ import annotations

import torch

from ..core.blemish import auto_blemish


class JHPixelProAutoBlemish:
    """Find faces and heal pimples, spots and small marks on their skin, keeping pores."""

    CATEGORY = "ComfyUI-JH-PixelPro/face"
    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("image", "spot_mask", "face_count")
    OUTPUT_TOOLTIPS = (
        "The retouched image.",
        "Pixels that were healed (check it with Mask Preview).",
        "Faces found (0 when a mask is given).",
    )
    FUNCTION = "retouch"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Portrait to retouch."}),
                "sensitivity": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Higher catches fainter marks; very high also removes freckles.",
                    },
                ),
                "spot_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.25,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": "Largest mark to remove, relative to the automatic size "
                        "(5 % of the face width). Raise it for bigger pimples.",
                    },
                ),
            },
            "optional": {
                "mask": (
                    "MASK",
                    {
                        "tooltip": "Skin area to clean instead of the detected faces (body, "
                        "back, hands)."
                    },
                ),
                "protect_mask": (
                    "MASK",
                    {"tooltip": "Never touch these areas (a beauty mark you keep, tattoos)."},
                ),
            },
        }

    def retouch(
        self,
        image: torch.Tensor,
        sensitivity: float,
        spot_scale: float,
        mask: torch.Tensor | None = None,
        protect_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        with torch.no_grad():
            out, spots, count = auto_blemish(
                image,
                mask=mask,
                protect_mask=protect_mask,
                sensitivity=float(sensitivity),
                spot_scale=float(spot_scale),
            )
        return out.to(image.device), spots.to(image.device), int(count)
