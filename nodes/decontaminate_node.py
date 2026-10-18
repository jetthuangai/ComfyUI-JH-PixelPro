"""ComfyUI wrapper for colour decontamination of cut-out edges."""

from __future__ import annotations

import torch

from ..core.decontaminate import decontaminate_colors


class JHPixelProDecontaminateColors:
    """Remove background colour fringes (green screen, sky) from a subject's soft edges."""

    CATEGORY = "ComfyUI-JH-PixelPro/mask"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = ("Image whose edge colours come from the subject, not the background.",)
    FUNCTION = "decontaminate"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image the mask was made from."}),
                "mask": (
                    "MASK",
                    {"tooltip": "Subject mask with soft edges (1 = subject), e.g. Alpha Matte."},
                ),
                "radius": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 256,
                        "step": 1,
                        "tooltip": "How far to look for the subject's own colour. Raise it for "
                        "wide, wispy edges such as hair.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "0 = unchanged, 1 = full correction.",
                    },
                ),
            },
        }

    def decontaminate(
        self, image: torch.Tensor, mask: torch.Tensor, radius: int, strength: float
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            out = decontaminate_colors(image, mask, radius=int(radius), strength=float(strength))
        return (out,)
