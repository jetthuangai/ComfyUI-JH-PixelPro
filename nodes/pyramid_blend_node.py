"""ComfyUI wrapper for multi-band Laplacian pyramid blending."""

from __future__ import annotations

import torch

from ..core.pyramid_blend import pyramid_blend


class JHPixelProPyramidBlend:
    """Seamlessly composite image_b over image_a through a mask using a Laplacian pyramid."""

    CATEGORY = "ComfyUI-JH-PixelPro/compositing"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "blend"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "image_a": ("IMAGE", {"tooltip": "Base image, kept where mask is 0."}),
                "image_b": ("IMAGE", {"tooltip": "Overlay image, taken where mask is 1."}),
                "mask": ("MASK", {"tooltip": "1 takes image_b, 0 keeps image_a."}),
                "levels": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": (
                            "Pyramid depth. Higher = wider, smoother seam. "
                            "1 = plain alpha composite. Auto-reduced on small images."
                        ),
                    },
                ),
            }
        }

    def blend(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        mask: torch.Tensor,
        levels: int,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            out = pyramid_blend(image_a, image_b, mask, levels=levels)
        return (out,)
