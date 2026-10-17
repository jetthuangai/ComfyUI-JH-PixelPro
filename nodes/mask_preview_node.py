"""ComfyUI wrapper for the mask preview overlay."""

from __future__ import annotations

import torch

from ..core.mask_preview import COLORMAPS, COLORS, MODES, mask_preview


class JHPixelProMaskPreview:
    """Visualise a MASK on top of an image for debugging."""

    CATEGORY = "ComfyUI-JH-PixelPro/mask"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "preview"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Mask to visualise."}),
                "mode": (
                    list(MODES),
                    {
                        "default": "overlay",
                        "tooltip": (
                            "overlay = colour tint, heatmap = colormap of mask values, "
                            "mask_only = greyscale mask, cutout = image over checkerboard."
                        ),
                    },
                ),
                "color": (
                    list(COLORS),
                    {"default": "red", "tooltip": "Tint colour (overlay) and edge colour."},
                ),
                "opacity": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Overlay strength (overlay and heatmap modes).",
                    },
                ),
                "colormap": (
                    list(COLORMAPS),
                    {"default": "turbo", "tooltip": "Colormap used by heatmap mode."},
                ),
                "show_edges": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Draw the 0.5 mask contour in the chosen colour.",
                    },
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Background image. Omit for a neutral grey background."},
                ),
            },
        }

    def preview(
        self,
        mask: torch.Tensor,
        mode: str,
        color: str,
        opacity: float,
        colormap: str,
        show_edges: bool,
        image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            out = mask_preview(
                image,
                mask,
                mode=mode,
                color=color,
                opacity=opacity,
                colormap=colormap,
                show_edges=show_edges,
            )
        return (out,)
