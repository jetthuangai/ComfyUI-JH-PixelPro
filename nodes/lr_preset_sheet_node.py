"""ComfyUI wrapper: contact sheet of Lightroom presets applied to an image."""

from __future__ import annotations

import torch

from ..core.lr_preset import list_presets
from ..core.lr_preset_library import ALL, collections, filter_presets, render_contact_sheet


class JHPixelProLightroomPresetSheet:
    """Compare many presets at once: one labelled grid, original first ([!] = partly supported)."""

    CATEGORY = "ComfyUI-JH-PixelPro/color"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("sheet", "presets")
    OUTPUT_TOOLTIPS = (
        "Labelled grid: the original, then each preset.",
        "Names of the presets on the sheet, one per line (same order).",
    )
    FUNCTION = "render"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Photo to preview the presets on (first frame)."}),
                "collection": (
                    collections(),
                    {"default": ALL, "tooltip": "All, Favorites, Recent or a preset sub-folder."},
                ),
                "columns": ("INT", {"default": 4, "min": 1, "max": 12}),
                "thumb_size": (
                    "INT",
                    {
                        "default": 320,
                        "min": 128,
                        "max": 768,
                        "step": 32,
                        "tooltip": "Tile long edge.",
                    },
                ),
                "max_presets": ("INT", {"default": 24, "min": 1, "max": 200}),
            },
            "optional": {
                "search": (
                    "STRING",
                    {"default": "", "tooltip": "Only names containing these words."},
                ),
                "tone_engine": (["local", "legacy"], {"default": "local"}),
            },
        }

    def render(
        self,
        image: torch.Tensor,
        collection: str,
        columns: int,
        thumb_size: int,
        max_presets: int,
        search: str = "",
        tone_engine: str = "local",
    ) -> tuple[torch.Tensor, str]:
        names = filter_presets(list_presets(), collection, search)[: int(max_presets)]
        with torch.no_grad():
            sheet = render_contact_sheet(
                image, names, columns=columns, thumb_size=thumb_size, tone_engine=tone_engine
            )
        return sheet, "\n".join(names)
