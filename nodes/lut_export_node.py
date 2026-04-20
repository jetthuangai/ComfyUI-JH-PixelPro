"""ComfyUI wrapper for ``core.lut.export_cube`` — writes Adobe Cube 1.0 (.cube)."""

from __future__ import annotations

import os

import torch

from ..core.lut import export_cube


class JHPixelProLUTExport:
    """Export a graded HALD image as a portable Adobe Cube 1.0 (``.cube``) 3D LUT."""

    CATEGORY = "ComfyUI-JH-PixelPro/color"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "export"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802 — ComfyUI node contract mandates UPPER_CASE.
        return {
            "required": {
                "image": ("IMAGE",),
                "level": (
                    "INT",
                    {
                        "default": 8,
                        "min": 2,
                        "max": 16,
                        "step": 1,
                        "tooltip": (
                            "HALD level L — MUST match the level used by the "
                            "upstream HALD Identity node. Image H==W==L³ is "
                            "validated; mismatch raises ValueError."
                        ),
                    },
                ),
                "filename": (
                    "STRING",
                    {
                        "default": "pack_lut.cube",
                        "tooltip": (
                            "Filename (relative to ComfyUI output/) or "
                            "absolute path. '.cube' extension auto-appended "
                            "if missing. Existing files are overwritten."
                        ),
                    },
                ),
                "title": (
                    "STRING",
                    {
                        "default": "JHPixelPro LUT",
                        "tooltip": (
                            "Adobe Cube TITLE header metadata — shown by "
                            "DaVinci Resolve, Premiere and other LUT tools "
                            "when browsing cube files."
                        ),
                    },
                ),
            },
        }

    def export(
        self,
        image: torch.Tensor,
        level: int,
        filename: str,
        title: str,
    ) -> tuple[str]:
        resolved = self._resolve_output_path(filename)
        written = export_cube(image, int(level), resolved, title=title)
        return (written,)

    @staticmethod
    def _resolve_output_path(filename: str) -> str:
        name = filename.strip()
        if not name.lower().endswith(".cube"):
            name = name + ".cube"
        if os.path.isabs(name):
            return name
        try:
            import folder_paths  # ComfyUI runtime module (not installed in unit tests).

            base = folder_paths.get_output_directory()
        except ImportError:
            base = os.path.abspath("output")
        return os.path.join(base, name)
