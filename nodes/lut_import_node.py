"""ComfyUI wrapper for ``core.lut.parse_cube`` + ``core.lut.apply_lut_3d``.

Reads an Adobe Cube 1.0 (``.cube``) 3D LUT from disk and applies it to a
BHWC image via trilinear ``grid_sample``. Pairs with ``N-13 LUT Export``
to close the round-trip loop: develop a grade in ComfyUI, export the
``.cube``, then re-apply it here (or in DaVinci / Premiere / OBS / OCIO).
"""

from __future__ import annotations

import os

import torch

from ..core.lut import apply_lut_3d, parse_cube


class JHPixelProLUTImport:
    """Import a portable Adobe Cube 1.0 (``.cube``) 3D LUT and apply trilinear."""

    CATEGORY = "ComfyUI-JH-PixelPro/color"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802 — ComfyUI node contract mandates UPPER_CASE.
        return {
            "required": {
                "image": ("IMAGE",),
                "filename": (
                    "STRING",
                    {
                        "default": "pack_lut.cube",
                        "tooltip": (
                            "Path to the .cube file. Relative paths resolve "
                            "against ComfyUI's input/ directory; absolute paths "
                            "are honored verbatim. The '.cube' extension is NOT "
                            "auto-appended on read (explicit, so the user sees "
                            "the exact filename being loaded). Pairs with the "
                            "'pack_lut.cube' default emitted by N-13 LUT Export "
                            "to close the round-trip loop."
                        ),
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Blend factor between the original image and the "
                            "LUT-applied result: 0.0 = pass-through (LUT "
                            "ignored), 1.0 = full LUT. Linear interpolation "
                            "out = in * (1 - s) + lut(in) * s. Use 0.5 – 0.8 "
                            "for a subtle creative-look dose."
                        ),
                    },
                ),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    def apply(
        self,
        image: torch.Tensor,
        filename: str,
        strength: float,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        resolved = self._resolve_input_path(filename)
        parsed = parse_cube(resolved)
        lut_grid = parsed["lut"].to(image.device)
        domain_min = parsed["domain_min"].to(image.device)
        domain_max = parsed["domain_max"].to(image.device)
        with torch.no_grad():
            out = apply_lut_3d(
                image,
                lut_grid,
                strength=float(strength),
                mask=mask,
                domain_min=domain_min,
                domain_max=domain_max,
            )
        return (out,)

    @staticmethod
    def _resolve_input_path(filename: str) -> str:
        name = filename.strip()
        if os.path.isabs(name):
            return name
        try:
            import folder_paths  # ComfyUI runtime module (not installed in unit tests).

            base = folder_paths.get_input_directory()
        except ImportError:
            base = os.path.abspath("input")
        return os.path.join(base, name)
