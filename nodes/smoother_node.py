"""ComfyUI wrapper for ``core.smoother.edge_aware_smooth``."""

from __future__ import annotations

import torch

from ..core import edge_aware_smooth


class JHPixelProEdgeAwareSmoother:
    """Apply edge-preserving bilateral smoothing to a ComfyUI IMAGE, optionally gated by a MASK."""

    CATEGORY = "image/pixelpro/filters"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802 — ComfyUI node contract mandates UPPER_CASE.
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Blend between smoothed and original. 0 = identity "
                            "(bypass), 1 = full smoothing. Typical pro dose "
                            "0.3–0.5."
                        ),
                    },
                ),
                "sigma_color": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.01,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": (
                            "Intensity sigma on the [0, 1] image scale — not the "
                            "8-bit 10–50 range from OpenCV docs. Small values "
                            "preserve edges; large values smooth across weak "
                            "edges."
                        ),
                    },
                ),
                "sigma_space": (
                    "FLOAT",
                    {
                        "default": 6.0,
                        "min": 1.0,
                        "max": 32.0,
                        "step": 0.1,
                        "tooltip": (
                            "Spatial sigma in pixels. Larger = wider spatial "
                            "influence = stronger smoothing. Kernel size is "
                            "auto-sized to 2*ceil(3*sigma_space)+1."
                        ),
                    },
                ),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    def run(
        self,
        image: torch.Tensor,
        strength: float,
        sigma_color: float,
        sigma_space: float,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            img_bchw = image.permute(0, 3, 1, 2).contiguous()
            mask_bc1hw = mask.unsqueeze(1).contiguous() if mask is not None else None
            out_bchw = edge_aware_smooth(
                img_bchw,
                strength=strength,
                sigma_color=sigma_color,
                sigma_space=sigma_space,
                mask_bchw=mask_bc1hw,
            )
            out = out_bchw.permute(0, 2, 3, 1).contiguous()
        return (out,)
