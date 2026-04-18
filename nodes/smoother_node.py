"""ComfyUI wrapper for ``core.smoother.edge_aware_smooth``."""

from __future__ import annotations

import torch

from ..core import edge_aware_smooth


class JHPixelProEdgeAwareSmoother:
    """Apply edge-preserving bilateral smoothing to a ComfyUI IMAGE, optionally gated by a MASK."""

    CATEGORY = "ComfyUI-JH-PixelPro/filters"
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
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": (
                            "Spatial sigma in pixels. Range 1.0–8.0 (v1.1 cap). "
                            "Kernel size auto-sized to 2*ceil(3*sigma_space)+1. "
                            "Need wider? Downsample the image first with a "
                            "Resize node upstream."
                        ),
                    },
                ),
                "device": (
                    ["auto", "cpu", "cuda"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Compute device. 'auto' picks CUDA if available, "
                            "else CPU. Explicit 'cuda' raises if CUDA "
                            "unavailable. 'cpu' forces CPU (slow but "
                            "deterministic)."
                        ),
                    },
                ),
                "tile_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Enable 512x512 tile processing to avoid OOM on "
                            "large images. Required for 4K+ or sigma_space > 4 "
                            "on most GPUs. Leave off for images ≤1K for max "
                            "speed."
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
        device: str,
        tile_mode: bool,
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
                device=device,
                tile_mode=tile_mode,
            )
            out = out_bchw.permute(0, 2, 3, 1).contiguous()
        return (out,)
