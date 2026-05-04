"""ComfyUI wrapper for region-aware LAB color matching."""

from __future__ import annotations

import torch

from ..core import color_matcher_region

_CHANNEL_CHOICES = ["ab", "lab"]


class JHPixelProColorMatcherRegion:
    """Match colors from optional reference mask into optional target mask."""

    CATEGORY = "ComfyUI-JH-PixelPro/color"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_matched",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "image_target": ("IMAGE",),
                "image_reference": ("IMAGE",),
                "channels": (
                    _CHANNEL_CHOICES,
                    {
                        "default": "ab",
                        "tooltip": (
                            "ab = match chroma only and preserve target luminance. "
                            "lab = match L + a + b for full tone transfer."
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
                        "tooltip": "Blend factor. 0 = bypass, 1 = full region match.",
                    },
                ),
            },
            "optional": {
                "target_mask": (
                    "MASK",
                    {
                        "tooltip": (
                            "Optional target region. It controls target stats and where "
                            "the matched result is applied. Unconnected = full target."
                        ),
                    },
                ),
                "reference_mask": (
                    "MASK",
                    {
                        "tooltip": (
                            "Optional reference sampling region. Unconnected = full reference."
                        ),
                    },
                ),
            },
        }

    def run(
        self,
        image_target: torch.Tensor,
        image_reference: torch.Tensor,
        channels: str,
        strength: float,
        target_mask: torch.Tensor | None = None,
        reference_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            target_bchw = image_target.permute(0, 3, 1, 2).contiguous()
            reference_bchw = image_reference.permute(0, 3, 1, 2).contiguous()
            out_bchw = color_matcher_region(
                target_bchw,
                reference_bchw,
                channels=channels,
                strength=float(strength),
                target_mask=target_mask,
                reference_mask=reference_mask,
            )
            out = out_bchw.permute(0, 2, 3, 1).contiguous()
        return (out,)
