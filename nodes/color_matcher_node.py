"""ComfyUI wrapper for ``core.color_matcher.color_matcher`` (Reinhard LAB color transfer)."""

from __future__ import annotations

import torch

from ..core import color_matcher

_CHANNEL_CHOICES = ["ab", "lab"]


class JHPixelProColorMatcher:
    """Match target RGB colors to a reference image in LAB space (Reinhard transfer)."""

    CATEGORY = "ComfyUI-JH-PixelPro/color"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_matched",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802 — ComfyUI node contract mandates UPPER_CASE.
        return {
            "required": {
                "image_target": ("IMAGE",),
                "image_reference": ("IMAGE",),
                "channels": (
                    _CHANNEL_CHOICES,
                    {
                        "default": "ab",
                        "tooltip": (
                            "ab = match chroma only, preserve target luminance "
                            "(pro retouch default — avoids washing out the AI "
                            "output's lighting). lab = match L + a + b (full tone "
                            "transfer including brightness)."
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
                            "Blend factor. 0 = identity target (bypass), 1 = full "
                            "match. Typical pro dose 0.6–0.8 for natural skin-tone "
                            "correction."
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
        image_target: torch.Tensor,
        image_reference: torch.Tensor,
        channels: str,
        strength: float,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            target_bchw = image_target.permute(0, 3, 1, 2).contiguous()
            reference_bchw = image_reference.permute(0, 3, 1, 2).contiguous()
            out_bchw = color_matcher(
                target_bchw,
                reference_bchw,
                channels=channels,
                strength=float(strength),
                mask=mask,
            )
            out = out_bchw.permute(0, 2, 3, 1).contiguous()
        return (out,)
