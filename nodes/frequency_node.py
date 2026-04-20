"""ComfyUI wrapper for ``core.frequency.frequency_separation``."""

from __future__ import annotations

import torch

from ..core import frequency_separation


class JHPixelProFrequencySeparation:
    """Split a ComfyUI IMAGE into Gaussian low/high frequency pins."""

    CATEGORY = "ComfyUI-JH-PixelPro/filters"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("low", "high")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802 — ComfyUI node contract mandates UPPER_CASE.
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 128,
                        "tooltip": "Gaussian blur radius in pixels.",
                    },
                ),
                "sigma": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 50.0,
                        "step": 0.1,
                        "tooltip": (
                            "Sigma override. 0.0 = auto-compute radius/2 (Photoshop convention)."
                        ),
                    },
                ),
                "precision": (
                    ["float32", "float16"],
                    {
                        "default": "float32",
                        "tooltip": (
                            "float32 = lossless reconstruction (atol 1e-5); "
                            "float16 = ~2x faster on modern GPU, reconstruction "
                            "error ~1e-3. 'high' may contain negative values — "
                            "use ImageAdd to reconstruct, preview may look wrong."
                        ),
                    },
                ),
            },
        }

    def run(
        self,
        image: torch.Tensor,
        radius: int,
        sigma: float,
        precision: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            img_bchw = image.permute(0, 3, 1, 2).contiguous()
            low_bchw, high_bchw = frequency_separation(
                img_bchw,
                radius=radius,
                sigma=sigma,
                precision=precision,
            )
            low = low_bchw.permute(0, 2, 3, 1).contiguous()
            high = high_bchw.permute(0, 2, 3, 1).contiguous()
        return (low, high)
