"""ComfyUI wrappers for ``core.frequency`` (separation and recombination)."""

from __future__ import annotations

import torch

from ..core import frequency_recombine, frequency_separation


class JHPixelProFrequencySeparation:
    """Split a ComfyUI IMAGE into Gaussian low/high frequency pins."""

    CATEGORY = "ComfyUI-JH-PixelPro/filters"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("low", "high")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802 — ComfyUI node contract mandates UPPER_CASE.
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
                            "error ~1e-3. 'high' may contain negative values, so its "
                            "preview looks dark; rebuild with Frequency Recombine."
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


class JHPixelProFrequencyRecombine:
    """Rebuild an image from Frequency Separation's low and high layers."""

    CATEGORY = "ComfyUI-JH-PixelPro/filters"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802 — ComfyUI node contract mandates UPPER_CASE.
        return {
            "required": {
                "low": ("IMAGE", {"tooltip": "Low layer (colour and tone), edited or not."}),
                "high": (
                    "IMAGE",
                    {"tooltip": "High layer (texture). Signed values; connect it directly."},
                ),
                "detail": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 3.0,
                        "step": 0.05,
                        "tooltip": (
                            "How much texture to add back. 1 = original, above 1 boosts "
                            "pores and fine detail, below 1 softens them."
                        ),
                    },
                ),
            },
        }

    def run(self, low: torch.Tensor, high: torch.Tensor, detail: float) -> tuple[torch.Tensor]:
        with torch.no_grad():
            out_bchw = frequency_recombine(
                low.permute(0, 3, 1, 2), high.permute(0, 3, 1, 2), detail=float(detail)
            )
        return (out_bchw.permute(0, 2, 3, 1).contiguous(),)
