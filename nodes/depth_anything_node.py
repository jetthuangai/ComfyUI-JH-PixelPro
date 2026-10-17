"""ComfyUI wrapper for ``core.depth_anything.estimate_depth`` (Depth Anything V2 via ONNX)."""

from __future__ import annotations

import torch

from ..core import depth_anything
from ..core.depth_anything import (
    MODEL_SIZES,
    RESOLUTION_MAX,
    RESOLUTION_MIN,
    estimate_depth,
)


class JHPixelProDepthAnything:
    """Estimate a relative depth map (1 = nearest) with Depth Anything V2."""

    CATEGORY = "ComfyUI-JH-PixelPro/depth"
    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("depth", "depth_preview")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802 - ComfyUI node contract mandates UPPER_CASE.
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (
                    list(MODEL_SIZES),
                    {
                        "default": "small",
                        "tooltip": (
                            "small = Apache-2.0 (free for commercial use), fastest. "
                            "base and large = CC-BY-NC-4.0 (NON-COMMERCIAL use only), more "
                            "detail. Models are saved in ComfyUI/models/depth_anything."
                        ),
                    },
                ),
                "resolution": (
                    "INT",
                    {
                        "default": 518,
                        "min": RESOLUTION_MIN,
                        "max": RESOLUTION_MAX,
                        "step": 14,
                        "tooltip": (
                            "Long side (px) the model works at, rounded to a multiple of 14. "
                            "Higher = finer depth edges but slower."
                        ),
                    },
                ),
            },
            "optional": {
                "keep_model_loaded": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "ON: keep the Depth Anything model in VRAM for the next run (faster). "
                            "OFF: free it right after this run (ComfyUI's Free memory "
                            "button cannot see it)."
                        ),
                    },
                ),
            },
        }

    def run(
        self,
        image: torch.Tensor,
        model: str,
        resolution: int,
        keep_model_loaded: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            depth = estimate_depth(image, model=model, resolution=int(resolution))
            preview = depth.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()
        if not keep_model_loaded:
            depth_anything.unload()
        return (depth, preview)
