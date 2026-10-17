"""ComfyUI wrapper for ``core.depth_lens_blur.depth_lens_blur``."""

from __future__ import annotations

import torch

from ..core import depth_anything
from ..core.depth_lens_blur import BOKEH_SHAPES, depth_lens_blur


class JHPixelProDepthLensBlur:
    """Photographic lens blur (bokeh) driven by a depth map."""

    CATEGORY = "ComfyUI-JH-PixelPro/depth"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "coc")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802 - ComfyUI node contract mandates UPPER_CASE.
        return {
            "required": {
                "image": ("IMAGE",),
                "focus_depth": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Depth value that stays sharp (1 = nearest). Ignored when "
                            "focus_mask is connected."
                        ),
                    },
                ),
                "depth_of_field": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Width of the sharp zone around the focus depth. "
                            "Small = shallow depth of field (f/1.4), large = deep (f/16)."
                        ),
                    },
                ),
                "max_blur": (
                    "INT",
                    {
                        "default": 24,
                        "min": 1,
                        "max": 96,
                        "step": 1,
                        "tooltip": "Blur radius in pixels at the farthest distance from focus.",
                    },
                ),
                "bokeh": (
                    list(BOKEH_SHAPES),
                    {
                        "default": "disc",
                        "tooltip": "disc = flat lens-like bokeh balls, gaussian = soft blur.",
                    },
                ),
                "layers": (
                    "INT",
                    {
                        "default": 8,
                        "min": 2,
                        "max": 16,
                        "step": 1,
                        "tooltip": (
                            "Number of blur steps blended together. More = smoother "
                            "transitions, slower."
                        ),
                    },
                ),
                "blur_foreground": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Off: only things behind the focus plane are blurred.",
                    },
                ),
            },
            "optional": {
                "depth": (
                    "MASK",
                    {
                        "tooltip": (
                            "Depth map, 1 = nearest. Not connected: estimated internally with "
                            "Depth Anything V2 small (needs onnxruntime)."
                        )
                    },
                ),
                "focus_mask": (
                    "MASK",
                    {
                        "tooltip": (
                            "Subject mask (e.g. a face mask). The median depth inside it "
                            "becomes the focus depth."
                        )
                    },
                ),
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
        focus_depth: float,
        depth_of_field: float,
        max_blur: int,
        bokeh: str,
        layers: int,
        blur_foreground: bool,
        depth: torch.Tensor | None = None,
        focus_mask: torch.Tensor | None = None,
        keep_model_loaded: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            result = depth_lens_blur(
                image,
                depth,
                focus_depth=float(focus_depth),
                depth_of_field=float(depth_of_field),
                max_blur=int(max_blur),
                bokeh=bokeh,
                layers=int(layers),
                focus_mask=focus_mask,
                blur_foreground=bool(blur_foreground),
            )
        if not keep_model_loaded and depth is None:
            depth_anything.unload()
        return result
