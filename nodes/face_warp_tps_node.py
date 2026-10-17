"""ComfyUI wrapper for the smooth thin-plate-spline face warp."""

from __future__ import annotations

import torch

from ..core.face_warp_tps import face_warp_tps
from .face_warp_node import _first_face


class JHPixelProFaceWarpTPS:
    """Warp a face image from source to destination landmarks with a smooth TPS field."""

    CATEGORY = "ComfyUI-JH-PixelPro/face"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("warped",)
    FUNCTION = "warp"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "image": ("IMAGE",),
                "src_landmarks": ("LANDMARKS",),
                "dst_landmarks": ("LANDMARKS",),
                "max_points": (
                    "INT",
                    {
                        "default": 468,
                        "min": 16,
                        "max": 468,
                        "step": 1,
                        "tooltip": (
                            "Maximum landmark pairs used for the spline. Moved landmarks are "
                            "always kept; the rest is filled with evenly spaced unmoved ones."
                        ),
                    },
                ),
                "anchor_border": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Pin points along the image border so the background does not swim."
                        ),
                    },
                ),
            },
            # Optional so API-format prompts saved with v1.5.0 (no smoothness) still queue.
            "optional": {
                "smoothness": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": (
                            "0 = every landmark lands exactly. 1 (default) absorbs detection "
                            "noise when target landmarks come from another photo (avoids "
                            "ripples/folds); clean edits stay within ~1 px. Raise if you still "
                            "see ripples."
                        ),
                    },
                ),
            },
        }

    def warp(
        self,
        image: torch.Tensor,
        src_landmarks: torch.Tensor,
        dst_landmarks: torch.Tensor,
        max_points: int = 468,
        anchor_border: bool = True,
        smoothness: float = 1.0,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            output = face_warp_tps(
                image,
                _first_face(src_landmarks),
                _first_face(dst_landmarks),
                max_points=max_points,
                anchor_border=anchor_border,
                smoothness=smoothness,
                device=image.device,
            )
        return (output,)
