"""ComfyUI wrapper for per-triangle face warp using dense landmarks."""

from __future__ import annotations

import torch

from ..core.face_pipeline import face_warp_delaunay


def _first_face(points: torch.Tensor) -> torch.Tensor:
    if points.ndim == 4:
        return points[:, 0, :, :]
    if points.ndim == 3:
        return points
    raise ValueError(
        f"LANDMARKS input must have shape (B,F,468,2) or (B,468,2), got {tuple(points.shape)}."
    )


class JHPixelProFaceWarp:
    """Warp a face image from source landmarks to destination landmarks."""

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
            },
        }

    def warp(
        self,
        image: torch.Tensor,
        src_landmarks: torch.Tensor,
        dst_landmarks: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            output = face_warp_delaunay(
                image,
                _first_face(src_landmarks),
                _first_face(dst_landmarks),
                device=image.device,
            )
        return (output,)
