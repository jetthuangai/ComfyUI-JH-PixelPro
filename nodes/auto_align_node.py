"""ComfyUI wrapper for ``core.auto_align.auto_align``."""

from __future__ import annotations

import json

import torch

from ..core.auto_align import auto_align


class JHPixelProAutoAlign:
    """Align an image to a reference with feature matching, RANSAC and a perspective warp."""

    CATEGORY = "ComfyUI-JH-PixelPro/geometry"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT")
    RETURN_NAMES = ("aligned", "valid_mask", "matrix_json", "inliers")
    # "inliers" reports the FIRST batch item only; matrix_json holds one matrix per item.
    OUTPUT_TOOLTIPS = (
        "Image warped into the reference frame.",
        "1 where the warped image has real pixels.",
        "JSON list of 3x3 matrices, one per batch item (image to reference pixels).",
        "Agreeing-match count of the first batch item only.",
    )
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802 — ComfyUI node contract mandates UPPER_CASE.
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to warp onto the reference."}),
                "reference": (
                    "IMAGE",
                    {"tooltip": "Target frame. Output has this size. Batch 1 or same as image."},
                ),
                "method": (
                    ["sift", "loftr"],
                    {
                        "default": "sift",
                        "tooltip": (
                            "sift: classic keypoints, no download. loftr: learned dense matcher "
                            "for hard pairs; downloads ~45 MB on first use."
                        ),
                    },
                ),
                "model": (
                    ["homography", "affine"],
                    {
                        "default": "homography",
                        "tooltip": (
                            "homography handles perspective change; affine (rotate/scale/shear/"
                            "shift) is steadier for near-flat scenes or few matches."
                        ),
                    },
                ),
                "max_side": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Longest side used for matching. The warp uses full resolution.",
                    },
                ),
                "ransac_threshold": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.5,
                        "max": 20.0,
                        "step": 0.5,
                        "tooltip": "Max error (pixels at matching size) for a match to count.",
                    },
                ),
                "min_inliers": (
                    "INT",
                    {
                        "default": 12,
                        "min": 4,
                        "max": 500,
                        "step": 1,
                        "tooltip": (
                            "Below this many agreeing matches the image is just resized to the "
                            "reference size (warning in the log)."
                        ),
                    },
                ),
            },
        }

    def run(
        self,
        image: torch.Tensor,
        reference: torch.Tensor,
        method: str,
        model: str,
        max_side: int,
        ransac_threshold: float,
        min_inliers: int,
    ) -> tuple[torch.Tensor, torch.Tensor, str, int]:
        with torch.no_grad():
            aligned, valid_mask, matrices, inliers = auto_align(
                image,
                reference,
                method=method,
                model=model,
                max_side=int(max_side),
                ransac_threshold=float(ransac_threshold),
                min_inliers=int(min_inliers),
            )
        return aligned, valid_mask, json.dumps(matrices), int(inliers[0])
