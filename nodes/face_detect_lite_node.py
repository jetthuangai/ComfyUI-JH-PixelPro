"""ComfyUI wrapper for ``core.face_detect_lite.face_detect_lite`` (Kornia YuNet)."""

from __future__ import annotations

import json

import torch

from ..core.face_detect_lite import face_detect_lite

_MODE_CHOICES = ["single_largest", "multi_top_k"]


class JHPixelProFaceDetectLite:
    """Fast GPU face detector (YuNet); landmarks JSON is compatible with the Facial Aligner."""

    CATEGORY = "ComfyUI-JH-PixelPro/face"
    RETURN_TYPES = ("STRING", "STRING", "INT", "MASK")
    RETURN_NAMES = ("landmarks_json", "bbox_json", "face_count", "face_mask")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802 — ComfyUI node contract mandates UPPER_CASE.
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (
                    _MODE_CHOICES,
                    {
                        "default": "single_largest",
                        "tooltip": (
                            "single_largest returns the face with the largest box per image. "
                            "multi_top_k returns up to max_faces ranked by box area."
                        ),
                    },
                ),
                "max_faces": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Cap on faces per image. Ignored when mode = single_largest.",
                    },
                ),
                "confidence_threshold": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.1,
                        "max": 0.95,
                        "step": 0.05,
                        "tooltip": "Minimum YuNet score. Raise to drop false positives.",
                    },
                ),
                "nms_threshold": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.1,
                        "max": 0.9,
                        "step": 0.05,
                        "tooltip": "IoU above which overlapping detections are merged.",
                    },
                ),
                "detect_size": (
                    "INT",
                    {
                        "default": 640,
                        "min": 160,
                        "max": 1024,
                        "step": 32,
                        "tooltip": (
                            "Long side of the copy used for detection (never upscaled). 640 is "
                            "best for portraits; raise only for group shots with small faces "
                            "(may give false hits on close-ups)."
                        ),
                    },
                ),
            },
        }

    def run(
        self,
        image: torch.Tensor,
        mode: str,
        max_faces: int,
        confidence_threshold: float,
        nms_threshold: float,
        detect_size: int,
    ) -> tuple[str, str, int, torch.Tensor]:
        with torch.no_grad():
            landmarks, boxes, count, mask = face_detect_lite(
                image,
                mode=mode,
                max_faces=int(max_faces),
                confidence_threshold=float(confidence_threshold),
                nms_threshold=float(nms_threshold),
                detect_size=int(detect_size),
            )
        return json.dumps(landmarks), json.dumps(boxes), int(count), mask
