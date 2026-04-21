"""ComfyUI wrapper for ``core.face_detect.face_detect`` (MediaPipe FaceLandmarker)."""

from __future__ import annotations

import json

import torch

from ..core import face_detect

_MODE_CHOICES = ["single_largest", "multi_top_k"]


class JHPixelProFaceDetect:
    """Detect 5-point face landmarks via MediaPipe; output JSON compatible with S-06."""

    CATEGORY = "ComfyUI-JH-PixelPro/face"
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("landmarks_json", "bbox_json", "face_count")
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
                            "single_largest (default, ~90% portrait use case) returns 1 face "
                            "with the largest bbox area. multi_top_k returns up to max_faces "
                            "ranked by bbox area."
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
                        "tooltip": (
                            "Cap on detected faces. Ignored when mode = single_largest. "
                            "Crowd scenes: bump to 5–10."
                        ),
                    },
                ),
                "confidence_threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 0.95,
                        "step": 0.05,
                        "tooltip": (
                            "MediaPipe min_face_detection_confidence gate. Default 0.5 is "
                            "balanced. WARNING: values > 0.85 may miss faces on typical "
                            "portraits (sample_portrait ceiling tested ~0.85). Raise only for "
                            "strict crowd-filtering scenarios."
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
    ) -> tuple[str, str, int]:
        with torch.no_grad():
            landmarks_list, boxes_list, face_count = face_detect(
                image,
                mode=mode,
                max_faces=int(max_faces),
                confidence_threshold=float(confidence_threshold),
            )
        # landmarks_json is downstream-compatible: paste landmarks_list[0] into S-06.
        # bbox_json[*].conf field = the threshold-gate metadata that admitted this detection,
        # NOT MediaPipe's per-face detector probability (tasks API does not expose per-face
        # score). Use mode=multi_top_k + bbox area for ranking, not conf.
        return (
            json.dumps(landmarks_list),
            json.dumps(boxes_list),
            int(face_count),
        )
