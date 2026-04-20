"""ComfyUI wrapper for 468-point face-landmark extraction + overlay preview."""

from __future__ import annotations

import numpy as np
import torch

from ..core.face_pipeline import extract_landmarks


def _draw_overlay(image: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
    overlay = image.detach().to("cpu", torch.float32).clone().numpy()
    batch, height, width, _ = overlay.shape
    points = landmarks.detach().to("cpu", torch.float32)

    for batch_index in range(batch):
        for face_points in points[batch_index]:
            finite = torch.isfinite(face_points).all(dim=-1)
            if not finite.any():
                continue
            valid = face_points[finite]
            if valid.numel() == 0:
                continue
            if valid.max().item() <= 1.5 and valid.min().item() >= -0.5:
                valid = valid * torch.tensor([max(width - 1, 1), max(height - 1, 1)])
            for x_value, y_value in valid.tolist():
                x_int = int(round(x_value))
                y_int = int(round(y_value))
                x0 = max(0, x_int - 1)
                y0 = max(0, y_int - 1)
                x1 = min(width, x_int + 2)
                y1 = min(height, y_int + 2)
                overlay[batch_index, y0:y1, x0:x1] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    return torch.from_numpy(overlay).to(device=image.device, dtype=image.dtype)


class JHPixelProFaceLandmarks:
    CATEGORY = "ComfyUI-JH-PixelPro/face"
    RETURN_TYPES = ("LANDMARKS", "IMAGE")
    RETURN_NAMES = ("landmarks", "overlay")
    FUNCTION = "extract"

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802
        return {
            "required": {
                "image": ("IMAGE",),
                "max_num_faces": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Max faces per image. Missing faces are padded with NaN.",
                    },
                ),
                "min_detection_confidence": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "MediaPipe min_face_detection_confidence threshold.",
                    },
                ),
                "refine_landmarks": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Kept for API compatibility; dense 468-point output is always used."
                        ),
                    },
                ),
                "draw_overlay": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Draw the detected landmarks as green dots on an overlay image.",
                    },
                ),
            },
        }

    def extract(
        self,
        image: torch.Tensor,
        max_num_faces: int,
        min_detection_confidence: float,
        refine_landmarks: bool,
        draw_overlay: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            landmarks, _visibility = extract_landmarks(
                image,
                max_num_faces=max_num_faces,
                min_detection_confidence=min_detection_confidence,
                refine_landmarks=refine_landmarks,
            )
        overlay = _draw_overlay(image, landmarks) if draw_overlay else image.clone()
        return (landmarks, overlay)
