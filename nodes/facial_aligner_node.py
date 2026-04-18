"""ComfyUI wrapper for ``core.facial_aligner.facial_align``."""

from __future__ import annotations

import json

import torch

from ..core import facial_align

_DEFAULT_LANDMARKS_JSON = "[[820, 650], [1420, 640], [1120, 900], [930, 1150], [1310, 1145]]"


def _parse_landmarks(landmarks_str: str) -> list[list[float]] | list[list[list[float]]]:
    try:
        parsed = json.loads(landmarks_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"landmarks must be valid JSON: {exc.msg}.") from exc

    if not isinstance(parsed, list):
        raise ValueError("landmarks JSON must be a list.")
    return parsed


def _serialize_inverse_matrix(matrix_bhw: torch.Tensor) -> str:
    matrices = matrix_bhw.detach().cpu().tolist()
    return json.dumps(matrices)


class JHPixelProFacialAligner:
    """Align a face to canonical frame via 5 landmarks; emit inverse transform for unwarp."""

    CATEGORY = "ComfyUI-JH-PixelPro/geometry"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image_aligned", "inverse_matrix_json")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802 — ComfyUI node contract mandates UPPER_CASE.
        return {
            "required": {
                "image": ("IMAGE",),
                "landmarks": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": _DEFAULT_LANDMARKS_JSON,
                        "tooltip": (
                            "5-point landmark JSON in order "
                            "[L-eye, R-eye, nose, L-mouth, R-mouth]. "
                            "Pixel-absolute or normalized (auto-detected: "
                            "values ≤ 1.5 treated as normalized). "
                            "Shape 5x2 for single image or Bx5x2 for batch."
                        ),
                    },
                ),
                "target_size": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 512,
                        "max": 1024,
                        "step": 256,
                        "tooltip": (
                            "Square output size in pixels (512 / 768 / 1024). "
                            "Smaller is faster; 1024 is SDXL-friendly."
                        ),
                    },
                ),
                "padding": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.05,
                        "tooltip": (
                            "Ratio of canonical frame reserved around the "
                            "face (hair/chin room). 0 = tight crop, 0.5 = "
                            "half-frame padding."
                        ),
                    },
                ),
            },
        }

    def run(
        self,
        image: torch.Tensor,
        landmarks: str,
        target_size: int,
        padding: float,
    ) -> tuple[torch.Tensor, str]:
        parsed_landmarks = _parse_landmarks(landmarks)
        with torch.no_grad():
            img_bchw = image.permute(0, 3, 1, 2).contiguous()
            aligned_bchw, inverse_matrix = facial_align(
                img_bchw,
                parsed_landmarks,
                target_size=int(target_size),
                padding=float(padding),
            )
            aligned_bhwc = aligned_bchw.permute(0, 2, 3, 1).contiguous()
        inverse_json = _serialize_inverse_matrix(inverse_matrix)
        return (aligned_bhwc, inverse_json)
