"""ComfyUI wrapper for ``core.unwrap_face.unwrap_face``."""

from __future__ import annotations

import json

import torch

from ..core import unwrap_face

_DEFAULT_INVERSE_MATRIX_JSON = "[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]"


def _parse_inverse_matrix(matrix_str: str) -> torch.Tensor:
    try:
        parsed = json.loads(matrix_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"inverse_matrix_json must be valid JSON: {exc.msg}.") from exc

    if not isinstance(parsed, list) or not parsed:
        raise ValueError("inverse_matrix_json must be a non-empty list of matrices.")

    try:
        tensor = torch.tensor(parsed, dtype=torch.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"inverse_matrix_json could not be coerced to a tensor: {exc}.") from exc

    if tensor.ndim != 3 or tensor.shape[-2:] not in {(2, 3), (3, 3)}:
        raise ValueError(
            f"inverse_matrix_json must shape Bx3x3 or Bx2x3, got {tuple(tensor.shape)}."
        )
    return tensor


class JHPixelProUnwrapFace:
    """Warp an edited aligned crop onto the original canvas via inverse affine + feather."""

    CATEGORY = "ComfyUI-JH-PixelPro/face"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image_composited", "mask_used")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802 — ComfyUI node contract mandates UPPER_CASE.
        return {
            "required": {
                "edited_aligned": ("IMAGE",),
                "original_image": ("IMAGE",),
                "inverse_matrix_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": _DEFAULT_INVERSE_MATRIX_JSON,
                        "tooltip": (
                            "Output of S-06 Facial Aligner — Bx3x3 (or Bx2x3) affine per "
                            "batch, JSON serialized. Default identity is a safe pass-through "
                            "when nothing is wired."
                        ),
                    },
                ),
                "feather_radius": (
                    "FLOAT",
                    {
                        "default": 16.0,
                        "min": 0.0,
                        "max": 128.0,
                        "step": 1.0,
                        "tooltip": (
                            "Gaussian edge blur (pixels) applied to the auto-generated mask. "
                            "Higher = smoother blend at cost of softer face edge; 0 = hard "
                            "edge. Ignored when mask_override is wired."
                        ),
                    },
                ),
            },
            "optional": {
                "mask_override": ("MASK",),
            },
        }

    def run(
        self,
        edited_aligned: torch.Tensor,
        original_image: torch.Tensor,
        inverse_matrix_json: str,
        feather_radius: float,
        mask_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inverse_matrix = _parse_inverse_matrix(inverse_matrix_json)
        with torch.no_grad():
            edited_bchw = edited_aligned.permute(0, 3, 1, 2).contiguous()
            original_bchw = original_image.permute(0, 3, 1, 2).contiguous()
            composited_bchw, mask_used_bhw = unwrap_face(
                edited_bchw,
                original_bchw,
                inverse_matrix,
                feather_radius=float(feather_radius),
                mask_override=mask_override,
            )
            composited = composited_bchw.permute(0, 2, 3, 1).contiguous()
        return (composited, mask_used_bhw)
