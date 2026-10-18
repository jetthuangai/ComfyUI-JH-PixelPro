"""Shadows/Highlights Pro: one entry point for the Camera Raw and Photoshop models.

Two sliders drive both modes (Lightroom convention):

- ``shadows`` -100..100: positive lifts shadows.
- ``highlights`` -100..100: negative recovers highlights.

``camera_raw`` uses both signs (local Laplacian on log luminance, see
``core.local_laplacian``). ``photoshop`` maps them to the Adjustments dialog's
amounts (Shadows = max(shadows, 0) %, Highlights = max(-highlights, 0) %); the
dialog cannot darken shadows or brighten highlights, so the other signs do nothing.
The dialog's Black/White Clip levels stretch is not applied.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .local_laplacian import shadows_highlights_camera_raw
from .shadows_highlights_ps import shadows_highlights_photoshop

MODES = ("camera_raw", "photoshop")
PS_OPTION_KEYS = (
    "shadow_tone",
    "shadow_radius",
    "highlight_tone",
    "highlight_radius",
    "color",
    "midtone",
)


def _validate_mask(mask: torch.Tensor | None, image: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor.")
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape (B,H,W), got {tuple(mask.shape)}.")
    if mask.shape[1:] != image.shape[1:3]:
        raise ValueError("mask and image spatial shapes must match.")
    if mask.shape[0] not in (1, image.shape[0]):
        raise ValueError("mask batch must match image batch or be 1.")
    return mask.to(dtype=torch.float32).clamp(0.0, 1.0)


def _run(
    image: torch.Tensor,
    mode: str,
    shadows: float,
    highlights: float,
    ps_options: Mapping[str, float],
) -> torch.Tensor:
    if mode == "camera_raw":
        return shadows_highlights_camera_raw(image, shadows, highlights)
    # No Black/White Clip: its levels stretch pushed lifted shadows back to black
    # (-1.2 stops on a backlit portrait at Shadows 100) and is unverified against Photoshop.
    return shadows_highlights_photoshop(
        image,
        shadow_amount=max(shadows, 0.0),
        highlight_amount=max(-highlights, 0.0),
        black_clip=0.0,
        white_clip=0.0,
        **ps_options,
    )


def shadows_highlights(
    image: torch.Tensor,
    *,
    mode: str = "camera_raw",
    shadows: float = 30.0,
    highlights: float = -30.0,
    mask: torch.Tensor | None = None,
    ps_options: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Apply Shadows/Highlights to a ComfyUI IMAGE ``(B,H,W,3)``.

    Args:
        image: sRGB float tensor in ``[0,1]``.
        mode: ``camera_raw`` or ``photoshop``.
        shadows: -100..100 (positive lifts shadows).
        highlights: -100..100 (negative recovers highlights).
        mask: Optional ``(B,H,W)`` mask (batch 1 broadcasts) limiting the effect.
        ps_options: Photoshop-only fine tuning, keys from ``PS_OPTION_KEYS``.

    Returns:
        ``(B,H,W,3)`` float32 tensor on the input device. Both sliders at 0 return
        the input unchanged. Runs on CUDA when the input is on CPU and a GPU is
        available, falling back to CPU on out-of-memory.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"image must have shape (B,H,W,3), got {tuple(image.shape)}.")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}.")
    for name, value in (("shadows", shadows), ("highlights", highlights)):
        if not -100.0 <= float(value) <= 100.0:
            raise ValueError(f"{name} must be in [-100, 100], got {value}.")
    options = dict(ps_options or {})
    unknown = set(options) - set(PS_OPTION_KEYS)
    if unknown:
        raise ValueError(f"unknown ps_options: {sorted(unknown)}")
    prepared_mask = _validate_mask(mask, image)
    if float(shadows) == 0.0 and float(highlights) == 0.0:
        return image
    if mode == "photoshop" and float(shadows) <= 0.0 and float(highlights) >= 0.0:
        return image  # both dialog amounts are 0 %: nothing to do (skip the clip stretch)

    out_device = image.device
    # ComfyUI hands IMAGE tensors over on the CPU; do the heavy work on the GPU when there is one.
    work_device = (
        torch.device("cuda")
        if out_device.type == "cpu" and torch.cuda.is_available()
        else out_device
    )
    prepared = image.to(dtype=torch.float32).clamp(0.0, 1.0)
    args = (mode, float(shadows), float(highlights), options)
    try:
        result = _run(prepared.to(work_device), *args).to(out_device)
    except torch.cuda.OutOfMemoryError:
        if work_device == out_device:
            raise
        torch.cuda.empty_cache()
        result = _run(prepared, *args)

    if prepared_mask is not None:
        weight = prepared_mask.to(out_device).unsqueeze(-1)
        result = prepared + (result - prepared) * weight
    return result.clamp(0.0, 1.0)
