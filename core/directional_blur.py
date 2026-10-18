"""Motion and zoom blur for ComfyUI IMAGE tensors.

``motion``: average along a straight line of ``distance`` pixels at ``angle`` (centred
on each pixel), as a camera or subject moving during the exposure. ``zoom``: average
along the ray toward a centre point, as zooming the lens during the exposure.

Both are a box filter along a 1-D path. A box of ``n1 * n2`` taps is the composition of
a box of ``n1`` taps at spacing 1 and one of ``n2`` taps at spacing ``n1``, so a blur
``D`` pixels long costs about ``2 * sqrt(D)`` bilinear warps (``grid_sample``) instead
of ``D``. Zoom works the same way in log-radius, where scaling compositions add. Each
warp holds one sampling grid (8 bytes per pixel), so memory stays O(H*W).
"""

from __future__ import annotations

import math
from numbers import Real

import torch
import torch.nn.functional as functional

MODES = ("motion", "zoom")


def _check(name: str, value: float, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not lower <= value <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return float(value)


def _split(length: float) -> tuple[int, int, float]:
    """Taps ``(n1, n2)`` and step so ``n1 * n2`` samples evenly span ``length`` pixels."""
    total = max(2, math.ceil(length) + 1)
    n2 = max(1, round(math.sqrt(total)))
    n1 = math.ceil(total / n2)
    return n1, n2, length / max(n1 * n2 - 1, 1)


def _base_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, height, device=device)
    xs = torch.linspace(-1.0, 1.0, width, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1,H,W,2), align_corners=True


def _box_pass(x: torch.Tensor, taps: int, offset_of: callable) -> torch.Tensor:
    """Mean of ``taps`` warped copies of ``x``; ``offset_of(k)`` gives the k-th grid."""
    if taps <= 1:
        return x
    total = torch.zeros_like(x)
    for k in range(taps):
        grid = offset_of(k - (taps - 1) / 2.0).expand(x.shape[0], -1, -1, -1)
        total += functional.grid_sample(
            x, grid, mode="bilinear", padding_mode="reflection", align_corners=True
        )
    return total / taps


def _motion(x: torch.Tensor, angle: float, distance: float) -> torch.Tensor:
    height, width = x.shape[-2:]
    base = _base_grid(height, width, x.device)
    n1, n2, step = _split(distance)
    theta = math.radians(angle)
    # One pixel along the direction, in normalised grid units (y down, angle counter-clockwise).
    unit = torch.tensor(
        [
            math.cos(theta) * 2.0 / max(width - 1, 1),
            -math.sin(theta) * 2.0 / max(height - 1, 1),
        ],
        device=x.device,
    )
    x = _box_pass(x, n1, lambda k: base + unit * (k * step))
    return _box_pass(x, n2, lambda k: base + unit * (k * step * n1))


def _zoom(x: torch.Tensor, amount: float, center: tuple[float, float]) -> torch.Tensor:
    height, width = x.shape[-2:]
    base = _base_grid(height, width, x.device)
    c = torch.tensor([2.0 * center[0] - 1.0, 2.0 * center[1] - 1.0], device=x.device)
    rel = base - c
    # Scale factors exp(u) with u spread evenly over [log(1 - amount), 0]: centred taps.
    span = -math.log(max(1.0 - amount, 1e-3))
    half = rel.new_tensor([(width - 1) / 2.0, (height - 1) / 2.0])
    longest = float((rel * half).norm(dim=-1).amax())  # farthest pixel from the centre, px
    n1, n2, step = _split(span * longest)
    du = step / max(longest, 1.0)
    shift = -span / 2.0  # centre the kernel so the mid tap is the average scale

    def grid(u: float) -> torch.Tensor:
        return c + rel * math.exp(u)

    x = _box_pass(x, n1, lambda k: grid(shift + k * du))
    return _box_pass(x, n2, lambda k: grid(k * du * n1))


def directional_blur(
    image: torch.Tensor,
    *,
    mode: str = "motion",
    angle: float = 0.0,
    distance: float = 20.0,
    zoom: float = 0.15,
    center_x: float = 0.5,
    center_y: float = 0.5,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Motion or zoom blur.

    Args:
        image: IMAGE ``(B,H,W,3)``.
        mode: ``motion`` or ``zoom``.
        angle: Motion direction in degrees (0 = horizontal, 90 = vertical).
        distance: Motion length in pixels, 0..1000.
        zoom: Zoom amount 0..0.9; the outer edge is averaged over this fraction of its
            distance to the centre.
        center_x, center_y: Zoom centre, 0..1 of the width/height.
        mask: Optional MASK ``(B,H,W)``; blur applies where it is 1.

    Returns:
        Blurred ``(B,H,W,3)`` float32 image (unchanged where ``mask`` is 0).
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"image must have shape (B,H,W,3), got {tuple(image.shape)}.")
    if not torch.is_floating_point(image):
        raise ValueError(f"image must be floating point, got {image.dtype}.")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}.")
    angle = _check("angle", angle, -360.0, 360.0)
    distance = _check("distance", distance, 0.0, 1000.0)
    zoom = _check("zoom", zoom, 0.0, 0.9)
    center = (_check("center_x", center_x, 0.0, 1.0), _check("center_y", center_y, 0.0, 1.0))
    if mask is not None:
        if not isinstance(mask, torch.Tensor) or mask.ndim != 3:
            raise ValueError("mask must have shape (B,H,W).")
        if mask.shape[1:] != image.shape[1:3] or mask.shape[0] not in (1, image.shape[0]):
            raise ValueError("mask must match the image size (batch 1 or the image batch).")

    source = image.float().clamp(0.0, 1.0)
    if (mode == "motion" and distance < 0.5) or (mode == "zoom" and zoom <= 0.0):
        return source
    x = source.permute(0, 3, 1, 2)
    out = _motion(x, angle, distance) if mode == "motion" else _zoom(x, zoom, center)
    out = out.permute(0, 2, 3, 1).clamp(0.0, 1.0)
    if mask is not None:
        weight = mask.to(device=out.device, dtype=out.dtype).clamp(0.0, 1.0).unsqueeze(-1)
        out = source + (out - source) * weight
    return out


def run(image: torch.Tensor, **kwargs: object) -> torch.Tensor:
    """``directional_blur`` on the GPU when there is one (ComfyUI images live on the CPU).

    Frames go to the GPU one at a time (about 0.5 GB per 4K frame), so a batch stays
    within the per-node VRAM budget.
    """
    if image.device.type == "cpu" and torch.cuda.is_available():
        mask = kwargs.get("mask")
        try:
            frames = []
            for index in range(image.shape[0]):
                frame_kwargs = dict(kwargs)
                if isinstance(mask, torch.Tensor):
                    pick = index if mask.shape[0] > 1 else 0
                    frame_kwargs["mask"] = mask[pick : pick + 1].cuda()
                frame = image[index : index + 1].cuda()
                frames.append(directional_blur(frame, **frame_kwargs).cpu())
            return torch.cat(frames, dim=0)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    return directional_blur(image, **kwargs)
