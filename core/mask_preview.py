"""Mask preview: visualise a MASK on top of an image (overlay, heatmap, mask-only, cutout)."""

from __future__ import annotations

import torch
from kornia.color import ColorMap, ColorMapType, apply_colormap
from torch.nn import functional as nnf

MODES: tuple[str, ...] = ("overlay", "heatmap", "mask_only", "cutout")

COLORS: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "white": (1.0, 1.0, 1.0),
    "black": (0.0, 0.0, 0.0),
}

# Built from kornia so the list never drifts from the installed version.
COLORMAPS: tuple[str, ...] = tuple(sorted(m.name for m in ColorMapType))

_CHECKER_SIZE = 8
_CHECKER_LO = 0.4
_CHECKER_HI = 0.6
_GREY = 0.5


def _checkerboard(h: int, w: int, device: torch.device) -> torch.Tensor:
    ys = torch.arange(h, device=device) // _CHECKER_SIZE
    xs = torch.arange(w, device=device) // _CHECKER_SIZE
    parity = (ys[:, None] + xs[None, :]) % 2
    board = torch.where(
        parity == 0,
        torch.tensor(_CHECKER_LO, device=device),
        torch.tensor(_CHECKER_HI, device=device),
    )
    return board[None, :, :, None].expand(1, h, w, 3)


def _edge_band(mask: torch.Tensor) -> torch.Tensor:
    """Boundary of the 0.5 contour as a (B,H,W,1) {0,1} map (dilation minus erosion)."""
    binary = (mask > 0.5).to(torch.float32).unsqueeze(1)
    dil = nnf.max_pool2d(binary, 3, stride=1, padding=1)
    ero = -nnf.max_pool2d(-binary, 3, stride=1, padding=1)
    return (dil - ero).permute(0, 2, 3, 1)


def mask_preview(
    image: torch.Tensor | None,
    mask: torch.Tensor,
    *,
    mode: str = "overlay",
    color: str = "red",
    opacity: float = 0.5,
    colormap: str = "turbo",
    show_edges: bool = False,
) -> torch.Tensor:
    """Render a mask on top of an image for visual debugging.

    Args:
        image: float (B,H,W,3) or (B,H,W,4) in [0,1], or None. None uses a neutral grey
            background (overlay/cutout) or the bare colormap (heatmap).
        mask: float (B,H,W) in [0,1].
        mode: "overlay" lerp(image, color, opacity*mask); "heatmap" lerp(image, colormap(mask),
            opacity); "mask_only" greyscale mask as RGB; "cutout" image*mask over a checkerboard.
        color: overlay / edge colour, one of COLORS.
        opacity: float in [0,1].
        colormap: kornia ColorMapType name used by "heatmap".
        show_edges: draw the 0.5 contour of the mask in `color` at full opacity.

    Returns:
        float32 (B,H,W,3) in [0,1] on the mask device. Batch 1 broadcasts against batch B.

    Raises:
        TypeError: image/mask are not tensors (or numeric args wrong type).
        ValueError: bad shapes, range, mode, colour or colormap names, mismatched sizes.
    """
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"mask must be a torch.Tensor, got {type(mask).__name__}")
    if image is not None and not isinstance(image, torch.Tensor):
        raise TypeError(f"image must be a torch.Tensor or None, got {type(image).__name__}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if color not in COLORS:
        raise ValueError(f"color must be one of {tuple(COLORS)}, got {color!r}")
    if colormap not in COLORMAPS:
        raise ValueError(f"colormap must be one of {COLORMAPS}, got {colormap!r}")
    if isinstance(opacity, bool) or not isinstance(opacity, (int, float)):
        raise TypeError(f"opacity must be a float, got {type(opacity).__name__}")
    if not 0.0 <= float(opacity) <= 1.0:
        raise ValueError(f"opacity must be in [0, 1], got {opacity}")
    if not isinstance(show_edges, bool):
        raise TypeError(f"show_edges must be bool, got {type(show_edges).__name__}")
    if mask.ndim != 3:
        raise ValueError(f"mask must be (B,H,W), got shape {tuple(mask.shape)}")
    if not mask.is_floating_point() and mask.dtype is not torch.bool:
        raise ValueError(f"mask must be a float or bool tensor, got {mask.dtype}")
    if mask.numel() == 0:
        raise ValueError("mask must not be empty")

    device = mask.device
    m = mask.to(torch.float32).clamp(0.0, 1.0)
    b, h, w = m.shape

    img = None
    if image is not None:
        if image.ndim != 4 or image.shape[-1] not in (3, 4):
            raise ValueError(f"image must be (B,H,W,3|4), got shape {tuple(image.shape)}")
        if not image.is_floating_point():
            raise ValueError(f"image must be a floating tensor, got {image.dtype}")
        if tuple(image.shape[1:3]) != (h, w):
            raise ValueError(
                f"image spatial size {tuple(image.shape[1:3])} must match mask {(h, w)}"
            )
        if image.shape[0] != b and image.shape[0] != 1 and b != 1:
            raise ValueError(f"batch mismatch: image {image.shape[0]} vs mask {b}")
        img = image[..., :3].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)

    out_b = max(b, img.shape[0]) if img is not None else b
    m4 = m.unsqueeze(-1)  # (B,H,W,1)
    rgb = torch.tensor(COLORS[color], device=device, dtype=torch.float32)
    op = float(opacity)

    if mode == "overlay":
        base = img if img is not None else torch.full((1, h, w, 3), _GREY, device=device)
        result = base + (rgb - base) * (op * m4)
    elif mode == "heatmap":
        cmap = ColorMap(base=ColorMapType[colormap], num_colors=256, device=device)
        heat = apply_colormap(m.unsqueeze(1).contiguous(), cmap).permute(0, 2, 3, 1)
        heat = heat.to(torch.float32).clamp(0.0, 1.0)
        result = heat if img is None else img + (heat - img) * op
    elif mode == "mask_only":
        result = m4.expand(b, h, w, 3)
    else:  # cutout
        board = _checkerboard(h, w, device)
        fg = img if img is not None else torch.full((1, h, w, 3), _GREY, device=device)
        result = fg * m4 + board * (1.0 - m4)

    if show_edges:
        band = _edge_band(m)
        result = result + (rgb - result) * band

    return result.expand(out_b, h, w, 3).contiguous().to(torch.float32)
