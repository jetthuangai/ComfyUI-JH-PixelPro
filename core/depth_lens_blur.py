"""Depth-driven photographic lens blur (bokeh) for ComfyUI IMAGE tensors.

Circle of confusion (CoC) per pixel::

    coc = clamp((|depth - focus| - dof / 2) / (1 - dof / 2), 0, 1)      # 0 = sharp, 1 = max blur

``K`` blur layers with increasing radius are rendered and blended per pixel by linear
interpolation between the two nearest radii. Each layer is a normalised convolution
(``blur(image * w) / blur(w)``) whose source weight ``w`` falls off for pixels that are sharper
than the layer, so a sharp foreground does not smear into a blurred background.

Memory is bounded: layers with a radius above ``_MAX_KERNEL_RADIUS`` pixels are blurred on a
downscaled copy and upsampled, so the kernel never grows with ``max_blur`` (no unfold).

Limitations: depth edges are only as good as the depth map (soft depth edges give soft CoC
edges); sharp pixels in front of a blurred background are protected, but a blurred *foreground*
does not spread over an in-focus background (no occlusion-aware foreground bleed).
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import kornia.filters as kf
import torch
import torch.nn.functional as tnf

from . import depth_anything

BOKEH_SHAPES = ("disc", "gaussian")
_MAX_KERNEL_RADIUS = 8.0
_EPS = 1e-3


def _validate_float(name: str, value: float, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    value_float = float(value)
    if not lower <= value_float <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_float


def _validate_int(name: str, value: int, *, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}].")
    value_int = int(value)
    if not lower <= value_int <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return value_int


def _validate_image(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"image must have shape (B,H,W,3), got {tuple(image.shape)}.")
    if not torch.is_floating_point(image):
        raise ValueError(f"image must be a floating point tensor, got {image.dtype}.")
    return torch.nan_to_num(image.detach().to(dtype=torch.float32)).clamp(0.0, 1.0)


def _prepare_map(name: str, value: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Validate a MASK ``(B,H,W)``, broadcast batch 1, resize to the image size if needed."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape (B,H,W), got {tuple(value.shape)}.")
    if not torch.is_floating_point(value) and value.dtype is not torch.bool:
        raise ValueError(f"{name} must be float or bool, got {value.dtype}.")
    batch, height, width = image.shape[0], image.shape[1], image.shape[2]
    if value.shape[0] not in (1, batch):
        raise ValueError(f"{name} batch must match image batch or be 1.")
    out = value.detach().to(device=image.device, dtype=torch.float32)
    # One NaN would otherwise spread through the median / blur to the whole output.
    out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if out.shape[1:] != (height, width):
        out = tnf.interpolate(out.unsqueeze(1), size=(height, width), mode="bilinear")[:, 0]
    return out.expand(batch, -1, -1)


def _disc_kernel(radius: float, device: torch.device) -> torch.Tensor:
    half = max(1, math.ceil(radius))
    coords = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
    dist = (coords.view(-1, 1) ** 2 + coords.view(1, -1) ** 2).sqrt()
    kernel = (radius + 0.5 - dist).clamp(0.0, 1.0)
    return (kernel / kernel.sum()).unsqueeze(0)


def _blur(tensor: torch.Tensor, radius: float, shape: str) -> torch.Tensor:
    """Blur ``(B,C,h,w)`` with a disc of ``radius`` px or a Gaussian of equal visual extent."""
    if shape == "disc":
        kernel = _disc_kernel(radius, tensor.device)
        half = kernel.shape[-1] // 2
        # Reflect padding needs pad < side; on tiny images fall back to edge replication.
        border = "reflect" if half < min(tensor.shape[-2:]) else "replicate"
        return kf.filter2d(tensor, kernel, border_type=border, normalized=False)
    return _gaussian(tensor, max(radius / 2.0, 0.3))


def _gaussian(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    size = 2 * math.ceil(3.0 * sigma) + 1
    size = min(size, 2 * (min(tensor.shape[-2:]) - 1) + 1) | 1
    return kf.gaussian_blur2d(tensor, (size, size), (sigma, sigma), border_type="reflect")


def _render_layer(
    image: torch.Tensor, radius_map: torch.Tensor, radius: float, shape: str
) -> torch.Tensor:
    """Normalised-convolution blur of ``(B,3,H,W)`` at ``radius`` px, weighted by source CoC."""
    height, width = image.shape[-2:]
    weight = (radius_map / radius).clamp(0.0, 1.0) ** 2
    packed = torch.cat([image * weight, weight], dim=1)
    work_radius = radius
    if radius > _MAX_KERNEL_RADIUS:
        scale = _MAX_KERNEL_RADIUS / radius
        small = (max(1, round(height * scale)), max(1, round(width * scale)))
        packed = tnf.interpolate(packed, size=small, mode="area")
        work_radius = radius * scale
    packed = _blur(packed, max(work_radius, 0.3), shape)
    if packed.shape[-2:] != (height, width):
        packed = tnf.interpolate(packed, size=(height, width), mode="bilinear", align_corners=False)
    return (packed[:, :3] + _EPS * image) / (packed[:, 3:] + _EPS)


def circle_of_confusion(
    depth: torch.Tensor,
    focus: torch.Tensor,
    depth_of_field: float,
    *,
    blur_foreground: bool = True,
) -> torch.Tensor:
    """Normalised CoC ``(B,H,W)`` in ``[0,1]``; ``focus`` is ``(B,1,1)``."""
    half = depth_of_field / 2.0
    coc = ((depth - focus).abs() - half) / (1.0 - half)
    if not blur_foreground:
        coc = torch.where(depth > focus, torch.zeros_like(coc), coc)
    return coc.clamp(0.0, 1.0)


def _smooth_coc(coc: torch.Tensor, max_blur: int) -> torch.Tensor:
    """Close narrow sharp gaps in the CoC map (grey closing, half-width ~ max_blur / 8 px).

    Where the depth crosses the focus plane between two blurred zones the CoC dips to 0 for a
    few pixels and leaves a thin sharp seam. A closing (dilate then erode with a square window,
    ``max_pool2d`` so memory does not grow with the window) fills such dips but leaves wide
    in-focus regions and their edges untouched, so the subject stays sharp.
    """
    half = max(1, math.ceil(max_blur / 8.0))
    size = 2 * half + 1
    x = coc.unsqueeze(1)
    dilated = tnf.max_pool2d(x, size, stride=1, padding=half)
    closed = -tnf.max_pool2d(-dilated, size, stride=1, padding=half)
    return closed[:, 0].clamp(0.0, 1.0)


def resolve_focus(
    depth: torch.Tensor, focus_depth: float, focus_mask: torch.Tensor | None
) -> torch.Tensor:
    """Per-image focus value ``(B,1,1)``: median depth inside ``focus_mask`` (>0.5) if given."""
    batch = depth.shape[0]
    focus = torch.full((batch,), focus_depth, dtype=torch.float32, device=depth.device)
    if focus_mask is not None:
        for i in range(batch):
            inside = depth[i][focus_mask[i] > 0.5]
            if inside.numel() > 0:
                focus[i] = inside.median()
    return focus.view(batch, 1, 1)


def depth_lens_blur(
    image: torch.Tensor,
    depth: torch.Tensor | None = None,
    *,
    focus_depth: float = 0.5,
    depth_of_field: float = 0.1,
    max_blur: int = 24,
    bokeh: str = "disc",
    layers: int = 8,
    focus_mask: torch.Tensor | None = None,
    blur_foreground: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a depth-driven lens blur.

    Args:
        image: IMAGE ``(B,H,W,3)`` float in ``[0,1]``.
        depth: MASK ``(B,H,W)``, ``1`` = nearest. ``None`` estimates it with Depth Anything V2
            small at 518 px. A depth map of another size is resized to the image.
        focus_depth: Depth value that stays sharp, ``[0,1]``. Replaced by the median depth inside
            ``focus_mask`` when that is given (and non-empty).
        depth_of_field: Width of the sharp zone around the focus, ``[0,1]``.
        max_blur: Blur radius in pixels at the farthest distance from focus, ``[1,96]``.
        bokeh: ``"disc"`` (flat, photographic) or ``"gaussian"`` (soft).
        layers: Number of blur layers (radius steps), ``[2,16]``.
        focus_mask: Optional MASK ``(B,H,W)``; pixels > 0.5 select the in-focus subject.
        blur_foreground: ``False`` blurs only what is behind the focus plane.

    Returns:
        ``(image_blurred (B,H,W,3), coc (B,H,W))`` float32; ``coc`` is the blur amount in
        ``[0,1]`` (multiply by ``max_blur`` for pixels).

    Raises:
        TypeError: If inputs are not tensors.
        ValueError: On invalid shapes, dtypes or parameters.
        RuntimeError: If ``depth`` is None and onnxruntime / the model is unavailable.

    Notes:
        The work runs on CUDA when available (CPU inputs are moved there and the result is
        returned on the input device). NaN / inf in ``depth`` and ``focus_mask`` are replaced by
        finite values. Narrow sharp gaps in the CoC map (thin seams where depth crosses the
        focus plane between two blurred zones) are closed, window ~ ``max_blur / 8`` px.
    """
    prepared = _validate_image(image)
    focus_depth = _validate_float("focus_depth", focus_depth, lower=0.0, upper=1.0)
    depth_of_field = _validate_float("depth_of_field", depth_of_field, lower=0.0, upper=1.0)
    max_blur = _validate_int("max_blur", max_blur, lower=1, upper=96)
    layers = _validate_int("layers", layers, lower=2, upper=16)
    if bokeh not in BOKEH_SHAPES:
        raise ValueError(f"bokeh must be one of {BOKEH_SHAPES}, got {bokeh!r}.")
    if not isinstance(blur_foreground, bool):
        raise ValueError("blur_foreground must be a bool.")

    out_device = prepared.device
    # ComfyUI hands IMAGE tensors over on the CPU; do the heavy work on the GPU when there is one.
    work_device = (
        torch.device("cuda")
        if out_device.type == "cpu" and torch.cuda.is_available()
        else out_device
    )
    args = (depth, focus_depth, depth_of_field, max_blur, bokeh, layers, focus_mask)
    try:
        result, coc = _run(prepared.to(work_device), *args, blur_foreground)
    except torch.cuda.OutOfMemoryError:
        if work_device == out_device:
            raise
        torch.cuda.empty_cache()
        result, coc = _run(prepared, *args, blur_foreground)
    return result.to(out_device), coc.to(out_device)


def _run(
    prepared: torch.Tensor,
    depth: torch.Tensor | None,
    focus_depth: float,
    depth_of_field: float,
    max_blur: int,
    bokeh: str,
    layers: int,
    focus_mask: torch.Tensor | None,
    blur_foreground: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if depth is None:
        depth_map = depth_anything.estimate_depth(prepared, "small", 518)
        depth_map = _prepare_map("depth", depth_map, prepared)
    else:
        depth_map = _prepare_map("depth", depth, prepared)
    mask_map = None if focus_mask is None else _prepare_map("focus_mask", focus_mask, prepared)

    focus = resolve_focus(depth_map, focus_depth, mask_map)
    coc = circle_of_confusion(depth_map, focus, depth_of_field, blur_foreground=blur_foreground)
    smoothed = _smooth_coc(coc, max_blur)
    # Keep pixels the user marked as subject exactly as sharp as their depth says.
    coc = smoothed if mask_map is None else torch.where(mask_map > 0.5, coc, smoothed)

    radius_map = (coc * max_blur).unsqueeze(1)
    nchw = prepared.permute(0, 3, 1, 2).contiguous()
    position = coc * (layers - 1)  # fractional layer index
    output = torch.zeros_like(nchw)
    for k in range(layers):
        tent = (1.0 - (position - k).abs()).clamp(min=0.0).unsqueeze(1)
        if k == 0:
            output += tent * nchw
            continue
        if not bool((tent > 0).any()):
            continue
        radius = max_blur * k / (layers - 1)
        output += tent * _render_layer(nchw, radius_map, radius, bokeh)
    result = output.permute(0, 2, 3, 1).clamp(0.0, 1.0).contiguous()
    return result, coc
