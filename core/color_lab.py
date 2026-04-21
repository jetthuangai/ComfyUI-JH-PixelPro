"""ACR-style ColorLab pipeline implemented with pure PyTorch."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F

HUE_ANCHORS = {
    "red": 0.0,
    "orange": 30.0,
    "yellow": 60.0,
    "green": 120.0,
    "aqua": 180.0,
    "blue": 240.0,
    "purple": 280.0,
    "magenta": 310.0,
}

BASIC_KEYS = (
    "basic_exposure",
    "basic_contrast",
    "basic_highlights",
    "basic_shadows",
    "basic_whites",
    "basic_blacks",
    "basic_texture",
    "basic_clarity",
    "basic_dehaze",
    "basic_vibrance",
    "basic_saturation",
)

GRADE_REGIONS = ("shadow", "mid", "highlight")


def apply_colorlab_pipeline(image: torch.Tensor, params: Mapping[str, object]) -> torch.Tensor:
    """Apply Basic -> HSL -> Color Grading -> Gray Mix to a ComfyUI IMAGE tensor."""

    _validate_image(image)
    if _is_identity(params):
        return image
    out = image.clamp(0.0, 1.0)
    out = _apply_basic(out, params)
    out = _apply_hsl(out, params)
    out = _apply_color_grading(out, params)
    if bool(params.get("gray_enable", False)):
        out = _apply_gray_mix(out, params)
    return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _validate_image(image: torch.Tensor) -> None:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"image must have shape (B,H,W,3), got {tuple(image.shape)}.")


def _is_identity(params: Mapping[str, object]) -> bool:
    if bool(params.get("gray_enable", False)):
        return False
    keys = list(BASIC_KEYS)
    for color in HUE_ANCHORS:
        keys.extend((f"hsl_{color}_hue", f"hsl_{color}_sat", f"hsl_{color}_lum"))
    for region in GRADE_REGIONS:
        keys.extend(
            (
                f"grade_{region}_hue",
                f"grade_{region}_sat",
                f"grade_{region}_lum",
                f"grade_{region}_bal",
            )
        )
    return all(abs(_float(params, key)) <= 1e-12 for key in keys)


def _float(params: Mapping[str, object], key: str, default: float = 0.0) -> float:
    return float(params.get(key, default))


def _luma(image: torch.Tensor) -> torch.Tensor:
    weights = image.new_tensor([0.299, 0.587, 0.114])
    return torch.sum(image * weights, dim=-1, keepdim=True)


def _smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
    if edge0 == edge1:
        return (x >= edge1).to(x.dtype)
    t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _apply_basic(image: torch.Tensor, params: Mapping[str, object]) -> torch.Tensor:
    out = image
    exposure = _float(params, "basic_exposure")
    if exposure:
        out = out * (2.0**exposure)

    contrast = _float(params, "basic_contrast") / 100.0
    if contrast:
        factor = 1.0 + contrast
        out = (out - 0.5) * factor + 0.5

    lum = _luma(out)
    out = _tone_region(out, lum, "basic_highlights", params, _smoothstep(0.5, 1.0, lum), True)
    out = _tone_region(out, lum, "basic_shadows", params, 1.0 - _smoothstep(0.0, 0.5, lum), True)
    out = _tone_region(out, lum, "basic_whites", params, _smoothstep(0.75, 1.0, lum), True)
    out = _tone_region(out, lum, "basic_blacks", params, 1.0 - _smoothstep(0.0, 0.25, lum), True)

    texture = _float(params, "basic_texture") / 100.0
    if texture:
        out = _local_contrast(out, texture * 0.25, kernel_size=5)

    clarity = _float(params, "basic_clarity") / 100.0
    if clarity:
        mid_mask = (1.0 - (lum - 0.5).abs() * 2.0).clamp(0.0, 1.0)
        out = out + (_local_contrast(out, clarity * 0.35, kernel_size=9) - out) * mid_mask

    dehaze = _float(params, "basic_dehaze") / 100.0
    if dehaze:
        out = (out - 0.5) * (1.0 + dehaze * 0.35) + 0.5 - dehaze * 0.03

    vibrance = _float(params, "basic_vibrance") / 100.0
    saturation = _float(params, "basic_saturation") / 100.0
    if vibrance or saturation:
        h, s, v = rgb_to_hsv(out.clamp(0.0, 1.0))
        if vibrance:
            skin = _hue_mask(h, 24.0, sigma=28.0) * s
            s = s * (1.0 + vibrance * (1.0 - s) * (1.0 - skin * 0.95))
        if saturation:
            s = s * (1.0 + saturation)
        out = hsv_to_rgb(h, s.clamp(0.0, 1.0), v)

    return out.clamp(0.0, 1.0)


def _tone_region(
    image: torch.Tensor,
    lum: torch.Tensor,
    key: str,
    params: Mapping[str, object],
    mask: torch.Tensor,
    protect_range: bool,
) -> torch.Tensor:
    amount = _float(params, key) / 100.0
    if not amount:
        return image
    if protect_range:
        delta = amount * (1.0 - image if amount > 0 else image) * 0.45
    else:
        delta = amount * lum * 0.45
    return image + delta * mask


def _local_contrast(image: torch.Tensor, amount: float, kernel_size: int) -> torch.Tensor:
    if not amount:
        return image
    x = image.permute(0, 3, 1, 2)
    blur = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return (x + (x - blur) * amount).permute(0, 2, 3, 1)


def _apply_hsl(image: torch.Tensor, params: Mapping[str, object]) -> torch.Tensor:
    h, s, v = rgb_to_hsv(image)
    hue_shift = torch.zeros_like(h)
    sat_delta = torch.zeros_like(s)
    lum_delta = torch.zeros_like(v)

    for color, center in HUE_ANCHORS.items():
        mask = _hue_mask(h, center)
        hue_shift = hue_shift + mask * (_float(params, f"hsl_{color}_hue") * 0.3)
        sat_delta = sat_delta + mask * (_float(params, f"hsl_{color}_sat") / 100.0)
        lum_delta = lum_delta + mask * (_float(params, f"hsl_{color}_lum") / 100.0 * 0.25)

    h = (h + hue_shift) % 360.0
    s = (s * (1.0 + sat_delta)).clamp(0.0, 1.0)
    v = (v + lum_delta).clamp(0.0, 1.0)
    return hsv_to_rgb(h, s, v)


def _apply_color_grading(image: torch.Tensor, params: Mapping[str, object]) -> torch.Tensor:
    out = image
    lum = _luma(image)
    for region in GRADE_REGIONS:
        sat = _float(params, f"grade_{region}_sat") / 100.0
        lum_shift = _float(params, f"grade_{region}_lum") / 100.0 * 0.25
        if not sat and not lum_shift:
            continue
        bal = _float(params, f"grade_{region}_bal") / 100.0 * 0.5
        region_lum = (lum + bal).clamp(0.0, 1.0)
        if region == "shadow":
            mask = 1.0 - _smoothstep(0.0, 0.55, region_lum)
        elif region == "highlight":
            mask = _smoothstep(0.45, 1.0, region_lum)
        else:
            mask = (1.0 - ((region_lum - 0.5).abs() / 0.35)).clamp(0.0, 1.0)

        hue = image.new_full(lum.shape, _float(params, f"grade_{region}_hue") % 360.0)
        tint = hsv_to_rgb(hue, image.new_full(lum.shape, sat), torch.ones_like(lum))
        graded = (out + (tint - 0.5) * sat * mask + lum_shift * mask).clamp(0.0, 1.0)
        out = torch.lerp(out, graded, mask.clamp(0.0, 1.0))
    return out


def _apply_gray_mix(image: torch.Tensor, params: Mapping[str, object]) -> torch.Tensor:
    h, s, _ = rgb_to_hsv(image)
    base = _luma(image)
    total = torch.zeros_like(base)
    weighted = torch.zeros_like(base)
    for color, center in HUE_ANCHORS.items():
        mask = _hue_mask(h, center)
        weight = 1.0 + _float(params, f"gray_{color}") / 100.0
        weighted = weighted + mask * weight
        total = total + mask
    chroma_factor = torch.where(total > 1e-6, weighted / total, torch.ones_like(total))
    gray = (base * (0.75 + 0.25 * chroma_factor) + s * 0.05 * (chroma_factor - 1.0)).clamp(0.0, 1.0)
    return gray.expand_as(image)


def _hue_distance(hue: torch.Tensor, center: float) -> torch.Tensor:
    return torch.remainder(hue - center + 180.0, 360.0) - 180.0


def _hue_mask(hue: torch.Tensor, center: float, sigma: float = 20.0) -> torch.Tensor:
    dist = _hue_distance(hue, center)
    return torch.exp(-0.5 * (dist / sigma) ** 2)


def rgb_to_hsv(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert RGB BHWC tensor in [0,1] to HSV with hue in degrees."""

    r, g, b = rgb.unbind(dim=-1)
    maxc = rgb.max(dim=-1).values
    minc = rgb.min(dim=-1).values
    delta = maxc - minc

    hue = torch.zeros_like(maxc)
    safe_delta = delta.clamp_min(1e-8)
    hue = torch.where(maxc == r, 60.0 * torch.remainder((g - b) / safe_delta, 6.0), hue)
    hue = torch.where(maxc == g, 60.0 * ((b - r) / safe_delta + 2.0), hue)
    hue = torch.where(maxc == b, 60.0 * ((r - g) / safe_delta + 4.0), hue)
    hue = torch.where(delta <= 1e-8, torch.zeros_like(hue), hue % 360.0)
    sat = torch.where(maxc <= 1e-8, torch.zeros_like(maxc), delta / maxc.clamp_min(1e-8))
    return hue.unsqueeze(-1), sat.unsqueeze(-1), maxc.unsqueeze(-1)


def hsv_to_rgb(hue: torch.Tensor, sat: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    """Convert HSV tensors with hue in degrees to RGB BHWC."""

    h = torch.remainder(hue / 60.0, 6.0)
    c = val * sat
    x = c * (1.0 - torch.abs(torch.remainder(h, 2.0) - 1.0))
    z = torch.zeros_like(c)

    r1 = torch.where((0.0 <= h) & (h < 1.0), c, z)
    g1 = torch.where((0.0 <= h) & (h < 1.0), x, z)
    b1 = z.clone()

    r1 = torch.where((1.0 <= h) & (h < 2.0), x, r1)
    g1 = torch.where((1.0 <= h) & (h < 2.0), c, g1)
    b1 = torch.where((1.0 <= h) & (h < 2.0), z, b1)

    r1 = torch.where((2.0 <= h) & (h < 3.0), z, r1)
    g1 = torch.where((2.0 <= h) & (h < 3.0), c, g1)
    b1 = torch.where((2.0 <= h) & (h < 3.0), x, b1)

    r1 = torch.where((3.0 <= h) & (h < 4.0), z, r1)
    g1 = torch.where((3.0 <= h) & (h < 4.0), x, g1)
    b1 = torch.where((3.0 <= h) & (h < 4.0), c, b1)

    r1 = torch.where((4.0 <= h) & (h < 5.0), x, r1)
    g1 = torch.where((4.0 <= h) & (h < 5.0), z, g1)
    b1 = torch.where((4.0 <= h) & (h < 5.0), c, b1)

    r1 = torch.where((5.0 <= h) & (h < 6.0), c, r1)
    g1 = torch.where((5.0 <= h) & (h < 6.0), z, g1)
    b1 = torch.where((5.0 <= h) & (h < 6.0), x, b1)

    m = val - c
    return torch.cat((r1 + m, g1 + m, b1 + m), dim=-1).clamp(0.0, 1.0)


def solid_hsv(hue: float, sat: float, val: float, like: torch.Tensor) -> torch.Tensor:
    """Small helper used by tests and docs-oriented workflows."""

    h = like.new_full((*like.shape[:-1], 1), hue)
    s = like.new_full((*like.shape[:-1], 1), sat)
    v = like.new_full((*like.shape[:-1], 1), val)
    return hsv_to_rgb(h, s, v)
