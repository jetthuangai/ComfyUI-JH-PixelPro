"""Auto Tone & White Balance: measure a photo and propose ColorLab settings.

Everything is measured on a proxy (long edge <= 512 px) so the result does not depend on
the image size. The output is a plain dict of ColorLab keys; applying it is left to the
ColorLab pipeline (``tone_engine = local``, ``exposure_mode = linear``), so the same
settings can be shown as ``params_json`` and fine-tuned in a ColorLab node.

- White balance: the mean of near-grey pixels in linear light (found iteratively, so large
  uniform colour areas such as skin never count), solved for ColorLab's relative
  Temp/Tint gains, scaled by how many near-grey pixels there are and limited to
  ``max_wb_shift``.
- Exposure: log-average luminance. Dark photos are brightened toward an 18 % key, capped
  so the brightest 0.5 % does not blow out by more than half a stop (Highlights recovers
  it); bright photos are only darkened above a 65 % key (high-key is bright on purpose),
  and when 40 % or more of the frame is near-white only the rest (the subject) is judged.
- Tone: Whites/Blacks move the 99.5th/0.5th luma percentiles toward 0.96/0.02 by inverting
  ColorLab's Whites/Blacks formula; Highlights/Shadows follow the clipped fractions.
- B&W mix (optional): hue bands brighter than the image mean get brighter in grey and
  darker bands darker, which keeps colour contrast as tonal contrast.

The constants are starting points chosen on test images, not fitted to Adobe's Auto.
"""

from __future__ import annotations

import math
from numbers import Real

import torch
from kornia.color import linear_rgb_to_rgb, rgb_to_linear_rgb

from .color_lab import GRAY_MIX_COLORS, HUE_ANCHORS, rgb_to_hsv
from .guided import make_proxy

PROXY_SIDE = 512
_REC709 = (0.2126, 0.7152, 0.0722)
_BT601 = (0.299, 0.587, 0.114)
# White balance
_WB_TOLERANCES = (1.08, 1.15, 1.25, 1.4)  # max/min channel ratio counted as near grey
_WB_FULL_CONFIDENCE = 0.25  # share of near-grey pixels that earns the full correction
_WB_DAMPING = 0.8  # correct most, not all, of the estimated cast
_TEMP_GAIN, _TINT_GAIN = 0.3, 0.22  # ColorLab's relative Temp/Tint gains (per 100)
# Exposure
_KEY = 0.18
_KEY_HIGH = 0.65  # log-average luminance above which a photo counts as overexposed
_NEAR_WHITE = 0.6  # linear luminance (about sRGB 0.8) counted as white background
_WHITE_BACKGROUND_SHARE = 0.4  # share of near-white pixels that marks a white background
_KEY_SUBJECT = 0.3  # on a white background, darken only a subject brighter than this
_EV_DAMPING = 0.6
_EV_DEADBAND = 0.1
_EV_UP, _EV_DOWN = 1.5, 1.0
# Tone
_WHITE_TARGET, _BLACK_TARGET = 0.96, 0.02
_TONE_GAIN = 0.45  # ColorLab _tone_region scale
_TONE_LIMIT = 30.0
_TONE_DAMPING = 0.5
_HIGH_CLIP, _LOW_CLIP = 0.92, 0.06
# B&W
_BW_GAIN = 250.0
_BW_LIMIT = 60.0
_BW_MIN_PRESENCE = 0.01


def _check_image(image: torch.Tensor) -> None:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"image must have shape (B,H,W,3), got {tuple(image.shape)}.")
    if not torch.is_floating_point(image):
        raise ValueError(f"image must be floating point, got {image.dtype}.")


def _quantile(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x.flatten().float(), q))


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = min(1.0, max(0.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def _white_balance(linear: torch.Tensor, max_shift: float) -> dict[str, float]:
    """Temp/Tint (-100..100) that make the photo's near-grey pixels grey. ``linear``: (1,3,h,w).

    Starts from no correction and averages only pixels whose channels are within a small
    ratio of each other, widening the tolerance over a few rounds while the estimate
    converges. Large uniform colour areas (skin, foliage, a sunset sky) are never near grey,
    so unlike the grey world they do not pull the balance. The correction is scaled by the
    share of near-grey pixels: a photo with none is left alone.
    """
    pixels = linear[0].flatten(1)
    luma = (pixels * pixels.new_tensor(_REC709).view(3, 1)).sum(0)
    usable = (luma > 0.02) & (pixels.amax(0) < 0.95)
    if int(usable.sum()) < 64:
        return {}
    gains = torch.ones(3, 1, dtype=pixels.dtype)
    share = 0.0
    for tolerance in _WB_TOLERANCES:
        balanced = pixels * gains
        ratio = balanced.amax(0) / balanced.amin(0).clamp_min(1e-6)
        grey = usable & (ratio < tolerance)
        if int(grey.sum()) < 50:
            continue
        mean = pixels[:, grey].mean(1)
        gains = (mean[1] / mean.clamp_min(1e-6)).view(3, 1)
        share = float(grey.sum()) / float(usable.sum())
    confidence = min(1.0, share / _WB_FULL_CONFIDENCE)
    if confidence <= 0.0:
        return {}
    r, g, b = (1.0 / float(v) for v in gains.flatten())
    # R(1 + 0.3t) = B(1 - 0.3t), then G(1 - 0.22n) = mean of the balanced R and B.
    temp = (b - r) / (_TEMP_GAIN * (r + b))
    rb = r * (1.0 + _TEMP_GAIN * temp)
    tint = (1.0 - rb / g) / _TINT_GAIN
    scale = _WB_DAMPING * confidence
    limit = max_shift / 100.0
    temp = max(-limit, min(limit, scale * temp))
    tint = max(-limit, min(limit, scale * tint))
    return {"wb_temperature": round(100.0 * temp, 1), "wb_tint": round(100.0 * tint, 1)}


def _exposure(luma: torch.Tensor) -> float:
    """EV from the log-average luminance (``luma`` linear).

    Dark photos are brightened toward an 18 % key. Bright photos are only darkened above a
    much higher key: white backgrounds and high-key portraits are bright on purpose.
    """
    log_mean = math.exp(float(torch.log(luma.clamp_min(1e-4)).mean()))
    if log_mean < _KEY:
        ev = _EV_DAMPING * math.log2(_KEY / max(log_mean, 1e-4))
        # Keep the brightest 0.5 % within half a stop of clipping.
        ev = min(ev, math.log2(1.0 / max(_quantile(luma, 0.995), 1e-4)) + 0.5)
    elif log_mean > _KEY_HIGH:
        # A frame that is mostly near-white is a white background (product, high-key), not
        # an overexposed photo: judge only the rest (the subject), and leave an all-white
        # frame alone.
        key = _KEY_HIGH
        near_white = luma > _NEAR_WHITE
        if float(near_white.float().mean()) >= _WHITE_BACKGROUND_SHARE:
            rest = luma[~near_white]
            if rest.numel() < 16:
                return 0.0
            log_mean = math.exp(float(torch.log(rest.clamp_min(1e-4)).mean()))
            key = _KEY_SUBJECT
        ev = _EV_DAMPING * math.log2(key / log_mean) if log_mean > key else 0.0
    else:
        ev = 0.0
    ev = max(-_EV_DOWN, min(_EV_UP, ev))
    return 0.0 if abs(ev) < _EV_DEADBAND else round(ev, 2)


def _region_amount(value: float, target: float, mask: float) -> float:
    """Invert ColorLab's ``delta = a * (1 - x | x) * 0.45 * mask`` for one luma level."""
    if mask <= 1e-3:
        return 0.0
    room = (1.0 - value) if target > value else value
    if room <= 1e-3:
        return 0.0
    amount = _TONE_DAMPING * 100.0 * (target - value) / (room * _TONE_GAIN * mask)
    return max(-_TONE_LIMIT, min(_TONE_LIMIT, amount))


def _tone(srgb_luma: torch.Tensor) -> dict[str, float]:
    high = _quantile(srgb_luma, 0.995)
    low = _quantile(srgb_luma, 0.005)
    whites = _region_amount(high, _WHITE_TARGET, _smoothstep(0.75, 1.0, high))
    blacks = _region_amount(low, _BLACK_TARGET, 1.0 - _smoothstep(0.0, 0.25, low))
    clipped_high = float((srgb_luma > _HIGH_CLIP).float().mean())
    clipped_low = float((srgb_luma < _LOW_CLIP).float().mean())
    highlights = -min(_TONE_LIMIT, 200.0 * clipped_high)
    shadows = min(_TONE_LIMIT, 150.0 * clipped_low)
    out = {
        "basic_whites": whites,
        "basic_blacks": blacks,
        "basic_highlights": highlights,
        "basic_shadows": shadows,
    }
    return {k: round(v, 1) for k, v in out.items() if abs(v) >= 1.0}


def _bw_mix(srgb: torch.Tensor) -> dict[str, float]:
    """Gray-mix weights from the mean luma of each hue band. ``srgb``: (1,h,w,3)."""
    hue, sat, _ = (c.squeeze(-1) for c in rgb_to_hsv(srgb))
    luma = (srgb * srgb.new_tensor(_REC709)).sum(-1)
    mean = float(luma.mean())
    out: dict[str, float] = {}
    for color in GRAY_MIX_COLORS:
        dist = torch.remainder(hue - HUE_ANCHORS[color] + 180.0, 360.0) - 180.0
        weight = torch.exp(-0.5 * (dist / 20.0) ** 2) * sat
        presence = float(weight.mean())
        if presence < _BW_MIN_PRESENCE:
            continue
        band = float((luma * weight).sum() / weight.sum())
        value = max(-_BW_LIMIT, min(_BW_LIMIT, _BW_GAIN * (band - mean)))
        if abs(value) >= 1.0:
            out[f"gray_{color}"] = round(value, 1)
    return out


def auto_tone_params(
    image: torch.Tensor,
    *,
    exposure: bool = True,
    tone: bool = True,
    white_balance: bool = True,
    bw: bool = False,
    max_wb_shift: float = 50.0,
) -> list[dict[str, object]]:
    """Measure each image of a batch and return ColorLab settings, one dict per image.

    Args:
        image: IMAGE ``(B,H,W,3)`` in ``[0,1]`` (sRGB).
        exposure: Propose ``basic_exposure`` (EV, linear light).
        tone: Propose Whites, Blacks, Highlights and Shadows.
        white_balance: Propose ``wb_temperature`` / ``wb_tint``.
        bw: Convert to black and white with an automatic gray mix.
        max_wb_shift: Largest Temp/Tint change, 0..100.

    Returns:
        One dict per image with ``tone_engine = local`` and ``exposure_mode = linear``
        plus the non-zero adjustments.
    """
    _check_image(image)
    if isinstance(max_wb_shift, bool) or not isinstance(max_wb_shift, Real):
        raise ValueError("max_wb_shift must be a number in [0, 100].")
    if not 0.0 <= float(max_wb_shift) <= 100.0:
        raise ValueError("max_wb_shift must be in [0, 100].")

    results: list[dict[str, object]] = []
    for index in range(image.shape[0]):
        frame = image[index : index + 1].float().clamp(0.0, 1.0).permute(0, 3, 1, 2)
        proxy = make_proxy(frame.cpu(), PROXY_SIDE)
        linear = rgb_to_linear_rgb(proxy)
        params: dict[str, object] = {"tone_engine": "local", "exposure_mode": "linear"}

        if white_balance and max_wb_shift > 0:
            wb = _white_balance(linear, float(max_wb_shift))
            params.update({k: v for k, v in wb.items() if abs(v) >= 0.5})
            if wb:
                t, n = wb["wb_temperature"] / 100.0, wb["wb_tint"] / 100.0
                gains = linear.new_tensor(
                    [1.0 + _TEMP_GAIN * t, 1.0 - _TINT_GAIN * n, max(1.0 - _TEMP_GAIN * t, 0.0)]
                ).view(1, 3, 1, 1)
                balanced = linear * gains
                weights = linear.new_tensor(_REC709).view(1, 3, 1, 1)
                ratio = (linear * weights).sum(1, keepdim=True) / (balanced * weights).sum(
                    1, keepdim=True
                ).clamp_min(1e-9)
                linear = (balanced * ratio).clamp(0.0, 1.0)

        luma = (linear * linear.new_tensor(_REC709).view(1, 3, 1, 1)).sum(1)
        if exposure:
            ev = _exposure(luma)
            if ev:
                params["basic_exposure"] = ev
                linear = (linear * 2.0**ev).clamp(0.0, 1.0)
        srgb = linear_rgb_to_rgb(linear)
        if tone:
            # ColorLab's Whites/Blacks masks use BT.601 luma on sRGB; invert with the same.
            srgb_luma = (srgb * srgb.new_tensor(_BT601).view(1, 3, 1, 1)).sum(1)
            params.update(_tone(srgb_luma))
        if bw:
            params["gray_enable"] = True
            params.update(_bw_mix(srgb.permute(0, 2, 3, 1)))
        results.append(params)
    return results
