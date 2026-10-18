"""ACR-style ColorLab pipeline implemented with pure PyTorch."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping

import numpy as np
import torch
from kornia.color import linear_rgb_to_rgb, rgb_to_linear_rgb
from kornia.filters import box_blur

from .guided import gaussian_blur_box, guided_filter, make_proxy
from .local_laplacian import shadows_highlights_camera_raw

logger = logging.getLogger(__name__)

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

GRAY_MIX_COLORS = tuple(color for color in HUE_ANCHORS if color != "magenta")

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

# Optional since v1.7.0; their defaults reproduce v1.6.0 exactly.
TONE_ENGINES = ("legacy", "local")
WB_KEYS = ("wb_temperature", "wb_tint")
# Point curves (Lightroom ToneCurvePV2012 style, [[x, y], ...] on 0..255). They have no
# widget: they travel in params_json (Lightroom Preset node, ColorLab preset output).
CURVE_KEYS = ("tone_curve", "tone_curve_red", "tone_curve_green", "tone_curve_blue")
# Camera calibration (Lightroom Calibration panel, -100..100), also params_json only.
CALIB_KEYS = (
    "calib_red_hue",
    "calib_red_sat",
    "calib_green_hue",
    "calib_green_sat",
    "calib_blue_hue",
    "calib_blue_sat",
    "calib_shadow_tint",
)
# Effects panel (applied last, Lightroom order: vignette, then grain) and the B&W magenta
# mixer; params_json only. Shape keys fall back to Lightroom's defaults when absent.
EFFECT_KEYS = (
    "vignette_amount",
    "vignette_midpoint",
    "vignette_feather",
    "vignette_roundness",
    "grain_amount",
    "grain_size",
    "grain_frequency",
)
_SHAPE_DEFAULTS = {
    "vignette_midpoint": 50.0,
    "vignette_feather": 50.0,
    "vignette_roundness": 0.0,
    "grain_size": 25.0,
    "grain_frequency": 50.0,
}
JSON_ONLY_KEYS = CURVE_KEYS + CALIB_KEYS + EFFECT_KEYS + ("gray_magenta",)
_GRAIN_SEED = 20261018  # fixed: the same preset gives the same grain on every run
# LightCraft colorops.rs: primary hue rotation (OkLCh radians) and chroma scale at +-100,
# and the green change in deep shadows at Shadows Tint +-100.
_CALIB_HUE = 0.5
_CALIB_SAT = 0.6
_SHADOW_TINT = 0.3
_REC709 = (0.2126, 0.7152, 0.0722)
_GREY = 0.18
# Clarity band and dehaze veil radii as fractions of the long edge (LightCraft local.rs).
_CLARITY_SIGMA = 0.012
_CLARITY_EPS = 0.8
_DEHAZE_SIGMA = 0.02
_AIRLIGHT_PROXY = 512
# Mid-grey slope change of the local engine's Contrast S-curve at +-100 (monotone below 1).
_CONTRAST_S = 0.6


def param_keys() -> tuple[str, ...]:
    """Every ColorLab parameter name (slider keys plus mode switches)."""

    keys = list(BASIC_KEYS)
    for color in HUE_ANCHORS:
        keys.extend((f"hsl_{color}_hue", f"hsl_{color}_sat", f"hsl_{color}_lum"))
    for region in GRADE_REGIONS:
        keys.extend(f"grade_{region}_{part}" for part in ("hue", "sat", "lum", "bal"))
    keys.append("gray_enable")
    keys.extend(f"gray_{color}" for color in GRAY_MIX_COLORS)
    keys.extend(("exposure_mode", "tone_engine", *WB_KEYS))
    keys.extend(JSON_ONLY_KEYS)
    return tuple(keys)


def _curve_points(key: str, value: object) -> list[list[float]]:
    """Validate a point curve: >= 2 [x, y] pairs in 0..255, sorted by x, distinct x."""

    if not isinstance(value, list | tuple) or len(value) < 2:
        raise ValueError(f"{key} must be a list of at least two [x, y] points.")
    points = []
    for point in value:
        if (
            not isinstance(point, list | tuple)
            or len(point) != 2
            or not all(isinstance(v, int | float) and not isinstance(v, bool) for v in point)
            or not all(0.0 <= float(v) <= 255.0 for v in point)
        ):
            raise ValueError(f"{key} points must be [x, y] numbers in 0..255, got {point!r}.")
        points.append([float(point[0]), float(point[1])])
    points.sort(key=lambda point: point[0])
    if any(b[0] <= a[0] for a, b in zip(points, points[1:], strict=False)):
        raise ValueError(f"{key} x values must be distinct.")
    return points


def _curve_is_identity(points: object) -> bool:
    return not points or all(abs(x - y) < 1e-6 for x, y in points)  # type: ignore[union-attr]


def merge_params_json(params: Mapping[str, object], text: str | None) -> dict[str, object]:
    """Merge a ``params_json`` object (from ColorLab / Lightroom Preset nodes) into ``params``.

    Numbers are added to the matching slider (so manual sliders act as offsets on
    top of the automatic values); booleans and strings replace the setting. Unknown
    keys are ignored with a warning. Empty text returns ``params`` unchanged.
    """

    merged = dict(params)
    if text is None or not str(text).strip():
        return merged
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"params_json is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("params_json must be a JSON object of ColorLab parameters.")
    known = set(param_keys())
    ignored = []
    for key, value in data.items():
        if key not in known:
            ignored.append(key)
        elif key in CURVE_KEYS:
            merged[key] = _curve_points(key, value)
        elif isinstance(value, bool | str):
            merged[key] = value
        elif isinstance(value, int | float) and math.isfinite(value):
            merged[key] = float(merged.get(key, 0.0)) + float(value)
        else:
            raise ValueError(f"params_json value for {key!r} must be a number, bool or string.")
    if ignored:
        logger.warning("ColorLab params_json: ignored unknown keys %s", sorted(ignored))
    return merged


_SWITCH_DEFAULTS = {"gray_enable": False, "exposure_mode": "display", "tone_engine": "legacy"}


def _is_angle(key: str) -> bool:
    # Color Grading hues are absolute angles and effect shapes are positions/sizes:
    # scaling them would change the look, not its amount.
    return (key.startswith("grade_") and key.endswith("_hue")) or key in _SHAPE_DEFAULTS


def params_to_json(params: Mapping[str, object]) -> str:
    """ColorLab settings as a ``params_json`` preset: only non-default values, slider order."""

    out: dict[str, object] = {}
    for key in param_keys():
        if key not in params:
            continue
        value = params[key]
        if key in CURVE_KEYS:
            points = _curve_points(key, value)
            if not _curve_is_identity(points):
                out[key] = [[round(x, 4), round(y, 4)] for x, y in points]
        elif key in _SWITCH_DEFAULTS:
            value = bool(value) if key == "gray_enable" else str(value)
            if value != _SWITCH_DEFAULTS[key]:
                out[key] = value
        elif abs(float(value)) > 1e-9:
            out[key] = round(float(value), 6)
    return json.dumps(out)


def scale_params_json(text: str | None, strength: float) -> str:
    """Scale a ``params_json`` preset's amounts by ``strength`` (switches and grading hues kept)."""

    merged = merge_params_json({}, text)
    scaled: dict[str, object] = {}
    for key, value in merged.items():
        if key in CURVE_KEYS:
            # Blend the curve toward the identity line.
            scaled[key] = [[x, x + (y - x) * strength] for x, y in value]  # type: ignore[union-attr]
        elif isinstance(value, float) and not _is_angle(key):
            scaled[key] = value * strength
        else:
            scaled[key] = value
    return params_to_json(scaled)


def apply_preset(
    image: torch.Tensor, params_json: str | None, strength: float
) -> tuple[torch.Tensor, str]:
    """Apply a ``params_json`` preset at ``strength`` -> (image, preset at that strength).

    B&W is a switch, so below strength 1 the black-and-white result is faded in over the
    colour one (like Lightroom's preset Amount) instead of turning fully grey at once.
    """

    preset = scale_params_json(params_json, strength)
    if strength == 0.0:
        return image, preset
    params = merge_params_json({}, preset)
    out = apply_colorlab_pipeline(image, params)
    if bool(params.get("gray_enable", False)) and strength < 1.0:
        colour = apply_colorlab_pipeline(image, {**params, "gray_enable": False})
        out = torch.lerp(colour, out, float(strength))
    return out, preset


def apply_colorlab_pipeline(
    image: torch.Tensor, params: Mapping[str, object], *, prefer_gpu: bool = True
) -> torch.Tensor:
    """Apply Basic -> HSL -> Color Grading -> Gray Mix to a ComfyUI IMAGE tensor.

    ``prefer_gpu=False`` keeps CPU inputs on the CPU (small previews while a workflow runs).
    """

    _validate_image(image)
    engine = str(params.get("tone_engine", "legacy"))
    if engine not in TONE_ENGINES:
        raise ValueError(f"tone_engine must be one of {TONE_ENGINES}, got {engine!r}.")
    if _is_identity(params):
        return image
    if (
        prefer_gpu
        and engine == "local"
        and image.device.type == "cpu"
        and torch.cuda.is_available()
    ):
        # The local engine's pyramids are ~100x faster on the GPU; ComfyUI images live on the CPU.
        try:
            return _run_pipeline(image.cuda(), params).to(image.device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    return _run_pipeline(image, params)


def _run_pipeline(image: torch.Tensor, params: Mapping[str, object]) -> torch.Tensor:
    out = image.clamp(0.0, 1.0)
    out = _apply_white_balance(out, params)
    out = _apply_calibration(out, params)
    out = _apply_basic(out, params)
    out = _apply_tone_curves(out, params)
    out = _apply_hsl(out, params)
    out = _apply_color_grading(out, params)
    if bool(params.get("gray_enable", False)):
        out = _apply_gray_mix(out, params)
    out = _apply_vignette(out, params)
    out = _apply_grain(out, params)
    return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _validate_image(image: torch.Tensor) -> None:
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"image must have shape (B,H,W,3), got {tuple(image.shape)}.")


def _is_identity(params: Mapping[str, object]) -> bool:
    if bool(params.get("gray_enable", False)):
        return False
    if any(not _curve_is_identity(params.get(key)) for key in CURVE_KEYS):
        return False
    keys = [*BASIC_KEYS, *WB_KEYS, *CALIB_KEYS, "vignette_amount", "grain_amount"]
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
    local = str(params.get("tone_engine", "legacy")) == "local"
    if local and _float(params, "basic_dehaze"):
        # Scene-referred dehaze runs before exposure, as in Lightroom.
        out = _dehaze_local(out, _float(params, "basic_dehaze") / 100.0)
    exposure = _float(params, "basic_exposure")
    exposure_mode = str(params.get("exposure_mode", "display"))
    if exposure_mode not in ("display", "linear"):
        raise ValueError(f"exposure_mode must be 'display' or 'linear', got {exposure_mode!r}.")
    if exposure:
        if exposure_mode == "linear":
            linear = rgb_to_linear_rgb(out.permute(0, 3, 1, 2).clamp(0.0, 1.0))
            out = linear_rgb_to_rgb((linear * (2.0**exposure)).clamp(0.0, 1.0))
            out = out.permute(0, 2, 3, 1)
        else:
            out = out * (2.0**exposure)

    contrast = _float(params, "basic_contrast") / 100.0
    if contrast and local:
        # Lightroom-like: an S-curve that keeps black and white fixed (slope 1 + a at mid-grey,
        # 1 - a at the ends), instead of a linear stretch that greys out whites.
        a = max(-1.0, min(1.0, contrast)) * _CONTRAST_S
        x = out.clamp(0.0, 1.0)
        out = x - a * torch.sin(2.0 * math.pi * x) / (2.0 * math.pi)
    elif contrast:
        factor = 1.0 + contrast
        out = (out - 0.5) * factor + 0.5

    if local:
        highlights, shadows = _float(params, "basic_highlights"), _float(params, "basic_shadows")
        if highlights or shadows:
            out = shadows_highlights_camera_raw(
                out, max(-100.0, min(100.0, shadows)), max(-100.0, min(100.0, highlights))
            )
        lum = _luma(out)
    else:
        lum = _luma(out)
        out = _tone_region(out, lum, "basic_highlights", params, _smoothstep(0.5, 1.0, lum), True)
        out = _tone_region(
            out, lum, "basic_shadows", params, 1.0 - _smoothstep(0.0, 0.5, lum), True
        )
    out = _tone_region(out, lum, "basic_whites", params, _smoothstep(0.75, 1.0, lum), True)
    out = _tone_region(out, lum, "basic_blacks", params, 1.0 - _smoothstep(0.0, 0.25, lum), True)

    texture = _float(params, "basic_texture") / 100.0
    if texture:
        out = _local_contrast(out, texture * 0.25, kernel_size=5)

    clarity = _float(params, "basic_clarity") / 100.0
    if clarity and local:
        out = _clarity_local(out, clarity)
    elif clarity:
        mid_mask = (1.0 - (lum - 0.5).abs() * 2.0).clamp(0.0, 1.0)
        out = out + (_local_contrast(out, clarity * 0.35, kernel_size=9) - out) * mid_mask

    dehaze = _float(params, "basic_dehaze") / 100.0
    if dehaze and not local:
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


_OKLAB_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_OKLAB_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)


def _oklab(rgb: np.ndarray) -> np.ndarray:
    return _OKLAB_M2 @ np.cbrt(_OKLAB_M1 @ rgb)


def _from_oklab(lab: np.ndarray) -> np.ndarray:
    return np.linalg.solve(_OKLAB_M1, np.linalg.solve(_OKLAB_M2, lab) ** 3)


def calibration_matrix(params: Mapping[str, object]) -> np.ndarray | None:
    """3x3 linear-RGB matrix moving each primary's OkLCh hue / chroma, white kept (LightCraft)."""

    primaries = [
        (_float(params, f"calib_{c}_hue"), _float(params, f"calib_{c}_sat"))
        for c in ("red", "green", "blue")
    ]
    if all(h == 0.0 and s == 0.0 for h, s in primaries):
        return None
    columns = []
    for i, (hue, sat) in enumerate(primaries):
        lab = _oklab(np.eye(3)[i])
        chroma, angle = np.hypot(lab[1], lab[2]), np.arctan2(lab[2], lab[1])
        angle += max(-1.0, min(1.0, hue / 100.0)) * _CALIB_HUE
        chroma *= 1.0 + max(-1.0, min(1.0, sat / 100.0)) * _CALIB_SAT
        columns.append(
            _from_oklab(np.array([lab[0], chroma * np.cos(angle), chroma * np.sin(angle)]))
        )
    matrix = np.stack(columns, axis=1)
    return matrix * np.linalg.solve(matrix, np.ones(3))[None, :]


def _apply_calibration(image: torch.Tensor, params: Mapping[str, object]) -> torch.Tensor:
    """Calibration in linear light: primaries matrix, then Shadows Tint (luminance kept)."""

    matrix = calibration_matrix(params)
    tint = max(-1.0, min(1.0, _float(params, "calib_shadow_tint") / 100.0))
    if matrix is None and not tint:
        return image
    linear = _to_linear(image)
    if matrix is not None:
        m = torch.as_tensor(matrix, dtype=linear.dtype, device=linear.device)
        linear = torch.einsum("rc,bchw->brhw", m, linear).clamp_min(0.0)
    if tint:
        y0 = _luminance(linear)
        shadows = 1.0 - _smoothstep(-5.0, -0.5, torch.log2(y0.clamp_min(1e-7) / _GREY))
        green = linear[:, 1:2] * (1.0 - _SHADOW_TINT * tint * shadows).clamp_min(0.0)
        linear = torch.cat((linear[:, :1], green, linear[:, 2:]), dim=1)
        linear = linear * y0 / _luminance(linear).clamp_min(1e-9)
    return _to_srgb(linear)


def _shape(params: Mapping[str, object], key: str) -> float:
    return _float(params, key, _SHAPE_DEFAULTS[key]) if key in params else _SHAPE_DEFAULTS[key]


def _apply_vignette(image: torch.Tensor, params: Mapping[str, object]) -> torch.Tensor:
    """Post-crop vignette: Amount (-100 darker .. 100 lighter), Midpoint, Feather, Roundness."""

    amount = max(-1.0, min(1.0, _float(params, "vignette_amount") / 100.0))
    if not amount:
        return image
    _, height, width, _ = image.shape
    midpoint = _shape(params, "vignette_midpoint") / 100.0
    feather = _shape(params, "vignette_feather") / 100.0
    roundness = max(-1.0, min(1.0, _shape(params, "vignette_roundness") / 100.0))
    ys = torch.linspace(-1.0, 1.0, height, device=image.device).view(-1, 1)
    xs = torch.linspace(-1.0, 1.0, width, device=image.device).view(1, -1)
    # Roundness 0 follows the frame; +100 is a circle on the short side, -100 a squarer ellipse.
    aspect = width / height
    ax = ay = 1.0
    if roundness > 0:  # shrink the long axis toward a circle in pixels
        ax = 1.0 + (min(1.0, 1.0 / aspect) - 1.0) * roundness
        ay = 1.0 + (min(1.0, aspect) - 1.0) * roundness
    power = 2.0 + 4.0 * max(0.0, -roundness)
    dist = ((xs.abs() / ax) ** power + (ys.abs() / ay) ** power) ** (1.0 / power)
    start = 0.15 + 0.7 * midpoint
    width_band = 0.05 + 0.9 * feather
    t = ((dist - start) / width_band).clamp(0.0, 1.0)
    mask = (t * t * (3.0 - 2.0 * t)).view(1, height, width, 1)
    linear = _to_linear(image).permute(0, 2, 3, 1)
    if amount < 0:
        linear = linear * (1.0 + amount * mask)
    else:
        linear = linear + (1.0 - linear) * amount * mask
    return _to_srgb(linear.permute(0, 3, 1, 2))


def _apply_grain(image: torch.Tensor, params: Mapping[str, object]) -> torch.Tensor:
    """Film grain on luminance: Amount, Size (grain radius) and Roughness (frequency)."""

    amount = max(0.0, min(1.0, _float(params, "grain_amount") / 100.0))
    if not amount:
        return image
    batch, height, width, _ = image.shape
    size = _shape(params, "grain_size") / 100.0
    rough = max(0.0, min(1.0, _shape(params, "grain_frequency") / 100.0))
    generator = torch.Generator().manual_seed(_GRAIN_SEED)
    noise = torch.randn((2, 1, height, width), generator=generator).to(image.device)
    sigma = 0.4 + 2.6 * size
    coarse = gaussian_blur_box(noise[:1], sigma)
    fine = gaussian_blur_box(noise[1:], max(0.3, sigma * 0.45))
    coarse = coarse / coarse.std().clamp_min(1e-6)
    fine = fine / fine.std().clamp_min(1e-6)
    grain = ((1.0 - rough) * coarse + rough * fine).permute(0, 2, 3, 1)
    grain = grain / grain.std().clamp_min(1e-6)
    lum = _luma(image)
    weight = (4.0 * lum * (1.0 - lum)).clamp(0.15, 1.0)  # quieter in deep shadows / highlights
    return (image + amount * 0.09 * grain * weight).clamp(0.0, 1.0)


def _monotone_lut(points: list[list[float]], size: int = 1024) -> np.ndarray:
    """Monotone cubic (Fritsch-Carlson) curve through ``points`` (0..255) as a 0..1 LUT."""

    xs = np.array([p[0] for p in points], dtype=np.float64) / 255.0
    ys = np.array([p[1] for p in points], dtype=np.float64) / 255.0
    grid = np.linspace(0.0, 1.0, size)
    if len(xs) == 2:
        return np.clip(np.interp(grid, xs, ys), 0.0, 1.0)
    slopes = np.diff(ys) / np.diff(xs)
    tangents = np.empty_like(ys)
    tangents[0], tangents[-1] = slopes[0], slopes[-1]
    tangents[1:-1] = np.where(slopes[:-1] * slopes[1:] > 0, (slopes[:-1] + slopes[1:]) / 2.0, 0.0)
    for i, slope in enumerate(slopes):
        if slope == 0.0:
            tangents[i] = tangents[i + 1] = 0.0
            continue
        a, b = tangents[i] / slope, tangents[i + 1] / slope
        norm = a * a + b * b
        if norm > 9.0:
            scale = 3.0 / np.sqrt(norm)
            tangents[i], tangents[i + 1] = scale * a * slope, scale * b * slope
    idx = np.clip(np.searchsorted(xs, grid) - 1, 0, len(xs) - 2)
    h = xs[idx + 1] - xs[idx]
    t = np.clip((grid - xs[idx]) / h, 0.0, 1.0)
    h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
    h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
    curve = (
        h00 * ys[idx] + h10 * h * tangents[idx] + h01 * ys[idx + 1] + h11 * h * tangents[idx + 1]
    )
    # Flat beyond the end points, as in Lightroom.
    curve = np.where(grid < xs[0], ys[0], np.where(grid > xs[-1], ys[-1], curve))
    return np.clip(curve, 0.0, 1.0)


def _sample(values: torch.Tensor, lut: np.ndarray) -> torch.Tensor:
    table = torch.as_tensor(lut, dtype=values.dtype, device=values.device)
    pos = values.clamp(0.0, 1.0) * (len(lut) - 1)
    lo = pos.floor().long().clamp(max=len(lut) - 2)
    return torch.lerp(table[lo], table[lo + 1], pos - lo)


def _apply_tone_curves(image: torch.Tensor, params: Mapping[str, object]) -> torch.Tensor:
    """Master then per-channel point curves on the display-referred image (Lightroom order)."""

    out = image
    master = params.get("tone_curve")
    if not _curve_is_identity(master):
        out = _sample(out, _monotone_lut(_curve_points("tone_curve", master)))
    channels = []
    for index, key in enumerate(CURVE_KEYS[1:]):
        points = params.get(key)
        channel = out[..., index]
        if not _curve_is_identity(points):
            channel = _sample(channel, _monotone_lut(_curve_points(key, points)))
        channels.append(channel)
    return torch.stack(channels, dim=-1)


def _to_linear(image: torch.Tensor) -> torch.Tensor:
    return rgb_to_linear_rgb(image.permute(0, 3, 1, 2).clamp(0.0, 1.0))


def _to_srgb(linear: torch.Tensor) -> torch.Tensor:
    return linear_rgb_to_rgb(linear.clamp(0.0, 1.0)).permute(0, 2, 3, 1)


def _luminance(linear: torch.Tensor) -> torch.Tensor:
    return (linear * linear.new_tensor(_REC709).view(1, 3, 1, 1)).sum(1, keepdim=True)


def _apply_white_balance(image: torch.Tensor, params: Mapping[str, object]) -> torch.Tensor:
    """Relative Temp/Tint (-100..100) on a rendered image, luminance preserving.

    LightCraft's local temp/tint: R x (1 + 0.3 t), G x (1 - 0.22 n), B x (1 - 0.3 t)
    in linear light, then rescaled to the original luminance.
    """

    temp = _float(params, "wb_temperature") / 100.0
    tint = _float(params, "wb_tint") / 100.0
    if not temp and not tint:
        return image
    linear = _to_linear(image)
    gains = linear.new_tensor([1.0 + 0.3 * temp, 1.0 - 0.22 * tint, max(1.0 - 0.3 * temp, 0.0)])
    balanced = linear * gains.clamp_min(0.0).view(1, 3, 1, 1)
    ratio = _luminance(linear) / _luminance(balanced).clamp_min(1e-9)
    return _to_srgb(balanced * ratio)


def _clarity_local(image: torch.Tensor, amount: float) -> torch.Tensor:
    """Clarity as an edge-aware band of log luminance (guided filter), weighted to midtones."""

    linear = _to_linear(image)
    log_l = torch.log2(_luminance(linear).clamp_min(1e-7) / _GREY)
    sigma = max(1.0, _CLARITY_SIGMA * max(image.shape[1:3]))
    base = guided_filter(
        log_l,
        log_l,
        radius=max(1, round(sigma * math.sqrt(3.0))),
        eps=_CLARITY_EPS,
        subsample=int(min(16, max(1, sigma // 3))),
    )
    detail = (log_l - base).clamp(-2.5, 2.5)
    midtones = torch.exp(-((base / 3.2) ** 2))
    delta = amount * 0.85 * detail * (0.35 + 0.65 * midtones)
    return _to_srgb(linear * torch.exp2(delta))


def _airlight(linear: torch.Tensor) -> torch.Tensor:
    """99.5th percentile of the blurred dark channel of the whole frame (proxy), per image."""

    proxy = make_proxy(linear, _AIRLIGHT_PROXY)
    dark = gaussian_blur_box(proxy.amin(1, keepdim=True), _DEHAZE_SIGMA * max(proxy.shape[-2:]))
    return torch.quantile(dark.flatten(1), 0.995, dim=1).clamp_min(0.05).view(-1, 1, 1, 1)


def _dehaze_local(image: torch.Tensor, amount: float) -> torch.Tensor:
    """Dark-channel-prior dehaze (He et al. 2009) with LightCraft's transmission model."""

    linear = _to_linear(image)
    air = _airlight(linear)
    dark = gaussian_blur_box(linear.amin(1, keepdim=True), _DEHAZE_SIGMA * max(image.shape[1:3]))
    density = (dark / air).clamp(0.0, 1.0)
    if amount > 0.0:
        t = (1.0 - 0.95 * min(amount, 1.0) * density).clamp_min(0.12)
        linear = ((linear - air * (1.0 - t)) / t).clamp_min(0.0)
    else:
        k = min(-amount, 1.0) * 0.7 * (0.35 + 0.65 * density)
        linear = linear + (air * 0.9 - linear) * k
    return _to_srgb(linear)


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
    # Replicate padding: zero padding brightened/darkened a halo around the frame.
    blur = box_blur(x, (kernel_size, kernel_size), border_type="replicate")
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
    # Magenta only joins when set (Lightroom presets), so v1.6 results stay identical.
    colors = GRAY_MIX_COLORS + (("magenta",) if _float(params, "gray_magenta") else ())
    for color in colors:
        center = HUE_ANCHORS[color]
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

    r1 = torch.where((h >= 0.0) & (h < 1.0), c, z)
    g1 = torch.where((h >= 0.0) & (h < 1.0), x, z)
    b1 = z.clone()

    r1 = torch.where((h >= 1.0) & (h < 2.0), x, r1)
    g1 = torch.where((h >= 1.0) & (h < 2.0), c, g1)
    b1 = torch.where((h >= 1.0) & (h < 2.0), z, b1)

    r1 = torch.where((h >= 2.0) & (h < 3.0), z, r1)
    g1 = torch.where((h >= 2.0) & (h < 3.0), c, g1)
    b1 = torch.where((h >= 2.0) & (h < 3.0), x, b1)

    r1 = torch.where((h >= 3.0) & (h < 4.0), z, r1)
    g1 = torch.where((h >= 3.0) & (h < 4.0), x, g1)
    b1 = torch.where((h >= 3.0) & (h < 4.0), c, b1)

    r1 = torch.where((h >= 4.0) & (h < 5.0), x, r1)
    g1 = torch.where((h >= 4.0) & (h < 5.0), z, g1)
    b1 = torch.where((h >= 4.0) & (h < 5.0), c, b1)

    r1 = torch.where((h >= 5.0) & (h < 6.0), c, r1)
    g1 = torch.where((h >= 5.0) & (h < 6.0), z, g1)
    b1 = torch.where((h >= 5.0) & (h < 6.0), x, b1)

    m = val - c
    return torch.cat((r1 + m, g1 + m, b1 + m), dim=-1).clamp(0.0, 1.0)


def solid_hsv(hue: float, sat: float, val: float, like: torch.Tensor) -> torch.Tensor:
    """Small helper used by tests and docs-oriented workflows."""

    h = like.new_full((*like.shape[:-1], 1), hue)
    s = like.new_full((*like.shape[:-1], 1), sat)
    v = like.new_full((*like.shape[:-1], 1), val)
    return hsv_to_rgb(h, s, v)
