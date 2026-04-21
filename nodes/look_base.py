"""Shared helpers for JSON-driven Look preset nodes."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from kornia.color import lab_to_rgb, rgb_to_lab

from ..core.selective_color import (
    apply_hue_sat_shift,
    hue_range_mask,
    saturation_range_mask,
)
from ..core.tone_curve import tone_curve

_PRESET_DIR = Path(__file__).resolve().parents[1] / "presets"


@lru_cache(maxsize=16)
def load_preset_json(preset_id: str) -> dict[str, Any]:
    """Load a versioned preset config by kebab-case id."""

    preset_path = _PRESET_DIR / f"{preset_id}.json"
    with preset_path.open("r", encoding="utf-8") as handle:
        preset = json.load(handle)

    if preset.get("schema_version") != 1:
        raise ValueError(f"Unsupported preset schema_version for {preset_id!r}.")
    if preset.get("id") != preset_id:
        raise ValueError(f"Preset id mismatch in {preset_path}.")
    if not isinstance(preset.get("compose_ops"), list):
        raise ValueError(f"Preset {preset_id!r} must define compose_ops list.")
    return preset


def apply_preset(
    image: torch.Tensor, preset_id: str, intensity: float, protect_skin: bool
) -> torch.Tensor:
    """Apply a preset and blend it against the original image by intensity."""

    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"image must have shape (B,H,W,3), got {tuple(image.shape)}.")

    intensity_float = float(intensity)
    if not 0.0 <= intensity_float <= 1.0:
        raise ValueError("intensity must be in [0.0, 1.0].")
    if intensity_float == 0.0:
        return image

    original = image.clamp(0.0, 1.0)
    preset = load_preset_json(preset_id)
    skin_mask = _skin_mask(original) if protect_skin else None

    processed = original.clone()
    for op in preset["compose_ops"]:
        processed = dispatch_op(op, processed, skin_mask)

    output = torch.lerp(original, processed, intensity_float)
    return torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def dispatch_op(
    op: dict[str, Any],
    image: torch.Tensor,
    skin_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Route a preset operation to existing batch-3/batch-6 primitives."""

    op_name = op.get("op")
    params = op.get("params", {})

    if op_name == "lab_color_shift":
        output = _apply_lab_color_shift(image, params)
    elif op_name == "tone_curve_s":
        output = _apply_tone_curve(image, params)
    elif op_name == "hue_sat_per_range":
        output = _apply_hue_sat_per_range(image, params)
    elif op_name == "saturation_mask":
        output = _apply_saturation_adjust(image, params, skin_mask)
    else:
        raise ValueError(f"Unsupported look preset op: {op_name!r}.")

    if skin_mask is None:
        return output
    delta = output - image
    return (image + delta * (1.0 - skin_mask).unsqueeze(-1)).clamp(0.0, 1.0)


def _apply_tone_curve(image: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    points = torch.tensor(params["curve_points"], dtype=torch.float32, device=image.device)
    image_bchw = image.permute(0, 3, 1, 2).contiguous()
    output_bchw = tone_curve(image_bchw, control_points=points, channel="rgb_master", strength=1.0)
    return output_bchw.permute(0, 2, 3, 1).contiguous()


def _apply_hue_sat_per_range(image: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    mask = hue_range_mask(
        image,
        float(params["hue_center"]),
        float(params["band_width"]),
        device=image.device,
    )
    return apply_hue_sat_shift(
        image,
        mask,
        float(params.get("hue_shift", 0.0)),
        float(params.get("sat_mult", 1.0)),
        float(params.get("sat_add", 0.0)),
        device=image.device,
    )


def _apply_saturation_adjust(
    image: torch.Tensor,
    params: dict[str, Any],
    skin_mask: torch.Tensor | None,
) -> torch.Tensor:
    sat_mask = saturation_range_mask(
        image,
        float(params.get("sat_min", 0.0)),
        float(params.get("sat_max", 1.0)),
        feather=float(params.get("feather", 0.0)),
        device=image.device,
    )
    if bool(params.get("exclude_skin", False)):
        base_skin_mask = skin_mask if skin_mask is not None else _skin_mask(image)
        sat_mask = sat_mask * (1.0 - base_skin_mask)

    sat_mult = float(params.get("sat_mult", 1.0))
    return apply_hue_sat_shift(
        image,
        sat_mask,
        0.0,
        sat_mult,
        float(params.get("sat_add", 0.0)),
        device=image.device,
    )


def _apply_lab_color_shift(image: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    image_bchw = image.permute(0, 3, 1, 2).contiguous()
    lab = rgb_to_lab(image_bchw)
    lightness = (lab[:, 0:1] / 100.0).clamp(0.0, 1.0)
    delta_a = torch.zeros_like(lightness)
    delta_b = torch.zeros_like(lightness)

    global_shift = params.get("global_shift")
    if global_shift:
        delta_a = delta_a + float(global_shift.get("a_shift", 0.0))
        delta_b = delta_b + float(global_shift.get("b_shift", 0.0))

    shadow_shift = params.get("shadow_shift")
    if shadow_shift:
        threshold = max(float(shadow_shift.get("L_threshold", 0.45)), 1e-6)
        weight = ((threshold - lightness) / threshold).clamp(0.0, 1.0)
        delta_a = delta_a + weight * float(shadow_shift.get("a_shift", 0.0))
        delta_b = delta_b + weight * float(shadow_shift.get("b_shift", 0.0))

    highlight_shift = params.get("highlight_shift")
    if highlight_shift:
        threshold = min(float(highlight_shift.get("L_threshold", 0.6)), 1.0 - 1e-6)
        weight = ((lightness - threshold) / (1.0 - threshold)).clamp(0.0, 1.0)
        delta_a = delta_a + weight * float(highlight_shift.get("a_shift", 0.0))
        delta_b = delta_b + weight * float(highlight_shift.get("b_shift", 0.0))

    midtone_shift = params.get("midtone_shift")
    if midtone_shift:
        center = float(midtone_shift.get("L_center", 0.5))
        width = max(float(midtone_shift.get("width", 0.35)), 1e-6)
        weight = (1.0 - (lightness - center).abs() / width).clamp(0.0, 1.0)
        delta_a = delta_a + weight * float(midtone_shift.get("a_shift", 0.0))
        delta_b = delta_b + weight * float(midtone_shift.get("b_shift", 0.0))

    shifted_lab = lab.clone()
    shifted_lab[:, 1:2] = shifted_lab[:, 1:2] + delta_a
    shifted_lab[:, 2:3] = shifted_lab[:, 2:3] + delta_b
    output_bchw = lab_to_rgb(shifted_lab)
    return output_bchw.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)


def _skin_mask(image: torch.Tensor) -> torch.Tensor:
    hue_mask = hue_range_mask(image, 24.0, 28.0, device=image.device)
    sat_mask = saturation_range_mask(image, 0.18, 0.95, feather=0.08, device=image.device)
    return (hue_mask * sat_mask).clamp(0.0, 1.0)
