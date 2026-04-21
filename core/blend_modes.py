"""Photoshop-style blend modes and LAYER_STACK composition."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as functional

from .color_lab import hsv_to_rgb, rgb_to_hsv

EPS = 1e-6


def _clamp(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def blend_normal(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return blend


def blend_dissolve(
    base: torch.Tensor,
    blend: torch.Tensor,
    opacity: float = 1.0,
    seed: int = 0,
) -> torch.Tensor:
    generator = torch.Generator(device=base.device)
    generator.manual_seed(seed)
    keep = torch.rand(base.shape[:-1], generator=generator, device=base.device) < float(opacity)
    return torch.where(keep.unsqueeze(-1), blend, base)


def blend_darken(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return torch.minimum(base, blend)


def blend_multiply(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return base * blend


def blend_color_burn(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.minimum(torch.ones_like(base), (1.0 - base) / blend.clamp_min(EPS))


def blend_linear_burn(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return base + blend - 1.0


def blend_darker_color(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    base_sum = base.sum(dim=-1, keepdim=True)
    blend_sum = blend.sum(dim=-1, keepdim=True)
    return torch.where(base_sum <= blend_sum, base, blend)


def blend_lighten(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return torch.maximum(base, blend)


def blend_screen(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return 1.0 - (1.0 - base) * (1.0 - blend)


def blend_color_dodge(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return torch.minimum(torch.ones_like(base), base / (1.0 - blend).clamp_min(EPS))


def blend_linear_dodge(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return base + blend


def blend_lighter_color(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    base_sum = base.sum(dim=-1, keepdim=True)
    blend_sum = blend.sum(dim=-1, keepdim=True)
    return torch.where(base_sum >= blend_sum, base, blend)


def blend_overlay(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return torch.where(base <= 0.5, 2.0 * base * blend, 1.0 - 2.0 * (1.0 - base) * (1.0 - blend))


def blend_soft_light(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    dark = base - (1.0 - 2.0 * blend) * base * (1.0 - base)
    light = base + (2.0 * blend - 1.0) * (torch.sqrt(base.clamp_min(0.0)) - base)
    return torch.where(blend <= 0.5, dark, light)


def blend_hard_light(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return torch.where(blend <= 0.5, 2.0 * base * blend, 1.0 - 2.0 * (1.0 - base) * (1.0 - blend))


def blend_vivid_light(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    burn = blend_color_burn(base, 2.0 * blend)
    dodge = blend_color_dodge(base, 2.0 * (blend - 0.5))
    return torch.where(blend <= 0.5, burn, dodge)


def blend_linear_light(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return base + 2.0 * blend - 1.0


def blend_pin_light(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return torch.where(
        blend <= 0.5, torch.minimum(base, 2.0 * blend), torch.maximum(base, 2.0 * (blend - 0.5))
    )


def blend_hard_mix(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return (blend_vivid_light(base, blend) >= 0.5).to(base.dtype)


def blend_difference(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return torch.abs(base - blend)


def blend_exclusion(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return base + blend - 2.0 * base * blend


def blend_subtract(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return base - blend


def blend_divide(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return base / blend.clamp_min(EPS)


def blend_hue(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    h_base, s_base, v_base = rgb_to_hsv(base)
    h_blend, _, _ = rgb_to_hsv(blend)
    return hsv_to_rgb(h_blend, s_base, v_base)


def blend_saturation(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    h_base, _, v_base = rgb_to_hsv(base)
    _, s_blend, _ = rgb_to_hsv(blend)
    return hsv_to_rgb(h_base, s_blend, v_base)


def blend_color(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    h_blend, s_blend, _ = rgb_to_hsv(blend)
    _, _, v_base = rgb_to_hsv(base)
    return hsv_to_rgb(h_blend, s_blend, v_base)


def blend_luminosity(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    h_base, s_base, _ = rgb_to_hsv(base)
    _, _, v_blend = rgb_to_hsv(blend)
    return hsv_to_rgb(h_base, s_base, v_blend)


BLEND_FUNCTIONS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "normal": blend_normal,
    "dissolve": blend_dissolve,
    "darken": blend_darken,
    "multiply": blend_multiply,
    "color_burn": blend_color_burn,
    "linear_burn": blend_linear_burn,
    "darker_color": blend_darker_color,
    "lighten": blend_lighten,
    "screen": blend_screen,
    "color_dodge": blend_color_dodge,
    "linear_dodge": blend_linear_dodge,
    "lighter_color": blend_lighter_color,
    "overlay": blend_overlay,
    "soft_light": blend_soft_light,
    "hard_light": blend_hard_light,
    "vivid_light": blend_vivid_light,
    "linear_light": blend_linear_light,
    "pin_light": blend_pin_light,
    "hard_mix": blend_hard_mix,
    "difference": blend_difference,
    "exclusion": blend_exclusion,
    "subtract": blend_subtract,
    "divide": blend_divide,
    "hue": blend_hue,
    "saturation": blend_saturation,
    "color": blend_color,
    "luminosity": blend_luminosity,
}

BLEND_MODES = list(BLEND_FUNCTIONS.keys())


def apply_blend(mode: str, base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    if mode not in BLEND_FUNCTIONS:
        raise ValueError(f"Unknown blend mode: {mode!r}")
    return _clamp(BLEND_FUNCTIONS[mode](base, blend))


def compose_stack(stack: list[dict]) -> torch.Tensor:
    """Compose a LAYER_STACK list of dict layers to a final IMAGE tensor."""

    if not stack:
        raise ValueError("compose_stack: empty stack")
    result = _clamp(stack[0]["image"])
    previous_mask = stack[0].get("mask")
    for layer in stack[1:]:
        layer_image = _resize_image(_clamp(layer["image"]), result.shape[-3:-1])
        layer_mask = layer.get("mask")
        mask = _resize_mask(layer_mask, result.shape[-3:-1], result.device, result.dtype)
        if layer.get("clip_to_below", False) and previous_mask is not None:
            below_mask = _resize_mask(
                previous_mask, result.shape[-3:-1], result.device, result.dtype
            )
            mask = mask * below_mask
        blend = apply_blend(str(layer.get("blend_mode", "normal")), result, layer_image)
        strength = float(layer.get("opacity", 1.0)) * float(layer.get("fill", 1.0))
        alpha = (mask * strength).clamp(0.0, 1.0).unsqueeze(-1)
        result = torch.lerp(result, blend, alpha)
        previous_mask = mask
    return _clamp(result)


def _resize_image(image: torch.Tensor, hw: torch.Size | tuple[int, int]) -> torch.Tensor:
    if tuple(image.shape[-3:-1]) == tuple(hw):
        return image
    x = image.permute(0, 3, 1, 2)
    resized = functional.interpolate(x, size=tuple(hw), mode="bilinear", align_corners=False)
    return resized.permute(0, 2, 3, 1).contiguous()


def _resize_mask(
    mask: torch.Tensor | None,
    hw: torch.Size | tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if mask is None:
        return torch.ones((1, int(hw[0]), int(hw[1])), device=device, dtype=dtype)
    out = mask.to(device=device, dtype=dtype)
    if out.ndim == 2:
        out = out.unsqueeze(0)
    if tuple(out.shape[-2:]) == tuple(hw):
        return out.clamp(0.0, 1.0)
    resized = functional.interpolate(
        out.unsqueeze(1), size=tuple(hw), mode="bilinear", align_corners=False
    )
    return resized.squeeze(1).clamp(0.0, 1.0)
