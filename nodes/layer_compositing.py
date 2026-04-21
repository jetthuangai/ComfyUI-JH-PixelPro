"""ComfyUI wrappers for Photoshop-style layer compositing."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from ..core.blend_modes import BLEND_MODES, compose_stack


def _resize_image(image: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    if tuple(image.shape[-3:-1]) == tuple(hw):
        return image
    x = image.permute(0, 3, 1, 2)
    out = functional.interpolate(x, size=hw, mode="bilinear", align_corners=False)
    return out.permute(0, 2, 3, 1).contiguous()


def _resize_mask(mask: torch.Tensor | None, hw: tuple[int, int]) -> torch.Tensor | None:
    if mask is None:
        return None
    out = mask
    if out.ndim == 2:
        out = out.unsqueeze(0)
    if tuple(out.shape[-2:]) == tuple(hw):
        return out.clamp(0.0, 1.0)
    return (
        functional.interpolate(out.unsqueeze(1), size=hw, mode="bilinear", align_corners=False)
        .squeeze(1)
        .clamp(0.0, 1.0)
    )


def _base_hw(stack: list[dict]) -> tuple[int, int]:
    if not stack:
        raise ValueError("LAYER_STACK must not be empty.")
    return tuple(int(v) for v in stack[0]["image"].shape[-3:-1])


class JHPixelProLayerStackStart:
    """Start a LAYER_STACK from a background image."""

    CATEGORY = "ComfyUI-JH-PixelPro/compositing"
    RETURN_TYPES = ("LAYER_STACK",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict[str, object]:  # noqa: N802
        return {"required": {"background": ("IMAGE",)}}

    def apply(self, background: torch.Tensor) -> tuple[list[dict[str, object]]]:
        layer = {
            "image": background,
            "mask": None,
            "blend_mode": "normal",
            "opacity": 1.0,
            "fill": 1.0,
            "clip_to_below": False,
            "is_group": False,
            "sub_stack": None,
        }
        return ([layer],)


class JHPixelProLayerAdd:
    """Append an image layer with blend, opacity, mask, and clipping controls."""

    CATEGORY = "ComfyUI-JH-PixelPro/compositing"
    RETURN_TYPES = ("LAYER_STACK",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict[str, object]:  # noqa: N802
        return {
            "required": {
                "stack": ("LAYER_STACK",),
                "layer_image": ("IMAGE",),
                "blend_mode": (BLEND_MODES, {"default": "normal"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fill": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "clip_to_below": ("BOOLEAN", {"default": False}),
            },
            "optional": {"layer_mask": ("MASK",)},
        }

    def apply(
        self,
        stack: list[dict[str, object]],
        layer_image: torch.Tensor,
        blend_mode: str,
        opacity: float,
        fill: float,
        clip_to_below: bool,
        layer_mask: torch.Tensor | None = None,
    ) -> tuple[list[dict[str, object]]]:
        hw = _base_hw(stack)
        layer = {
            "image": _resize_image(layer_image, hw),
            "mask": _resize_mask(layer_mask, hw),
            "blend_mode": blend_mode,
            "opacity": float(opacity),
            "fill": float(fill),
            "clip_to_below": bool(clip_to_below),
            "is_group": False,
            "sub_stack": None,
        }
        return (list(stack) + [layer],)


class JHPixelProLayerGroup:
    """Flatten a sub-stack into a grouped compositing layer."""

    CATEGORY = "ComfyUI-JH-PixelPro/compositing"
    RETURN_TYPES = ("LAYER_STACK",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict[str, object]:  # noqa: N802
        return {
            "required": {
                "parent_stack": ("LAYER_STACK",),
                "sub_stack": ("LAYER_STACK",),
                "group_blend_mode": (BLEND_MODES, {"default": "normal"}),
                "group_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {"group_mask": ("MASK",)},
        }

    def apply(
        self,
        parent_stack: list[dict[str, object]],
        sub_stack: list[dict[str, object]],
        group_blend_mode: str,
        group_opacity: float,
        group_mask: torch.Tensor | None = None,
    ) -> tuple[list[dict[str, object]]]:
        hw = _base_hw(parent_stack)
        flattened = _resize_image(compose_stack(sub_stack), hw)
        layer = {
            "image": flattened,
            "mask": _resize_mask(group_mask, hw),
            "blend_mode": group_blend_mode,
            "opacity": float(group_opacity),
            "fill": float(group_opacity),
            "clip_to_below": False,
            "is_group": True,
            "sub_stack": sub_stack,
        }
        return (list(parent_stack) + [layer],)


class JHPixelProLayerFlatten:
    """Render a LAYER_STACK into a final ComfyUI IMAGE."""

    CATEGORY = "ComfyUI-JH-PixelPro/compositing"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict[str, object]:  # noqa: N802
        return {"required": {"stack": ("LAYER_STACK",)}}

    def apply(self, stack: list[dict[str, object]]) -> tuple[torch.Tensor]:
        return (compose_stack(stack),)
