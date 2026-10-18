"""ComfyUI wrapper: apply a Lightroom / Camera Raw preset (.xmp, .lrtemplate) via ColorLab."""

from __future__ import annotations

import torch

from ..core.color_lab import TONE_ENGINES, apply_preset, params_to_json
from ..core.lr_preset import list_presets, load_lightroom_preset, preset_dirs, resolve_preset
from ..core.lr_preset_library import add_recent, remember_source

_NONE = "(no presets found)"


class JHPixelProLightroomPreset:
    """Read a Lightroom preset, apply it with ColorLab and output its ColorLab settings."""

    CATEGORY = "ComfyUI-JH-PixelPro/color"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "params_json", "report")
    OUTPUT_TOOLTIPS = (
        "Image with the preset applied.",
        "The preset as ColorLab settings: connect to ColorLab or Apply ColorLab Preset.",
        "Which Lightroom settings were applied and which are not supported.",
    )
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        presets = list_presets() or [_NONE]
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to grade."}),
                "preset": (
                    presets,
                    {"tooltip": f"Presets in {preset_dirs()[0]} (.xmp / .lrtemplate)."},
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Like Lightroom's preset Amount. 0 = original.",
                    },
                ),
            },
            "optional": {
                "preset_path": (
                    "STRING",
                    {"default": "", "tooltip": "Full path to a preset file; overrides the list."},
                ),
                "tone_engine": (
                    list(reversed(TONE_ENGINES)),
                    {
                        "default": "local",
                        "tooltip": "local = edge-aware tone tools, closer to Lightroom.",
                    },
                ),
                "mask": ("MASK", {"tooltip": "Optional mask limiting where the preset applies."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, preset: str, preset_path: str = "", **_: object) -> str:  # noqa: N802
        # Re-run when the preset file is edited on disk.
        try:
            path = resolve_preset(preset, preset_path)
            return f"{path}:{path.stat().st_mtime_ns}"
        except (ValueError, OSError):
            return f"{preset}:{preset_path}"

    def apply(
        self,
        image: torch.Tensor,
        preset: str,
        strength: float,
        preset_path: str = "",
        tone_engine: str = "local",
        mask: torch.Tensor | None = None,
        unique_id: str | None = None,
    ) -> dict:
        if preset == _NONE and not preset_path.strip():
            raise ValueError(
                f"No preset selected. Copy .xmp / .lrtemplate presets into {preset_dirs()[0]} "
                "and refresh, or fill preset_path."
            )
        loaded = load_lightroom_preset(resolve_preset(preset, preset_path))
        # The browser's thumbnails show presets on the image this node last processed.
        ui = {"jh_lr_source": [remember_source(str(unique_id), image)]} if unique_id else {}
        if not preset_path.strip():
            add_recent(preset)
        # Lightroom renders exposure in linear light.
        base = params_to_json(
            {**loaded.params, "tone_engine": tone_engine, "exposure_mode": "linear"}
        )
        out, params_json = apply_preset(image, base, float(strength))
        if float(strength) == 0.0:
            return {"ui": ui, "result": (image, params_json, loaded.report())}
        if mask is not None:
            if mask.ndim != 3 or mask.shape[1:] != image.shape[1:3]:
                raise ValueError("mask must have shape (B,H,W) matching the image.")
            weight = mask.to(device=out.device, dtype=out.dtype).clamp(0.0, 1.0).unsqueeze(-1)
            out = image + (out - image) * weight
        return {"ui": ui, "result": (out, params_json, loaded.report())}
