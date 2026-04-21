"""ComfyUI wrapper for ``core.tone_curve.tone_curve`` (8-control-point Catmull-Rom LUT)."""

from __future__ import annotations

import json

import torch

from ..core import tone_curve

_PRESETS: dict[str, list[list[float]]] = {
    "linear": [
        [0.0, 0.0],
        [0.14, 0.14],
        [0.29, 0.29],
        [0.43, 0.43],
        [0.57, 0.57],
        [0.71, 0.71],
        [0.86, 0.86],
        [1.0, 1.0],
    ],
    "s_curve_mild": [
        [0.0, 0.0],
        [0.15, 0.10],
        [0.30, 0.25],
        [0.45, 0.43],
        [0.55, 0.57],
        [0.70, 0.75],
        [0.85, 0.90],
        [1.0, 1.0],
    ],
    "s_curve_strong": [
        [0.0, 0.0],
        [0.15, 0.05],
        [0.30, 0.18],
        [0.45, 0.38],
        [0.55, 0.62],
        [0.70, 0.82],
        [0.85, 0.95],
        [1.0, 1.0],
    ],
    "lift_shadows": [
        [0.0, 0.0],
        [0.10, 0.15],
        [0.25, 0.30],
        [0.40, 0.45],
        [0.55, 0.58],
        [0.70, 0.72],
        [0.85, 0.87],
        [1.0, 1.0],
    ],
    "crush_blacks": [
        [0.0, 0.0],
        [0.15, 0.05],
        [0.30, 0.22],
        [0.45, 0.40],
        [0.60, 0.57],
        [0.75, 0.75],
        [0.90, 0.90],
        [1.0, 1.0],
    ],
}

_PRESET_CHOICES = [*_PRESETS.keys(), "custom"]
_CHANNEL_CHOICES = ["rgb_master", "r", "g", "b"]
_DEFAULT_CUSTOM_JSON = (
    "[[0.0, 0.0], [0.15, 0.10], [0.30, 0.25], [0.45, 0.43], "
    "[0.55, 0.57], [0.70, 0.75], [0.85, 0.90], [1.0, 1.0]]"
)


def _parse_control_points(points_json: str) -> torch.Tensor:
    try:
        raw = json.loads(points_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"points_json must be valid JSON: {exc.msg}. "
            "Expected [[x1,y1],...,[x8,y8]] in [0,1]^2 with endpoints (0,0)+(1,1)."
        ) from exc

    if not isinstance(raw, list) or len(raw) != 8:
        actual = len(raw) if isinstance(raw, list) else type(raw).__name__
        raise ValueError(f"points_json must be a list of exactly 8 control points, got {actual}.")

    for index, point in enumerate(raw):
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"points_json[{index}] must be a [x, y] pair, got {point!r}.")

    try:
        tensor = torch.tensor(raw, dtype=torch.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"points_json could not be coerced to a tensor: {exc}.") from exc

    return tensor


class JHPixelProToneCurve:
    """Apply an 8-control-point Catmull-Rom tone curve via a 1024-step LUT."""

    CATEGORY = "ComfyUI-JH-PixelPro/color"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_toned",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802 — ComfyUI node contract mandates UPPER_CASE.
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (
                    _PRESET_CHOICES,
                    {
                        "default": "linear",
                        "tooltip": (
                            "Tone curve preset. s_curve_mild / s_curve_strong add "
                            "contrast; lift_shadows opens up dark regions; "
                            "crush_blacks deepens shadows for dramatic look; "
                            "linear = identity. Choose 'custom' to paste your own "
                            "8-point JSON below."
                        ),
                    },
                ),
                "channel": (
                    _CHANNEL_CHOICES,
                    {
                        "default": "rgb_master",
                        "tooltip": (
                            "rgb_master = apply the curve equally to R, G, B "
                            "(global contrast). r / g / b = apply only to that "
                            "channel (color balance / white-balance correction)."
                        ),
                    },
                ),
                "points_json": (
                    "STRING",
                    {
                        "default": _DEFAULT_CUSTOM_JSON,
                        "multiline": True,
                        "tooltip": (
                            "Custom 8 control points [[x1,y1],...,[x8,y8]] in "
                            "[0,1]^2. Endpoints MUST be (0,0) and (1,1). x must "
                            "be strictly increasing (monotone). Used only when "
                            "preset = custom; ignored otherwise."
                        ),
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Blend factor. 0 = identity (bypass), 1 = full curve. "
                            "Use 0.5–0.8 for subtle grading."
                        ),
                    },
                ),
            },
        }

    def run(
        self,
        image: torch.Tensor,
        preset: str,
        channel: str,
        points_json: str,
        strength: float,
    ) -> tuple[torch.Tensor]:
        if preset == "custom":
            points = _parse_control_points(points_json)
        else:
            points = torch.tensor(_PRESETS[preset], dtype=torch.float32)
        with torch.no_grad():
            image_bchw = image.permute(0, 3, 1, 2).contiguous()
            out_bchw = tone_curve(
                image_bchw,
                control_points=points.to(image_bchw.device),
                channel=channel,
                strength=float(strength),
            )
            out = out_bchw.permute(0, 2, 3, 1).contiguous()
        return (out,)
