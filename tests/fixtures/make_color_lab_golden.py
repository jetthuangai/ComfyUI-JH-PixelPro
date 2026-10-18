"""Regenerate ``color_lab_v160_golden.pt``: ColorLab outputs frozen at v1.6.0.

The golden guards the legacy path: new optional inputs (``tone_engine``,
``wb_temperature``, ``wb_tint``, ``params_json``) must leave old workflows'
results unchanged. Only regenerate from the v1.6.0 code
(``git show v1.6.0:core/color_lab.py``), never from a later version.

Run from the pack root: ``python tests/fixtures/make_color_lab_golden.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from core.color_lab import apply_colorlab_pipeline  # noqa: E402

GOLDEN = Path(__file__).with_name("color_lab_v160_golden.pt")

CASES: dict[str, dict[str, object]] = {
    "basic_mix": {
        "basic_exposure": 0.4,
        "basic_contrast": 20.0,
        "basic_highlights": -40.0,
        "basic_shadows": 35.0,
        "basic_whites": 10.0,
        "basic_blacks": -15.0,
        "basic_vibrance": 25.0,
        "basic_saturation": -10.0,
    },
    "local_ops": {"basic_texture": 30.0, "basic_clarity": 45.0, "basic_dehaze": 40.0},
    "negative_local": {"basic_clarity": -50.0, "basic_dehaze": -30.0},
    "linear_exposure": {"basic_exposure": -0.7, "exposure_mode": "linear"},
    "hsl_grade": {
        "hsl_red_hue": 20.0,
        "hsl_blue_sat": -40.0,
        "hsl_green_lum": 30.0,
        "grade_shadow_hue": 210.0,
        "grade_shadow_sat": 30.0,
        "grade_highlight_hue": 40.0,
        "grade_highlight_sat": 25.0,
        "grade_mid_lum": 10.0,
        "grade_shadow_bal": -20.0,
    },
    "gray_mix": {"gray_enable": True, "gray_red": 40.0, "gray_blue": -30.0, "basic_contrast": 15.0},
}


def golden_image() -> torch.Tensor:
    """Deterministic (1,24,32,3) test image: colour ramps, a dark corner and a bright patch."""

    y = torch.linspace(0.0, 1.0, 24).view(24, 1).expand(24, 32)
    x = torch.linspace(0.0, 1.0, 32).view(1, 32).expand(24, 32)
    img = torch.stack((x, 0.3 + 0.5 * y, 1.0 - 0.8 * x * y), dim=-1)
    img[:6, :8] *= 0.15
    img[16:, 22:] = torch.tensor([0.95, 0.9, 0.85])
    img[10:14, 4:28:2] += 0.08  # fine texture
    return img.clamp(0.0, 1.0).unsqueeze(0)


def main() -> None:
    image = golden_image()
    outputs = {name: apply_colorlab_pipeline(image, params) for name, params in CASES.items()}
    torch.save(outputs, GOLDEN)
    print(f"wrote {GOLDEN} ({len(outputs)} cases)")


if __name__ == "__main__":
    main()
