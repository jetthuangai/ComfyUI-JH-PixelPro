"""Regenerate ``refine_dust_v170_golden.pt``: Edge-Aware Refiner and Dust & Spot outputs
frozen at v1.7.0.

The golden guards the legacy path: the new optional ``guide_mode`` (Refiner) and
``fill_mode`` (Dust & Spot) inputs must leave old workflows' results unchanged at their
defaults. Only regenerate from the v1.7.0 code, never from a later version.

Run from the pack root: ``python tests/fixtures/make_refine_dust_golden.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from core.dust_spot import dust_spot_removal  # noqa: E402
from core.mask_edge_refine import edge_aware_refine  # noqa: E402

GOLDEN = Path(__file__).with_name("refine_dust_v170_golden.pt")


def inputs() -> dict[str, torch.Tensor]:
    """Deterministic test inputs: textured image with dark and bright specks, soft mask."""
    gen = torch.Generator().manual_seed(1707)
    h, w = 96, 128
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    base = torch.stack([0.55 + 0.2 * xx, 0.45 + 0.1 * yy, 0.35 + 0.15 * xx * yy], dim=-1)
    image = (base + 0.04 * torch.randn(h, w, 3, generator=gen)).clamp(0, 1)
    for y, x in ((20, 30), (50, 90), (70, 40)):
        image[y - 2 : y + 3, x - 2 : x + 3] *= 0.4
    image[40:43, 60:63] = 0.97
    mask = ((xx - 0.5) ** 2 + (yy - 0.5) ** 2 < 0.1).float()
    return {"image": image.unsqueeze(0), "mask": mask.unsqueeze(0)}


def outputs() -> dict[str, torch.Tensor]:
    data = inputs()
    image, mask = data["image"], data["mask"]
    out = {
        "refine_default": edge_aware_refine(mask, image),
        "refine_r4_feather": edge_aware_refine(mask, image, radius=4, eps=0.01, feather_sigma=2.0),
    }
    for mode in ("dark", "bright", "both"):
        cleaned, spots = dust_spot_removal(image, spot_size=10, sensitivity=0.6, mode=mode)
        out[f"dust_{mode}"] = cleaned
        out[f"dust_{mode}_spots"] = spots
    out["dust_default"] = dust_spot_removal(image)[0]
    return out


if __name__ == "__main__":
    torch.save(outputs(), GOLDEN)
    print(f"wrote {GOLDEN}")
