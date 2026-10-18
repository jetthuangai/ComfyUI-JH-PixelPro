"""HDR merge core (``core/hdr_merge.py``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from core.hdr_merge import hdr_merge

_CAFE = Path(__file__).resolve().parents[1] / "workflows" / "sample_cafe.jpg"


def _scene(h: int = 160, w: int = 240) -> torch.Tensor:
    """Linear radiance: a dark textured room (left) up to a bright textured window (right)."""
    gen = torch.Generator().manual_seed(0)
    blocks = torch.rand(1, 1, h // 8, w // 8, generator=gen)  # structure MTB can lock onto
    blocks = torch.nn.functional.interpolate(blocks, size=(h, w), mode="nearest")[0, 0, ..., None]
    texture = 0.6 + 0.4 * blocks
    # Brightness rises smoothly from a dark room (left) to a bright window (right), so every
    # pair of exposures shares well-exposed midtones, as real scenes do.
    radiance = (2.0 ** torch.linspace(-6.0, 2.0, w)).view(1, w, 1).expand(h, w, 1)
    return (radiance * texture).expand(h, w, 3)


def _shot(radiance: torch.Tensor, ev: float) -> torch.Tensor:
    return (radiance * 2.0**ev).clamp(0, 1) ** (1 / 2.2)


def _bracket() -> tuple[torch.Tensor, torch.Tensor]:
    scene = _scene()
    return torch.stack([_shot(scene, ev) for ev in (-3.0, 0.0, 3.0)]), scene


def _detail(x: torch.Tensor) -> float:
    """Local texture: mean distance from the 9x9 neighbourhood mean."""
    chw = x.permute(2, 0, 1)[None]
    local = torch.nn.functional.avg_pool2d(chw, 9, stride=1, padding=4, count_include_pad=False)
    return float((chw - local).abs().mean())


def test_fusion_keeps_detail_in_shadows_and_highlights() -> None:
    shots, _ = _bracket()
    out = hdr_merge(shots[[2, 0, 1]], deghost=False)  # order must not matter
    assert out.shape == (1, 160, 240, 3)
    dark, bright = out[0, :, :60], out[0, :, 180:]
    mid_dark, mid_bright = shots[1, :, :60], shots[1, :, 180:]
    assert _detail(dark) > 1.4 * _detail(mid_dark)  # shadows open up
    assert _detail(bright) > 1.3 * _detail(mid_bright)  # window not clipped
    assert float(bright.mean()) > float(dark.mean())  # scene order kept


def _cafe_bracket() -> torch.Tensor:
    """Three exposures (-2, 0, +2 EV) simulated from the sample cafe photo."""
    arr = np.asarray(Image.open(_CAFE).convert("RGB").resize((300, 400)), dtype=np.float32)
    linear = (torch.from_numpy(arr) / 255.0) ** 2.2
    return torch.stack([(linear * 2.0**ev).clamp(0, 1) ** (1 / 2.2) for ev in (-2, 0, 2)])


def test_alignment_fixes_a_handheld_shift() -> None:
    shots = _cafe_bracket()
    moved = shots.clone()
    moved[2] = torch.roll(moved[2], shifts=(3, -4), dims=(0, 1))
    aligned = hdr_merge(moved, align=True, deghost=False)
    reference = hdr_merge(shots, align=False, deghost=False)
    unaligned = hdr_merge(moved, align=False, deghost=False)
    inner = (slice(None), slice(8, -8), slice(8, -8))
    err_aligned = float((aligned - reference)[inner].abs().mean())
    err_unaligned = float((unaligned - reference)[inner].abs().mean())
    assert err_aligned < 0.5 * err_unaligned


def test_deghost_suppresses_a_moving_object() -> None:
    shots = _cafe_bracket()
    ghosted = shots.clone()
    ghosted[2, 250:300, 60:120] = torch.tensor([0.9, 0.2, 0.2])  # appears in one shot only
    with_deghost = hdr_merge(ghosted, align=False, deghost=True)
    without = hdr_merge(ghosted, align=False, deghost=False)
    clean = hdr_merge(shots, align=False, deghost=False)
    region = (0, slice(258, 292), slice(68, 112))
    err_with = float((with_deghost[region] - clean[region]).abs().mean())
    err_without = float((without[region] - clean[region]).abs().mean())
    assert err_with < 0.5 * err_without


def test_validates() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        hdr_merge(torch.rand(1, 16, 16, 3))
    with pytest.raises(ValueError, match="shape"):
        hdr_merge(torch.rand(2, 16, 16))


def test_all_white_bracket_and_tiny_images() -> None:
    white = torch.ones(3, 32, 32, 3)
    assert torch.equal(hdr_merge(white), white[:1])
    with pytest.raises(ValueError, match="at least 16"):
        hdr_merge(torch.rand(2, 4, 4, 3))
