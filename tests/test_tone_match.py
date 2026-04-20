from __future__ import annotations

from pathlib import Path

import torch

from core.lut import apply_lut_3d, identity_hald, parse_cube
from core.tone_match import compute_lab_histogram_match, tone_match_lut


def _gradient_image(batch: int, size: int) -> torch.Tensor:
    xx = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, 1, size, 1)
    yy = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, size, 1, 1)
    return torch.cat(
        [
            xx.expand(batch, size, size, 1),
            yy.expand(batch, size, size, 1),
            ((xx + yy) / 2.0).expand(batch, size, size, 1),
        ],
        dim=-1,
    )


def test_compute_lab_histogram_match_preserves_shape_and_dtype() -> None:
    reference = _gradient_image(1, 32)
    source = _gradient_image(2, 32)

    matched = compute_lab_histogram_match(reference, source)

    assert matched.shape == source.shape
    assert matched.dtype == source.dtype
    assert torch.isfinite(matched).all()


def test_compute_lab_histogram_match_identity_when_reference_equals_source() -> None:
    source = _gradient_image(1, 32)

    matched = compute_lab_histogram_match(source, source)

    assert torch.mean(torch.abs(matched - source)).item() < 5e-3


def test_tone_match_lut_writes_cube_with_required_headers(tmp_path: Path) -> None:
    reference = _gradient_image(1, 32)

    lut_path = tone_match_lut(reference, 4, str(tmp_path / "tone_match.cube"), title="My Match")

    assert Path(lut_path).is_file()
    header = Path(lut_path).read_text(encoding="utf-8").splitlines()[:4]
    assert header[0] == 'TITLE "My Match"'
    assert header[1] == "LUT_3D_SIZE 16"
    assert header[2] == "DOMAIN_MIN 0.0 0.0 0.0"
    assert header[3] == "DOMAIN_MAX 1.0 1.0 1.0"


def test_tone_match_lut_identity_reference_stays_near_identity(tmp_path: Path) -> None:
    level = 4
    reference = identity_hald(level)

    lut_path = tone_match_lut(reference, level, str(tmp_path / "identity_match.cube"))
    parsed = parse_cube(lut_path)
    image = _gradient_image(1, 32)
    applied = apply_lut_3d(image, parsed["lut"])

    assert torch.mean(torch.abs(applied - image)).item() < 1e-2
