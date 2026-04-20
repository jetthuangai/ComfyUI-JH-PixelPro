"""Functional tests for ``core.lut`` (identity HALD + Adobe Cube 1.0 exporter)."""

from __future__ import annotations

import math

import pytest
import torch

from core.lut import export_cube, identity_hald

EPS_CORNER = 1e-6
EPS_ROUNDTRIP = 1e-5


def _parse_cube_body(path: str) -> tuple[list[str], list[tuple[float, float, float]]]:
    """Split a .cube file into (header_lines, body_triplets).

    Header = every line until the first line that parses as ``r g b`` floats.
    Body = every non-blank line after that parses as three floats.
    """
    with open(path, encoding="utf-8") as fh:
        raw = fh.read().splitlines()

    header: list[str] = []
    body: list[tuple[float, float, float]] = []
    in_body = False
    for line in raw:
        stripped = line.strip()
        parts = stripped.split()
        is_triplet = False
        if len(parts) == 3:
            try:
                triplet = (float(parts[0]), float(parts[1]), float(parts[2]))
                is_triplet = True
            except ValueError:
                is_triplet = False
        if is_triplet:
            in_body = True
            body.append(triplet)
        else:
            if in_body:
                # Blank lines inside body are tolerated (some writers add them).
                if stripped:
                    header.append(line)
            else:
                header.append(line)
    return header, body


def test_identity_hald_shape_level_8() -> None:
    hald = identity_hald(8)
    assert hald.shape == (1, 512, 512, 3)
    assert hald.dtype == torch.float32
    assert float(hald.min()) >= 0.0
    assert float(hald.max()) <= 1.0


def test_identity_hald_black_corner_level_8() -> None:
    hald = identity_hald(8)
    corner = hald[0, 0, 0]
    assert torch.allclose(corner, torch.tensor([0.0, 0.0, 0.0]), atol=EPS_CORNER)


def test_identity_hald_white_corner_level_8() -> None:
    hald = identity_hald(8)
    corner = hald[0, 511, 511]
    assert torch.allclose(corner, torch.tensor([1.0, 1.0, 1.0]), atol=EPS_CORNER)


def test_roundtrip_identity_cube_body(tmp_path) -> None:
    level = 8
    n = level * level  # 64
    hald = identity_hald(level)
    path = str(tmp_path / "identity.cube")
    export_cube(hald, level, path, title="roundtrip")

    _, body = _parse_cube_body(path)
    assert len(body) == n**3 == 262144

    # Spot-check 4 entries distributed across the cube — full 262144 compare is slow.
    for idx in (0, 37, 64 * 64 - 1, 262143):
        b_idx, rem = divmod(idx, n * n)
        g_idx, r_idx = divmod(rem, n)
        expected = (r_idx / (n - 1), g_idx / (n - 1), b_idx / (n - 1))
        got = body[idx]
        for e, g in zip(expected, got, strict=True):
            assert math.isclose(g, e, abs_tol=EPS_ROUNDTRIP), (
                f"idx={idx} expected={expected} got={got}"
            )


def test_identity_hald_shape_level_12() -> None:
    hald = identity_hald(12)
    assert hald.shape == (1, 1728, 1728, 3)
    corner = hald[0, 1727, 1727]
    assert torch.allclose(corner, torch.tensor([1.0, 1.0, 1.0]), atol=EPS_CORNER)


def test_export_cube_header_and_body_count(tmp_path) -> None:
    level = 8
    hald = identity_hald(level) * 0.9 + 0.05  # arbitrary "graded" hald
    path = str(tmp_path / "graded.cube")
    export_cube(hald, level, path, title="My Title")

    header, body = _parse_cube_body(path)
    assert any('TITLE "My Title"' in line for line in header)
    assert any("LUT_3D_SIZE 64" in line for line in header)
    assert any("DOMAIN_MIN 0.0 0.0 0.0" in line for line in header)
    assert any("DOMAIN_MAX 1.0 1.0 1.0" in line for line in header)
    assert len(body) == 64**3 == 262144


def test_export_cube_shape_mismatch_raises(tmp_path) -> None:
    bad = torch.zeros((1, 256, 256, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match=r"hald shape must be \(512,512,3\)"):
        export_cube(bad, 8, str(tmp_path / "nope.cube"))


def test_export_cube_non_finite_raises(tmp_path) -> None:
    hald = identity_hald(8).clone()
    hald[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite values"):
        export_cube(hald, 8, str(tmp_path / "nan.cube"))
