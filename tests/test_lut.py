"""Functional tests for ``core.lut`` (identity HALD + Adobe Cube 1.0 exporter)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from core.lut import apply_lut_3d, export_cube, identity_hald, parse_cube

EPS_CORNER = 1e-6
EPS_ROUNDTRIP = 1e-5


def _identity_lut_grid(size: int) -> torch.Tensor:
    coords = torch.linspace(0.0, 1.0, size, dtype=torch.float32)
    blue, green, red = torch.meshgrid(coords, coords, coords, indexing="ij")
    return torch.stack([red, green, blue], dim=-1)


def _invert_lut_grid(size: int) -> torch.Tensor:
    coords = torch.linspace(0.0, 1.0, size, dtype=torch.float32)
    blue, green, red = torch.meshgrid(coords, coords, coords, indexing="ij")
    return torch.stack([1.0 - red, 1.0 - green, 1.0 - blue], dim=-1)


def _gradient_image(batch: int, size: int) -> torch.Tensor:
    xx = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, 1, size, 1)
    yy = torch.linspace(0.0, 1.0, size, dtype=torch.float32).view(1, size, 1, 1)
    image = torch.cat(
        [
            xx.expand(batch, size, size, 1),
            yy.expand(batch, size, size, 1),
            ((xx + yy) / 2.0).expand(batch, size, size, 1),
        ],
        dim=-1,
    )
    return image


def _write_cube(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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


def test_parse_cube_identity_l8(tmp_path) -> None:
    path = tmp_path / "identity_l8.cube"
    export_cube(identity_hald(8), 8, str(path), title="identity")

    result = parse_cube(path)

    assert result["size"] == 64
    assert result["lut"].shape == (64, 64, 64, 3)
    assert torch.allclose(result["lut"][0, 0, 0], torch.tensor([0.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(result["lut"][63, 63, 63], torch.tensor([1.0, 1.0, 1.0]), atol=1e-6)


def test_parse_cube_header_title_and_defaults(tmp_path) -> None:
    path = _write_cube(
        tmp_path / "header_defaults.cube",
        [
            'TITLE "Test LUT"',
            "LUT_3D_SIZE 2",
            "0.0 0.0 0.0",
            "1.0 0.0 0.0",
            "0.0 1.0 0.0",
            "1.0 1.0 0.0",
            "0.0 0.0 1.0",
            "1.0 0.0 1.0",
            "0.0 1.0 1.0",
            "1.0 1.0 1.0",
        ],
    )

    result = parse_cube(path)

    assert result["title"] == "Test LUT"
    assert torch.allclose(result["domain_min"], torch.zeros(3, dtype=torch.float32))
    assert torch.allclose(result["domain_max"], torch.ones(3, dtype=torch.float32))


def test_parse_cube_custom_domain(tmp_path) -> None:
    path = _write_cube(
        tmp_path / "custom_domain.cube",
        [
            "LUT_3D_SIZE 2",
            "DOMAIN_MIN -0.1 -0.1 -0.1",
            "DOMAIN_MAX 1.1 1.1 1.1",
            "0.0 0.0 0.0",
            "1.0 0.0 0.0",
            "0.0 1.0 0.0",
            "1.0 1.0 0.0",
            "0.0 0.0 1.0",
            "1.0 0.0 1.0",
            "0.0 1.0 1.0",
            "1.0 1.0 1.0",
        ],
    )

    result = parse_cube(path)

    assert torch.allclose(result["domain_min"], torch.full((3,), -0.1, dtype=torch.float32))
    assert torch.allclose(result["domain_max"], torch.full((3,), 1.1, dtype=torch.float32))


def test_parse_cube_missing_size_raises(tmp_path) -> None:
    path = _write_cube(
        tmp_path / "missing_size.cube",
        [
            "DOMAIN_MIN 0.0 0.0 0.0",
            "0.0 0.0 0.0",
        ],
    )

    with pytest.raises(ValueError, match="missing LUT_3D_SIZE header"):
        parse_cube(path)


def test_parse_cube_body_mismatch_raises(tmp_path) -> None:
    path = _write_cube(
        tmp_path / "bad_body_count.cube",
        ["LUT_3D_SIZE 4"] + ["0.0 0.0 0.0"] * 30,
    )

    with pytest.raises(ValueError, match="expected 64 body lines, got 30"):
        parse_cube(path)


def test_parse_cube_malformed_line_raises(tmp_path) -> None:
    path = _write_cube(
        tmp_path / "bad_line.cube",
        [
            "LUT_3D_SIZE 2",
            "0.0 0.0 0.0",
            "1.0 0.0 0.0",
            "0.0 1.0 0.0",
            "1.0 1.0 0.0",
            "0.0 0.0 1.0",
            "not a number 0.5 0.5",
            "0.0 1.0 1.0",
            "1.0 1.0 1.0",
        ],
    )

    with pytest.raises(ValueError, match=r"line 7:"):
        parse_cube(path)


def test_parse_cube_comments_and_blanks(tmp_path) -> None:
    clean = _write_cube(
        tmp_path / "clean.cube",
        [
            'TITLE "Clean"',
            "LUT_3D_SIZE 2",
            "DOMAIN_MIN 0.0 0.0 0.0",
            "DOMAIN_MAX 1.0 1.0 1.0",
            "0.0 0.0 0.0",
            "1.0 0.0 0.0",
            "0.0 1.0 0.0",
            "1.0 1.0 0.0",
            "0.0 0.0 1.0",
            "1.0 0.0 1.0",
            "0.0 1.0 1.0",
            "1.0 1.0 1.0",
        ],
    )
    noisy = _write_cube(
        tmp_path / "noisy.cube",
        [
            "# comment",
            "",
            'TITLE "Clean"',
            "LUT_3D_SIZE 2",
            "",
            "DOMAIN_MIN 0.0 0.0 0.0",
            "DOMAIN_MAX 1.0 1.0 1.0",
            "",
            "0.0\t0.0\t0.0   ",
            "1.0 0.0 0.0",
            "",
            "# comment in body area",
            "0.0 1.0 0.0",
            "1.0 1.0 0.0",
            "0.0 0.0 1.0",
            "1.0 0.0 1.0",
            "0.0 1.0 1.0",
            "1.0 1.0 1.0",
        ],
    )

    clean_result = parse_cube(clean)
    noisy_result = parse_cube(noisy)

    assert clean_result["title"] == noisy_result["title"]
    assert torch.allclose(clean_result["lut"], noisy_result["lut"], atol=1e-6)
    assert torch.allclose(clean_result["domain_min"], noisy_result["domain_min"], atol=1e-6)
    assert torch.allclose(clean_result["domain_max"], noisy_result["domain_max"], atol=1e-6)


@pytest.mark.parametrize("batch,size", [(1, 64), (2, 128)])
def test_apply_lut_3d_identity_invariant(batch: int, size: int, rng: torch.Generator) -> None:
    image = torch.rand((batch, size, size, 3), generator=rng, dtype=torch.float32)
    identity_lut = _identity_lut_grid(8)

    output = apply_lut_3d(image, identity_lut)

    assert torch.mean(torch.abs(output - image)).item() <= 1e-5


def test_apply_lut_3d_invert_known_transform() -> None:
    image = _gradient_image(1, 64)
    invert_lut = _invert_lut_grid(16)

    output = apply_lut_3d(image, invert_lut)

    assert torch.allclose(output, 1.0 - image, atol=1e-3)


def test_apply_lut_3d_strength_zero_noop() -> None:
    image = _gradient_image(1, 32)
    invert_lut = _invert_lut_grid(8)

    output = apply_lut_3d(image, invert_lut, strength=0.0)

    assert torch.max(torch.abs(output - image)).item() < 1e-6


def test_apply_lut_3d_mask_gating() -> None:
    image = _gradient_image(1, 32)
    invert_lut = _invert_lut_grid(8)

    zero_mask = torch.zeros((1, 32, 32), dtype=torch.float32)
    zero_output = apply_lut_3d(image, invert_lut, strength=1.0, mask=zero_mask)
    assert torch.allclose(zero_output, image, atol=1e-6)

    half_mask = torch.full((1, 32, 32), 0.5, dtype=torch.float32)
    half_output = apply_lut_3d(image, invert_lut, strength=1.0, mask=half_mask)
    expected = 0.5 * image + 0.5 * (1.0 - image)
    assert torch.allclose(half_output, expected, atol=1e-5)


def test_apply_lut_3d_domain_clamp() -> None:
    image = torch.full((1, 4, 4, 3), 1.5, dtype=torch.float32)
    identity_lut = _identity_lut_grid(8)

    output = apply_lut_3d(
        image,
        identity_lut,
        domain_min=torch.zeros(3, dtype=torch.float32),
        domain_max=torch.ones(3, dtype=torch.float32),
    )

    assert torch.isfinite(output).all()
    assert torch.allclose(output, torch.ones_like(output), atol=1e-6)


def test_apply_lut_3d_shape_mismatch_raises() -> None:
    image = _gradient_image(1, 16)
    bad_lut = torch.zeros((8, 8, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match=r"lut_grid must have shape \(N,N,N,3\)"):
        apply_lut_3d(image, bad_lut)


@pytest.mark.parametrize("strength", [-0.1, 1.5])
def test_apply_lut_3d_strength_out_of_range_raises(strength: float) -> None:
    image = _gradient_image(1, 16)
    identity_lut = _identity_lut_grid(8)

    with pytest.raises(ValueError, match="strength must be in"):
        apply_lut_3d(image, identity_lut, strength=strength)
