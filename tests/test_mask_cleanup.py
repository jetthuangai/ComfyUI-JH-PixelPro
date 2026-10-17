from __future__ import annotations

import time

import pytest
import torch

from core.mask_cleanup import mask_cleanup


def _blank(h: int = 64, w: int = 64) -> torch.Tensor:
    return torch.zeros(1, h, w)


def test_speck_removed_big_blob_kept() -> None:
    m = _blank()
    m[0, 10:30, 10:30] = 1.0  # 400 px blob
    m[0, 50, 50] = 1.0  # 1 px speck
    m[0, 40:43, 40:43] = 1.0  # 9 px speck
    out = mask_cleanup(m, min_island_area=64, max_hole_area=0)
    assert out.sum().item() == 400
    assert out[0, 10:30, 10:30].min().item() == 1.0
    assert out[0, 50, 50].item() == 0.0


def test_island_area_boundary_is_inclusive_keep() -> None:
    m = _blank()
    m[0, 5:8, 5:8] = 1.0  # exactly 9 px
    assert mask_cleanup(m, min_island_area=9, max_hole_area=0).sum().item() == 9
    assert mask_cleanup(m, min_island_area=10, max_hole_area=0).sum().item() == 0


def test_hole_filled_border_background_not_filled() -> None:
    m = _blank()
    m[0, 10:40, 10:40] = 1.0
    m[0, 20:22, 20:22] = 0.0  # 4 px interior hole
    out = mask_cleanup(m, min_island_area=0, max_hole_area=64)
    assert out[0, 20:22, 20:22].min().item() == 1.0
    assert out.sum().item() == 900
    # big hole stays
    m2 = m.clone()
    m2[0, 15:35, 15:35] = 0.0
    out2 = mask_cleanup(m2, min_island_area=0, max_hole_area=64)
    assert out2.sum().item() == m2.sum().item()


def test_border_touching_background_pocket_not_filled() -> None:
    m = torch.ones(1, 32, 32)
    m[0, 0:2, 0:2] = 0.0  # small background in the corner
    out = mask_cleanup(m, min_island_area=0, max_hole_area=1000)
    assert out[0, 0:2, 0:2].max().item() == 0.0
    assert out.sum().item() == 32 * 32 - 4


def test_disabled_steps_equal_binarized_input() -> None:
    torch.manual_seed(0)
    m = torch.rand(2, 40, 48)
    out = mask_cleanup(m, threshold=0.4, min_island_area=0, max_hole_area=0, feather_px=0.0)
    assert torch.equal(out, (m > 0.4).float())
    assert out.dtype == torch.float32


def test_threshold_is_strict_greater() -> None:
    m = torch.full((1, 8, 8), 0.5)
    assert mask_cleanup(m, threshold=0.5, min_island_area=0, max_hole_area=0).sum().item() == 0


def test_feather_ramp_monotonic_width_and_half_at_edge() -> None:
    f = 4.0
    m = _blank(16, 64)
    m[0, :, :32] = 1.0
    out = mask_cleanup(m, min_island_area=0, max_hole_area=0, feather_px=f)
    row = out[0, 8]
    assert torch.all(row[1:] <= row[:-1] + 1e-6)  # non-increasing left to right
    assert row[0].item() == 1.0 and row[-1].item() == 0.0
    # edge pixels straddle 0.5 symmetrically
    assert row[31].item() == pytest.approx(0.5 + 0.5 / (2 * f))
    assert row[32].item() == pytest.approx(0.5 - 0.5 / (2 * f))
    assert row[31].item() + row[32].item() == pytest.approx(1.0)
    # ramp (strictly between 0 and 1) spans 2 * feather_px pixels
    assert int(((row > 0) & (row < 1)).sum().item()) == int(2 * f)
    # 0.5 contour sits on the binary boundary
    assert torch.equal(row > 0.5, m[0, 8] > 0.5)


def test_feather_zero_stays_binary() -> None:
    m = _blank()
    m[0, 10:30, 10:30] = 1.0
    out = mask_cleanup(m, feather_px=0.0)
    assert set(out.unique().tolist()) <= {0.0, 1.0}


def test_connectivity_4_vs_8_on_diagonal() -> None:
    m = _blank(16, 16)
    m[0, 4, 4] = m[0, 5, 5] = m[0, 6, 6] = 1.0  # 3 diagonal pixels
    out8 = mask_cleanup(m, min_island_area=3, max_hole_area=0, connectivity=8)
    out4 = mask_cleanup(m, min_island_area=3, max_hole_area=0, connectivity=4)
    assert out8.sum().item() == 3
    assert out4.sum().item() == 0


def test_all_zero_and_all_one() -> None:
    z = torch.zeros(1, 20, 20)
    o = torch.ones(1, 20, 20)
    assert mask_cleanup(z, feather_px=3.0).sum().item() == 0
    assert torch.equal(mask_cleanup(o, feather_px=3.0), o)
    assert torch.equal(mask_cleanup(o), o)  # no hole fill at the border
    assert mask_cleanup(z).sum().item() == 0


def test_batch_per_item_and_bool_input() -> None:
    m = torch.zeros(2, 32, 32, dtype=torch.bool)
    m[0, 4:20, 4:20] = True
    m[1, 4, 4] = True
    out = mask_cleanup(m)
    assert out.shape == (2, 32, 32)
    assert out[0].sum().item() == 256
    assert out[1].sum().item() == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": -0.1},
        {"threshold": 1.5},
        {"threshold": float("nan")},
        {"threshold": True},
        {"min_island_area": -1},
        {"min_island_area": 1_000_001},
        {"min_island_area": 1.5},
        {"max_hole_area": -1},
        {"max_hole_area": True},
        {"feather_px": -1.0},
        {"feather_px": 257.0},
        {"feather_px": float("inf")},
        {"connectivity": 6},
        {"connectivity": 8.0 + 0.5},
    ],
)
def test_invalid_params(kwargs) -> None:
    with pytest.raises(ValueError):
        mask_cleanup(torch.zeros(1, 8, 8), **kwargs)


def test_invalid_mask() -> None:
    with pytest.raises(TypeError):
        mask_cleanup([[0.0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        mask_cleanup(torch.zeros(8, 8))
    with pytest.raises(ValueError):
        mask_cleanup(torch.zeros(1, 8, 8, dtype=torch.int64))
    with pytest.raises(ValueError):
        mask_cleanup(torch.zeros(1, 0, 8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_device_preserved() -> None:
    out = mask_cleanup(torch.zeros(1, 16, 16, device="cuda"))
    assert out.device.type == "cuda"


def test_2048_runs_fast_on_cpu() -> None:
    torch.manual_seed(0)
    m = torch.zeros(1, 2048, 2048)
    m[0, 300:1700, 300:1700] = 1.0
    m[0, torch.randint(0, 2048, (500,)), torch.randint(0, 2048, (500,))] = 1.0
    start = time.perf_counter()
    out = mask_cleanup(m, feather_px=8.0)
    elapsed = time.perf_counter() - start
    assert out.shape == m.shape
    assert elapsed < 5.0  # generous; typically well under 1 s


def test_hole_inside_diagonal_outline_is_filled() -> None:
    # An 8-connected diamond ring encloses its centre only if holes use 4-connectivity.
    mask = torch.zeros(1, 15, 15)
    for i in range(6):
        for y, x in ((7 - i, 2 + i), (7 + i, 2 + i), (7 - i, 12 - i), (7 + i, 12 - i)):
            mask[0, y, x] = 1.0
    out = mask_cleanup(mask, min_island_area=0, max_hole_area=100, connectivity=8)
    assert out[0, 7, 7] == 1.0
    assert out[0, 0, 0] == 0.0
