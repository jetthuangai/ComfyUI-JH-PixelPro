from __future__ import annotations

import pytest
import torch

from core.pyramid_blend import effective_levels, pyramid_blend


def _two_tone(h: int = 64, w: int = 96) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.full((1, h, w, 3), 0.2)
    b = torch.full((1, h, w, 3), 0.8)
    return a, b


def _step_mask(h: int = 64, w: int = 96) -> torch.Tensor:
    m = torch.zeros(1, h, w)
    m[:, :, w // 2 :] = 1.0
    return m


def _rand(*shape: int, seed: int = 0) -> torch.Tensor:
    return torch.rand(*shape, generator=torch.Generator().manual_seed(seed))


def _mid_count(t: torch.Tensor) -> int:
    row = t[0, 32, :, 0]
    return int(((row > 0.21) & (row < 0.79)).sum())


def test_shape_dtype() -> None:
    a, b = _rand(2, 40, 56, 3), _rand(2, 40, 56, 3, seed=1)
    out = pyramid_blend(a, b, _rand(2, 40, 56, seed=2), levels=4)
    assert out.shape == (2, 40, 56, 3)
    assert out.dtype == torch.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.parametrize("hw", [(37, 51), (65, 33), (100, 100), (9, 9), (1, 1), (130, 7)])
def test_arbitrary_sizes(hw: tuple[int, int]) -> None:
    h, w = hw
    a, b = _rand(1, h, w, 3), _rand(1, h, w, 3, seed=1)
    out = pyramid_blend(a, b, torch.zeros(1, h, w), levels=6)
    assert out.shape == (1, h, w, 3)
    assert torch.allclose(out, a, atol=1e-5)


def test_levels_one_is_alpha_composite() -> None:
    a, b, m = _rand(1, 33, 47, 3), _rand(1, 33, 47, 3, seed=1), _rand(1, 33, 47, seed=2)
    expected = a * (1 - m.unsqueeze(-1)) + b * m.unsqueeze(-1)
    assert torch.allclose(pyramid_blend(a, b, m, levels=1), expected, atol=1e-6)


def test_mask_zero_and_one() -> None:
    a, b = _rand(1, 50, 70, 3), _rand(1, 50, 70, 3, seed=1)
    assert torch.allclose(pyramid_blend(a, b, torch.zeros(1, 50, 70)), a, atol=1e-5)
    assert torch.allclose(pyramid_blend(a, b, torch.ones(1, 50, 70)), b, atol=1e-5)


def test_identical_images_any_mask() -> None:
    a = _rand(1, 45, 61, 3)
    out = pyramid_blend(a, a.clone(), _rand(1, 45, 61, seed=3), levels=5)
    assert torch.allclose(out, a, atol=1e-5)


def test_seam_is_smoother_than_hard_composite() -> None:
    a, b = _two_tone()
    m = _step_mask()
    hard = a * (1 - m.unsqueeze(-1)) + b * m.unsqueeze(-1)
    soft = pyramid_blend(a, b, m, levels=5)
    grad_hard = (hard[0, 32, 1:, 0] - hard[0, 32, :-1, 0]).abs().max()
    grad_soft = (soft[0, 32, 1:, 0] - soft[0, 32, :-1, 0]).abs().max()
    assert grad_soft < 0.5 * grad_hard
    assert _mid_count(soft) > _mid_count(hard)
    assert torch.allclose(soft[:, :, :8], a[:, :, :8], atol=0.02)
    assert torch.allclose(soft[:, :, -8:], b[:, :, -8:], atol=0.02)


def test_batch_broadcast() -> None:
    a, b = _rand(1, 40, 40, 3), _rand(3, 40, 40, 3, seed=1)
    m = _rand(1, 40, 40, seed=2)
    out = pyramid_blend(a, b, m)
    assert out.shape[0] == 3
    single = pyramid_blend(a, b[1:2], m)
    assert torch.allclose(out[1:2], single, atol=1e-6)


def test_bool_mask_accepted() -> None:
    a, b = _rand(1, 32, 32, 3), _rand(1, 32, 32, 3, seed=1)
    out = pyramid_blend(a, b, torch.ones(1, 32, 32, dtype=torch.bool))
    assert torch.allclose(out, b, atol=1e-5)


def test_effective_levels_clamp() -> None:
    assert effective_levels(1024, 1024, 6) == 6
    assert effective_levels(64, 64, 10) == 4  # 64 / 2**3 = 8
    assert effective_levels(7, 100, 5) == 1
    assert effective_levels(100, 100, 1) == 1


def test_validation_errors() -> None:
    a, m = torch.rand(1, 16, 16, 3), torch.rand(1, 16, 16)
    with pytest.raises(TypeError):
        pyramid_blend("x", a, m)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        pyramid_blend(a, a, "m")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="image_a"):
        pyramid_blend(torch.rand(1, 16, 16), a, m)
    with pytest.raises(ValueError, match="floating"):
        pyramid_blend(a.to(torch.int32), a, m)
    with pytest.raises(ValueError, match="mask must have shape"):
        pyramid_blend(a, a, torch.rand(1, 1, 16, 16))
    with pytest.raises(ValueError, match="spatial"):
        pyramid_blend(a, torch.rand(1, 8, 16, 3), m)
    with pytest.raises(ValueError, match="spatial"):
        pyramid_blend(a, a, torch.rand(1, 8, 16))
    with pytest.raises(ValueError, match="batches"):
        pyramid_blend(torch.rand(2, 16, 16, 3), torch.rand(3, 16, 16, 3), m)
    for bad in (0, 11, True, 2.5):
        with pytest.raises(ValueError, match="levels"):
            pyramid_blend(a, a, m, levels=bad)  # type: ignore[arg-type]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu() -> None:
    a, b, m = _rand(1, 70, 90, 3), _rand(1, 70, 90, 3, seed=1), _rand(1, 70, 90, seed=2)
    cpu = pyramid_blend(a, b, m)
    gpu = pyramid_blend(a.cuda(), b.cuda(), m.cuda())
    assert gpu.device.type == "cuda"
    assert torch.allclose(cpu, gpu.cpu(), atol=1e-4)
